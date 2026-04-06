# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import re
import math
from itertools import chain

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer


def _resolve_checkpoint_path(log_path: str, run_dir: str = ".*", checkpoint: str = "model_.*.pt") -> str:
    log_path = os.path.abspath(log_path)
    runs = [os.path.join(log_path, run.name) for run in os.scandir(log_path) if run.is_dir() and re.match(run_dir, run.name)]
    if not runs:
        raise ValueError(f"No runs present in the directory: '{log_path}' match '{run_dir}'.")
    runs.sort()
    run_path = runs[-1]
    checkpoints = [name for name in os.listdir(run_path) if re.match(checkpoint, name)]
    if not checkpoints:
        raise ValueError(f"No checkpoints in the directory: '{run_path}' match '{checkpoint}'.")
    checkpoints.sort(key=lambda name: f"{name:0>15}")
    return os.path.join(run_path, checkpoints[-1])


def _cosine_anneal(start: float, end: float, progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


class DaggerPPO(PPO):
    """Teacher-guided PPO with behavior cloning loss and future-aware actor support."""

    teacher: MLPModel

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        teacher: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 1e-4,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        dagger_coef: float = 0.2,
        dagger_coef_min: float = 0.1,
        dagger_coef_anneal_steps: int = 60000,
        dagger_coef_anneal_updates: int = 0,
        behavior_loss_type: str = "mse",
        warm_start_steps: int = 0,
        warm_start_updates: int = 0,
        teacher_rollout_prob_start: float = 1.0,
        teacher_rollout_prob_end: float = 0.0,
        teacher_rollout_anneal_steps: int = 0,
        teacher_rollout_anneal_updates: int = 0,
        future_consistency_coef: float = 0.0,
        future_consistency_keep_steps: int = 0,
        future_weight_decay: float = 1.0,
        grad_penalty_coef_schedule: tuple[float, float, int, int] | list[float] = (0.0, 0.0, 0, 1),
        std_schedule: tuple[float, float, int, int] | list[float] = (1.0, 1.0, 0, 1),
        load_balancing_loss_weight: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            actor,
            critic,
            storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )
        self.teacher = teacher.to(self.device)
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

        self.dagger_coef_start = float(dagger_coef)
        self.dagger_coef_min = float(dagger_coef_min)
        self.dagger_coef_anneal_steps = max(int(dagger_coef_anneal_steps), 1)
        self.dagger_coef_anneal_updates = max(int(dagger_coef_anneal_updates), 0)
        self.warm_start_steps = max(int(warm_start_steps), 0)
        self.warm_start_updates = max(int(warm_start_updates), 0)
        self.teacher_rollout_prob_start = float(teacher_rollout_prob_start)
        self.teacher_rollout_prob_end = float(teacher_rollout_prob_end)
        self.teacher_rollout_anneal_steps = max(int(teacher_rollout_anneal_steps), 0)
        self.teacher_rollout_anneal_updates = max(int(teacher_rollout_anneal_updates), 0)
        self.future_consistency_coef = float(future_consistency_coef)
        self.future_consistency_keep_steps = max(int(future_consistency_keep_steps), 0)
        self.future_weight_decay = min(max(float(future_weight_decay), 0.0), 1.0)
        self.grad_penalty_coef_schedule = self._normalize_schedule(grad_penalty_coef_schedule)
        self.std_schedule = self._normalize_schedule(std_schedule)
        self.load_balancing_loss_weight = float(load_balancing_loss_weight)
        self.total_env_steps = 0
        self.total_updates = 0
        self.teacher_loaded = False
        self._initial_std = self._capture_initial_std()

        loss_fn_dict = {
            "mse": nn.functional.mse_loss,
            "huber": nn.functional.huber_loss,
        }
        if behavior_loss_type not in loss_fn_dict:
            raise ValueError(f"Unsupported behavior_loss_type: {behavior_loss_type}")
        self.behavior_loss_fn = loss_fn_dict[behavior_loss_type]

    def load_teacher_checkpoint(self, path: str, strict: bool = True) -> None:
        loaded_dict = torch.load(path, weights_only=False, map_location=self.device)
        teacher_state_dict = loaded_dict.get("teacher_state_dict") or loaded_dict.get("actor_state_dict")
        if teacher_state_dict is None:
            raise ValueError(f"Checkpoint does not contain teacher or actor weights: {path}")
        self.teacher.load_state_dict(teacher_state_dict, strict=strict)
        self.teacher.eval()
        self.teacher_loaded = True

    def act(self, obs: TensorDict) -> torch.Tensor:
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())

        student_actions = self.actor(obs, stochastic_output=True)
        self.transition.values = self.critic(obs).detach()

        actions = student_actions.detach()
        teacher_rollout_prob = self._current_teacher_rollout_prob()
        if self.teacher_loaded and teacher_rollout_prob > 0.0:
            teacher_actions = self.teacher(obs).detach()
            if teacher_rollout_prob >= 1.0:
                actions = teacher_actions
            else:
                teacher_mask = (torch.rand(actions.shape[0], device=actions.device) < teacher_rollout_prob).unsqueeze(-1)
                actions = torch.where(teacher_mask, teacher_actions, actions)

        self.transition.actions = actions
        self.transition.actions_log_prob = self.actor.get_output_log_prob(actions).detach()
        self.transition.distribution_params = tuple(param.detach() for param in self.actor.output_distribution_params)
        self.transition.observations = obs
        return actions

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        self.total_env_steps += rewards.shape[0]
        super().process_env_step(obs, rewards, dones, extras)

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_behavior_loss = 0.0
        mean_future_consistency_loss = 0.0
        mean_grad_penalty_loss = 0.0
        mean_load_balancing_loss = 0.0
        mean_gate_entropy = 0.0
        current_dagger_coef = self._current_dagger_coef()
        current_grad_penalty_coef = self._current_grad_penalty_coef()

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore[operator]

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore[arg-type]
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(param for param in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore[arg-type]
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore[arg-type]
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore[arg-type]
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore[arg-type]
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            with torch.no_grad():
                teacher_actions = self.teacher(batch.observations)
            behavior_loss = self.behavior_loss_fn(self.actor.output_mean, teacher_actions)

            load_balancing_loss = torch.zeros((), device=self.device)
            gate_entropy = torch.zeros((), device=self.device)
            if self.load_balancing_loss_weight > 0.0:
                if hasattr(self.actor, "get_load_balancing_loss"):
                    load_balancing_loss = self.actor.get_load_balancing_loss()
                if hasattr(self.actor, "get_gate_entropy"):
                    gate_entropy = self.actor.get_gate_entropy()

            future_consistency_loss = torch.zeros((), device=self.device)
            if self.future_consistency_coef > 0.0:
                future_consistency_loss = self._compute_future_consistency_loss(batch.observations)

            grad_penalty_loss = torch.zeros((), device=self.device)
            if current_grad_penalty_coef > 0.0:
                grad_penalty_loss = self._compute_grad_penalty(batch.observations)

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + current_dagger_coef * behavior_loss
                + self.load_balancing_loss_weight * load_balancing_loss
                + self.future_consistency_coef * future_consistency_loss
                + current_grad_penalty_coef * grad_penalty_loss
            )

            self.optimizer.zero_grad()
            loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_behavior_loss += behavior_loss.item()
            mean_load_balancing_loss += load_balancing_loss.item()
            mean_gate_entropy += gate_entropy.item()
            mean_future_consistency_loss += future_consistency_loss.item()
            mean_grad_penalty_loss += grad_penalty_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_behavior_loss /= num_updates
        mean_load_balancing_loss /= num_updates
        mean_gate_entropy /= num_updates
        mean_future_consistency_loss /= num_updates
        mean_grad_penalty_loss /= num_updates
        self.storage.clear()
        self.total_updates += 1
        self._apply_std_schedule()

        return {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "behavior": mean_behavior_loss,
            "dagger_coef": current_dagger_coef,
            "teacher_rollout_prob": self._current_teacher_rollout_prob(),
            "moe_load_balancing": mean_load_balancing_loss,
            "moe_gate_entropy": mean_gate_entropy,
            "future_consistency": mean_future_consistency_loss,
            "grad_penalty": mean_grad_penalty_loss,
            "grad_penalty_coef": current_grad_penalty_coef,
            "scheduled_std": self._current_std_target(),
        }

    def train_mode(self) -> None:
        self.actor.train()
        self.critic.train()
        self.teacher.eval()
        if self.rnd:
            self.rnd.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic.eval()
        self.teacher.eval()
        if self.rnd:
            self.rnd.eval()

    def save(self) -> dict:
        saved_dict = super().save()
        saved_dict["teacher_state_dict"] = self.teacher.state_dict()
        saved_dict["total_env_steps"] = self.total_env_steps
        saved_dict["total_updates"] = self.total_updates
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
                "teacher": True,
            }

        if load_cfg.get("teacher"):
            teacher_state_dict = loaded_dict.get("teacher_state_dict") or loaded_dict.get("actor_state_dict")
            if teacher_state_dict is not None:
                self.teacher.load_state_dict(teacher_state_dict, strict=strict)
                self.teacher_loaded = True

        if "total_env_steps" in loaded_dict:
            self.total_env_steps = int(loaded_dict["total_env_steps"])
        if "total_updates" in loaded_dict:
            self.total_updates = int(loaded_dict["total_updates"])

        return super().load(loaded_dict, load_cfg, strict)

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "DaggerPPO":
        alg_class: type[DaggerPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore[arg-type]
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore[arg-type]
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore[arg-type]
        teacher_class: type[MLPModel] = resolve_callable(cfg["teacher"].pop("class_name"))  # type: ignore[arg-type]

        default_sets = ["actor", "critic", "teacher"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")
        teacher: MLPModel = teacher_class(obs, cfg["obs_groups"], "teacher", env.num_actions, **cfg["teacher"]).to(device)
        print(f"Teacher Model: {teacher}")

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        if "warm_start_iterations" in cfg["algorithm"]:
            cfg["algorithm"]["warm_start_updates"] = int(cfg["algorithm"].pop("warm_start_iterations"))
        if "dagger_coef_anneal_iterations" in cfg["algorithm"]:
            cfg["algorithm"]["dagger_coef_anneal_updates"] = int(cfg["algorithm"].pop("dagger_coef_anneal_iterations"))
        if "teacher_rollout_anneal_iterations" in cfg["algorithm"]:
            cfg["algorithm"]["teacher_rollout_anneal_updates"] = int(
                cfg["algorithm"].pop("teacher_rollout_anneal_iterations")
            )

        alg = alg_class(
            actor,
            critic,
            teacher,
            storage,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )

        teacher_load_root_path = cfg.get("teacher_load_root_path")
        teacher_load_run = cfg.get("teacher_load_run", ".*")
        teacher_load_checkpoint = cfg.get("teacher_load_checkpoint", "model_.*.pt")
        if teacher_load_root_path:
            teacher_checkpoint_path = _resolve_checkpoint_path(
                teacher_load_root_path,
                teacher_load_run,
                teacher_load_checkpoint,
            )
            print(f"[INFO]: Loading teacher checkpoint from: {teacher_checkpoint_path}")
            alg.load_teacher_checkpoint(teacher_checkpoint_path)

        return alg

    def broadcast_parameters(self) -> None:
        model_params = [self.actor.state_dict(), self.critic.state_dict(), self.teacher.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.actor.load_state_dict(model_params[0])
        self.critic.load_state_dict(model_params[1])
        self.teacher.load_state_dict(model_params[2])

    def reduce_parameters(self) -> None:
        all_params = list(chain(self.actor.parameters(), self.critic.parameters()))
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel

    def _current_dagger_coef(self) -> float:
        if self.dagger_coef_anneal_updates > 0:
            progress = min(self.total_updates / float(self.dagger_coef_anneal_updates), 1.0)
            return _cosine_anneal(self.dagger_coef_start, self.dagger_coef_min, progress)
        progress = min(self.total_env_steps / float(self.dagger_coef_anneal_steps), 1.0)
        return _cosine_anneal(self.dagger_coef_start, self.dagger_coef_min, progress)

    def _current_teacher_rollout_prob(self) -> float:
        if not self.teacher_loaded:
            return 0.0
        if self.warm_start_updates > 0 or self.teacher_rollout_anneal_updates > 0:
            if self.total_updates < self.warm_start_updates:
                return 1.0
            if self.teacher_rollout_anneal_updates <= 0:
                return self.teacher_rollout_prob_end

            updates_after_warm_start = max(self.total_updates - self.warm_start_updates, 0)
            progress = min(updates_after_warm_start / float(self.teacher_rollout_anneal_updates), 1.0)
            return self.teacher_rollout_prob_start + (
                self.teacher_rollout_prob_end - self.teacher_rollout_prob_start
            ) * progress
        if self.total_env_steps < self.warm_start_steps:
            return 1.0
        if self.teacher_rollout_anneal_steps <= 0:
            return self.teacher_rollout_prob_end

        steps_after_warm_start = max(self.total_env_steps - self.warm_start_steps, 0)
        progress = min(steps_after_warm_start / float(self.teacher_rollout_anneal_steps), 1.0)
        return self.teacher_rollout_prob_start + (
            self.teacher_rollout_prob_end - self.teacher_rollout_prob_start
        ) * progress

    def _compute_future_consistency_loss(self, obs: TensorDict) -> torch.Tensor:
        if not hasattr(self.actor, "future_obs_dim") or getattr(self.actor, "future_obs_dim", 0) <= 0:
            return torch.zeros((), device=self.device)

        policy_group = self.actor.obs_groups[0]
        masked_obs = obs.clone()
        masked_policy_obs = masked_obs[policy_group].clone()

        current_obs_dim = int(getattr(self.actor, "current_obs_dim"))
        history_obs_dim = int(getattr(self.actor, "history_obs_dim"))
        future_obs_dim = int(getattr(self.actor, "future_obs_dim"))
        num_future_steps = int(getattr(self.actor, "num_future_steps"))
        future_step_obs_dim = int(getattr(self.actor, "future_step_obs_dim"))

        keep_steps = self.future_consistency_keep_steps
        if keep_steps <= 0:
            keep_steps = max(1, num_future_steps // 2)
        keep_steps = min(keep_steps, num_future_steps)

        future_start = current_obs_dim + history_obs_dim
        if keep_steps < num_future_steps:
            future_values = masked_policy_obs[:, future_start : future_start + future_obs_dim].reshape(
                -1, num_future_steps, future_step_obs_dim
            )
            decay_weights = torch.ones(num_future_steps, device=self.device, dtype=masked_policy_obs.dtype)
            masked_steps = num_future_steps - keep_steps
            if masked_steps > 0:
                masked_indices = torch.arange(masked_steps, device=self.device, dtype=masked_policy_obs.dtype)
                decay_weights[keep_steps:] = self.future_weight_decay ** (masked_indices + 1.0)
            future_values = future_values * decay_weights.view(1, num_future_steps, 1)
            masked_policy_obs[:, future_start : future_start + future_obs_dim] = future_values.reshape(-1, future_obs_dim)
        masked_obs[policy_group] = masked_policy_obs

        reference_actions = self.actor.output_mean.detach()
        masked_actions = self.actor(masked_obs)
        return torch.nn.functional.mse_loss(masked_actions, reference_actions)

    @staticmethod
    def _normalize_schedule(schedule: tuple[float, float, int, int] | list[float]) -> tuple[float, float, int, int]:
        if len(schedule) != 4:
            raise ValueError(f"Expected 4 schedule values, got {len(schedule)}.")
        return float(schedule[0]), float(schedule[1]), int(schedule[2]), max(int(schedule[3]), 1)

    def _current_grad_penalty_coef(self) -> float:
        return self._schedule_value(self.grad_penalty_coef_schedule)

    def _current_std_target(self) -> float:
        return self._schedule_value(self.std_schedule)

    def _schedule_value(self, schedule: tuple[float, float, int, int]) -> float:
        start_value, end_value, start_update, duration = schedule
        if self.total_updates <= start_update:
            return start_value
        progress = min((self.total_updates - start_update) / float(duration), 1.0)
        return start_value + (end_value - start_value) * progress

    def _capture_initial_std(self) -> torch.Tensor | None:
        distribution = getattr(self.actor, "distribution", None)
        if distribution is None:
            return None
        if hasattr(distribution, "std_param"):
            return distribution.std_param.detach().clone()
        if hasattr(distribution, "log_std_param"):
            return torch.exp(distribution.log_std_param.detach()).clone()
        return None

    def _apply_std_schedule(self) -> None:
        if self._initial_std is None:
            return
        distribution = getattr(self.actor, "distribution", None)
        if distribution is None:
            return
        target_std = self._current_std_target()
        if hasattr(distribution, "std_param"):
            distribution.std_param.data.copy_(torch.full_like(distribution.std_param.data, target_std))
        elif hasattr(distribution, "log_std_param"):
            distribution.log_std_param.data.copy_(torch.full_like(distribution.log_std_param.data, math.log(target_std)))

    def _compute_grad_penalty(self, obs: TensorDict) -> torch.Tensor:
        grad_obs = obs.clone()
        grad_inputs = []
        for obs_group in self.actor.obs_groups:
            grad_tensor = grad_obs[obs_group].detach().clone().requires_grad_(True)
            grad_obs[obs_group] = grad_tensor
            grad_inputs.append(grad_tensor)

        actor_output = self.actor(grad_obs)
        output_mean = self.actor.output_mean if getattr(self.actor, "distribution", None) is not None else actor_output
        gradients = torch.autograd.grad(
            outputs=output_mean.sum(),
            inputs=grad_inputs,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )
        return torch.stack([gradient.pow(2).sum(dim=-1) for gradient in gradients], dim=0).sum(dim=0).mean()
