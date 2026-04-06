# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import EmpiricalNormalization, MLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


class FutureActorModel(MLPModel):
    """Actor model with explicit history and future-motion encoding."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        current_motion_obs_dim: int = 0,
        proprio_obs_dim: int = 0,
        history_length: int = 0,
        future_step_obs_dim: int = 0,
        num_future_steps: int = 0,
        history_encoder_dims: tuple[int, ...] | list[int] = (256, 128),
        history_latent_dim: int = 128,
        future_token_dim: int = 128,
        future_encoder_dims: tuple[int, ...] | list[int] = (256, 128),
        future_latent_dim: int = 128,
        future_attention_heads: int = 4,
        future_dropout: float = 0.1,
        num_experts: int = 1,
        expert_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        gating_hidden_dim: int = 128,
        moe_temperature: float = 1.0,
        moe_topk: int | None = None,
    ) -> None:
        nn.Module.__init__(self)

        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = nn.Identity()

        if distribution_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore[arg-type]
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        self.current_motion_obs_dim = int(current_motion_obs_dim)
        self.proprio_obs_dim = int(proprio_obs_dim)
        self.current_obs_dim = self.current_motion_obs_dim + self.proprio_obs_dim
        self.history_length = int(history_length)
        self.future_step_obs_dim = int(future_step_obs_dim)
        self.num_future_steps = int(num_future_steps)
        self.history_obs_dim = self.current_obs_dim * self.history_length
        self.future_obs_dim = self.future_step_obs_dim * self.num_future_steps
        self.history_latent_dim = int(history_latent_dim)
        self.future_latent_dim = int(future_latent_dim)
        self.future_token_dim = int(future_token_dim)
        self.num_experts = max(int(num_experts), 1)
        self.expert_hidden_dims = tuple(int(dim) for dim in expert_hidden_dims)
        self.gating_hidden_dim = int(gating_hidden_dim)
        self.moe_temperature = max(float(moe_temperature), 1.0e-6)
        self.moe_topk = None if moe_topk is None else max(int(moe_topk), 1)
        self._last_gate_probs: torch.Tensor | None = None

        expected_obs_dim = self.current_obs_dim + self.history_obs_dim + self.future_obs_dim
        if expected_obs_dim != self.obs_dim:
            raise ValueError(
                f"FutureActorModel expected observation dim {expected_obs_dim}, but resolved {self.obs_dim} "
                f"from observation groups {self.obs_groups}."
            )

        if self.history_length > 0:
            self.history_encoder = MLP(
                self.history_obs_dim,
                self.history_latent_dim,
                history_encoder_dims,
                activation,
            )
        else:
            self.history_encoder = nn.Identity()

        if self.num_future_steps > 0 and self.future_step_obs_dim > 0:
            if self.future_token_dim % future_attention_heads != 0:
                raise ValueError("future_token_dim must be divisible by future_attention_heads.")
            self.future_input_proj = nn.Linear(self.future_step_obs_dim, self.future_token_dim)
            self.future_pos_embedding = nn.Parameter(torch.zeros(1, self.num_future_steps, self.future_token_dim))
            self.future_attention = nn.MultiheadAttention(
                self.future_token_dim,
                future_attention_heads,
                dropout=future_dropout,
                batch_first=True,
            )
            self.future_norm = nn.LayerNorm(self.future_token_dim)
            self.future_encoder = MLP(
                self.future_token_dim,
                self.future_latent_dim,
                future_encoder_dims,
                activation,
            )
        else:
            self.future_input_proj = None
            self.future_pos_embedding = None
            self.future_attention = None
            self.future_norm = None
            self.future_encoder = None

        latent_dim = self._get_latent_dim()
        hidden_dims = tuple(int(dim) for dim in hidden_dims)
        if self.num_experts > 1:
            trunk_output_dim = hidden_dims[-1] if hidden_dims else latent_dim
            trunk_hidden_dims = hidden_dims[:-1]
            self.shared_mlp = self._build_head(latent_dim, trunk_output_dim, trunk_hidden_dims, activation)
            self.gating_mlp = self._build_head(
                trunk_output_dim,
                self.num_experts,
                (self.gating_hidden_dim,),
                activation,
            )
            self.experts = nn.ModuleList(
                self._build_head(trunk_output_dim, mlp_output_dim, self.expert_hidden_dims, activation)
                for _ in range(self.num_experts)
            )
            self.mlp = None
            if self.distribution is not None:
                for expert in self.experts:
                    self.distribution.init_mlp_weights(expert)
        else:
            self.shared_mlp = None
            self.gating_mlp = None
            self.experts = None
            self.mlp = self._build_head(latent_dim, mlp_output_dim, hidden_dims, activation)
            if self.distribution is not None:
                self.distribution.init_mlp_weights(self.mlp)

    @staticmethod
    def _build_head(
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str,
    ) -> nn.Module:
        hidden_dims = tuple(int(dim) for dim in hidden_dims)
        if hidden_dims:
            return MLP(input_dim, output_dim, hidden_dims, activation)
        return nn.Linear(input_dim, output_dim)

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        policy_obs = torch.cat(obs_list, dim=-1)
        policy_obs = self.obs_normalizer(policy_obs)

        current_end = self.current_obs_dim
        history_end = current_end + self.history_obs_dim

        current_obs = policy_obs[:, :current_end]
        history_obs = policy_obs[:, current_end:history_end]
        future_obs = policy_obs[:, history_end:]

        if self.history_length > 0:
            history_latent = self.history_encoder(history_obs)
        else:
            history_latent = torch.zeros(policy_obs.shape[0], 0, device=policy_obs.device, dtype=policy_obs.dtype)

        if self.future_encoder is not None and self.future_obs_dim > 0:
            future_tokens = future_obs.reshape(-1, self.num_future_steps, self.future_step_obs_dim)
            future_tokens = self.future_input_proj(future_tokens) + self.future_pos_embedding[:, : self.num_future_steps]
            future_tokens, _ = self.future_attention(future_tokens, future_tokens, future_tokens, need_weights=False)
            future_tokens = self.future_norm(future_tokens)
            future_latent = self.future_encoder(future_tokens.mean(dim=1))
        else:
            future_latent = torch.zeros(policy_obs.shape[0], 0, device=policy_obs.device, dtype=policy_obs.dtype)

        return torch.cat((current_obs, history_latent, future_latent), dim=-1)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        latent = self.get_latent(obs, masks, hidden_state)
        mlp_output = self._forward_head(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def get_load_balancing_loss(self) -> torch.Tensor:
        if self._last_gate_probs is None or self.num_experts <= 1:
            return torch.zeros((), device=self._get_module_device())
        mean_gate_probs = self._last_gate_probs.mean(dim=0)
        uniform = torch.full_like(mean_gate_probs, 1.0 / float(self.num_experts))
        return torch.mean((mean_gate_probs - uniform) ** 2)

    def get_gate_entropy(self) -> torch.Tensor:
        if self._last_gate_probs is None or self.num_experts <= 1:
            return torch.zeros((), device=self._get_module_device())
        gate_probs = self._last_gate_probs.clamp_min(1.0e-8)
        return -(gate_probs * gate_probs.log()).sum(dim=-1).mean()

    def _forward_head(self, latent: torch.Tensor) -> torch.Tensor:
        if self.num_experts <= 1:
            return self.mlp(latent)

        shared_latent = self.shared_mlp(latent)
        gate_probs = self._compute_gate_probs(shared_latent)
        self._last_gate_probs = gate_probs
        expert_outputs = torch.stack([expert(shared_latent) for expert in self.experts], dim=1)
        return torch.sum(expert_outputs * gate_probs.unsqueeze(-1), dim=1)

    def _compute_gate_probs(self, shared_latent: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gating_mlp(shared_latent) / self.moe_temperature
        if self.moe_topk is not None and self.moe_topk < self.num_experts:
            topk = min(self.moe_topk, self.num_experts)
            topk_values, topk_indices = torch.topk(gate_logits, k=topk, dim=-1)
            masked_logits = torch.full_like(gate_logits, float("-inf"))
            masked_logits.scatter_(dim=-1, index=topk_indices, src=topk_values)
            gate_logits = masked_logits
        return torch.softmax(gate_logits, dim=-1)

    def _get_module_device(self) -> torch.device:
        return next(self.parameters()).device

    def _get_latent_dim(self) -> int:
        return self.current_obs_dim + self.history_latent_dim + self.future_latent_dim

    def as_jit(self) -> nn.Module:
        return _TorchFutureActorModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxFutureActorModel(self, verbose)


class _FutureActorExportBase(nn.Module):
    def __init__(self, model: FutureActorModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.shared_mlp = copy.deepcopy(model.shared_mlp)
        self.gating_mlp = copy.deepcopy(model.gating_mlp)
        self.experts = copy.deepcopy(model.experts)
        self.current_obs_dim = model.current_obs_dim
        self.history_length = model.history_length
        self.history_obs_dim = model.history_obs_dim
        self.future_step_obs_dim = model.future_step_obs_dim
        self.num_future_steps = model.num_future_steps
        self.future_obs_dim = model.future_obs_dim
        self.history_latent_dim = model.history_latent_dim
        self.future_latent_dim = model.future_latent_dim
        self.num_experts = model.num_experts
        self.moe_temperature = model.moe_temperature
        self.moe_topk = model.moe_topk
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        self.history_encoder = copy.deepcopy(model.history_encoder)
        self.future_input_proj = copy.deepcopy(model.future_input_proj)
        self.future_attention = copy.deepcopy(model.future_attention)
        self.future_norm = copy.deepcopy(model.future_norm)
        self.future_encoder = copy.deepcopy(model.future_encoder)
        if model.future_pos_embedding is not None:
            self.future_pos_embedding = nn.Parameter(model.future_pos_embedding.detach().clone(), requires_grad=False)
        else:
            self.future_pos_embedding = None

    def _forward_latent(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        current_end = self.current_obs_dim
        history_end = current_end + self.history_obs_dim

        current_obs = x[:, :current_end]
        history_obs = x[:, current_end:history_end]
        future_obs = x[:, history_end:]

        if self.history_length > 0:
            history_latent = self.history_encoder(history_obs)
        else:
            history_latent = torch.zeros(x.shape[0], 0, device=x.device, dtype=x.dtype)

        if self.future_encoder is not None and self.future_obs_dim > 0:
            future_tokens = future_obs.reshape(-1, self.num_future_steps, self.future_step_obs_dim)
            future_tokens = self.future_input_proj(future_tokens) + self.future_pos_embedding[:, : self.num_future_steps]
            future_tokens, _ = self.future_attention(future_tokens, future_tokens, future_tokens, need_weights=False)
            future_tokens = self.future_norm(future_tokens)
            future_latent = self.future_encoder(future_tokens.mean(dim=1))
        else:
            future_latent = torch.zeros(x.shape[0], 0, device=x.device, dtype=x.dtype)

        return torch.cat((current_obs, history_latent, future_latent), dim=-1)

    def _forward_head(self, latent: torch.Tensor) -> torch.Tensor:
        if self.num_experts <= 1:
            return self.mlp(latent)

        shared_latent = self.shared_mlp(latent)
        gate_logits = self.gating_mlp(shared_latent) / self.moe_temperature
        if self.moe_topk is not None and self.moe_topk < self.num_experts:
            topk = min(self.moe_topk, self.num_experts)
            topk_values, topk_indices = torch.topk(gate_logits, k=topk, dim=-1)
            masked_logits = torch.full_like(gate_logits, float("-inf"))
            masked_logits.scatter_(dim=-1, index=topk_indices, src=topk_values)
            gate_logits = masked_logits
        gate_probs = torch.softmax(gate_logits, dim=-1)
        expert_outputs = torch.stack([expert(shared_latent) for expert in self.experts], dim=1)
        return torch.sum(expert_outputs * gate_probs.unsqueeze(-1), dim=1)


class _TorchFutureActorModel(_FutureActorExportBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self._forward_latent(x)
        out = self._forward_head(latent)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxFutureActorModel(_FutureActorExportBase):
    is_recurrent: bool = False

    def __init__(self, model: FutureActorModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.input_size = model.obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self._forward_latent(x)
        out = self._forward_head(latent)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
