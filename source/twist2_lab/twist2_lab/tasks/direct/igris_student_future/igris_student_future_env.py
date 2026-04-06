# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_from_euler_xyz, quat_mul

from twist2_lab.robots import (
    IGRIS_STANDING_BODY_POSE,
    IGRIS_STUDENT_KEY_BODY_MOTION_ORDER,
    IGRIS_STUDENT_KEY_BODY_ROBOT_ORDER,
)
from twist2_lab.tasks.direct.igris_mimic_teacher.igris_mimic_teacher_env import IgrisMimicTeacherEnv

from .igris_student_future_env_cfg import IgrisStudentFutureEnvCfg


def _roll_pitch_from_quat(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w, x, y, z = quat.unbind(dim=-1)

    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sin_roll, cos_roll)

    sin_pitch = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sin_pitch, min=-1.0, max=1.0))

    return roll, pitch


def _roll_pitch_yaw_from_quat(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    roll, pitch = _roll_pitch_from_quat(quat)
    w, x, y, z = quat.unbind(dim=-1)
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(sin_yaw, cos_yaw)
    return roll, pitch, yaw


class IgrisStudentFutureEnv(IgrisMimicTeacherEnv):
    cfg: IgrisStudentFutureEnvCfg

    def __init__(self, cfg: IgrisStudentFutureEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._student_mimic_obs_dim = self.cfg.student_mimic_obs_dim
        self._proprio_obs_dim = self.cfg.proprio_obs_dim
        self._student_obs_dim = self.cfg.student_obs_dim
        self._history_length = int(self.cfg.history_length)
        self._future_motion_steps = torch.tensor(self.cfg.future_motion_steps, dtype=torch.float32, device=self.device)
        if self._future_motion_steps.ndim != 1:
            raise ValueError("future_motion_steps must be a 1D sequence.")
        if torch.any(self._future_motion_steps < 0):
            raise ValueError("future_motion_steps must be non-negative.")
        self._future_curriculum_steps = self._future_motion_steps.numel()
        self._future_mask = torch.ones(
            self.num_envs,
            self._future_curriculum_steps,
            dtype=torch.bool,
            device=self.device,
        )

        self._default_body_joint_pos = torch.tensor(
            [IGRIS_STANDING_BODY_POSE[name] for name in self._body_joint_names],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        self._ankle_joint_indices = [
            self._body_joint_names.index(joint_name)
            for joint_name in (
                "Joint_Ankle_Pitch_Left",
                "Joint_Ankle_Roll_Left",
                "Joint_Ankle_Pitch_Right",
                "Joint_Ankle_Roll_Right",
            )
        ]
        self._critic_key_body_ids, self._critic_key_body_names = self.robot.find_bodies(
            IGRIS_STUDENT_KEY_BODY_ROBOT_ORDER, preserve_order=True
        )
        self._motion_key_body_indices = self._motion_lib.get_key_body_idx(list(IGRIS_STUDENT_KEY_BODY_MOTION_ORDER))
        if len(self._critic_key_body_ids) != len(self._motion_key_body_indices):
            raise ValueError("Student critic key-body configuration is inconsistent between robot and motion data.")
        self._num_critic_key_bodies = len(self._critic_key_body_ids)

        self._obs_history = torch.zeros(
            self.num_envs,
            self._history_length,
            self._student_obs_dim,
            dtype=torch.float32,
            device=self.device,
        )

        expected_obs_dim = self._student_obs_dim * (1 + self._history_length)
        expected_obs_dim += self._student_mimic_obs_dim * self._future_motion_steps.numel()
        if expected_obs_dim != self.cfg.observation_space:
            raise ValueError(
                f"Configured observation space ({self.cfg.observation_space}) does not match "
                f"student-future observation dim ({expected_obs_dim})."
            )
        expected_state_dim = expected_obs_dim + self.cfg.critic_privileged_obs_dim
        if expected_state_dim != self.cfg.state_space:
            raise ValueError(
                f"Configured state space ({self.cfg.state_space}) does not match "
                f"student-future critic dim ({expected_state_dim})."
            )

        self._last_root_pos_w = self.robot.data.root_link_pos_w.clone()
        self._last_root_quat_w = self.robot.data.root_link_quat_w.clone()
        self._motion_dr_enabled = bool(getattr(self.cfg, "motion_dr_enabled", False))
        self._motion_dr_resampling = bool(getattr(self.cfg, "motion_dr_resampling", True))
        self._enable_force_curriculum = bool(getattr(self.cfg, "enable_force_curriculum", False))

        if self._motion_dr_enabled and not self._motion_dr_resampling:
            self._root_position_noise = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
            self._root_orientation_noise = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
            self._root_velocity_noise = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
            self._joint_position_noise = torch.zeros(self.num_envs, self._num_body_joints, dtype=torch.float32, device=self.device)
        else:
            self._root_position_noise = None
            self._root_orientation_noise = None
            self._root_velocity_noise = None
            self._joint_position_noise = None

        if self._enable_force_curriculum:
            self._force_body_ids, self._force_body_names = self.robot.find_bodies(
                list(self.cfg.force_apply_body_names), preserve_order=True
            )
            if len(self._force_body_ids) == 0:
                raise ValueError("Force curriculum enabled, but no force application bodies were found.")
            self._num_force_bodies = len(self._force_body_ids)
            self._force_scale = torch.full(
                (self.num_envs,),
                float(self.cfg.force_scale_initial),
                dtype=torch.float32,
                device=self.device,
            )
            self._force_duration = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._force_vectors = torch.zeros(
                self.num_envs,
                self._num_force_bodies,
                3,
                dtype=torch.float32,
                device=self.device,
            )
            self._force_torques = torch.zeros_like(self._force_vectors)
            self._resample_force_commands(self.robot._ALL_INDICES)
        else:
            self._force_body_ids = []
            self._num_force_bodies = 0

    def _build_student_mimic_obs(
        self,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        root_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        dof_pos: torch.Tensor,
    ) -> torch.Tensor:
        batch_shape = root_pos.shape[:-1]

        root_vel_local = quat_apply_inverse(root_rot.reshape(-1, 4), root_vel.reshape(-1, 3)).reshape(*batch_shape, 3)
        root_ang_vel_local = quat_apply_inverse(root_rot.reshape(-1, 4), root_ang_vel.reshape(-1, 3)).reshape(
            *batch_shape, 3
        )
        roll, pitch = _roll_pitch_from_quat(root_rot.reshape(-1, 4))
        roll = roll.reshape(*batch_shape, 1)
        pitch = pitch.reshape(*batch_shape, 1)

        return torch.cat(
            (
                root_vel_local[..., :2],
                root_pos[..., 2:3],
                roll,
                pitch,
                root_ang_vel_local[..., 2:3],
                dof_pos,
            ),
            dim=-1,
        )

    def _compute_future_motion_obs(self) -> torch.Tensor:
        if self._future_motion_steps.numel() == 0:
            return torch.zeros(self.num_envs, 0, dtype=torch.float32, device=self.device)

        motion_ids = self._motion_ids.unsqueeze(-1).expand(-1, self._future_motion_steps.numel())
        motion_times = self._motion_times().unsqueeze(-1) + self._future_motion_steps.unsqueeze(0) * self.step_dt
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, _ = self._motion_lib.calc_motion_frame_batch(
            motion_ids, motion_times
        )

        root_pos = root_pos.clone()
        root_pos[..., :2] += self._motion_xy_offsets.unsqueeze(1)
        root_pos[..., 2] += self.cfg.motion_height_offset
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos = self._apply_motion_domain_randomization(
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
        )

        future_obs = self._build_student_mimic_obs(root_pos, root_rot, root_vel, root_ang_vel, dof_pos)
        future_obs = future_obs * self._compute_future_mask().unsqueeze(-1)
        return future_obs.reshape(self.num_envs, -1)

    def _sample_uniform_range(self, value_range: tuple[float, float], shape: tuple[int, ...]) -> torch.Tensor:
        low, high = value_range
        return torch.empty(shape, device=self.device, dtype=torch.float32).uniform_(low, high)

    def _resample_motion_dr_noise(self, env_ids: torch.Tensor) -> None:
        if self._root_position_noise is None:
            return
        self._root_position_noise[env_ids] = self._sample_uniform_range(
            (-self.cfg.root_position_noise_range[1], self.cfg.root_position_noise_range[1]),
            (len(env_ids), 3),
        )
        self._root_orientation_noise[env_ids] = self._sample_uniform_range(
            (-self.cfg.root_orientation_noise_range[1], self.cfg.root_orientation_noise_range[1]),
            (len(env_ids), 3),
        )
        self._root_velocity_noise[env_ids] = self._sample_uniform_range(
            (-self.cfg.root_velocity_noise_range[1], self.cfg.root_velocity_noise_range[1]),
            (len(env_ids), 3),
        )
        self._joint_position_noise[env_ids] = self._sample_uniform_range(
            (-self.cfg.joint_position_noise_range[1], self.cfg.joint_position_noise_range[1]),
            (len(env_ids), self._num_body_joints),
        )

    def _apply_motion_domain_randomization(
        self,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        root_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        dof_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._motion_dr_enabled:
            return root_pos, root_rot, root_vel, root_ang_vel, dof_pos

        batch_shape = root_pos.shape[:-1]
        num_steps = 1 if len(batch_shape) == 1 else batch_shape[1]

        if self._motion_dr_resampling or self._root_position_noise is None:
            root_position_noise = self._sample_uniform_range(
                (-self.cfg.root_position_noise_range[1], self.cfg.root_position_noise_range[1]),
                (self.num_envs, 3),
            )
            root_orientation_noise = self._sample_uniform_range(
                (-self.cfg.root_orientation_noise_range[1], self.cfg.root_orientation_noise_range[1]),
                (self.num_envs, 3),
            )
            root_velocity_noise = self._sample_uniform_range(
                (-self.cfg.root_velocity_noise_range[1], self.cfg.root_velocity_noise_range[1]),
                (self.num_envs, 3),
            )
            joint_position_noise = self._sample_uniform_range(
                (-self.cfg.joint_position_noise_range[1], self.cfg.joint_position_noise_range[1]),
                (self.num_envs, self._num_body_joints),
            )
        else:
            root_position_noise = self._root_position_noise
            root_orientation_noise = self._root_orientation_noise
            root_velocity_noise = self._root_velocity_noise
            joint_position_noise = self._joint_position_noise

        root_position_noise = root_position_noise.unsqueeze(1).expand(-1, num_steps, -1)
        root_orientation_noise = root_orientation_noise.unsqueeze(1).expand(-1, num_steps, -1)
        root_velocity_noise = root_velocity_noise.unsqueeze(1).expand(-1, num_steps, -1)
        joint_position_noise = joint_position_noise.unsqueeze(1).expand(-1, num_steps, -1)

        root_pos = root_pos + root_position_noise
        roll, pitch, yaw = _roll_pitch_yaw_from_quat(root_rot.reshape(-1, 4))
        euler_noise = root_orientation_noise.reshape(-1, 3)
        noisy_root_rot = quat_from_euler_xyz(
            roll + euler_noise[:, 0],
            pitch + euler_noise[:, 1],
            yaw + euler_noise[:, 2],
        ).reshape(*batch_shape, 4)
        root_vel = root_vel + root_velocity_noise
        root_ang_vel = root_ang_vel + root_velocity_noise
        dof_pos = dof_pos + joint_position_noise
        return root_pos, noisy_root_rot, root_vel, root_ang_vel, dof_pos

    def _resample_force_commands(self, env_ids: torch.Tensor) -> None:
        if not self._enable_force_curriculum or env_ids.numel() == 0:
            return

        self._force_duration[env_ids] = torch.randint(
            int(self.cfg.randomize_force_duration[0]),
            int(self.cfg.randomize_force_duration[1]) + 1,
            (len(env_ids),),
            device=self.device,
        )

        force_scale = self._force_scale[env_ids].view(-1, 1, 1)
        x_force = self._sample_uniform_range(self.cfg.apply_force_x_range, (len(env_ids), self._num_force_bodies, 1))
        y_force = self._sample_uniform_range(self.cfg.apply_force_y_range, (len(env_ids), self._num_force_bodies, 1))
        z_force = self._sample_uniform_range(self.cfg.apply_force_z_range, (len(env_ids), self._num_force_bodies, 1))
        force_vectors = torch.cat((x_force, y_force, z_force), dim=-1)

        zero_prob = torch.tensor(self.cfg.zero_force_prob, device=self.device, dtype=torch.float32).view(1, 1, 3)
        zero_mask = torch.rand_like(force_vectors) < zero_prob
        force_vectors = force_vectors.masked_fill(zero_mask, 0.0)

        if self.cfg.only_apply_resistance_when_walking:
            root_xy_vel = self.robot.data.root_lin_vel_b[env_ids, :2].unsqueeze(1)
            moving_mask = torch.norm(root_xy_vel, dim=-1, keepdim=True) > 0.05
            resistance_sign = -torch.sign(root_xy_vel)
            resistance_sign[resistance_sign == 0.0] = 1.0
            force_vectors[:, :, :2] = torch.where(
                moving_mask.expand(-1, self._num_force_bodies, -1),
                resistance_sign * torch.abs(force_vectors[:, :, :2]),
                force_vectors[:, :, :2],
            )

        self._force_vectors[env_ids] = force_vectors * force_scale
        self._force_torques[env_ids] = 0.0

    def _update_force_curriculum(self, env_ids: torch.Tensor) -> None:
        if not self._enable_force_curriculum or env_ids.numel() == 0:
            return

        episode_lengths = self.episode_length_buf[env_ids].float()
        if self.cfg.force_scale_curriculum:
            scale_up_mask = episode_lengths >= float(self.cfg.force_scale_up_threshold)
            scale_down_mask = episode_lengths <= float(self.cfg.force_scale_down_threshold)
            self._force_scale[env_ids[scale_up_mask]] += float(self.cfg.force_scale_up)
            self._force_scale[env_ids[scale_down_mask]] -= float(self.cfg.force_scale_down)
            self._force_scale.clamp_(min=float(self.cfg.force_scale_min), max=float(self.cfg.force_scale_max))

        self._resample_force_commands(env_ids)

    def _update_force_application(self) -> None:
        if not self._enable_force_curriculum:
            return

        self._force_duration -= 1
        resample_env_ids = (self._force_duration <= 0).nonzero(as_tuple=False).squeeze(-1)
        if resample_env_ids.numel() > 0:
            self._resample_force_commands(resample_env_ids)

        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=self._force_vectors,
            torques=self._force_torques,
            body_ids=self._force_body_ids,
        )

    def _future_curriculum_progress(self) -> float:
        if not self.cfg.enable_future_curriculum or self._future_curriculum_steps == 0:
            return 1.0
        if self.num_envs == 1 and self.cfg.full_future_for_single_env:
            return 1.0

        start_step = int(self.cfg.future_curriculum_start_step)
        full_step = int(self.cfg.future_curriculum_full_step)
        if full_step <= start_step:
            return 1.0
        if self.common_step_counter <= start_step:
            return 0.0

        return min((self.common_step_counter - start_step) / float(full_step - start_step), 1.0)

    def _visible_future_count(self) -> int:
        if self._future_curriculum_steps == 0:
            return 0
        if not self.cfg.enable_future_curriculum:
            return self._future_curriculum_steps
        if self.num_envs == 1 and self.cfg.full_future_for_single_env:
            return self._future_curriculum_steps

        min_visible = max(1, int(self.cfg.future_curriculum_min_visible))
        min_visible = min(min_visible, self._future_curriculum_steps)
        progress = self._future_curriculum_progress()
        additional_frames = self._future_curriculum_steps - min_visible
        return min_visible + int(progress * additional_frames + 1e-6)

    def _compute_future_mask(self) -> torch.Tensor:
        visible_count = self._visible_future_count()
        if visible_count <= 0:
            self._future_mask.zero_()
            return self._future_mask
        if visible_count >= self._future_curriculum_steps:
            self._future_mask.fill_(True)
            return self._future_mask

        self._future_mask.zero_()
        self._future_mask[:, :visible_count] = True
        return self._future_mask

    def _update_obs_history(self, current_obs: torch.Tensor) -> None:
        if self._history_length == 0:
            return

        reset_mask = self.episode_length_buf <= 1
        if reset_mask.any():
            reset_ids = reset_mask.nonzero(as_tuple=False).squeeze(-1)
            self._obs_history[reset_ids] = current_obs[reset_ids].unsqueeze(1).expand(-1, self._history_length, -1)

        continue_mask = ~reset_mask
        if continue_mask.any():
            continue_ids = continue_mask.nonzero(as_tuple=False).squeeze(-1)
            self._obs_history[continue_ids, :-1] = self._obs_history[continue_ids, 1:]
            self._obs_history[continue_ids, -1] = current_obs[continue_ids]

    def _build_proprio_obs(self) -> torch.Tensor:
        roll, pitch = _roll_pitch_from_quat(self.robot.data.root_link_quat_w)
        joint_pos_offset = (self.robot.data.joint_pos[:, self._body_joint_ids] - self._default_body_joint_pos) * self.cfg.dof_pos_scale
        joint_vel = self.robot.data.joint_vel[:, self._body_joint_ids] * self.cfg.dof_vel_scale
        joint_vel[:, self._ankle_joint_indices] = 0.0

        return torch.cat(
            (
                self.robot.data.root_ang_vel_b * self.cfg.ang_vel_scale,
                roll.unsqueeze(-1),
                pitch.unsqueeze(-1),
                joint_pos_offset,
                joint_vel,
                self._actions,
            ),
            dim=-1,
        )

    def _compute_current_root_rot_delta_local(self) -> torch.Tensor:
        delta_quat = quat_mul(self.robot.data.root_link_quat_w, quat_conjugate(self._last_root_quat_w))
        roll, pitch, yaw = _roll_pitch_yaw_from_quat(delta_quat)
        delta_euler = torch.stack((roll, pitch, yaw), dim=-1)
        return quat_apply_inverse(self._last_root_quat_w, delta_euler)

    def _compute_current_key_body_local(self) -> torch.Tensor:
        key_body_offsets = self.robot.data.body_pos_w[:, self._critic_key_body_ids] - self.robot.data.root_link_pos_w.unsqueeze(1)
        return self._key_body_local_from_offsets(self.robot.data.root_link_quat_w, key_body_offsets)

    def _compute_reference_privileged_motion_features(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        motion_times = self._motion_times()
        _, root_rot, _, _, _, _, local_body_pos, root_pos_delta_local, root_rot_delta_local = (
            self._motion_lib.calc_motion_frame_with_local_info(self._motion_ids, motion_times)
        )
        ref_key_body_local = self._reference_key_body_local_from_motion(
            self._ref_root_pos,
            root_rot,
            local_body_pos,
        )
        return ref_key_body_local, root_pos_delta_local, root_rot_delta_local

    def _build_critic_privileged_obs(self, tracking_obs: torch.Tensor) -> torch.Tensor:
        current_key_body_local = self._compute_current_key_body_local()
        ref_key_body_local, ref_root_pos_delta_local, ref_root_rot_delta_local = (
            self._compute_reference_privileged_motion_features()
        )
        current_root_pos_delta_local = quat_apply_inverse(
            self._last_root_quat_w,
            self.robot.data.root_link_pos_w - self._last_root_pos_w,
        )
        current_root_rot_delta_local = self._compute_current_root_rot_delta_local()

        reset_mask = self.episode_length_buf <= 1
        if reset_mask.any():
            current_root_pos_delta_local = current_root_pos_delta_local.clone()
            current_root_rot_delta_local = current_root_rot_delta_local.clone()
            current_root_pos_delta_local[reset_mask] = 0.0
            current_root_rot_delta_local[reset_mask] = 0.0

        key_body_local_error = ref_key_body_local - current_key_body_local
        root_pos_delta_error = ref_root_pos_delta_local - current_root_pos_delta_local
        root_rot_delta_error = ref_root_rot_delta_local - current_root_rot_delta_local

        return torch.cat(
            (
                tracking_obs,
                current_key_body_local,
                ref_key_body_local,
                key_body_local_error,
                current_root_pos_delta_local,
                ref_root_pos_delta_local,
                root_pos_delta_error,
                current_root_rot_delta_local,
                ref_root_rot_delta_local,
                root_rot_delta_error,
            ),
            dim=-1,
        )

    def _update_error_aware_sampling_metrics(self) -> None:
        if not self.cfg.use_error_aware_sampling:
            return

        current_key_body_local = self._compute_current_key_body_local().reshape(self.num_envs, self._num_critic_key_bodies, 3)
        ref_key_body_local, _, _ = self._compute_reference_privileged_motion_features()
        ref_key_body_local = ref_key_body_local.reshape(self.num_envs, self._num_critic_key_bodies, 3)

        key_body_error_per_part = torch.mean(torch.abs(current_key_body_local - ref_key_body_local), dim=-1)
        current_max_error = torch.amax(key_body_error_per_part, dim=-1)

        motion_max_error = torch.full_like(self.max_key_body_error, float("-inf"))
        motion_max_error.scatter_reduce_(0, self._motion_ids, current_max_error, reduce="amax", include_self=False)
        valid_motion_mask = torch.isfinite(motion_max_error)
        self.max_key_body_error[valid_motion_mask] = torch.maximum(
            self.max_key_body_error[valid_motion_mask],
            motion_max_error[valid_motion_mask],
        )

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)

        if env_ids is None:
            env_ids_tensor = self.robot._ALL_INDICES
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids
        else:
            env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        self._obs_history[env_ids_tensor] = 0.0
        self._last_root_pos_w[env_ids_tensor] = self._ref_root_pos[env_ids_tensor]
        self._last_root_quat_w[env_ids_tensor] = self._ref_root_rot[env_ids_tensor]
        if self._motion_dr_enabled and not self._motion_dr_resampling:
            self._resample_motion_dr_noise(env_ids_tensor)
        if self._enable_force_curriculum:
            self._update_force_curriculum(env_ids_tensor)
        self.extras.setdefault("log", {})
        self.extras["log"]["Curriculum/future_visible_frames"] = float(self._visible_future_count())
        self.extras["log"]["Curriculum/future_progress"] = self._future_curriculum_progress()
        if self._enable_force_curriculum:
            self.extras["log"]["Curriculum/force_scale_mean"] = self._force_scale.mean().item()
            self.extras["log"]["Curriculum/force_scale_min"] = self._force_scale.min().item()
            self.extras["log"]["Curriculum/force_scale_max"] = self._force_scale.max().item()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)
        self._update_force_application()

    def _get_observations(self) -> dict:
        self._update_reference_motion()
        tracking_obs = self._compute_tracking_observation()

        current_mimic_obs = self._build_student_mimic_obs(
            self._ref_root_pos.unsqueeze(1),
            self._ref_root_rot.unsqueeze(1),
            self._ref_root_vel.unsqueeze(1),
            self._ref_root_ang_vel.unsqueeze(1),
            self._ref_dof_pos.unsqueeze(1),
        ).squeeze(1)

        proprio_obs = self._build_proprio_obs()

        current_obs = torch.cat((current_mimic_obs, proprio_obs), dim=-1)
        future_obs = self._compute_future_motion_obs()

        policy_obs = torch.cat(
            (
                current_obs,
                self._obs_history.reshape(self.num_envs, -1),
                future_obs,
            ),
            dim=-1,
        )
        critic_obs = torch.cat((policy_obs, self._build_critic_privileged_obs(tracking_obs)), dim=-1)
        teacher_priv_obs = self._build_teacher_privileged_obs()

        self._update_obs_history(current_obs)
        self._previous_actions.copy_(self._actions)
        self._last_root_pos_w.copy_(self.robot.data.root_link_pos_w)
        self._last_root_quat_w.copy_(self.robot.data.root_link_quat_w)
        return {"policy": policy_obs, "critic": critic_obs, "teacher": tracking_obs, "teacher_priv": teacher_priv_obs}
