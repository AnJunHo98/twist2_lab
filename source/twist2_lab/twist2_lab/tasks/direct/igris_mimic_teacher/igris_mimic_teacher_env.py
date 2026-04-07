# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_mul,
)

from twist2_lab.motions import PklMotionLibrary
from twist2_lab.robots import (
    IGRIS_BODY_JOINT_ORDER,
    IGRIS_STANDING_BODY_POSE,
    IGRIS_STUDENT_KEY_BODY_MOTION_ORDER,
    IGRIS_STUDENT_KEY_BODY_ROBOT_ORDER,
)

from .igris_mimic_teacher_env_cfg import IgrisMimicTeacherEnvCfg


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


def _yaw_only_quat(quat: torch.Tensor) -> torch.Tensor:
    _, _, yaw = _roll_pitch_yaw_from_quat(quat)
    zero = torch.zeros_like(yaw)
    return quat_from_euler_xyz(zero, zero, yaw)


class IgrisMimicTeacherEnv(DirectRLEnv):
    cfg: IgrisMimicTeacherEnvCfg

    def __init__(self, cfg: IgrisMimicTeacherEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._body_joint_ids, self._body_joint_names = self.robot.find_joints(
            IGRIS_BODY_JOINT_ORDER, preserve_order=True
        )
        self._num_body_joints = len(self._body_joint_ids)
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

        self._motion_lib = PklMotionLibrary(
            self.cfg.motion_file,
            device=self.device,
            storage_device=getattr(self.cfg, "motion_storage_device", None),
            storage_dtype=getattr(self.cfg, "motion_storage_dtype", None),
        )
        self._max_motion_curriculum_level = self._motion_lib.max_curriculum_level()
        self._motion_curriculum_mode = str(getattr(self.cfg, "motion_level_curriculum_mode", "success")).lower()
        if self._motion_curriculum_mode not in {"step", "success"}:
            raise ValueError(
                f"Unsupported motion_level_curriculum_mode='{self.cfg.motion_level_curriculum_mode}'."
                " Expected 'step' or 'success'."
            )
        if self._motion_lib.num_dofs != self._num_body_joints:
            raise RuntimeError(
                f"Motion DOF count ({self._motion_lib.num_dofs}) does not match IGRIS body joints ({self._num_body_joints})."
            )
        self._motion_key_body_indices = self._motion_lib.get_key_body_idx(list(IGRIS_STUDENT_KEY_BODY_MOTION_ORDER))
        self._motion_key_body_idx_tensor = torch.tensor(
            self._motion_key_body_indices, dtype=torch.long, device=self.device
        )
        self._teacher_key_body_ids, self._teacher_key_body_names = self.robot.find_bodies(
            IGRIS_STUDENT_KEY_BODY_ROBOT_ORDER, preserve_order=True
        )
        if len(self._teacher_key_body_ids) != len(self._motion_key_body_indices):
            raise ValueError("Teacher privileged key-body configuration is inconsistent between robot and motion data.")
        self._num_teacher_key_bodies = len(self._teacher_key_body_ids)
        self._teacher_privileged_motion_steps = torch.tensor(
            self.cfg.teacher_privileged_motion_steps,
            dtype=torch.float32,
            device=self.device,
        )
        self._teacher_foot_ids = torch.tensor(
            self._contact_sensor.find_bodies(["Link_Ankle_Roll_Left", "Link_Ankle_Roll_Right"])[0],
            dtype=torch.long,
            device=self.device,
        )
        self._teacher_mass_feature = self._compute_teacher_mass_feature()
        self._teacher_friction_feature = self.robot.data.default_joint_friction_coeff[:, self._body_joint_ids].mean(
            dim=1, keepdim=True
        )
        self._reference_marker_enabled = bool(getattr(self.cfg, "enable_reference_markers", False))
        self._reference_marker_translate_ops = []
        self._teacher_motor_strength_offset = torch.zeros(
            self.num_envs,
            2 * self._num_body_joints,
            dtype=torch.float32,
            device=self.device,
        )
        self.motion_difficulty = torch.full(
            (self._motion_lib.num_motions(),),
            float(self.cfg.motion_difficulty_max),
            dtype=torch.float32,
            device=self.device,
        )
        self.mean_motion_difficulty = float(self.cfg.motion_difficulty_max)
        self.max_key_body_error = torch.zeros(self._motion_lib.num_motions(), dtype=torch.float32, device=self.device)
        self._motion_visible_level = max(1, min(int(self.cfg.motion_level_curriculum_min), self._max_motion_curriculum_level))
        self._motion_level_success_rate = 0.0
        self._motion_level_success_total = 0
        self._motion_level_success_count = 0
        self._motion_level_batch_success_rate = 0.0
        self._motion_level_completion_rate = 0.0
        self._motion_level_root_pos_error = 0.0
        self._motion_level_dof_pos_error = 0.0
        self._motion_level_key_body_local_error = 0.0

        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._joint_pos_targets = torch.zeros_like(self._actions)

        self._motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._motion_time_offsets = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._motion_xy_offsets = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self.device)
        self._last_root_pos_w = self.robot.data.root_link_pos_w.clone()
        self._last_root_quat_w = self.robot.data.root_link_quat_w.clone()

        self._ref_root_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self._ref_root_rot = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        self._ref_root_vel = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self._ref_root_ang_vel = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self._ref_dof_pos = torch.zeros(self.num_envs, self._num_body_joints, dtype=torch.float32, device=self.device)
        self._ref_dof_vel = torch.zeros(self.num_envs, self._num_body_joints, dtype=torch.float32, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "alive",
                "track_root_pos",
                "track_root_rot",
                "track_root_vel",
                "track_root_ang_vel",
                "track_root_path_pos",
                "track_root_path_rot",
                "track_dof_pos",
                "track_dof_vel",
                "track_key_body",
                "track_key_body_global",
                "joint_acc",
                "ang_vel_xy",
                "ankle_dof_vel",
                "action_rate",
                "terminated",
            ]
        }
        self._done_reason_names = (
            "invalid_state",
            "root_height",
            "root_height_diff",
            "tilt",
            "max_root_lin_vel",
            "pose_tracking",
            "root_xy_tracking",
            "contact",
            "motion_end",
            "time_limit",
        )
        self._done_reason_buffers = {
            name: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) for name in self._done_reason_names
        }
        if self._reference_marker_enabled:
            self._initialize_reference_markers()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=True)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _motion_times(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            return self.episode_length_buf * self.step_dt + self._motion_time_offsets
        return self.episode_length_buf[env_ids] * self.step_dt + self._motion_time_offsets[env_ids]

    def _clear_done_reason_buffers(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            for mask in self._done_reason_buffers.values():
                mask.zero_()
            return
        for mask in self._done_reason_buffers.values():
            mask[env_ids] = False

    def _motion_level_curriculum_progress(self) -> float:
        if not self.cfg.enable_motion_level_curriculum or self._max_motion_curriculum_level <= 1:
            return 1.0
        if self.num_envs == 1 and self.cfg.full_motion_level_curriculum_for_single_env:
            return 1.0
        if self._motion_curriculum_mode == "success":
            min_visible = max(1, int(self.cfg.motion_level_curriculum_min))
            min_visible = min(min_visible, self._max_motion_curriculum_level)
            if self._max_motion_curriculum_level <= min_visible:
                return 1.0
            return (self._motion_visible_level - min_visible) / float(self._max_motion_curriculum_level - min_visible)

        start_step = int(self.cfg.motion_level_curriculum_start_step)
        full_step = int(self.cfg.motion_level_curriculum_full_step)
        if full_step <= start_step:
            return 1.0
        common_step_counter = int(getattr(self, "common_step_counter", 0))
        if common_step_counter <= start_step:
            return 0.0

        return min((common_step_counter - start_step) / float(full_step - start_step), 1.0)

    def _visible_motion_curriculum_level(self) -> int:
        if self._max_motion_curriculum_level <= 1:
            return 1
        if not self.cfg.enable_motion_level_curriculum:
            return self._max_motion_curriculum_level
        if self.num_envs == 1 and self.cfg.full_motion_level_curriculum_for_single_env:
            return self._max_motion_curriculum_level
        if self._motion_curriculum_mode == "success":
            return self._motion_visible_level

        min_visible = max(1, int(self.cfg.motion_level_curriculum_min))
        min_visible = min(min_visible, self._max_motion_curriculum_level)
        progress = self._motion_level_curriculum_progress()
        additional_levels = self._max_motion_curriculum_level - min_visible
        return min_visible + int(progress * additional_levels + 1.0e-6)

    def _update_motion_level_curriculum(self, env_ids: torch.Tensor) -> None:
        if not self.cfg.enable_motion_level_curriculum or self._max_motion_curriculum_level <= 1:
            return
        if self._motion_curriculum_mode != "success":
            return
        if self.num_envs == 1 and self.cfg.full_motion_level_curriculum_for_single_env:
            self._motion_visible_level = self._max_motion_curriculum_level
            self._motion_level_success_rate = 1.0
            self._motion_level_success_total = 0
            self._motion_level_success_count = 0
            return

        current_level = self._motion_visible_level
        reset_motion_levels = self._motion_lib.get_motion_curriculum_levels(self._motion_ids[env_ids])
        current_level_mask = reset_motion_levels == current_level
        if not torch.any(current_level_mask):
            return

        current_level_env_ids = env_ids[current_level_mask]
        completion_rate = self._compute_motion_completion_rate(current_level_env_ids)
        root_pos_error, dof_pos_error, key_body_local_error, _ = self._compute_tracking_error_terms(current_level_env_ids)
        success_mask = completion_rate >= float(self.cfg.motion_level_curriculum_success_completion)
        root_pos_threshold = float(self.cfg.motion_level_curriculum_success_root_pos_threshold)
        if root_pos_threshold > 0.0:
            success_mask &= root_pos_error <= root_pos_threshold
        dof_pos_threshold = float(self.cfg.motion_level_curriculum_success_dof_pos_threshold)
        if dof_pos_threshold > 0.0:
            success_mask &= dof_pos_error <= dof_pos_threshold
        key_body_local_threshold = float(self.cfg.motion_level_curriculum_success_key_body_local_threshold)
        if key_body_local_threshold > 0.0:
            success_mask &= key_body_local_error <= key_body_local_threshold
        success_mask &= torch.isfinite(completion_rate)
        success_mask &= torch.isfinite(root_pos_error)
        success_mask &= torch.isfinite(dof_pos_error)
        success_mask &= torch.isfinite(key_body_local_error)

        episode_count = int(completion_rate.numel())
        if episode_count <= 0:
            return
        success_count = int(torch.count_nonzero(success_mask).item())

        self._motion_level_batch_success_rate = success_count / float(episode_count)
        self._motion_level_completion_rate = torch.nan_to_num(
            completion_rate, nan=0.0, posinf=0.0, neginf=0.0
        ).mean().item()
        self._motion_level_root_pos_error = torch.nan_to_num(
            root_pos_error, nan=0.0, posinf=0.0, neginf=0.0
        ).mean().item()
        self._motion_level_dof_pos_error = torch.nan_to_num(
            dof_pos_error, nan=0.0, posinf=0.0, neginf=0.0
        ).mean().item()
        self._motion_level_key_body_local_error = torch.nan_to_num(
            key_body_local_error, nan=0.0, posinf=0.0, neginf=0.0
        ).mean().item()

        total_successes = self._motion_level_success_total + success_count
        total_episodes = self._motion_level_success_count + episode_count
        self._motion_level_success_total = total_successes
        self._motion_level_success_count = total_episodes
        self._motion_level_success_rate = total_successes / float(total_episodes)

        min_level_episodes = max(1, int(self.cfg.motion_level_curriculum_min_level_episodes))
        success_rate_threshold = float(self.cfg.motion_level_curriculum_success_threshold)
        if (
            current_level < self._max_motion_curriculum_level
            and total_episodes >= min_level_episodes
            and self._motion_level_success_rate >= success_rate_threshold
        ):
            self._motion_visible_level = min(self._motion_visible_level + 1, self._max_motion_curriculum_level)
            self._motion_level_success_rate = 0.0
            self._motion_level_success_total = 0
            self._motion_level_success_count = 0

    def _sample_motion_ids(self, n: int) -> torch.Tensor:
        motion_difficulty = None
        if self.cfg.motion_curriculum or self.cfg.use_error_aware_sampling:
            motion_difficulty = self.motion_difficulty
        visible_level = self._visible_motion_curriculum_level()

        return self._motion_lib.sample_motions(
            n,
            motion_difficulty=motion_difficulty,
            max_curriculum_level=visible_level,
            frontier_curriculum_level=visible_level if self.cfg.enable_motion_level_curriculum else None,
            frontier_level_boost=self.cfg.motion_level_frontier_sampling_boost,
            balance_motion_groups=self.cfg.balance_motion_groups,
            motion_group_balance_power=self.cfg.motion_group_balance_power,
            max_key_body_error=self.max_key_body_error if self.cfg.use_error_aware_sampling else None,
            use_error_aware_sampling=self.cfg.use_error_aware_sampling,
            error_sampling_power=self.cfg.error_sampling_power,
            error_sampling_threshold=self.cfg.error_sampling_threshold,
        )

    def _update_error_aware_sampling_metrics(self) -> None:
        if not self.cfg.use_error_aware_sampling:
            return

        current_key_body_local = self._compute_current_key_body_local().reshape(self.num_envs, self._num_teacher_key_bodies, 3)
        ref_key_body_local, _ = self._compute_reference_key_body_features()

        key_body_error_per_part = torch.mean(torch.abs(current_key_body_local - ref_key_body_local), dim=-1)
        current_max_error = torch.amax(key_body_error_per_part, dim=-1)

        motion_max_error = torch.full_like(self.max_key_body_error, float("-inf"))
        motion_max_error.scatter_reduce_(0, self._motion_ids, current_max_error, reduce="amax", include_self=False)
        valid_motion_mask = torch.isfinite(motion_max_error)
        self.max_key_body_error[valid_motion_mask] = torch.maximum(
            self.max_key_body_error[valid_motion_mask],
            motion_max_error[valid_motion_mask],
        )

    def _compute_motion_completion_rate(self, env_ids: torch.Tensor) -> torch.Tensor:
        motion_lengths = self._motion_lib.get_motion_length(self._motion_ids[env_ids])
        episode_horizon = torch.full_like(motion_lengths, self.max_episode_length_s)
        target_duration = torch.minimum(motion_lengths, episode_horizon)
        elapsed_time = self.episode_length_buf[env_ids].float() * self.step_dt
        return torch.clamp(elapsed_time / torch.clamp(target_duration, min=self.step_dt), min=0.0, max=1.0)

    def _update_motion_difficulty(self, env_ids: torch.Tensor) -> None:
        reset_motion_ids = self._motion_ids[env_ids]
        completion_rate = self._compute_motion_completion_rate(env_ids)

        motion_completion_rate_sum = torch.zeros_like(self.motion_difficulty)
        motion_completion_rate_sum.scatter_add_(0, reset_motion_ids, completion_rate)
        motion_completion_rate_count = torch.zeros_like(self.motion_difficulty)
        motion_completion_rate_count.scatter_add_(0, reset_motion_ids, torch.ones_like(completion_rate))

        motion_completion_rate = motion_completion_rate_sum / torch.clamp(motion_completion_rate_count, min=1.0)
        motion_completion_rate[motion_completion_rate_count == 0] = 0.7

        add_idx = motion_completion_rate <= 0.5
        sub_idx = (motion_completion_rate >= 0.95) & (motion_completion_rate < 0.99)
        super_sub_idx = motion_completion_rate >= 0.99

        self.motion_difficulty[add_idx] *= 1.0 + self.cfg.motion_curriculum_gamma
        self.motion_difficulty[sub_idx] *= 1.0 - self.cfg.motion_curriculum_gamma
        self.motion_difficulty[super_sub_idx] *= 1.0 - self.cfg.motion_curriculum_gamma * 20.0
        self.motion_difficulty.clamp_(min=self.cfg.motion_difficulty_min, max=self.cfg.motion_difficulty_max)
        self.mean_motion_difficulty = self.motion_difficulty.mean().item()

    def _set_reference_buffers(
        self,
        env_ids: torch.Tensor,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        root_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        dof_pos: torch.Tensor,
        dof_vel: torch.Tensor,
    ) -> None:
        root_pos = root_pos.clone()
        root_pos[:, :2] += self._motion_xy_offsets[env_ids]
        root_pos[:, 2] += self.cfg.motion_height_offset

        self._ref_root_pos[env_ids] = root_pos
        self._ref_root_rot[env_ids] = root_rot
        self._ref_root_vel[env_ids] = root_vel
        self._ref_root_ang_vel[env_ids] = root_ang_vel
        self._ref_dof_pos[env_ids] = dof_pos
        self._ref_dof_vel[env_ids] = dof_vel
        self._update_reference_markers()

    def _reset_reference_motion(self, env_ids: torch.Tensor) -> None:
        motion_ids = self._sample_motion_ids(len(env_ids))
        motion_times = self._motion_lib.sample_time(motion_ids)

        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel = self._motion_lib.calc_motion_frame(
            motion_ids, motion_times
        )

        self._motion_ids[env_ids] = motion_ids
        self._motion_time_offsets[env_ids] = motion_times
        desired_xy = self.scene.env_origins[env_ids, :2]
        desired_xy += torch.empty_like(desired_xy).uniform_(-self.cfg.reset_root_xy_noise, self.cfg.reset_root_xy_noise)
        self._motion_xy_offsets[env_ids] = desired_xy - root_pos[:, :2]
        self._set_reference_buffers(env_ids, root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel)

    def _update_reference_motion(self) -> None:
        motion_times = self._motion_times()
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel = self._motion_lib.calc_motion_frame(
            self._motion_ids, motion_times
        )
        self._set_reference_buffers(
            self.robot._ALL_INDICES, root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        unclipped_targets = self._ref_dof_pos + self.cfg.action_scale * self._actions
        joint_limits = self.robot.data.soft_joint_pos_limits[:, self._body_joint_ids]
        self._joint_pos_targets = torch.clamp(unclipped_targets, joint_limits[..., 0], joint_limits[..., 1])

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._joint_pos_targets, joint_ids=self._body_joint_ids)

    def _compute_teacher_mass_feature(self) -> torch.Tensor:
        default_mass = self.robot.data.default_mass
        total_mass = torch.clamp(default_mass.sum(dim=1, keepdim=True), min=1.0e-6)
        root_mass = default_mass[:, :1]
        max_body_mass = default_mass.amax(dim=1, keepdim=True)
        body_mass_std = default_mass.std(dim=1, keepdim=True)
        mean_total_mass = torch.clamp(total_mass.mean(), min=1.0e-6)
        return torch.cat(
            (
                total_mass / mean_total_mass,
                root_mass / total_mass,
                max_body_mass / total_mass,
                body_mass_std / total_mass,
            ),
            dim=-1,
        )

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

    def _compute_current_root_pos_delta_local(self) -> torch.Tensor:
        return quat_apply_inverse(self._last_root_quat_w, self.robot.data.root_link_pos_w - self._last_root_pos_w)

    def _key_body_local_from_offsets(self, root_quat: torch.Tensor, key_body_offsets: torch.Tensor) -> torch.Tensor:
        yaw_only_root_quat = _yaw_only_quat(root_quat)
        root_quat_expand = yaw_only_root_quat.unsqueeze(-2).expand(*key_body_offsets.shape[:-1], 4)
        key_body_local = quat_apply_inverse(
            root_quat_expand.reshape(-1, 4),
            key_body_offsets.reshape(-1, 3),
        )
        return key_body_local.reshape(*key_body_offsets.shape[:-2], -1)

    def _reference_key_body_local_from_motion(
        self,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        local_body_pos: torch.Tensor,
    ) -> torch.Tensor:
        key_body_local_full = torch.index_select(local_body_pos, dim=-2, index=self._motion_key_body_idx_tensor)
        root_quat_expand = root_rot.unsqueeze(-2).expand(*key_body_local_full.shape[:-1], 4)
        key_body_offsets_world = quat_apply(
            root_quat_expand.reshape(-1, 4),
            key_body_local_full.reshape(-1, 3),
        ).reshape_as(key_body_local_full)
        return self._key_body_local_from_offsets(root_rot, key_body_offsets_world)

    def _compute_current_key_body_local(self) -> torch.Tensor:
        key_body_offsets = self.robot.data.body_pos_w[:, self._teacher_key_body_ids] - self.robot.data.root_link_pos_w.unsqueeze(1)
        return self._key_body_local_from_offsets(self.robot.data.root_link_quat_w, key_body_offsets)

    def _compute_latest_foot_contacts(self) -> torch.Tensor:
        contact_force_history_z = self._contact_sensor.data.net_forces_w_history[:, :, self._teacher_foot_ids, 2]
        latest_contacts = torch.max(contact_force_history_z, dim=1)[0] > self.cfg.teacher_contact_force_threshold
        return latest_contacts.float()

    def _build_teacher_motion_step_obs(
        self,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        root_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        dof_pos: torch.Tensor,
        local_body_pos: torch.Tensor,
        root_pos_delta_local: torch.Tensor,
        root_rot_delta_local: torch.Tensor,
    ) -> torch.Tensor:
        batch_shape = root_pos.shape[:-1]
        root_vel_local = quat_apply_inverse(root_rot.reshape(-1, 4), root_vel.reshape(-1, 3)).reshape(*batch_shape, 3)
        root_ang_vel_local = quat_apply_inverse(root_rot.reshape(-1, 4), root_ang_vel.reshape(-1, 3)).reshape(
            *batch_shape, 3
        )
        roll, pitch, yaw = _roll_pitch_yaw_from_quat(root_rot.reshape(-1, 4))
        roll = roll.reshape(*batch_shape, 1)
        pitch = pitch.reshape(*batch_shape, 1)
        yaw = yaw.reshape(*batch_shape, 1)
        key_body_local = self._reference_key_body_local_from_motion(root_pos, root_rot, local_body_pos)
        current_root_pos = self.robot.data.root_link_pos_w.view(self.num_envs, *([1] * max(root_pos.dim() - 2, 0)), 3)
        root_pos_distance_to_target = root_pos - current_root_pos
        return torch.cat(
            (
                root_pos,
                root_pos_distance_to_target,
                roll,
                pitch,
                yaw,
                root_vel_local,
                root_ang_vel_local,
                root_pos_delta_local,
                root_rot_delta_local,
                dof_pos,
                key_body_local,
            ),
            dim=-1,
        )

    def _build_current_teacher_motion_obs(self) -> torch.Tensor:
        (
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            _,
            local_body_pos,
            root_pos_delta_local,
            root_rot_delta_local,
        ) = self._motion_lib.calc_motion_frame_with_local_info(self._motion_ids, self._motion_times())
        root_pos = root_pos.clone()
        root_pos[:, :2] += self._motion_xy_offsets
        root_pos[:, 2] += self.cfg.motion_height_offset
        return self._build_teacher_motion_step_obs(
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            local_body_pos,
            root_pos_delta_local,
            root_rot_delta_local,
        )

    def _build_teacher_privileged_motion_obs(self) -> torch.Tensor:
        if self._teacher_privileged_motion_steps.numel() == 0:
            return torch.zeros(self.num_envs, 0, dtype=torch.float32, device=self.device)

        motion_ids = self._motion_ids.unsqueeze(-1).expand(-1, self._teacher_privileged_motion_steps.numel())
        motion_times = self._motion_times().unsqueeze(-1) + self._teacher_privileged_motion_steps.unsqueeze(0) * self.step_dt
        (
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            _,
            local_body_pos,
            root_pos_delta_local,
            root_rot_delta_local,
        ) = self._motion_lib.calc_motion_frame_batch_with_local_info(motion_ids, motion_times)

        root_pos = root_pos.clone()
        root_pos[..., :2] += self._motion_xy_offsets.unsqueeze(1)
        root_pos[..., 2] += self.cfg.motion_height_offset
        teacher_motion_obs = self._build_teacher_motion_step_obs(
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            local_body_pos,
            root_pos_delta_local,
            root_rot_delta_local,
        )
        return teacher_motion_obs.reshape(self.num_envs, -1)

    def _build_teacher_privileged_info(self) -> torch.Tensor:
        teacher_privileged_info = (
            self.robot.data.root_lin_vel_b,
            self.robot.data.root_link_pos_w,
            self.robot.data.root_link_quat_w,
            self._compute_current_key_body_local(),
            self._compute_latest_foot_contacts(),
            self._teacher_mass_feature,
            self._teacher_friction_feature,
            self._teacher_motor_strength_offset,
        )
        return torch.cat(
            tuple(tensor.to(device=self.device, dtype=torch.float32) for tensor in teacher_privileged_info),
            dim=-1,
        )

    def _build_teacher_privileged_obs(self) -> torch.Tensor:
        return torch.cat(
            (
                self._build_current_teacher_motion_obs(),
                self._build_proprio_obs(),
                self._build_teacher_privileged_info(),
                self._build_teacher_privileged_motion_obs(),
            ),
            dim=-1,
        )

    def _compute_reference_privileged_motion_features(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        motion_times = self._motion_times()
        _, root_rot, _, _, _, _, local_body_pos, root_pos_delta_local, root_rot_delta_local = (
            self._motion_lib.calc_motion_frame_with_local_info(self._motion_ids, motion_times)
        )
        ref_key_body_local = self._reference_key_body_local_from_motion(self._ref_root_pos, root_rot, local_body_pos)
        return ref_key_body_local, root_pos_delta_local, root_rot_delta_local

    def _compute_reference_key_body_features(self) -> tuple[torch.Tensor, torch.Tensor]:
        motion_times = self._motion_times()
        _, root_rot, _, _, _, _, local_body_pos, _, _ = self._motion_lib.calc_motion_frame_with_local_info(
            self._motion_ids, motion_times
        )
        ref_key_body_local = self._reference_key_body_local_from_motion(self._ref_root_pos, root_rot, local_body_pos).reshape(
            self.num_envs, self._num_teacher_key_bodies, 3
        )
        ref_key_body_full_local = torch.index_select(local_body_pos, dim=1, index=self._motion_key_body_idx_tensor)
        root_quat_expand = root_rot.unsqueeze(1).expand(-1, self._num_teacher_key_bodies, -1)
        ref_key_body_global = quat_apply(
            root_quat_expand.reshape(-1, 4),
            ref_key_body_full_local.reshape(-1, 3),
        ).reshape(self.num_envs, self._num_teacher_key_bodies, 3)
        ref_key_body_global = ref_key_body_global + self._ref_root_pos.unsqueeze(1)
        return ref_key_body_local, ref_key_body_global

    def _initialize_reference_markers(self) -> None:
        try:
            import omni.usd
            from pxr import Gf, UsdGeom
        except ImportError as exc:
            print(f"[WARN] Reference marker visualization unavailable: {exc}")
            self._reference_marker_enabled = False
            return

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[WARN] USD stage unavailable, disabling reference marker visualization.")
            self._reference_marker_enabled = False
            return

        marker_root_path = "/World/ReferenceMotionMarkers"
        UsdGeom.Xform.Define(stage, marker_root_path)
        marker_color = Gf.Vec3f(*[float(value) for value in self.cfg.reference_marker_color])
        marker_radius = float(self.cfg.reference_marker_radius)

        for env_id in range(self.num_envs):
            env_marker_root_path = f"{marker_root_path}/env_{env_id}"
            UsdGeom.Xform.Define(stage, env_marker_root_path)
            env_translate_ops = []
            for key_body_name in self._teacher_key_body_names:
                prim_path = f"{env_marker_root_path}/{key_body_name}"
                sphere = UsdGeom.Sphere.Define(stage, prim_path)
                sphere.CreateRadiusAttr(marker_radius)
                sphere.CreateDisplayColorAttr([marker_color])
                translate_op = sphere.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(0.0, 0.0, -10.0))
                env_translate_ops.append(translate_op)
            self._reference_marker_translate_ops.append(env_translate_ops)

    def _update_reference_markers(self) -> None:
        if not self._reference_marker_enabled or not self._reference_marker_translate_ops or self.num_envs == 0:
            return

        try:
            from pxr import Gf
        except ImportError:
            return

        _, ref_key_body_global = self._compute_reference_key_body_features()
        marker_positions = ref_key_body_global.detach().cpu()
        for env_translate_ops, env_positions in zip(self._reference_marker_translate_ops, marker_positions, strict=True):
            for translate_op, position in zip(env_translate_ops, env_positions, strict=True):
                translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))

    def _compute_tracking_error_terms(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if env_ids.numel() == 0:
            empty = torch.empty(0, dtype=torch.float32, device=self.device)
            return empty, empty, empty, empty

        root_pos_error = torch.sum(torch.square(self.robot.data.root_link_pos_w[env_ids] - self._ref_root_pos[env_ids]), dim=1)
        dof_pos_error = torch.sum(
            torch.square(self.robot.data.joint_pos[env_ids][:, self._body_joint_ids] - self._ref_dof_pos[env_ids]),
            dim=1,
        )
        current_key_body_local = self._compute_current_key_body_local().reshape(self.num_envs, self._num_teacher_key_bodies, 3)
        ref_key_body_local, ref_key_body_global = self._compute_reference_key_body_features()
        current_key_body_global = self.robot.data.body_pos_w[:, self._teacher_key_body_ids]
        key_body_local_error = torch.mean(
            torch.norm(current_key_body_local[env_ids] - ref_key_body_local[env_ids], dim=-1),
            dim=1,
        )
        key_body_global_error = torch.mean(
            torch.norm(current_key_body_global[env_ids] - ref_key_body_global[env_ids], dim=-1),
            dim=1,
        )
        return root_pos_error, dof_pos_error, key_body_local_error, key_body_global_error

    def _compute_eval_metrics(self, env_ids: torch.Tensor) -> dict[str, float]:
        if env_ids.numel() == 0:
            return {}

        root_pos_error, dof_pos_error, key_body_local_error, key_body_global_error = self._compute_tracking_error_terms(
            env_ids
        )

        return {
            "Eval/root_pos_error": torch.nan_to_num(root_pos_error, nan=0.0, posinf=0.0, neginf=0.0).mean().item(),
            "Eval/dof_pos_error": torch.nan_to_num(dof_pos_error, nan=0.0, posinf=0.0, neginf=0.0).mean().item(),
            "Eval/key_body_local_error": torch.nan_to_num(
                key_body_local_error, nan=0.0, posinf=0.0, neginf=0.0
            ).mean().item(),
            "Eval/key_body_global_error": torch.nan_to_num(
                key_body_global_error, nan=0.0, posinf=0.0, neginf=0.0
            ).mean().item(),
        }

    def _compute_tracking_observation(self) -> torch.Tensor:
        root_pos_error_b = quat_apply_inverse(self._ref_root_rot, self.robot.data.root_link_pos_w - self._ref_root_pos)
        ref_root_vel_local = quat_apply_inverse(self._ref_root_rot, self._ref_root_vel)
        ref_root_ang_vel_local = quat_apply_inverse(self._ref_root_rot, self._ref_root_ang_vel)
        joint_pos_error = self.robot.data.joint_pos[:, self._body_joint_ids] - self._ref_dof_pos
        joint_vel_error = self.robot.data.joint_vel[:, self._body_joint_ids] - self._ref_dof_vel

        return torch.cat(
            (
                self.robot.data.projected_gravity_b,
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
                root_pos_error_b,
                ref_root_vel_local,
                ref_root_ang_vel_local,
                joint_pos_error,
                joint_vel_error,
                self._actions,
            ),
            dim=-1,
        )

    def _get_observations(self) -> dict:
        self._update_reference_motion()
        obs = self._compute_tracking_observation()
        self._previous_actions.copy_(self._actions)
        self._last_root_pos_w.copy_(self.robot.data.root_link_pos_w)
        self._last_root_quat_w.copy_(self.robot.data.root_link_quat_w)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self._update_error_aware_sampling_metrics()

        invalid_state = ~torch.isfinite(self.robot.data.root_pose_w).all(dim=1)
        invalid_state |= ~torch.isfinite(self.robot.data.joint_pos).all(dim=1)
        invalid_state |= ~torch.isfinite(self.robot.data.joint_vel).all(dim=1)

        rewards = {key: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device) for key in self._episode_sums}
        ref_key_body_local, ref_root_pos_delta_local, ref_root_rot_delta_local = (
            self._compute_reference_privileged_motion_features()
        )
        current_root_pos_delta_local = self._compute_current_root_pos_delta_local()
        current_root_rot_delta_local = self._compute_current_root_rot_delta_local()
        reset_mask = self.episode_length_buf <= 1
        if reset_mask.any():
            current_root_pos_delta_local = current_root_pos_delta_local.clone()
            current_root_rot_delta_local = current_root_rot_delta_local.clone()
            current_root_pos_delta_local[reset_mask] = 0.0
            current_root_rot_delta_local[reset_mask] = 0.0

        if self.cfg.rew_scale_alive != 0.0:
            rewards["alive"] = torch.full_like(rewards["alive"], self.cfg.rew_scale_alive * self.step_dt)

        if self.cfg.rew_scale_track_root_pos != 0.0:
            root_pos_error = torch.sum(torch.square(self.robot.data.root_link_pos_w - self._ref_root_pos), dim=1)
            rewards["track_root_pos"] = self.cfg.rew_scale_track_root_pos * torch.exp(-4.0 * root_pos_error) * self.step_dt
        if self.cfg.rew_scale_track_root_rot != 0.0:
            root_rot_error = quat_error_magnitude(self.robot.data.root_link_quat_w, self._ref_root_rot)
            rewards["track_root_rot"] = (
                self.cfg.rew_scale_track_root_rot * torch.exp(-2.0 * torch.square(root_rot_error)) * self.step_dt
            )
        if self.cfg.rew_scale_track_root_vel != 0.0:
            root_vel_error = torch.sum(torch.square(self.robot.data.root_link_lin_vel_w - self._ref_root_vel), dim=1)
            rewards["track_root_vel"] = self.cfg.rew_scale_track_root_vel * torch.exp(-2.0 * root_vel_error) * self.step_dt
        if self.cfg.rew_scale_track_root_ang_vel != 0.0:
            root_ang_vel_error = torch.sum(
                torch.square(self.robot.data.root_link_ang_vel_w - self._ref_root_ang_vel), dim=1
            )
            rewards["track_root_ang_vel"] = (
                self.cfg.rew_scale_track_root_ang_vel * torch.exp(-0.5 * root_ang_vel_error) * self.step_dt
            )
        if self.cfg.rew_scale_track_root_path_pos != 0.0:
            root_path_pos_error = torch.sum(torch.square(ref_root_pos_delta_local - current_root_pos_delta_local), dim=1)
            rewards["track_root_path_pos"] = (
                self.cfg.rew_scale_track_root_path_pos * torch.exp(-1.0 * root_path_pos_error) * self.step_dt
            )
        if self.cfg.rew_scale_track_root_path_rot != 0.0:
            root_path_rot_error = torch.sum(torch.square(ref_root_rot_delta_local - current_root_rot_delta_local), dim=1)
            rewards["track_root_path_rot"] = (
                self.cfg.rew_scale_track_root_path_rot * torch.exp(-1.0 * root_path_rot_error) * self.step_dt
            )
        if self.cfg.rew_scale_track_dof_pos != 0.0:
            dof_pos_error = torch.sum(
                torch.square(self.robot.data.joint_pos[:, self._body_joint_ids] - self._ref_dof_pos), dim=1
            )
            rewards["track_dof_pos"] = self.cfg.rew_scale_track_dof_pos * torch.exp(-0.25 * dof_pos_error) * self.step_dt
        if self.cfg.rew_scale_track_dof_vel != 0.0:
            dof_vel_error = torch.sum(
                torch.square(self.robot.data.joint_vel[:, self._body_joint_ids] - self._ref_dof_vel), dim=1
            )
            rewards["track_dof_vel"] = self.cfg.rew_scale_track_dof_vel * torch.exp(-0.05 * dof_vel_error) * self.step_dt
        if self.cfg.rew_scale_track_key_body != 0.0 or self.cfg.rew_scale_track_key_body_global != 0.0:
            current_key_body_local = self._compute_current_key_body_local().reshape(
                self.num_envs, self._num_teacher_key_bodies, 3
            )
            ref_key_body_local = ref_key_body_local.reshape(self.num_envs, self._num_teacher_key_bodies, 3)
            _, ref_key_body_global = self._compute_reference_key_body_features()
            if self.cfg.rew_scale_track_key_body != 0.0:
                key_body_local_error = torch.mean(torch.square(current_key_body_local - ref_key_body_local), dim=(1, 2))
                rewards["track_key_body"] = (
                    self.cfg.rew_scale_track_key_body * torch.exp(-2.0 * key_body_local_error) * self.step_dt
                )
            if self.cfg.rew_scale_track_key_body_global != 0.0:
                current_key_body_global = self.robot.data.body_pos_w[:, self._teacher_key_body_ids]
                key_body_global_error = torch.mean(torch.square(current_key_body_global - ref_key_body_global), dim=(1, 2))
                rewards["track_key_body_global"] = (
                    self.cfg.rew_scale_track_key_body_global
                    * torch.exp(-2.0 * key_body_global_error)
                    * self.step_dt
                )
        if self.cfg.rew_scale_joint_acc != 0.0:
            joint_acc_penalty = torch.sum(torch.square(self.robot.data.joint_acc[:, self._body_joint_ids]), dim=1)
            rewards["joint_acc"] = self.cfg.rew_scale_joint_acc * joint_acc_penalty * self.step_dt
        if self.cfg.rew_scale_ang_vel_xy != 0.0:
            ang_vel_xy_penalty = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)
            rewards["ang_vel_xy"] = self.cfg.rew_scale_ang_vel_xy * ang_vel_xy_penalty * self.step_dt
        if self.cfg.rew_scale_ankle_dof_vel != 0.0:
            ankle_dof_vel_penalty = torch.sum(
                torch.square(self.robot.data.joint_vel[:, self._body_joint_ids][:, self._ankle_joint_indices]), dim=1
            )
            rewards["ankle_dof_vel"] = self.cfg.rew_scale_ankle_dof_vel * ankle_dof_vel_penalty * self.step_dt
        if self.cfg.rew_scale_action_rate != 0.0:
            action_rate_penalty = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
            rewards["action_rate"] = self.cfg.rew_scale_action_rate * action_rate_penalty * self.step_dt
        if self.cfg.rew_scale_terminated != 0.0:
            rewards["terminated"] = self.cfg.rew_scale_terminated * self.reset_terminated.float() * self.step_dt

        reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for key, value in rewards.items():
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            value[invalid_state] = 0.0
            rewards[key] = value
            reward += value
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_limit = self.episode_length_buf >= self.max_episode_length - 1
        invalid_state = ~torch.isfinite(self.robot.data.root_pose_w).all(dim=1)
        invalid_state |= ~torch.isfinite(self.robot.data.joint_pos).all(dim=1)
        invalid_state |= ~torch.isfinite(self.robot.data.joint_vel).all(dim=1)
        invalid_state |= ~torch.isfinite(self.robot.data.body_pos_w).all(dim=(1, 2))
        invalid_state |= ~torch.isfinite(self.robot.data.joint_acc).all(dim=1)

        motion_end = self._motion_times() >= self._motion_lib.get_motion_length(self._motion_ids)
        motion_end_time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = time_limit.clone()
        if getattr(self.cfg, "terminate_on_motion_end", False) and not self.sim.has_gui():
            motion_end_time_out = motion_end
            time_out |= motion_end_time_out

        root_height = self.robot.data.root_link_pos_w[:, 2]
        root_height_fail = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.termination_root_height > 0.0:
            root_height_fail = root_height < self.cfg.termination_root_height

        root_height_diff_fail = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if getattr(self.cfg, "termination_root_height_diff", 0.0) > 0.0:
            root_height_diff_fail = (self._ref_root_pos[:, 2] - root_height) > float(self.cfg.termination_root_height_diff)

        tilt_fail = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.termination_tilt > 0.0:
            flat_orientation = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
            tilt_fail = flat_orientation > self.cfg.termination_tilt

        root_lin_vel_fail = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if getattr(self.cfg, "termination_max_root_lin_vel", 0.0) > 0.0:
            root_lin_vel_fail = (
                torch.linalg.vector_norm(self.robot.data.root_link_lin_vel_w, dim=1)
                > float(self.cfg.termination_max_root_lin_vel)
            )

        pose_tracking_fail = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if getattr(self.cfg, "termination_pose_tracking_dist", 0.0) > 0.0:
            current_key_body_local = self._compute_current_key_body_local().reshape(
                self.num_envs, self._num_teacher_key_bodies, 3
            )
            ref_key_body_local, _ = self._compute_reference_key_body_features()
            pose_error = torch.sum(torch.square(current_key_body_local - ref_key_body_local), dim=-1).amax(dim=1)
            pose_tracking_fail = pose_error > float(self.cfg.termination_pose_tracking_dist) ** 2

        root_xy_tracking_fail = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if getattr(self.cfg, "termination_root_pos_tracking_dist", 0.0) > 0.0:
            root_xy_error = torch.sum(torch.square(self.robot.data.root_link_pos_w[:, :2] - self._ref_root_pos[:, :2]), dim=1)
            root_xy_tracking_fail = root_xy_error > float(self.cfg.termination_root_pos_tracking_dist) ** 2

        terminated = (
            root_height_fail
            | root_height_diff_fail
            | tilt_fail
            | root_lin_vel_fail
            | pose_tracking_fail
            | root_xy_tracking_fail
            | invalid_state
        )

        self._done_reason_buffers["invalid_state"].copy_(invalid_state)
        self._done_reason_buffers["root_height"].copy_(root_height_fail)
        self._done_reason_buffers["root_height_diff"].copy_(root_height_diff_fail)
        self._done_reason_buffers["tilt"].copy_(tilt_fail)
        self._done_reason_buffers["max_root_lin_vel"].copy_(root_lin_vel_fail)
        self._done_reason_buffers["pose_tracking"].copy_(pose_tracking_fail)
        self._done_reason_buffers["root_xy_tracking"].copy_(root_xy_tracking_fail)
        self._done_reason_buffers["contact"].zero_()
        self._done_reason_buffers["motion_end"].copy_(motion_end_time_out)
        self._done_reason_buffers["time_limit"].copy_(time_limit)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        env_ids_tensor = env_ids if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=self.device)
        eval_metrics = self._compute_eval_metrics(env_ids_tensor)
        self._update_motion_level_curriculum(env_ids_tensor)
        if self.cfg.motion_curriculum:
            self._update_motion_difficulty(env_ids_tensor)

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self._reset_reference_motion(env_ids_tensor)

        self._actions[env_ids_tensor] = 0.0
        self._previous_actions[env_ids_tensor] = 0.0

        joint_pos = self.robot.data.default_joint_pos[env_ids_tensor].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids_tensor].clone()
        joint_pos[:, self._body_joint_ids] = self._ref_dof_pos[env_ids_tensor]
        joint_vel[:, self._body_joint_ids] = self._ref_dof_vel[env_ids_tensor]

        body_noise = torch.empty((len(env_ids_tensor), self._num_body_joints), device=self.device)
        body_noise.uniform_(-self.cfg.reset_joint_noise, self.cfg.reset_joint_noise)
        joint_pos[:, self._body_joint_ids] += body_noise

        joint_limits = self.robot.data.soft_joint_pos_limits[env_ids_tensor][:, self._body_joint_ids]
        joint_pos[:, self._body_joint_ids] = torch.clamp(
            joint_pos[:, self._body_joint_ids], joint_limits[..., 0], joint_limits[..., 1]
        )

        root_pose = self.robot.data.default_root_state[env_ids_tensor].clone()
        root_pose[:, :3] = self._ref_root_pos[env_ids_tensor]
        root_pose[:, 3:7] = self._ref_root_rot[env_ids_tensor]
        root_vel = torch.zeros_like(root_pose[:, 7:])
        root_vel[:, :3] = self._ref_root_vel[env_ids_tensor]
        root_vel[:, 3:6] = self._ref_root_ang_vel[env_ids_tensor]

        self.robot.write_root_pose_to_sim(root_pose[:, :7], env_ids_tensor)
        self.robot.write_root_velocity_to_sim(root_vel, env_ids_tensor)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids_tensor)
        self._joint_pos_targets[env_ids_tensor] = joint_pos[:, self._body_joint_ids]
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids_tensor)
        self._last_root_pos_w[env_ids_tensor] = self._ref_root_pos[env_ids_tensor]
        self._last_root_quat_w[env_ids_tensor] = self._ref_root_rot[env_ids_tensor]

        extras = {}
        for key in self._episode_sums:
            extras[f"Episode_Reward/{key}"] = torch.mean(self._episode_sums[key][env_ids_tensor]) / self.max_episode_length_s
            self._episode_sums[key][env_ids_tensor] = 0.0
        extras["Episode_Termination/terminated"] = torch.count_nonzero(self.reset_terminated[env_ids_tensor]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids_tensor]).item()
        terminated_envs = self.reset_terminated[env_ids_tensor]
        time_out_envs = self.reset_time_outs[env_ids_tensor]
        terminated_reason_order = (
            "invalid_state",
            "root_height",
            "root_height_diff",
            "tilt",
            "max_root_lin_vel",
            "pose_tracking",
            "root_xy_tracking",
            "contact",
        )
        time_out_reason_order = ("motion_end", "time_limit")
        for reason in terminated_reason_order:
            reason_mask = self._done_reason_buffers[reason][env_ids_tensor] & terminated_envs
            extras[f"Episode_TerminationReason/{reason}"] = torch.count_nonzero(reason_mask).item()
        for reason in time_out_reason_order:
            reason_mask = self._done_reason_buffers[reason][env_ids_tensor] & time_out_envs
            extras[f"Episode_TerminationReason/{reason}"] = torch.count_nonzero(reason_mask).item()

        remaining_terminated = terminated_envs.clone()
        for reason in terminated_reason_order:
            primary_mask = self._done_reason_buffers[reason][env_ids_tensor] & remaining_terminated
            extras[f"Episode_TerminationPrimary/{reason}"] = torch.count_nonzero(primary_mask).item()
            remaining_terminated &= ~primary_mask
        extras["Episode_TerminationPrimary/unassigned"] = torch.count_nonzero(remaining_terminated).item()

        remaining_time_out = time_out_envs.clone()
        for reason in time_out_reason_order:
            primary_mask = self._done_reason_buffers[reason][env_ids_tensor] & remaining_time_out
            extras[f"Episode_TimeOutPrimary/{reason}"] = torch.count_nonzero(primary_mask).item()
            remaining_time_out &= ~primary_mask
        extras["Episode_TimeOutPrimary/unassigned"] = torch.count_nonzero(remaining_time_out).item()
        if self.cfg.motion_curriculum:
            extras["Curriculum/mean_motion_difficulty"] = self.mean_motion_difficulty
            extras["Curriculum/max_motion_difficulty"] = self.motion_difficulty.max().item()
            extras["Curriculum/min_motion_difficulty"] = self.motion_difficulty.min().item()
        if self.cfg.enable_motion_level_curriculum and self._max_motion_curriculum_level > 1:
            extras["Curriculum/motion_visible_level"] = float(self._visible_motion_curriculum_level())
            extras["Curriculum/motion_level_progress"] = self._motion_level_curriculum_progress()
            extras["Curriculum/motion_level_max_available"] = float(self._max_motion_curriculum_level)
            extras["Curriculum/motion_level_frontier_boost"] = float(self.cfg.motion_level_frontier_sampling_boost)
            if self._motion_curriculum_mode == "success":
                extras["Curriculum/motion_level_success_rate"] = self._motion_level_success_rate
                extras["Curriculum/motion_level_success_count"] = float(self._motion_level_success_count)
                extras["Curriculum/motion_level_batch_success_rate"] = self._motion_level_batch_success_rate
                extras["Curriculum/motion_level_completion_rate"] = self._motion_level_completion_rate
                extras["Curriculum/motion_level_root_pos_error"] = self._motion_level_root_pos_error
                extras["Curriculum/motion_level_dof_pos_error"] = self._motion_level_dof_pos_error
                extras["Curriculum/motion_level_key_body_local_error"] = self._motion_level_key_body_local_error
        if self.cfg.balance_motion_groups:
            extras["Curriculum/motion_group_balance_power"] = float(self.cfg.motion_group_balance_power)
        if self.cfg.use_error_aware_sampling:
            extras["Curriculum/max_key_body_error_mean"] = self.max_key_body_error.mean().item()
            extras["Curriculum/max_key_body_error_max"] = self.max_key_body_error.max().item()
            extras["Curriculum/high_error_motion_count"] = int(
                torch.count_nonzero(self.max_key_body_error >= self.cfg.error_sampling_threshold).item()
            )
        extras.update(eval_metrics)
        self.extras["log"] = extras
        self._clear_done_reason_buffers(env_ids_tensor)
