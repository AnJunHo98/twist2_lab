from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from .igris_base_teacher_env_cfg import (
    _TEACHER_PRIVILEGED_PROPRIO_DIM,
    _TEACHER_PRIVILEGED_SINGLE_STEP_DIM,
    IgrisBaseTeacherEnvCfg,
)
from .igris_mimic_teacher_env import IgrisMimicTeacherEnv


class IgrisBaseTeacherEnv(IgrisMimicTeacherEnv):
    cfg: IgrisBaseTeacherEnvCfg

    def __init__(self, cfg: IgrisBaseTeacherEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        history_len = int(self.cfg.history_len)
        if history_len > 0:
            self._obs_history_buf = torch.zeros(
                self.num_envs,
                history_len,
                _TEACHER_PRIVILEGED_PROPRIO_DIM,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self._obs_history_buf = None

        self._teacher_mass_feature = self._teacher_mass_feature.to(device=self.device, dtype=torch.float32)
        self._teacher_friction_feature = self._teacher_friction_feature.to(device=self.device, dtype=torch.float32)

        self._enable_teacher_phys_rand = bool(getattr(self.cfg, "enable_teacher_phys_rand", False))
        self._enable_teacher_root_com_rand = bool(getattr(self.cfg, "enable_teacher_root_com_rand", False))
        self._enable_teacher_gravity_rand = bool(getattr(self.cfg, "enable_teacher_gravity_rand", False))
        self._enable_teacher_robot_push = bool(getattr(self.cfg, "enable_teacher_robot_push", False))
        self._enable_teacher_end_effector_push = bool(getattr(self.cfg, "enable_teacher_end_effector_push", False))
        self._enable_teacher_action_delay = bool(getattr(self.cfg, "enable_teacher_action_delay", False))
        self._enable_teacher_obs_noise = bool(getattr(self.cfg, "enable_teacher_obs_noise", False))
        self._enable_teacher_force_curriculum = bool(getattr(self.cfg, "enable_teacher_force_curriculum", False))

        self._raw_actions = torch.zeros_like(self._actions)
        self._default_body_com_pose_b = self.robot.data.body_com_pose_b.clone()
        self._teacher_mass_scale = torch.ones(self.num_envs, 1, dtype=torch.float32, device=self.device)
        self._teacher_joint_friction_scale = torch.ones(self.num_envs, 1, dtype=torch.float32, device=self.device)
        self._teacher_root_com_offset = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self._teacher_default_gravity = torch.tensor(self.sim.cfg.gravity, dtype=torch.float32, device=self.device)
        self._teacher_current_gravity = self._teacher_default_gravity.clone()
        self._teacher_gravity_interval_steps = max(
            1,
            int(math.ceil(float(self.cfg.teacher_gravity_rand_interval_s) / self.step_dt)),
        )
        self._teacher_obs_noise_ramp_denom = max(
            1,
            int(float(self.cfg.teacher_obs_noise_increasing_steps) * float(self.cfg.teacher_obs_noise_step_multiplier)),
        )
        self._teacher_robot_push_interval_steps = max(
            1,
            int(math.ceil(float(self.cfg.teacher_robot_push_interval_s) / self.step_dt)),
        )
        self._teacher_end_effector_push_interval_steps = max(
            1,
            int(math.ceil(float(self.cfg.teacher_end_effector_push_interval_s) / self.step_dt)),
        )
        self._teacher_obs_noise_scale_vec = self._build_teacher_obs_noise_scale_vec()
        self._teacher_last_push_speed = 0.0
        self._teacher_last_end_effector_push_force = 0.0

        if self._enable_teacher_action_delay:
            if int(self.cfg.teacher_action_delay_buffer_len) < int(self.cfg.teacher_action_delay_max_steps) + 1:
                raise ValueError("teacher_action_delay_buffer_len must be at least teacher_action_delay_max_steps + 1.")
            self._teacher_action_history = torch.zeros(
                self.num_envs,
                int(self.cfg.teacher_action_delay_buffer_len),
                self.cfg.action_space,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self._teacher_action_history = torch.zeros(
                self.num_envs,
                1,
                self.cfg.action_space,
                dtype=torch.float32,
                device=self.device,
            )
        self._teacher_action_delay_steps = 0
        self._teacher_last_foot_contacts = torch.zeros(
            self.num_envs, len(self._teacher_foot_ids), dtype=torch.bool, device=self.device
        )
        self._teacher_feet_air_time = torch.zeros(
            self.num_envs, len(self._teacher_foot_ids), dtype=torch.float32, device=self.device
        )

        self._episode_sums["lin_vel_z"] = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_sums["orientation"] = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_sums["dof_pos_limits"] = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_sums["dof_vel"] = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_sums["feet_slip"] = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_sums["feet_contact_forces"] = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_sums["feet_stumble"] = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_sums["feet_air_time"] = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_sums["collision"] = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_sums["dof_torque_limits"] = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_sums["torque_penalty"] = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        penalized_contact_names = list(self.cfg.teacher_penalized_contact_body_names)
        termination_contact_names = list(self.cfg.teacher_termination_contact_body_names)
        penalized_contact_ids, _ = self.robot.find_bodies(penalized_contact_names, preserve_order=True)
        termination_contact_ids, _ = self.robot.find_bodies(termination_contact_names, preserve_order=True)
        if len(penalized_contact_ids) != len(self.cfg.teacher_penalized_contact_body_names):
            raise ValueError("Teacher penalized contact body mapping is inconsistent with the IGRIS asset.")
        if len(termination_contact_ids) != len(self.cfg.teacher_termination_contact_body_names):
            raise ValueError("Teacher termination contact body mapping is inconsistent with the IGRIS asset.")
        self._teacher_penalized_contact_ids = torch.tensor(
            self._contact_sensor.find_bodies(penalized_contact_names)[0],
            dtype=torch.long,
            device=self.device,
        )
        if termination_contact_names:
            self._teacher_termination_contact_ids = torch.tensor(
                self._contact_sensor.find_bodies(termination_contact_names)[0],
                dtype=torch.long,
                device=self.device,
            )
        else:
            self._teacher_termination_contact_ids = torch.zeros(0, dtype=torch.long, device=self.device)

        if self._enable_teacher_end_effector_push and self._enable_teacher_force_curriculum:
            raise ValueError("teacher_end_effector_push and teacher_force_curriculum cannot be enabled together.")

        if self._enable_teacher_end_effector_push:
            self._teacher_end_effector_push_body_ids, self._teacher_end_effector_push_body_names = self.robot.find_bodies(
                list(self.cfg.teacher_end_effector_push_body_names), preserve_order=True
            )
            if len(self._teacher_end_effector_push_body_ids) == 0:
                raise ValueError("Teacher end-effector push enabled, but no push bodies were found.")
            self._teacher_end_effector_push_forces = torch.zeros(
                self.num_envs,
                len(self._teacher_end_effector_push_body_ids),
                3,
                dtype=torch.float32,
                device=self.device,
            )
            self._teacher_end_effector_push_torques = torch.zeros_like(self._teacher_end_effector_push_forces)
        else:
            self._teacher_end_effector_push_body_ids = []
            self._teacher_end_effector_push_body_names = []
            self._teacher_end_effector_push_forces = None
            self._teacher_end_effector_push_torques = None

        if self._enable_teacher_force_curriculum:
            self._force_body_ids, self._force_body_names = self.robot.find_bodies(
                list(self.cfg.teacher_force_apply_body_names), preserve_order=True
            )
            if len(self._force_body_ids) == 0:
                raise ValueError("Teacher force curriculum enabled, but no force application bodies were found.")
            self._num_force_bodies = len(self._force_body_ids)
            self._force_scale = torch.full(
                (self.num_envs,),
                float(self.cfg.teacher_force_scale_initial),
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
            self._resample_teacher_force_commands(self.robot._ALL_INDICES)
        else:
            self._force_body_ids = []
            self._num_force_bodies = 0
            self._force_scale = None
            self._force_duration = None
            self._force_vectors = None
            self._force_torques = None

    def _sample_uniform_range(self, value_range: tuple[float, float], shape: tuple[int, ...]) -> torch.Tensor:
        low, high = value_range
        return torch.empty(shape, device=self.device, dtype=torch.float32).uniform_(low, high)

    def _env_ids_for_tensor(self, env_ids: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        return env_ids.to(dtype=torch.long, device=tensor.device)

    def _gather_env_tensor(self, tensor: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        return tensor[self._env_ids_for_tensor(env_ids, tensor)]

    def _build_teacher_obs_noise_scale_vec(self) -> torch.Tensor:
        if not self._enable_teacher_obs_noise:
            return torch.zeros(0, dtype=torch.float32, device=self.device)
        proprio_noise = torch.cat(
            (
                torch.full((3,), float(self.cfg.teacher_obs_noise_ang_vel), dtype=torch.float32, device=self.device),
                torch.full((2,), float(self.cfg.teacher_obs_noise_imu), dtype=torch.float32, device=self.device),
                torch.full(
                    (self._num_body_joints,),
                    float(self.cfg.teacher_obs_noise_dof_pos),
                    dtype=torch.float32,
                    device=self.device,
                ),
                torch.full(
                    (self._num_body_joints,),
                    float(self.cfg.teacher_obs_noise_dof_vel),
                    dtype=torch.float32,
                    device=self.device,
                ),
                torch.zeros(self._num_body_joints, dtype=torch.float32, device=self.device),
            ),
            dim=0,
        )
        return proprio_noise

    def _teacher_obs_noise_progress(self) -> float:
        if not self._enable_teacher_obs_noise:
            return 0.0
        if self.sim.has_gui():
            return 1.0
        return min(float(self.common_step_counter) / float(self._teacher_obs_noise_ramp_denom), 1.0)

    def _apply_teacher_obs_noise(self, obs: torch.Tensor) -> torch.Tensor:
        if not self._enable_teacher_obs_noise or self._teacher_obs_noise_scale_vec.numel() == 0:
            return obs
        noise_progress = self._teacher_obs_noise_progress()
        if noise_progress <= 0.0:
            return obs
        noise = (2.0 * torch.rand_like(obs) - 1.0) * self._teacher_obs_noise_scale_vec.unsqueeze(0) * noise_progress
        return obs + noise

    def _set_teacher_gravity(self, gravity: torch.Tensor) -> None:
        import carb

        gravity = gravity.to(dtype=torch.float32, device=self.device)
        gravity_cpu = gravity.detach().cpu()
        self.sim.physics_sim_view.set_gravity(carb.Float3(*[float(value) for value in gravity_cpu.tolist()]))
        self.sim.cfg.gravity = tuple(float(value) for value in gravity_cpu.tolist())

        gravity_dir = gravity / torch.clamp(torch.linalg.norm(gravity), min=1.0e-6)
        self.robot.data.GRAVITY_VEC_W[:] = gravity_dir.unsqueeze(0)
        self._teacher_current_gravity = gravity

    def _update_teacher_gravity_randomization(self) -> None:
        if not self._enable_teacher_gravity_rand:
            return
        if self.common_step_counter % self._teacher_gravity_interval_steps != 0:
            return
        gravity_offset = self._sample_uniform_range(self.cfg.teacher_gravity_range, (3,))
        self._set_teacher_gravity(self._teacher_default_gravity + gravity_offset)

    def _update_teacher_robot_push(self) -> None:
        if not self._enable_teacher_robot_push:
            return
        if self.common_step_counter <= 0 or self.common_step_counter % self._teacher_robot_push_interval_steps != 0:
            return

        pushed_root_vel = self.robot.data.root_vel_w.clone()
        pushed_root_vel[:, :2] = self._sample_uniform_range(
            (-float(self.cfg.teacher_robot_push_max_vel_xy), float(self.cfg.teacher_robot_push_max_vel_xy)),
            (self.num_envs, 2),
        )
        self.robot.write_root_velocity_to_sim(pushed_root_vel, self.robot._ALL_INDICES)
        self._teacher_last_push_speed = torch.linalg.vector_norm(pushed_root_vel[:, :2], dim=1).mean().item()

    def _update_teacher_end_effector_push(self) -> None:
        if not self._enable_teacher_end_effector_push:
            return

        self._teacher_end_effector_push_forces.zero_()
        if (
            self.common_step_counter > 0
            and self.common_step_counter % self._teacher_end_effector_push_interval_steps == 0
        ):
            self._teacher_end_effector_push_forces.uniform_(
                -float(self.cfg.teacher_end_effector_push_max_force),
                float(self.cfg.teacher_end_effector_push_max_force),
            )
            self._teacher_last_end_effector_push_force = (
                torch.linalg.vector_norm(self._teacher_end_effector_push_forces, dim=-1).mean().item()
            )
        else:
            self._teacher_last_end_effector_push_force = 0.0

        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=self._teacher_end_effector_push_forces,
            torques=self._teacher_end_effector_push_torques,
            body_ids=self._teacher_end_effector_push_body_ids,
        )

    def _compute_teacher_mass_feature_from_masses(self, masses: torch.Tensor) -> torch.Tensor:
        total_mass = torch.clamp(masses.sum(dim=1, keepdim=True), min=1.0e-6)
        root_mass = masses[:, :1]
        max_body_mass = masses.amax(dim=1, keepdim=True)
        body_mass_std = masses.std(dim=1, keepdim=True)
        default_mass = self.robot.data.default_mass
        if default_mass.device != masses.device:
            default_mass = default_mass.to(device=masses.device)
        mean_total_mass = torch.clamp(default_mass.sum(dim=1, keepdim=True).mean(), min=1.0e-6)
        return torch.cat(
            (
                total_mass / mean_total_mass,
                root_mass / total_mass,
                max_body_mass / total_mass,
                body_mass_std / total_mass,
            ),
            dim=-1,
        )

    def _update_teacher_privileged_physical_features(
        self, env_ids: torch.Tensor, masses_cpu: torch.Tensor | None = None
    ) -> None:
        if masses_cpu is None:
            masses_cpu = self.robot.root_physx_view.get_masses()
        masses = masses_cpu.to(device=self.device, dtype=torch.float32)
        masses_env_ids = self._env_ids_for_tensor(env_ids, masses)
        self._teacher_mass_feature[env_ids] = self._compute_teacher_mass_feature_from_masses(masses[masses_env_ids])

        joint_friction_coeff = self._gather_env_tensor(self.robot.data.joint_friction_coeff, env_ids).to(device=self.device)
        self._teacher_friction_feature[env_ids] = joint_friction_coeff[:, self._body_joint_ids].mean(dim=1, keepdim=True)

    def _randomize_teacher_physical_properties(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        if not self._enable_teacher_phys_rand and not self._enable_teacher_root_com_rand:
            return

        env_ids_cpu = env_ids.to(dtype=torch.long, device="cpu")
        num_envs = len(env_ids_cpu)

        if self._enable_teacher_phys_rand:
            masses = self.robot.root_physx_view.get_masses()
            default_masses = self._gather_env_tensor(self.robot.data.default_mass, env_ids).detach().to(device="cpu")
            mass_scale = torch.ones((num_envs, self.robot.num_bodies), dtype=torch.float32, device="cpu")
            if bool(getattr(self.cfg, "teacher_randomize_root_mass_only", False)):
                root_mass_scale = torch.empty((num_envs, 1), dtype=torch.float32, device="cpu").uniform_(
                    *self.cfg.teacher_mass_scale_range
                )
                mass_scale[:, :1] = root_mass_scale
            else:
                mass_scale.uniform_(*self.cfg.teacher_mass_scale_range)
            randomized_masses = torch.clamp(default_masses * mass_scale, min=1.0e-6)
            masses[env_ids_cpu] = randomized_masses
            self.robot.root_physx_view.set_masses(masses, env_ids_cpu)

            inertias = self.robot.root_physx_view.get_inertias()
            default_inertias = self._gather_env_tensor(self.robot.data.default_inertia, env_ids).detach().to(device="cpu")
            mass_ratios = randomized_masses / torch.clamp(default_masses, min=1.0e-6)
            inertias[env_ids_cpu] = default_inertias * mass_ratios.unsqueeze(-1)
            self.robot.root_physx_view.set_inertias(inertias, env_ids_cpu)

            if bool(getattr(self.cfg, "teacher_motor_scale_per_joint", True)):
                stiffness_scale = self._sample_uniform_range(
                    self.cfg.teacher_stiffness_scale_range, (num_envs, self._num_body_joints)
                )
                damping_scale = self._sample_uniform_range(
                    self.cfg.teacher_damping_scale_range, (num_envs, self._num_body_joints)
                )
                friction_scale = self._sample_uniform_range(
                    self.cfg.teacher_joint_friction_scale_range, (num_envs, self._num_body_joints)
                )
            else:
                stiffness_scalar = self._sample_uniform_range(self.cfg.teacher_stiffness_scale_range, (num_envs, 1))
                damping_scalar = self._sample_uniform_range(self.cfg.teacher_damping_scale_range, (num_envs, 1))
                friction_scalar = self._sample_uniform_range(self.cfg.teacher_joint_friction_scale_range, (num_envs, 1))
                stiffness_scale = stiffness_scalar.expand(-1, self._num_body_joints)
                damping_scale = damping_scalar.expand(-1, self._num_body_joints)
                friction_scale = friction_scalar.expand(-1, self._num_body_joints)

            joint_stiffness = self._gather_env_tensor(self.robot.data.default_joint_stiffness, env_ids).clone().to(self.device)
            joint_damping = self._gather_env_tensor(self.robot.data.default_joint_damping, env_ids).clone().to(self.device)
            joint_stiffness[:, self._body_joint_ids] *= stiffness_scale
            joint_damping[:, self._body_joint_ids] *= damping_scale
            self.robot.write_joint_stiffness_to_sim(
                stiffness=joint_stiffness[:, self._body_joint_ids], joint_ids=self._body_joint_ids, env_ids=env_ids
            )
            self.robot.write_joint_damping_to_sim(
                damping=joint_damping[:, self._body_joint_ids], joint_ids=self._body_joint_ids, env_ids=env_ids
            )

            joint_friction_coeff = self._gather_env_tensor(self.robot.data.default_joint_friction_coeff, env_ids).clone().to(
                self.device
            )
            joint_friction_coeff[:, self._body_joint_ids] *= friction_scale
            joint_friction_coeff[:, self._body_joint_ids].clamp_(min=0.0)
            self.robot.write_joint_friction_coefficient_to_sim(
                joint_friction_coeff=joint_friction_coeff[:, self._body_joint_ids],
                joint_ids=self._body_joint_ids,
                env_ids=env_ids,
            )

            for actuator in self.robot.actuators.values():
                actuator_env_ids = self._env_ids_for_tensor(env_ids, actuator.stiffness)
                actuator.stiffness[actuator_env_ids] = joint_stiffness[:, actuator.joint_indices].to(
                    device=actuator.stiffness.device
                )
                actuator.damping[actuator_env_ids] = joint_damping[:, actuator.joint_indices].to(
                    device=actuator.damping.device
                )

            self._teacher_mass_scale[env_ids] = mass_scale.mean(dim=1, keepdim=True).to(device=self.device)
            self._teacher_joint_friction_scale[env_ids] = friction_scale.mean(dim=1, keepdim=True)
            self._teacher_motor_strength_offset[env_ids] = torch.cat((stiffness_scale - 1.0, damping_scale - 1.0), dim=1)
            self._update_teacher_privileged_physical_features(env_ids, masses_cpu=masses)
        else:
            self._teacher_mass_scale[env_ids] = 1.0
            self._teacher_joint_friction_scale[env_ids] = 1.0
            self._teacher_motor_strength_offset[env_ids] = 0.0

        if self._enable_teacher_root_com_rand:
            coms = self.robot.root_physx_view.get_coms().clone()
            default_coms = self._gather_env_tensor(self._default_body_com_pose_b, env_ids).detach().to(device="cpu")
            root_com_offset = torch.empty((num_envs, 3), dtype=torch.float32, device="cpu").uniform_(
                *self.cfg.teacher_root_com_offset_range
            )
            coms[env_ids_cpu, 0, :3] = default_coms[:, 0, :3] + root_com_offset
            self.robot.root_physx_view.set_coms(coms, env_ids_cpu)
            self._teacher_root_com_offset[env_ids] = root_com_offset.to(device=self.device)
        else:
            self._teacher_root_com_offset[env_ids] = 0.0

    def _resample_teacher_force_commands(self, env_ids: torch.Tensor) -> None:
        if not self._enable_teacher_force_curriculum or env_ids.numel() == 0:
            return

        self._force_duration[env_ids] = torch.randint(
            int(self.cfg.teacher_randomize_force_duration[0]),
            int(self.cfg.teacher_randomize_force_duration[1]) + 1,
            (len(env_ids),),
            device=self.device,
        )

        force_scale = self._force_scale[env_ids].view(-1, 1, 1)
        x_force = self._sample_uniform_range(
            self.cfg.teacher_apply_force_x_range, (len(env_ids), self._num_force_bodies, 1)
        )
        y_force = self._sample_uniform_range(
            self.cfg.teacher_apply_force_y_range, (len(env_ids), self._num_force_bodies, 1)
        )
        z_force = self._sample_uniform_range(
            self.cfg.teacher_apply_force_z_range, (len(env_ids), self._num_force_bodies, 1)
        )
        force_vectors = torch.cat((x_force, y_force, z_force), dim=-1)

        zero_prob = torch.tensor(self.cfg.teacher_zero_force_prob, device=self.device, dtype=torch.float32).view(1, 1, 3)
        zero_mask = torch.rand_like(force_vectors) < zero_prob
        force_vectors = force_vectors.masked_fill(zero_mask, 0.0)

        if self.cfg.teacher_only_apply_resistance_when_walking:
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

    def _update_teacher_force_curriculum(self, env_ids: torch.Tensor) -> None:
        if not self._enable_teacher_force_curriculum or env_ids.numel() == 0:
            return

        episode_lengths = self.episode_length_buf[env_ids].float()
        if self.cfg.teacher_force_scale_curriculum:
            scale_up_mask = episode_lengths >= float(self.cfg.teacher_force_scale_up_threshold)
            scale_down_mask = episode_lengths <= float(self.cfg.teacher_force_scale_down_threshold)
            self._force_scale[env_ids[scale_up_mask]] += float(self.cfg.teacher_force_scale_up)
            self._force_scale[env_ids[scale_down_mask]] -= float(self.cfg.teacher_force_scale_down)
            self._force_scale.clamp_(
                min=float(self.cfg.teacher_force_scale_min),
                max=float(self.cfg.teacher_force_scale_max),
            )

        self._resample_teacher_force_commands(env_ids)

    def _update_teacher_force_application(self) -> None:
        if not self._enable_teacher_force_curriculum:
            return

        self._force_duration -= 1
        resample_env_ids = (self._force_duration <= 0).nonzero(as_tuple=False).squeeze(-1)
        if resample_env_ids.numel() > 0:
            self._resample_teacher_force_commands(resample_env_ids)

        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=self._force_vectors,
            torques=self._force_torques,
            body_ids=self._force_body_ids,
        )

    def _build_proprio_obs(self) -> torch.Tensor:
        proprio_obs = super()._build_proprio_obs()
        if self._enable_teacher_action_delay:
            proprio_obs[:, -self._num_body_joints :] = self._raw_actions
        return proprio_obs

    def _build_teacher_privileged_obs(self) -> torch.Tensor:
        current_motion_obs = self._build_current_teacher_motion_obs()
        future_motion_obs = self._build_teacher_privileged_motion_obs()
        proprio_obs = self._apply_teacher_obs_noise(self._build_proprio_obs())
        current_info = torch.cat((proprio_obs, self._build_teacher_privileged_info()), dim=-1)
        return torch.cat((current_motion_obs, current_info, future_motion_obs), dim=-1)

    def _get_observations(self) -> dict:
        self._update_reference_motion()
        obs = self._build_teacher_privileged_obs()

        if self._obs_history_buf is not None:
            current_motion_obs_dim = _TEACHER_PRIVILEGED_SINGLE_STEP_DIM
            noisy_proprio = obs[:, current_motion_obs_dim : current_motion_obs_dim + _TEACHER_PRIVILEGED_PROPRIO_DIM]
            history_flat = self._obs_history_buf.view(self.num_envs, -1)
            obs = torch.cat((obs, history_flat), dim=-1)
            self._obs_history_buf[:, :-1].copy_(self._obs_history_buf[:, 1:].clone())
            self._obs_history_buf[:, -1].copy_(noisy_proprio)

        self._previous_actions.copy_(self._actions)
        self._last_root_pos_w.copy_(self.robot.data.root_link_pos_w)
        self._last_root_quat_w.copy_(self.robot.data.root_link_quat_w)
        return {"policy": obs}

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._raw_actions = actions.clone()

        if self._enable_teacher_action_delay:
            self._teacher_action_history = torch.roll(self._teacher_action_history, shifts=-1, dims=1)
            self._teacher_action_history[:, -1] = self._raw_actions
            warmup_done = self.common_step_counter > self._teacher_obs_noise_ramp_denom
            if warmup_done:
                self._teacher_action_delay_steps = int(
                    torch.randint(
                        low=0,
                        high=int(self.cfg.teacher_action_delay_max_steps) + 1,
                        size=(1,),
                        device=self.device,
                    ).item()
                )
            else:
                self._teacher_action_delay_steps = 0
            self._actions = self._teacher_action_history[:, -(self._teacher_action_delay_steps + 1)].clone()
        else:
            self._teacher_action_delay_steps = 0
            self._actions = self._raw_actions.clone()

        unclipped_targets = self._ref_dof_pos + self.cfg.action_scale * self._actions
        joint_limits = self.robot.data.soft_joint_pos_limits[:, self._body_joint_ids]
        self._joint_pos_targets = torch.clamp(unclipped_targets, joint_limits[..., 0], joint_limits[..., 1])

        self._update_teacher_gravity_randomization()
        self._update_teacher_robot_push()
        self._update_teacher_end_effector_push()
        self._update_teacher_force_application()

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()

        extra_rewards = {
            "lin_vel_z": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "orientation": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "dof_pos_limits": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "dof_vel": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "feet_slip": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "feet_contact_forces": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "feet_stumble": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "feet_air_time": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "collision": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "dof_torque_limits": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "torque_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
        }

        foot_forces = self._contact_sensor.data.net_forces_w[:, self._teacher_foot_ids]
        foot_force_z = foot_forces[:, :, 2]
        foot_force_z_abs = torch.abs(foot_force_z)
        foot_contact = foot_force_z_abs > float(self.cfg.teacher_contact_force_threshold)

        if self.cfg.rew_scale_lin_vel_z != 0.0:
            lin_vel_z_penalty = torch.square(self.robot.data.root_link_lin_vel_w[:, 2])
            extra_rewards["lin_vel_z"] = self.cfg.rew_scale_lin_vel_z * lin_vel_z_penalty * self.step_dt

        if self.cfg.rew_scale_orientation != 0.0:
            orientation_penalty = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
            extra_rewards["orientation"] = self.cfg.rew_scale_orientation * orientation_penalty * self.step_dt

        if self.cfg.rew_scale_dof_pos_limits != 0.0:
            joint_pos = self.robot.data.joint_pos[:, self._body_joint_ids]
            joint_limits = self.robot.data.soft_joint_pos_limits[:, self._body_joint_ids]
            out_of_limits = -(joint_pos - joint_limits[..., 0]).clamp(max=0.0)
            out_of_limits += (joint_pos - joint_limits[..., 1]).clamp(min=0.0)
            extra_rewards["dof_pos_limits"] = (
                self.cfg.rew_scale_dof_pos_limits * torch.sum(out_of_limits, dim=1) * self.step_dt
            )

        if self.cfg.rew_scale_dof_vel != 0.0:
            dof_vel_penalty = torch.sum(torch.square(self.robot.data.joint_vel[:, self._body_joint_ids]), dim=1)
            extra_rewards["dof_vel"] = self.cfg.rew_scale_dof_vel * dof_vel_penalty * self.step_dt

        if self.cfg.rew_scale_feet_slip != 0.0:
            foot_speed_norm = torch.norm(self.robot.data.body_link_lin_vel_w[:, self._teacher_foot_ids, :2], dim=2)
            slip_penalty = torch.sqrt(torch.clamp(foot_speed_norm, min=0.0)) * foot_contact.float()
            extra_rewards["feet_slip"] = self.cfg.rew_scale_feet_slip * torch.sum(slip_penalty, dim=1) * self.step_dt

        if self.cfg.rew_scale_feet_contact_forces != 0.0:
            excess_force = torch.clamp(foot_force_z_abs - float(self.cfg.teacher_max_contact_force), min=0.0)
            extra_rewards["feet_contact_forces"] = (
                self.cfg.rew_scale_feet_contact_forces * torch.sum(excess_force, dim=1) * self.step_dt
            )

        if self.cfg.rew_scale_feet_stumble != 0.0:
            stumble = torch.any(torch.norm(foot_forces[:, :, :2], dim=2) > 4.0 * torch.abs(foot_force_z), dim=1)
            extra_rewards["feet_stumble"] = self.cfg.rew_scale_feet_stumble * stumble.float() * self.step_dt

        if self.cfg.rew_scale_feet_air_time != 0.0:
            contact_filt = torch.logical_or(foot_contact, self._teacher_last_foot_contacts)
            first_contact = (self._teacher_feet_air_time > 0.0) & contact_filt
            self._teacher_feet_air_time += self.step_dt
            air_time = (self._teacher_feet_air_time - float(self.cfg.teacher_feet_air_time_target)) * first_contact.float()
            air_time = air_time.clamp(max=0.0)
            self._teacher_feet_air_time *= (~contact_filt).float()
            self._teacher_last_foot_contacts = foot_contact
            moving_mask = (torch.norm(self._ref_root_vel[:, :2], dim=1) > 0.05).float()
            extra_rewards["feet_air_time"] = (
                self.cfg.rew_scale_feet_air_time * torch.sum(air_time, dim=1) * moving_mask * self.step_dt
            )

        if self.cfg.rew_scale_collision != 0.0:
            penalized_contact_forces = self._contact_sensor.data.net_forces_w[:, self._teacher_penalized_contact_ids]
            collision = torch.sum((torch.norm(penalized_contact_forces, dim=-1) > 0.1).float(), dim=1)
            extra_rewards["collision"] = self.cfg.rew_scale_collision * collision * self.step_dt

        if self.cfg.rew_scale_dof_torque_limits != 0.0:
            effort_limits = torch.clamp(self.robot.data.joint_effort_limits[:, self._body_joint_ids], min=1.0e-6)
            applied_torque = torch.abs(self.robot.data.applied_torque[:, self._body_joint_ids])
            torque_limit_penalty = torch.sum(
                (applied_torque / effort_limits - float(self.cfg.teacher_soft_torque_limit)).clamp(min=0.0),
                dim=1,
            )
            extra_rewards["dof_torque_limits"] = (
                self.cfg.rew_scale_dof_torque_limits * torque_limit_penalty * self.step_dt
            )

        if self.cfg.rew_scale_torque_penalty != 0.0:
            torque_penalty = torch.sum(torch.square(self.robot.data.applied_torque[:, self._body_joint_ids]), dim=1)
            extra_rewards["torque_penalty"] = self.cfg.rew_scale_torque_penalty * torque_penalty * self.step_dt

        invalid_state = ~torch.isfinite(self.robot.data.root_pose_w).all(dim=1)
        invalid_state |= ~torch.isfinite(self.robot.data.joint_pos).all(dim=1)
        invalid_state |= ~torch.isfinite(self.robot.data.joint_vel).all(dim=1)
        invalid_state |= ~torch.isfinite(self.robot.data.joint_acc).all(dim=1)
        invalid_state |= ~torch.isfinite(self.robot.data.body_link_lin_vel_w).all(dim=(1, 2))
        invalid_state |= ~torch.isfinite(self._contact_sensor.data.net_forces_w).all(dim=(1, 2))
        invalid_state |= ~torch.isfinite(self.robot.data.applied_torque).all(dim=1)

        for key, value in extra_rewards.items():
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            value[invalid_state] = 0.0
            reward += value
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, time_out = super()._get_dones()
        if self._teacher_termination_contact_ids.numel() > 0:
            termination_contact_forces = self._contact_sensor.data.net_forces_w[:, self._teacher_termination_contact_ids]
            terminated |= torch.any(torch.norm(termination_contact_forces, dim=-1) > 1.0, dim=1)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)

        if env_ids is None:
            env_ids_tensor = self.robot._ALL_INDICES
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids
        else:
            env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        self._raw_actions[env_ids_tensor] = 0.0
        self._teacher_action_history[env_ids_tensor] = 0.0
        if self._obs_history_buf is not None:
            self._obs_history_buf[env_ids_tensor] = 0.0
        self._teacher_last_foot_contacts[env_ids_tensor] = False
        self._teacher_feet_air_time[env_ids_tensor] = 0.0
        self._randomize_teacher_physical_properties(env_ids_tensor)
        if self._enable_teacher_force_curriculum:
            self._update_teacher_force_curriculum(env_ids_tensor)

        self.extras.setdefault("log", {})
        self.extras["log"]["Randomization/teacher_mass_scale_mean"] = self._teacher_mass_scale.mean().item()
        self.extras["log"]["Randomization/teacher_joint_friction_scale_mean"] = (
            self._teacher_joint_friction_scale.mean().item()
        )
        self.extras["log"]["Randomization/teacher_motor_strength_offset_abs_mean"] = (
            self._teacher_motor_strength_offset.abs().mean().item()
        )
        self.extras["log"]["Randomization/teacher_root_com_offset_abs_mean"] = (
            self._teacher_root_com_offset.abs().mean().item()
        )
        self.extras["log"]["Randomization/teacher_gravity_xy_norm"] = (
            torch.linalg.vector_norm(self._teacher_current_gravity[:2]).item()
        )
        self.extras["log"]["Randomization/teacher_gravity_z"] = float(self._teacher_current_gravity[2].item())
        self.extras["log"]["Randomization/teacher_robot_push_xy_speed_mean"] = self._teacher_last_push_speed
        self.extras["log"]["Randomization/teacher_end_effector_push_force_mean"] = (
            self._teacher_last_end_effector_push_force
        )
        self.extras["log"]["Randomization/teacher_action_delay_steps"] = float(self._teacher_action_delay_steps)
        self.extras["log"]["Noise/teacher_obs_noise_progress"] = self._teacher_obs_noise_progress()
        if self._enable_teacher_force_curriculum:
            self.extras["log"]["Curriculum/teacher_force_scale_mean"] = self._force_scale.mean().item()
            self.extras["log"]["Curriculum/teacher_force_scale_min"] = self._force_scale.min().item()
            self.extras["log"]["Curriculum/teacher_force_scale_max"] = self._force_scale.max().item()
