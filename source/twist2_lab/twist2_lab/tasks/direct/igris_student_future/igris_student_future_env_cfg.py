# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from twist2_lab.robots import IGRIS_BODY_JOINT_ORDER, IGRIS_STUDENT_KEY_BODY_ROBOT_ORDER
from twist2_lab.tasks.direct.igris_mimic_teacher.igris_mimic_teacher_env_cfg import IgrisMimicTeacherEnvCfg


@configclass
class IgrisStudentFutureEnvCfg(IgrisMimicTeacherEnvCfg):
    # student observation layout
    history_length = 10
    future_motion_steps = (5, 10, 15, 20)
    enable_future_curriculum = True
    future_curriculum_min_visible = 1
    future_curriculum_start_step = 0
    future_curriculum_full_step = 320
    full_future_for_single_env = True
    motion_curriculum = True
    motion_curriculum_gamma = 0.01
    motion_difficulty_min = 1.0
    motion_difficulty_max = 10.0
    use_error_aware_sampling = True
    error_sampling_power = 5.0
    error_sampling_threshold = 0.15
    ang_vel_scale = 0.25
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05
    motion_dr_enabled = False
    motion_dr_resampling = True
    root_position_noise_range = (0.01, 0.05)
    root_orientation_noise_range = (0.1, 0.2)
    root_velocity_noise_range = (0.05, 0.1)
    joint_position_noise_range = (0.05, 0.1)
    enable_force_curriculum = False
    force_apply_body_names = ("Left_Hand", "Right_Hand")
    force_scale_curriculum = True
    force_scale_initial = 0.0
    force_scale_up_threshold = 220
    force_scale_down_threshold = 180
    force_scale_up = 0.02
    force_scale_down = 0.02
    force_scale_max = 1.0
    force_scale_min = 0.0
    apply_force_x_range = (-40.0, 40.0)
    apply_force_y_range = (-40.0, 40.0)
    apply_force_z_range = (-50.0, 5.0)
    zero_force_prob = (0.25, 0.25, 0.25)
    randomize_force_duration = (10, 50)
    only_apply_resistance_when_walking = True

    student_mimic_obs_dim = 6 + len(IGRIS_BODY_JOINT_ORDER)
    proprio_obs_dim = 3 + 2 + len(IGRIS_BODY_JOINT_ORDER) * 3
    student_obs_dim = student_mimic_obs_dim + proprio_obs_dim

    observation_space = (
        student_obs_dim
        + history_length * student_obs_dim
        + len(future_motion_steps) * student_mimic_obs_dim
    )
    critic_tracking_obs_dim = 18 + len(IGRIS_BODY_JOINT_ORDER) * 3
    critic_key_body_obs_dim = len(IGRIS_STUDENT_KEY_BODY_ROBOT_ORDER) * 3 * 3
    critic_root_delta_obs_dim = 3 * 3 + 3 * 3
    critic_privileged_obs_dim = critic_tracking_obs_dim + critic_key_body_obs_dim + critic_root_delta_obs_dim
    state_space = observation_space + critic_privileged_obs_dim
