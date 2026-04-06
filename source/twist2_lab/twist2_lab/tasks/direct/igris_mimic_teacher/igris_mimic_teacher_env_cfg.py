# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from twist2_lab.robots import IGRIS_BODY_JOINT_ORDER, IGRIS_C_FLOATING_CFG


_PROJECT_SOURCE_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_MOTION_FILE = _PROJECT_SOURCE_ROOT / "data" / "motion_data_configs" / "twist2_local_dataset.yaml"
_DEFAULT_TEACHER_PRIVILEGED_MOTION_STEPS = (
    1,
    5,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    75,
    80,
    85,
    90,
    95,
)


@configclass
class IgrisMimicTeacherEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 10
    episode_length_s = 10.0
    action_space = len(IGRIS_BODY_JOINT_ORDER)
    observation_space = 3 + 3 + 3 + 3 + 3 + 3 + action_space + action_space + action_space
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=0.002, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # robot
    robot_cfg: ArticulationCfg = IGRIS_C_FLOATING_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.init_state.pos = (0.0, 0.0, 1.0)
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/Link_Ankle_Roll_.*",
        history_length=3,
        update_period=sim.dt,
    )

    # motion
    motion_file = str(_DEFAULT_MOTION_FILE)
    motion_storage_device = None
    motion_storage_dtype = None
    motion_height_offset = 0.0
    action_scale = 0.5
    enable_motion_level_curriculum = False
    motion_level_curriculum_mode = "success"
    motion_level_curriculum_min = 1
    motion_level_curriculum_start_step = 0
    motion_level_curriculum_full_step = 20000
    motion_level_curriculum_success_threshold = 0.8
    motion_level_curriculum_success_completion = 0.95
    motion_level_curriculum_min_level_episodes = 128
    motion_level_curriculum_success_root_pos_threshold = 2.5
    motion_level_curriculum_success_dof_pos_threshold = 1.5
    motion_level_curriculum_success_key_body_local_threshold = 0.3
    motion_level_frontier_sampling_boost = 2.0
    full_motion_level_curriculum_for_single_env = True
    balance_motion_groups = True
    motion_group_balance_power = 1.0
    motion_curriculum = True
    motion_curriculum_gamma = 0.01
    motion_difficulty_min = 1.0
    motion_difficulty_max = 10.0
    use_error_aware_sampling = False
    error_sampling_power = 5.0
    error_sampling_threshold = 0.15
    teacher_privileged_motion_steps = _DEFAULT_TEACHER_PRIVILEGED_MOTION_STEPS
    teacher_contact_force_threshold = 5.0
    ang_vel_scale = 0.25
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05
    enable_reference_markers = False
    reference_marker_radius = 0.03
    reference_marker_color = (0.95, 0.1, 0.1)

    # reset / termination
    reset_joint_noise = 0.02
    reset_root_xy_noise = 0.05
    termination_root_height = 0.65
    termination_root_height_diff = 0.0
    termination_tilt = 0.8
    termination_pose_tracking_dist = 0.0
    termination_root_pos_tracking_dist = 0.0
    termination_max_root_lin_vel = 0.0
    terminate_on_motion_end = False

    # reward scales
    rew_scale_alive = 0.25
    rew_scale_track_root_pos = 1.0
    rew_scale_track_root_rot = 0.75
    rew_scale_track_root_vel = 0.5
    rew_scale_track_root_ang_vel = 0.5
    rew_scale_track_dof_pos = 1.0
    rew_scale_track_dof_vel = 0.35
    rew_scale_track_root_path_pos = 0.0
    rew_scale_track_root_path_rot = 0.0
    rew_scale_track_key_body = 0.0
    rew_scale_track_key_body_global = 0.0
    rew_scale_joint_acc = 0.0
    rew_scale_ang_vel_xy = 0.0
    rew_scale_ankle_dof_vel = 0.0
    rew_scale_action_rate = -5.0e-2
    rew_scale_terminated = -2.0
