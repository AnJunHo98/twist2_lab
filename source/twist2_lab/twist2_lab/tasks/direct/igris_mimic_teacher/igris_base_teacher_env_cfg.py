from pathlib import Path

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from twist2_lab.robots import IGRIS_BODY_JOINT_ORDER, IGRIS_STUDENT_KEY_BODY_ROBOT_ORDER

from .igris_mimic_teacher_env_cfg import _DEFAULT_TEACHER_PRIVILEGED_MOTION_STEPS, IgrisMimicTeacherEnvCfg


_PROJECT_SOURCE_ROOT = Path(__file__).resolve().parents[4]
_BASE_MOTION_FILE = _PROJECT_SOURCE_ROOT / "data" / "motion_data_configs" / "twist2_teacher_upright_dataset.yaml"
_TEACHER_PRIVILEGED_SINGLE_STEP_DIM = 21 + len(IGRIS_BODY_JOINT_ORDER) + 3 * len(IGRIS_STUDENT_KEY_BODY_ROBOT_ORDER)
_TEACHER_PRIVILEGED_PROPRIO_DIM = 3 + 2 + 3 * len(IGRIS_BODY_JOINT_ORDER)
_TEACHER_PRIVILEGED_INFO_DIM = 3 + 3 + 4 + 3 * len(IGRIS_STUDENT_KEY_BODY_ROBOT_ORDER) + 2 + 4 + 1 + 2 * len(
    IGRIS_BODY_JOINT_ORDER
)
_TEACHER_HISTORY_LEN = 0
_TEACHER_CURRENT_INFO_DIM = _TEACHER_PRIVILEGED_PROPRIO_DIM + _TEACHER_PRIVILEGED_INFO_DIM
_TEACHER_NUM_FUTURE_STEPS = len(_DEFAULT_TEACHER_PRIVILEGED_MOTION_STEPS)


@configclass
class IgrisBaseTeacherEnvCfg(IgrisMimicTeacherEnvCfg):
    """Canonical teacher baseline used as the starting point for teacher experiments."""

    history_len: int = _TEACHER_HISTORY_LEN
    observation_space = (
        _TEACHER_PRIVILEGED_SINGLE_STEP_DIM
        + _TEACHER_CURRENT_INFO_DIM
        + _TEACHER_NUM_FUTURE_STEPS * _TEACHER_PRIVILEGED_SINGLE_STEP_DIM
    )
    state_space = 0
    enable_motion_level_curriculum = True

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.002,
    )
    motion_file = str(_BASE_MOTION_FILE)
    motion_storage_device = "cpu"
    motion_storage_dtype = "float16"

    # Canonical old-TWIST2 teacher baseline.
    rew_scale_alive = 0.5
    rew_scale_track_root_pos = 1.0
    rew_scale_track_root_rot = 1.0
    rew_scale_track_root_vel = 1.0
    rew_scale_track_root_ang_vel = 1.0
    rew_scale_track_root_path_pos = 1.0
    rew_scale_track_root_path_rot = 1.0
    rew_scale_track_dof_pos = 2.0
    rew_scale_track_dof_vel = 0.2
    rew_scale_track_key_body = 2.0
    rew_scale_track_key_body_global = 2.0
    rew_scale_lin_vel_z = -1.0
    rew_scale_orientation = -1.0
    rew_scale_joint_acc = -5.0e-8
    rew_scale_ang_vel_xy = -1.0e-2
    rew_scale_dof_pos_limits = -5.0
    rew_scale_dof_vel = -1.0e-4
    rew_scale_ankle_dof_vel = -2.0e-4
    rew_scale_action_rate = -1.0e-2
    rew_scale_feet_slip = -1.0e-1
    rew_scale_feet_contact_forces = -5.0e-4
    rew_scale_feet_stumble = -1.25
    rew_scale_feet_air_time = 5.0
    rew_scale_collision = -10.0
    rew_scale_dof_torque_limits = -1.0
    rew_scale_torque_penalty = 0.0
    rew_scale_terminated = -2.0

    # Keep the same tightened success gates that worked well during Isaac Lab migration.
    motion_level_curriculum_success_root_pos_threshold = 2.0
    motion_level_curriculum_success_dof_pos_threshold = 10.0
    motion_level_curriculum_success_key_body_local_threshold = 0.4
    motion_level_curriculum_success_completion = 0.1
    motion_level_curriculum_success_threshold = 0.2
    motion_level_curriculum_min_level_episodes = 64

    teacher_soft_torque_limit = 0.95
    teacher_max_contact_force = 500.0
    teacher_feet_air_time_target = 0.5
    teacher_penalized_contact_body_names = (
        "Link_Shoulder_Pitch_Left",
        "Link_Shoulder_Roll_Left",
        "Link_Shoulder_Yaw_Left",
        "Link_Elbow_Pitch_Left",
        "Link_Hip_Pitch_Left",
        "Link_Hip_Roll_Left",
        "Link_Hip_Yaw_Left",
        "Link_Knee_Pitch_Left",
        "Link_Shoulder_Pitch_Right",
        "Link_Shoulder_Roll_Right",
        "Link_Shoulder_Yaw_Right",
        "Link_Elbow_Pitch_Right",
        "Link_Hip_Pitch_Right",
        "Link_Hip_Roll_Right",
        "Link_Hip_Yaw_Right",
        "Link_Knee_Pitch_Right",
    )
    teacher_termination_contact_body_names = ()

    # Old TWIST2 used reference-relative height cutoffs and pose tracking failure, not strict upright-only tilt gates.
    termination_root_height = 0.0
    termination_root_height_diff = 0.4
    termination_tilt = 4.0
    termination_pose_tracking_dist = 0.9
    termination_root_pos_tracking_dist = 2.0
    termination_max_root_lin_vel = 5.0
    terminate_on_motion_end = True

    # Old TWIST2 teacher ran with broad domain randomization and privileged observation noise enabled.
    enable_teacher_phys_rand = True
    teacher_mass_scale_range = (0.9, 1.1)
    teacher_randomize_root_mass_only = True
    teacher_joint_friction_scale_range = (1.0, 1.0)
    teacher_stiffness_scale_range = (0.8, 1.2)
    teacher_damping_scale_range = (0.8, 1.2)
    teacher_motor_scale_per_joint = False

    enable_teacher_root_com_rand = True
    teacher_root_com_offset_range = (-0.05, 0.05)

    enable_teacher_gravity_rand = True
    teacher_gravity_rand_interval_s = 4.0
    teacher_gravity_range = (-0.1, 0.1)

    enable_teacher_robot_push = False
    teacher_robot_push_interval_s = 4.0
    teacher_robot_push_max_vel_xy = 1.0

    enable_teacher_action_delay = True
    teacher_action_delay_max_steps = 1
    teacher_action_delay_buffer_len = 2

    enable_teacher_end_effector_push = False
    teacher_end_effector_push_body_names = ("Left_Hand", "Right_Hand")
    teacher_end_effector_push_interval_s = 2.0
    teacher_end_effector_push_max_force = 10.0

    enable_teacher_force_curriculum = False
    teacher_force_apply_body_names = ("Left_Hand", "Right_Hand")
    teacher_force_scale_curriculum = True
    teacher_force_scale_initial = 0.08
    teacher_force_scale_up_threshold = 260
    teacher_force_scale_down_threshold = 180
    teacher_force_scale_up = 0.02
    teacher_force_scale_down = 0.02
    teacher_force_scale_max = 0.6
    teacher_force_scale_min = 0.0
    teacher_apply_force_x_range = (-25.0, 25.0)
    teacher_apply_force_y_range = (-25.0, 25.0)
    teacher_apply_force_z_range = (-35.0, 5.0)
    teacher_zero_force_prob = (0.35, 0.35, 0.35)
    teacher_randomize_force_duration = (10, 40)
    teacher_only_apply_resistance_when_walking = True

    enable_teacher_obs_noise = True
    teacher_obs_noise_increasing_steps = 50000
    teacher_obs_noise_step_multiplier = 24
    teacher_obs_noise_root_pos = 0.0
    teacher_obs_noise_orientation = 0.0
    teacher_obs_noise_root_vel = 0.0
    teacher_obs_noise_root_ang_vel = 0.0
    teacher_obs_noise_dof_pos = 0.01
    teacher_obs_noise_dof_vel = 0.1
    teacher_obs_noise_key_body = 0.0
    teacher_obs_noise_ang_vel = 0.1
    teacher_obs_noise_imu = 0.1
