from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ROBOT_ROOT = _PROJECT_ROOT / "data" / "robots" / "igris_c_v2"
_USD_PATH = _ROBOT_ROOT / "igris_c_v2.usd"
_FLOATING_USD_PATH = _ROBOT_ROOT / "igris_c_v2_floating.usd"
if not _FLOATING_USD_PATH.is_file():
    _FLOATING_USD_PATH = _USD_PATH
_DEFAULT_PRIM_PATH = "/World/envs/env_.*/Robot"


def _require_local_assets() -> None:
    missing_paths = [path for path in (_USD_PATH, _FLOATING_USD_PATH) if not path.is_file()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing IGRIS USD assets: {missing_text}")


_require_local_assets()


IGRIS_BODY_JOINT_ORDER = (
    "Joint_Waist_Yaw",
    "Joint_Waist_Roll",
    "Joint_Waist_Pitch",
    "Joint_Hip_Pitch_Left",
    "Joint_Hip_Roll_Left",
    "Joint_Hip_Yaw_Left",
    "Joint_Knee_Pitch_Left",
    "Joint_Ankle_Pitch_Left",
    "Joint_Ankle_Roll_Left",
    "Joint_Hip_Pitch_Right",
    "Joint_Hip_Roll_Right",
    "Joint_Hip_Yaw_Right",
    "Joint_Knee_Pitch_Right",
    "Joint_Ankle_Pitch_Right",
    "Joint_Ankle_Roll_Right",
    "Joint_Shoulder_Pitch_Left",
    "Joint_Shoulder_Roll_Left",
    "Joint_Shoulder_Yaw_Left",
    "Joint_Elbow_Pitch_Left",
    "Joint_Wrist_Yaw_Left",
    "Joint_Wrist_Roll_Left",
    "Joint_Wrist_Pitch_Left",
    "Joint_Shoulder_Pitch_Right",
    "Joint_Shoulder_Roll_Right",
    "Joint_Shoulder_Yaw_Right",
    "Joint_Elbow_Pitch_Right",
    "Joint_Wrist_Yaw_Right",
    "Joint_Wrist_Roll_Right",
    "Joint_Wrist_Pitch_Right",
    "Joint_Neck_Yaw",
    "Joint_Neck_Pitch",
)

_REAL_ROBOT_JOINTINDEX_BODY_ORDER = (
    "Joint_Waist_Yaw",
    "Joint_Waist_Roll",
    "Joint_Waist_Pitch",
    "Joint_Hip_Pitch_Left",
    "Joint_Hip_Roll_Left",
    "Joint_Hip_Yaw_Left",
    "Joint_Knee_Pitch_Left",
    "Joint_Ankle_Pitch_Left",
    "Joint_Ankle_Roll_Left",
    "Joint_Hip_Pitch_Right",
    "Joint_Hip_Roll_Right",
    "Joint_Hip_Yaw_Right",
    "Joint_Knee_Pitch_Right",
    "Joint_Ankle_Pitch_Right",
    "Joint_Ankle_Roll_Right",
    "Joint_Shoulder_Pitch_Left",
    "Joint_Shoulder_Roll_Left",
    "Joint_Shoulder_Yaw_Left",
    "Joint_Elbow_Pitch_Left",
    "Joint_Wrist_Yaw_Left",
    "Joint_Wrist_Roll_Left",
    "Joint_Wrist_Pitch_Left",
    "Joint_Shoulder_Pitch_Right",
    "Joint_Shoulder_Roll_Right",
    "Joint_Shoulder_Yaw_Right",
    "Joint_Elbow_Pitch_Right",
    "Joint_Wrist_Yaw_Right",
    "Joint_Wrist_Roll_Right",
    "Joint_Wrist_Pitch_Right",
    "Joint_Neck_Yaw",
    "Joint_Neck_Pitch",
)

if IGRIS_BODY_JOINT_ORDER != _REAL_ROBOT_JOINTINDEX_BODY_ORDER:
    raise ValueError("IGRIS_BODY_JOINT_ORDER must match the real-robot JointIndex ordering.")


IGRIS_HAND_JOINTS = ()

IGRIS_STUDENT_KEY_BODY_MOTION_ORDER = (
    "Left_Hand",
    "Right_Hand",
    "Link_Ankle_Roll_Left",
    "Link_Ankle_Roll_Right",
    "Link_Knee_Pitch_Left",
    "Link_Knee_Pitch_Right",
    "Link_Elbow_Pitch_Left",
    "Link_Elbow_Pitch_Right",
    "Link_Neck_Pitch",
)

IGRIS_STUDENT_KEY_BODY_ROBOT_ORDER = (
    "Left_Hand",
    "Right_Hand",
    "Link_Ankle_Roll_Left",
    "Link_Ankle_Roll_Right",
    "Link_Knee_Pitch_Left",
    "Link_Knee_Pitch_Right",
    "Link_Elbow_Pitch_Left",
    "Link_Elbow_Pitch_Right",
    "Link_Neck_Pitch",
)

if len(IGRIS_STUDENT_KEY_BODY_MOTION_ORDER) != len(IGRIS_STUDENT_KEY_BODY_ROBOT_ORDER):
    raise ValueError("IGRIS student key-body motion/robot mappings must have the same length.")

# Canonical PD gains shared across the IGRIS configs.
_BODY_KP = (
    960.0,
    288.0,
    480.0,
    200.0,
    250.0,
    150.0,
    300.0,
    50.0,
    50.0,
    200.0,
    250.0,
    150.0,
    300.0,
    50.0,
    50.0,
    75.0,
    75.0,
    45.0,
    45.0,
    7.5,
    7.5,
    7.5,
    75.0,
    75.0,
    45.0,
    45.0,
    7.5,
    7.5,
    7.5,
    2.0,
    5.0,
)

_BODY_KD = (
    2.0,
    0.8,
    1.15,
    5.0,
    5.0,
    2.0,
    5.0,
    2.5,
    2.5,
    5.0,
    5.0,
    2.0,
    5.0,
    2.5,
    2.5,
    0.75,
    0.75,
    0.225,
    0.225,
    0.15,
    0.15,
    0.15,
    0.75,
    0.75,
    0.225,
    0.225,
    0.15,
    0.15,
    0.15,
    0.05,
    0.1,
)

_BODY_STIFFNESS = dict(zip(IGRIS_BODY_JOINT_ORDER, _BODY_KP, strict=True))
_BODY_DAMPING = dict(zip(IGRIS_BODY_JOINT_ORDER, _BODY_KD, strict=True))
_HIGH_PD_BODY_STIFFNESS = dict(_BODY_STIFFNESS)
_HIGH_PD_BODY_DAMPING = dict(_BODY_DAMPING)

_HAND_STIFFNESS = {
    "Joint_Hand_.*Thumb_Proximal": 3000.0,
    "Joint_Hand_.*Thumb_Middle": 2500.0,
    "Joint_Hand_.*Thumb_Distal": 2000.0,
    "Joint_Hand_.*_(Index|Middle|Ring|Little)_Middle": 2500.0,
    "Joint_Hand_.*_(Index|Middle|Ring|Little)_Distal": 1800.0,
}
_HAND_DAMPING = {
    "Joint_Hand_.*Thumb_Proximal": 150.0,
    "Joint_Hand_.*Thumb_Middle": 120.0,
    "Joint_Hand_.*Thumb_Distal": 90.0,
    "Joint_Hand_.*_(Index|Middle|Ring|Little)_Middle": 120.0,
    "Joint_Hand_.*_(Index|Middle|Ring|Little)_Distal": 80.0,
}

IGRIS_STANDING_BODY_POSE = {
    "Joint_Waist_Yaw": 0.0,
    "Joint_Waist_Roll": 0.0,
    "Joint_Waist_Pitch": 0.0,
    "Joint_Hip_Pitch_Left": -0.2,
    "Joint_Hip_Roll_Left": 0.0,
    "Joint_Hip_Yaw_Left": 0.0,
    "Joint_Knee_Pitch_Left": 0.4,
    "Joint_Ankle_Pitch_Left": -0.2,
    "Joint_Ankle_Roll_Left": 0.0,
    "Joint_Hip_Pitch_Right": -0.2,
    "Joint_Hip_Roll_Right": 0.0,
    "Joint_Hip_Yaw_Right": 0.0,
    "Joint_Knee_Pitch_Right": 0.4,
    "Joint_Ankle_Pitch_Right": -0.2,
    "Joint_Ankle_Roll_Right": 0.0,
    "Joint_Shoulder_Pitch_Left": 0.0,
    "Joint_Shoulder_Roll_Left": 0.35,
    "Joint_Shoulder_Yaw_Left": 0.0,
    "Joint_Elbow_Pitch_Left": 1.0,
    "Joint_Wrist_Yaw_Left": 0.0,
    "Joint_Wrist_Roll_Left": 0.0,
    "Joint_Wrist_Pitch_Left": 0.0,
    "Joint_Shoulder_Pitch_Right": 0.0,
    "Joint_Shoulder_Roll_Right": -0.35,
    "Joint_Shoulder_Yaw_Right": 0.0,
    "Joint_Elbow_Pitch_Right": 1.0,
    "Joint_Wrist_Yaw_Right": 0.0,
    "Joint_Wrist_Roll_Right": 0.0,
    "Joint_Wrist_Pitch_Right": 0.0,
    "Joint_Neck_Yaw": 0.0,
    "Joint_Neck_Pitch": 0.0,
}


def _build_cfg(
    *,
    usd_path: Path,
    body_stiffness: dict[str, float],
    body_damping: dict[str, float],
    fix_root_link: bool,
    activate_contact_sensors: bool,
    include_hand_actuators: bool = True,
) -> ArticulationCfg:
    actuators = {
        "body_pd": ImplicitActuatorCfg(
            joint_names_expr=list(IGRIS_BODY_JOINT_ORDER),
            stiffness=body_stiffness,
            damping=body_damping,
            armature=None,
            effort_limit_sim=10000.0,
            velocity_limit_sim=10000.0,
            friction=None,
        ),
    }
    if include_hand_actuators and IGRIS_HAND_JOINTS:
        actuators["hands_pd"] = ImplicitActuatorCfg(
            joint_names_expr=["Joint_Hand_.*"],
            stiffness=_HAND_STIFFNESS,
            damping=_HAND_DAMPING,
            effort_limit_sim=10000.0,
            velocity_limit_sim=10000.0,
            armature=0.01,
            friction=None,
        )

    return ArticulationCfg(
        prim_path=_DEFAULT_PRIM_PATH,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            activate_contact_sensors=activate_contact_sensors,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=10000.0,
                max_angular_velocity=10000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=fix_root_link,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators=actuators,
    )


IGRIS_C_CFG = _build_cfg(
    usd_path=_USD_PATH,
    body_stiffness=_BODY_STIFFNESS,
    body_damping=_BODY_DAMPING,
    fix_root_link=True,
    activate_contact_sensors=False,
)

IGRIS_C_HIGH_PD_CFG = _build_cfg(
    usd_path=_USD_PATH,
    body_stiffness=_HIGH_PD_BODY_STIFFNESS,
    body_damping=_HIGH_PD_BODY_DAMPING,
    fix_root_link=True,
    activate_contact_sensors=False,
)

IGRIS_C_FLOATING_CFG = _build_cfg(
    usd_path=_FLOATING_USD_PATH,
    body_stiffness=_BODY_STIFFNESS,
    body_damping=_BODY_DAMPING,
    fix_root_link=False,
    activate_contact_sensors=True,
)

IGRIS_C_FLOATING_HIGH_PD_CFG = _build_cfg(
    usd_path=_FLOATING_USD_PATH,
    body_stiffness=_HIGH_PD_BODY_STIFFNESS,
    body_damping=_HIGH_PD_BODY_DAMPING,
    fix_root_link=False,
    activate_contact_sensors=True,
)
