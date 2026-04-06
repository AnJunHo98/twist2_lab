# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .igris_student_future_env_cfg import IgrisStudentFutureEnvCfg


@configclass
class IgrisStudentFutureDaggerEnvCfg(IgrisStudentFutureEnvCfg):
    """Student future configuration tuned for teacher-guided PPO/DAgger training."""

    motion_dr_enabled = True
    enable_force_curriculum = True
    force_scale_initial = 0.1
    rew_scale_action_rate = -1.0e-2
    rew_scale_track_key_body = 2.0
    rew_scale_track_key_body_global = 2.0
    rew_scale_joint_acc = -5.0e-8
    rew_scale_ang_vel_xy = -1.0e-2
    rew_scale_ankle_dof_vel = -2.0e-4
