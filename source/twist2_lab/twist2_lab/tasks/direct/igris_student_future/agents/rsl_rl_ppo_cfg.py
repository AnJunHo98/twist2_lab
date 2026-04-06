# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg
from twist2_lab.tasks.direct.igris_mimic_teacher.agents.rsl_rl_ppo_cfg import IgrisBaseTeacherPPORunnerCfg


@configclass
class IgrisStudentFuturePPORunnerCfg(IgrisBaseTeacherPPORunnerCfg):
    experiment_name = "igris_student_future_direct"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
