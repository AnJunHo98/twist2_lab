# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IGRIS direct mimic teacher environment."""

import gymnasium as gym

from . import agents


gym.register(
    id="TWIST2-IGRIS-Base-Teacher-Direct-v0",
    entry_point=f"{__name__}.igris_base_teacher_env:IgrisBaseTeacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.igris_base_teacher_env_cfg:IgrisBaseTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:IgrisBaseTeacherPPORunnerCfg",
    },
)
