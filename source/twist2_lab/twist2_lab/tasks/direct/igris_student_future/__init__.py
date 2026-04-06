# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IGRIS direct student environment with future-motion observations."""

import gymnasium as gym

from . import agents


gym.register(
    id="TWIST2-IGRIS-Student-Future-Direct-v0",
    entry_point=f"{__name__}.igris_student_future_env:IgrisStudentFutureEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.igris_student_future_env_cfg:IgrisStudentFutureEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:IgrisStudentFuturePPORunnerCfg",
    },
)

gym.register(
    id="TWIST2-IGRIS-Student-Future-Distill-Direct-v0",
    entry_point=f"{__name__}.igris_student_future_env:IgrisStudentFutureEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.igris_student_future_env_cfg:IgrisStudentFutureEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distill_cfg:IgrisStudentFutureDistillationRunnerCfg",
    },
)

gym.register(
    id="TWIST2-IGRIS-Student-Future-Dagger-Direct-v0",
    entry_point=f"{__name__}.igris_student_future_env:IgrisStudentFutureEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.igris_student_future_dagger_env_cfg:IgrisStudentFutureDaggerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dagger_cfg:IgrisStudentFutureDaggerRunnerCfg",
    },
)

gym.register(
    id="TWIST2-IGRIS-Student-Future-Dagger-NoMoE-Direct-v0",
    entry_point=f"{__name__}.igris_student_future_env:IgrisStudentFutureEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.igris_student_future_dagger_env_cfg:IgrisStudentFutureDaggerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dagger_cfg:IgrisStudentFutureDaggerNoMoeRunnerCfg",
    },
)

gym.register(
    id="TWIST2-IGRIS-Student-Future-Priv-Distill-Direct-v0",
    entry_point=f"{__name__}.igris_student_future_env:IgrisStudentFutureEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.igris_student_future_env_cfg:IgrisStudentFutureEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distill_cfg:IgrisStudentFuturePrivDistillationRunnerCfg",
    },
)

gym.register(
    id="TWIST2-IGRIS-Student-Future-Priv-Dagger-Direct-v0",
    entry_point=f"{__name__}.igris_student_future_env:IgrisStudentFutureEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.igris_student_future_dagger_env_cfg:IgrisStudentFutureDaggerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dagger_cfg:IgrisStudentFuturePrivDaggerRunnerCfg",
    },
)
