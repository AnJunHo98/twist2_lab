# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlDistillationAlgorithmCfg, RslRlDistillationRunnerCfg, RslRlMLPModelCfg

from twist2_lab.tasks.direct.igris_mimic_teacher.agents.rsl_rl_ppo_cfg import IgrisBaseTeacherFutureModelCfg


@configclass
class IgrisStudentFutureDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 50
    experiment_name = "igris_student_future_distill_direct"
    obs_groups = {"student": ["policy"], "teacher": ["teacher_priv"]}
    teacher_load_root_path = "logs/rsl_rl/igris_base_teacher_direct"
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
    student = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    teacher = IgrisBaseTeacherFutureModelCfg(
        hidden_dims=[512, 512, 256, 128],
        activation="swish",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1.0e-4,
        gradient_length=16,
        max_grad_norm=1.0,
        loss_type="mse",
    )


@configclass
class IgrisStudentFuturePrivDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 50
    experiment_name = "igris_student_future_priv_distill_direct"
    obs_groups = {"student": ["policy"], "teacher": ["teacher_priv"]}
    teacher_load_root_path = "logs/rsl_rl/igris_base_teacher_direct"
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
    student = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    teacher = IgrisBaseTeacherFutureModelCfg(
        hidden_dims=[512, 512, 256, 128],
        activation="swish",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1.0e-4,
        gradient_length=16,
        max_grad_norm=1.0,
        loss_type="mse",
    )
