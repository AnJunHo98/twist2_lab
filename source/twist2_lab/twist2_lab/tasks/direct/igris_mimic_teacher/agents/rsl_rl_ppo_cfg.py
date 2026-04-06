# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from twist2_lab.tasks.direct.igris_mimic_teacher.igris_base_teacher_env_cfg import (
    _TEACHER_CURRENT_INFO_DIM,
    _TEACHER_NUM_FUTURE_STEPS,
    _TEACHER_PRIVILEGED_SINGLE_STEP_DIM,
)


@configclass
class IgrisBaseTeacherFutureModelCfg(RslRlMLPModelCfg):
    class_name = "twist2_lab.rl.future_model:FutureActorModel"
    current_motion_obs_dim = _TEACHER_PRIVILEGED_SINGLE_STEP_DIM
    proprio_obs_dim = _TEACHER_CURRENT_INFO_DIM
    history_length = 0
    future_step_obs_dim = _TEACHER_PRIVILEGED_SINGLE_STEP_DIM
    num_future_steps = _TEACHER_NUM_FUTURE_STEPS
    history_encoder_dims = []
    history_latent_dim = 0
    future_token_dim = 128
    future_encoder_dims = [256, 256, 128]
    future_latent_dim = 128
    future_attention_heads = 4
    future_dropout = 0.1
    num_experts = 1
    expert_hidden_dims = [256, 128]
    gating_hidden_dim = 128
    moe_temperature = 1.0
    moe_topk = None


@configclass
class IgrisBaseTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    experiment_name = "igris_base_teacher_direct"
    max_iterations = 1_000_002
    save_interval = 500
    actor = IgrisBaseTeacherFutureModelCfg(
        hidden_dims=[512, 512, 256, 128],
        activation="swish",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = IgrisBaseTeacherFutureModelCfg(
        hidden_dims=[512, 512, 256, 128],
        activation="swish",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
