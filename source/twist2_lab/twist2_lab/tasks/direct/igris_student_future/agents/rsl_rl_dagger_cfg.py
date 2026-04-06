# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from twist2_lab.robots import IGRIS_BODY_JOINT_ORDER
from twist2_lab.tasks.direct.igris_mimic_teacher.agents.rsl_rl_ppo_cfg import IgrisBaseTeacherFutureModelCfg


_STUDENT_MIMIC_OBS_DIM = 6 + len(IGRIS_BODY_JOINT_ORDER)
_PROPRIO_OBS_DIM = 3 + 2 + len(IGRIS_BODY_JOINT_ORDER) * 3
_HISTORY_LENGTH = 10
_FUTURE_MOTION_STEPS = (5, 10, 15, 20)


@configclass
class IgrisStudentFutureActorModelCfg(RslRlMLPModelCfg):
    class_name = "twist2_lab.rl.future_model:FutureActorModel"
    current_motion_obs_dim = _STUDENT_MIMIC_OBS_DIM
    proprio_obs_dim = _PROPRIO_OBS_DIM
    history_length = _HISTORY_LENGTH
    future_step_obs_dim = _STUDENT_MIMIC_OBS_DIM
    num_future_steps = len(_FUTURE_MOTION_STEPS)
    history_encoder_dims = [256, 256]
    history_latent_dim = 128
    future_token_dim = 128
    future_encoder_dims = [256, 256, 128]
    future_latent_dim = 128
    future_attention_heads = 4
    future_dropout = 0.1
    num_experts = 4
    expert_hidden_dims = [256, 128]
    gating_hidden_dim = 128
    moe_temperature = 1.0
    moe_topk = None


@configclass
class IgrisStudentFutureNoMoeActorModelCfg(IgrisStudentFutureActorModelCfg):
    num_experts = 1
    moe_topk = None


@configclass
class IgrisStudentFutureDaggerAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name = "twist2_lab.rl.dagger_ppo:DaggerPPO"
    dagger_coef = 0.2
    dagger_coef_min = 0.1
    dagger_coef_anneal_iterations = 60000
    behavior_loss_type = "mse"
    warm_start_iterations = 100
    teacher_rollout_prob_start = 1.0
    teacher_rollout_prob_end = 0.05
    teacher_rollout_anneal_iterations = 1500
    future_consistency_coef = 0.05
    future_consistency_keep_steps = 2
    future_weight_decay = 0.95
    grad_penalty_coef_schedule = [0.0, 0.0, 700, 1000]
    std_schedule = [1.0, 0.4, 4000, 1500]
    load_balancing_loss_weight = 0.01


@configclass
class IgrisStudentFutureDaggerNoMoeAlgorithmCfg(IgrisStudentFutureDaggerAlgorithmCfg):
    load_balancing_loss_weight = 0.0


@configclass
class IgrisStudentFutureDaggerRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 50
    experiment_name = "igris_student_future_dagger_direct"
    obs_groups = {"actor": ["policy"], "critic": ["critic"], "teacher": ["teacher_priv"]}
    teacher_load_root_path = "logs/rsl_rl/igris_base_teacher_direct"
    teacher_load_run = ".*"
    teacher_load_checkpoint = "model_.*.pt"
    actor = IgrisStudentFutureActorModelCfg(
        hidden_dims=[512, 512, 256, 128],
        activation="swish",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 512, 256, 128],
        activation="swish",
        obs_normalization=True,
    )
    teacher = IgrisBaseTeacherFutureModelCfg(
        hidden_dims=[512, 512, 256, 128],
        activation="swish",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    algorithm = IgrisStudentFutureDaggerAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


@configclass
class IgrisStudentFutureDaggerNoMoeRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 50
    experiment_name = "igris_student_future_dagger_nomoe_direct"
    obs_groups = {"actor": ["policy"], "critic": ["critic"], "teacher": ["teacher_priv"]}
    teacher_load_root_path = "logs/rsl_rl/igris_base_teacher_direct"
    teacher_load_run = ".*"
    teacher_load_checkpoint = "model_.*.pt"
    actor = IgrisStudentFutureNoMoeActorModelCfg(
        hidden_dims=[512, 512, 256, 128],
        activation="swish",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 512, 256, 128],
        activation="swish",
        obs_normalization=True,
    )
    teacher = IgrisBaseTeacherFutureModelCfg(
        hidden_dims=[512, 512, 256, 128],
        activation="swish",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    algorithm = IgrisStudentFutureDaggerNoMoeAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


@configclass
class IgrisStudentFuturePrivDaggerRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 50
    experiment_name = "igris_student_future_priv_dagger_direct"
    obs_groups = {"actor": ["policy"], "critic": ["critic"], "teacher": ["teacher_priv"]}
    teacher_load_root_path = "logs/rsl_rl/igris_base_teacher_direct"
    teacher_load_run = ".*"
    teacher_load_checkpoint = "model_.*.pt"
    actor = IgrisStudentFutureActorModelCfg(
        hidden_dims=[512, 512, 256, 128],
        activation="swish",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 512, 256, 128],
        activation="swish",
        obs_normalization=True,
    )
    teacher = IgrisBaseTeacherFutureModelCfg(
        hidden_dims=[512, 512, 256, 128],
        activation="swish",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    algorithm = IgrisStudentFutureDaggerAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
