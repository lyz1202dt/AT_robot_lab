# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class CSDog1HIMActorCriticCfg(RslRlPpoActorCriticCfg):
    """Cyberstar CSDog1 282-D HIM policy matching the source project."""

    class_name = "HIMActorCritic"
    num_one_step_obs = 47
    history_length = 6
    history_order = "oldest_to_latest"
    policy_frame_term_order = [
        "phase",
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos",
        "joint_vel",
        "actions",
    ]
    estimator_target_term_names = [
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "base_lin_vel",
    ]
    actor_current_frame = "last"
    estimator_encoder_hidden_dims = [128, 64, 16]
    estimator_target_hidden_dims = [128, 64]
    estimator_learning_rate = 1.0e-3
    estimator_max_grad_norm = 10.0
    estimator_num_prototypes = 32
    estimator_temperature = 3.0
    estimator_vel_slice = None
    estimator_target_slice = None


@configclass
class CSDog1HIMPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    estimator_learning_rate = 1.0e-3
    estimator_velocity_loss_coef = 1.0
    estimator_swap_loss_coef = 0.1
    estimator_substeps = 1
    estimator_max_grad_norm = 10.0


@configclass
class CSDog1Flat282HIMRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "HIMOnPolicyRunner"
    num_steps_per_env = 48
    max_iterations = 1000
    save_interval = 50
    experiment_name = "csdog1_flat_282"
    policy = CSDog1HIMActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = CSDog1HIMPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class CSDog1Flat282DomainRandHIMRunnerCfg(CSDog1Flat282HIMRunnerCfg):
    # Explicit source checkpoint starts the robustness curriculum at iteration 1000.
    # A portable path is resolved by train.py; override with --checkpoint for another source.
    resume = True
    load_optimizer: bool = False
    source_checkpoint: str = "logs/rsl_rl/csdog1_flat_282/flat_282/model_1000.pt"
    seed = 1

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1000
        self.save_interval = 50
        self.experiment_name = "csdog1_flat_282_domain_rand"
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.entropy_coef = 0.003
        self.algorithm.schedule = "fixed"
        self.clip_actions = 100.0


@configclass
class CSDog1Stairs282HIMRunnerCfg(CSDog1Flat282DomainRandHIMRunnerCfg):
    seed = 42

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 500
        self.save_interval = 50
        self.experiment_name = "csdog1_stairs_282"
        self.resume = False
        self.source_checkpoint = ""
        self.load_optimizer = True
        self.clip_actions = None
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.entropy_coef = 0.0015
        self.algorithm.schedule = "adaptive"
