# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from ..cyberstar.stairs_runtime import stairs_iterations


@configclass
class Dog03RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "dog03_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Dog03FlatPPORunnerCfg(Dog03RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 2000
        self.experiment_name = "dog03_flat_47"


@configclass
class Dog03Flat282PPORunnerCfg(Dog03RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 48
        self.max_iterations = 1000
        self.save_interval = 50
        self.experiment_name = "dog03_flat_282"
        self.algorithm.learning_rate = 5.0e-4
        self.algorithm.entropy_coef = 0.01
        self.algorithm.schedule = "adaptive"


@configclass
class Dog03Flat282DomainRandPPORunnerCfg(Dog03Flat282PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1000
        self.save_interval = 50
        self.experiment_name = "dog03_flat_282_domain_rand"
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.entropy_coef = 0.0015
        self.algorithm.schedule = "adaptive"


@configclass
class Dog03Stairs282PPORunnerCfg(Dog03Flat282DomainRandPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = stairs_iterations()
        self.save_interval = 50
        self.experiment_name = "dog03_stairs_282"
