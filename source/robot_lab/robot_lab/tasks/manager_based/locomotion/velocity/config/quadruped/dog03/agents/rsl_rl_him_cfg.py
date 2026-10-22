# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from ...csdog1.agents.rsl_rl_him_cfg import CSDog1Flat282HIMRunnerCfg
from ..curriculum import stairs_config
from .rsl_rl_ppo_cfg import Dog03RunnerCfgMixin


@configclass
class Dog03Flat282HIMRunnerCfg(Dog03RunnerCfgMixin, CSDog1Flat282HIMRunnerCfg):
    """47 x 6 observations, using the shared HIM estimator and export layout."""


@configclass
class Dog03Flat282DomainRandHIMRunnerCfg(Dog03Flat282HIMRunnerCfg):
    # Checkpoints are selected explicitly with --resume --checkpoint=... .
    load_optimizer: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "dog03_flat_282_domain_rand"
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.entropy_coef = 0.003
        self.algorithm.schedule = "fixed"
        self.clip_actions = 100.0


@configclass
class Dog03Stairs282HIMRunnerCfg(Dog03Flat282DomainRandHIMRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = stairs_config()["iterations"]
        self.experiment_name = "dog03_stairs_282"
