# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from ...csdog1.agents.rsl_rl_ppo_cfg import (
    CSDog1Flat282DomainRandPPORunnerCfg,
    CSDog1Flat282PPORunnerCfg,
    CSDog1FlatPPORunnerCfg,
    CSDog1RoughPPORunnerCfg,
)
from ..curriculum import stairs_config


class Dog03RunnerCfgMixin:
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = self.experiment_name.replace("csdog1_", "dog03_", 1)


@configclass
class Dog03RoughPPORunnerCfg(Dog03RunnerCfgMixin, CSDog1RoughPPORunnerCfg):
    """Shared PPO hyperparameters with independent 03dog logs."""


@configclass
class Dog03FlatPPORunnerCfg(Dog03RunnerCfgMixin, CSDog1FlatPPORunnerCfg):
    """PPO for the 47-D flat task."""


@configclass
class Dog03Flat282PPORunnerCfg(Dog03RunnerCfgMixin, CSDog1Flat282PPORunnerCfg):
    """Optional standard PPO backend for six-frame observations."""


@configclass
class Dog03Flat282DomainRandPPORunnerCfg(Dog03RunnerCfgMixin, CSDog1Flat282DomainRandPPORunnerCfg):
    """PPO for robust flat terrain."""


@configclass
class Dog03Stairs282PPORunnerCfg(Dog03Flat282DomainRandPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = stairs_config()["iterations"]
        self.experiment_name = "dog03_stairs_282"
