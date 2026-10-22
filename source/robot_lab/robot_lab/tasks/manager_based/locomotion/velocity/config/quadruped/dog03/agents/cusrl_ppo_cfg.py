# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from ...csdog1.agents.cusrl_ppo_cfg import (
    CSDog1Flat282DomainRandTrainerCfg,
    CSDog1Flat282TrainerCfg,
    CSDog1FlatTrainerCfg,
    CSDog1RoughTrainerCfg,
)
from ..curriculum import stairs_config


@dataclass
class Dog03RoughTrainerCfg(CSDog1RoughTrainerCfg):
    experiment_name = "dog03_rough"


@dataclass
class Dog03FlatTrainerCfg(CSDog1FlatTrainerCfg):
    experiment_name = "dog03_flat_47"


@dataclass
class Dog03Flat282TrainerCfg(CSDog1Flat282TrainerCfg):
    experiment_name = "dog03_flat_282"


@dataclass
class Dog03Flat282DomainRandTrainerCfg(CSDog1Flat282DomainRandTrainerCfg):
    experiment_name = "dog03_flat_282_domain_rand"


@dataclass
class Dog03Stairs282TrainerCfg(Dog03Flat282DomainRandTrainerCfg):
    max_iterations: int = field(default_factory=lambda: stairs_config()["iterations"])
    experiment_name = "dog03_stairs_282"
