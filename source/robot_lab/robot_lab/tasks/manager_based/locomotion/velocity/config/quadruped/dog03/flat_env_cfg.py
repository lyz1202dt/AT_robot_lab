# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from ..csdog1.flat_env_cfg import CSDog1Flat282DomainRandEnvCfg, CSDog1Flat282EnvCfg, CSDog1FlatEnvCfg
from .rough_env_cfg import Dog03EnvCfgMixin


def disable_dynamics_randomization(cfg):
    """Keep the initial flat tasks at nominal mass, COM and PD gains."""
    cfg.events.randomize_rigid_body_mass_base = None
    cfg.events.randomize_com_positions = None
    cfg.events.randomize_actuator_gains = None


@configclass
class Dog03FlatEnvCfg(Dog03EnvCfgMixin, CSDog1FlatEnvCfg):
    """47-D flat task using 03dog's CAD reference posture."""

    def _configure_task(self):
        disable_dynamics_randomization(self)


@configclass
class Dog03Flat282EnvCfg(Dog03EnvCfgMixin, CSDog1Flat282EnvCfg):
    """Six 47-D frames; the shared HIM wrapper orders frames for deployment."""

    def _configure_task(self):
        disable_dynamics_randomization(self)


@configclass
class Dog03Flat282DomainRandEnvCfg(Dog03EnvCfgMixin, CSDog1Flat282DomainRandEnvCfg):
    """Manager-based robustness task with low obstacles and dynamics variation."""
