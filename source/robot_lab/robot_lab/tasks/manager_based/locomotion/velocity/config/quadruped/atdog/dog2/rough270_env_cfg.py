# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import ATDogDog2RoughEnvCfg


@configclass
class ATDogDog2Rough270EnvCfg(ATDogDog2RoughEnvCfg):
    """Rough Dog2 task with six historical 45-D actor observations."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 6
        self.observations.policy.flatten_history_dim = True
