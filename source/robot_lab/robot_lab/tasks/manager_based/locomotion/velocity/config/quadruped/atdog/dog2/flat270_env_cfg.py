# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .flat_env_cfg import ATDogDog2FlatEnvCfg


@configclass
class ATDogDog2Flat270EnvCfg(ATDogDog2FlatEnvCfg):
    """Flat Dog2 task with six historical 45-D actor observations."""

    def __post_init__(self):
        super().__post_init__()

        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-1.0, 1.0),
                "pitch": (-1.0, 1.0),
                "yaw": (-0.0, 0.0),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.3),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        }


        self.observations.policy.history_length = 6
        self.observations.policy.flatten_history_dim = True
        self.observations.critic.history_length = 6
        self.observations.critic.flatten_history_dim = True
        self.disable_zero_weight_rewards()
