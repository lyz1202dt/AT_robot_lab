# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import ATDogDog2RoughEnvCfg


@configclass
class ATDogDog2FlatEnvCfg(ATDogDog2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Keep flat-task resets very close to the nominal standing state.
        # Small pose perturbations help robustness without spawning the robot
        # in obviously unstable configurations.
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
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (-1.0, 0.3),
                "roll": (-0.8, 0.8),
                "pitch": (-0.8, 0.8),
                "yaw": (-0.8, 0.8),
            },
        }

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "ATDogDog2FlatEnvCfg":
            self.disable_zero_weight_rewards()
