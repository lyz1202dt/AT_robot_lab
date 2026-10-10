# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from .flat_env_cfg import CSDog1Flat282DomainRandEnvCfg


CSDOG1_STAIRS_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=16,
    num_cols=16,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.2,
    use_cache=False,
    sub_terrains={
        "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.65,
            step_height_range=(0.03, 0.05),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.35,
            step_height_range=(0.03, 0.05),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class CSDog1Stairs282EnvCfg(CSDog1Flat282DomainRandEnvCfg):
    """Blind 282-D CSDog1 stair task for the first 3-5 cm curriculum stage."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = CSDOG1_STAIRS_TERRAIN_CFG

        self.commands.base_velocity.rel_standing_envs = 0.15
        self.commands.base_velocity.ranges.lin_vel_x = (0.25, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.75, 0.75)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)

        self.rewards.lin_vel_z_l2.weight = -0.5
        self.rewards.feet_height.weight = -0.3
        self.rewards.feet_height.params["target_height"] = 0.08
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0

        if self.__class__.__name__ == "CSDog1Stairs282EnvCfg":
            self.disable_zero_weight_rewards()
