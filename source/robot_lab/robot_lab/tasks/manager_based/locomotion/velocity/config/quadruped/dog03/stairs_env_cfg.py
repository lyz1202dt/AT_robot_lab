# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from ..csdog1.cyberstar.stairs_terrain import CyberstarStairPatchCfg, CyberstarStairsTerrainGeneratorCfg
from .curriculum import stairs_config
from .flat_env_cfg import Dog03Flat282DomainRandEnvCfg


@configclass
class Dog03Stairs282EnvCfg(Dog03Flat282DomainRandEnvCfg):
    """Blind stairs, keeping the same actor and critic as robust flat training."""

    def _configure_task(self):
        super()._configure_task()
        stage = stairs_config()
        self.scene.terrain.terrain_generator = CyberstarStairsTerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=25.0,
            num_rows=16,
            num_cols=16,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=False,
            flat_ratio=0.30,
            up_ratio=0.65,
            height_bands=tuple(tuple(band) for band in stage["height_bands"]),
            sub_terrains={"stairs": CyberstarStairPatchCfg(platform_size=3.0)},
        )
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None
        self.sim.physx.gpu_collision_stack_size = max(self.sim.physx.gpu_collision_stack_size, 2**27)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
