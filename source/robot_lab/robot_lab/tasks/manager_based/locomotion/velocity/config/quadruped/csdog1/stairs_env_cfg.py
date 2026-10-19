# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .cyberstar.env_cfg import CSDog1CyberstarDomainRandEnvCfg
from .cyberstar.stairs_runtime import stairs_config
from .cyberstar.stairs_terrain import CyberstarStairPatchCfg, CyberstarStairsTerrainGeneratorCfg


@configclass
class CSDog1Stairs282EnvCfg(CSDog1CyberstarDomainRandEnvCfg):
    """Cyberstar blind stairs: 282-D actor history and unchanged 491-D critic."""

    cyberstar_config: dict = stairs_config()["env"]

    def __post_init__(self):
        super().__post_init__()
        terrain = self.cyberstar_config["terrain"]
        self.scene.terrain.terrain_generator = CyberstarStairsTerrainGeneratorCfg(
            size=(terrain["terrain_length"], terrain["terrain_width"]),
            border_width=terrain["border_size"],
            num_rows=terrain["num_rows"], num_cols=terrain["num_cols"],
            horizontal_scale=terrain["horizontal_scale"], vertical_scale=terrain["vertical_scale"],
            slope_threshold=terrain["slope_treshold"], use_cache=False, curriculum=False,
            flat_ratio=terrain["flat_terrain_ratio"], up_ratio=terrain["stair_up_ratio"],
            height_bands=tuple(tuple(band) for band in terrain["stair_height_bands"]),
            sub_terrains={"stairs": CyberstarStairPatchCfg(platform_size=terrain["stair_platform_size"])},
        )
        self.scene.terrain.max_init_terrain_level = None
