"""Source stair meshes and patch metadata on Isaac Lab's terrain importer."""
import numpy as np
import trimesh
from isaaclab.terrains import TerrainGenerator, TerrainGeneratorCfg
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh
from isaaclab.utils import configclass

from .stairs_geometry import pyramid_stair_heights, stair_patch_specs


def stair_mesh(difficulty, cfg):
    heights = pyramid_stair_heights(
        cfg.size, cfg.horizontal_scale, cfg.vertical_scale,
        cfg.step_height, cfg.step_width, cfg.platform_size,
    )
    vertices, faces = convert_height_field_to_mesh(
        heights, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold,
    )
    center = np.array(heights.shape) // 2
    radius = int(1 / cfg.horizontal_scale)
    z = heights[center[0]-radius:center[0]+radius, center[1]-radius:center[1]+radius].max() * cfg.vertical_scale
    return [trimesh.Trimesh(vertices=vertices, faces=faces)], np.array([cfg.size[0]/2, cfg.size[1]/2, z])


@configclass
class CyberstarStairPatchCfg(HfTerrainBaseCfg):
    function = stair_mesh
    step_height: float = 0.
    step_width: float = .35
    platform_size: float = 3.


class CyberstarStairsTerrainGenerator(TerrainGenerator):
    def _generate_random_terrains(self):
        specs = stair_patch_specs(
            self.cfg.num_rows * self.cfg.num_cols, self.cfg.flat_ratio,
            self.cfg.up_ratio, self.cfg.height_bands, self.np_rng,
        )
        # Stored on the shared config for the environment adapter. The importer
        # discards the generator after importing its mesh. Values are serializable.
        self.cfg.patch_step_heights = [height for height, _ in specs]
        template = next(iter(self.cfg.sub_terrains.values()))
        for index, (height, width) in enumerate(specs):
            row, col = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))
            patch = template.replace(step_height=height, step_width=width or .35)
            mesh, origin = self._get_terrain_mesh(0., patch)
            self._add_sub_terrain(mesh, origin, row, col, patch)


@configclass
class CyberstarStairsTerrainGeneratorCfg(TerrainGeneratorCfg):
    class_type = CyberstarStairsTerrainGenerator
    flat_ratio: float = .30
    up_ratio: float = .65
    height_bands: tuple = ((.03, .05, 1.),)
    patch_step_heights: list[float] = []
