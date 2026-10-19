"""Isaac Lab stair patch assignment and source reset placement."""
import torch

from .lab import CSDog1CyberstarEnv
from .stairs_runtime import CyberstarStairsRuntime


class CSDog1CyberstarStairsEnv(CSDog1CyberstarEnv):
    runtime_type = CyberstarStairsRuntime

    def _configure_cyberstar_terrain(self):
        terrain = self.scene["terrain"]
        origins = terrain.terrain_origins.reshape(-1, 3)
        patch_ids = torch.arange(self.num_envs, device=self.device) % len(origins)
        # Uniformly reuse all 256 shuffled patches, as in the source task.
        terrain.env_origins[:] = origins[patch_ids]
        heights = torch.tensor(terrain.cfg.terrain_generator.patch_step_heights, device=self.device)
        if len(heights) != len(origins):
            raise RuntimeError("Cyberstar stair patch metadata does not match the generated terrain")
        self.cyberstar.stair_step_heights[:] = heights[patch_ids]
        self.cyberstar.flat_env_mask[:] = heights[patch_ids] == 0

    def _place_stair_reset(self, state, ids):
        runtime = self.cyberstar
        cfg = runtime.cfg.terrain
        count = len(ids)
        distance = torch.empty(count, device=self.device).uniform_(*cfg.stair_spawn_distance_range)
        state[:, 0] = self.scene.env_origins[ids, 0] + .5 * cfg.stair_platform_size - distance
        state[:, 1] = self.scene.env_origins[ids, 1] + torch.empty(count, device=self.device).uniform_(
            -cfg.stair_spawn_lateral_range, cfg.stair_spawn_lateral_range,
        )
        yaw = torch.empty(count, device=self.device).uniform_(-cfg.stair_spawn_yaw_range, cfg.stair_spawn_yaw_range)
        # Isaac Lab root state uses wxyz; tensor runtime uses xyzw.
        state[:, 3:7] = 0
        state[:, 3] = torch.cos(yaw / 2)
        state[:, 6] = torch.sin(yaw / 2)
        state[:, 2] = self._terrain_height_at(state[:, :3]) + runtime.cfg.init_state.pos[2]
        runtime.stair_start_x[ids] = state[:, 0]
        runtime.stair_heading_reference[ids] = yaw
        runtime.translation_heading_reference[ids] = yaw
        runtime.translation_heading_active[ids] = True
