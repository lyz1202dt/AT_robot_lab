"""Stair-specific tensor state layered over the migrated robust-flat runtime."""
import json
from pathlib import Path

import torch

from .runtime import CyberstarRuntime
from .stairs_core import Csdog1Stairs282Mixin


def stairs_config():
    return json.loads(Path(__file__).with_name("stairs_config.json").read_text())


class CyberstarStairsRuntime(Csdog1Stairs282Mixin, CyberstarRuntime):
    def __init__(self, num_envs, device, dt=.02, config=None):
        self.flat_env_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.stair_step_heights = torch.zeros(num_envs, device=device)
        self.stair_success = torch.zeros((), device=device)
        super().__init__(num_envs, device, dt, config or stairs_config()["env"])
        self.ppo_iteration = 2000

    def _terrain_height_at(self, positions):
        return self._height_query(positions)

    def _resample_commands(self, env_ids):
        if not getattr(self, "external_commands", False):
            super()._resample_commands(env_ids)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        stair = ~self.flat_env_mask[env_ids]
        valid = self.episode_length_buf[env_ids] > 0
        success = (self.stair_command_progress[env_ids] > self.cfg.rewards.success_distance) & stair & valid
        self.stair_success = success.float().sum() / (stair & valid).sum().clamp_min(1)
        super().reset_idx(env_ids)
        self._reset_stair_landing_buffers(env_ids)
        targets = self._command_mode_targets()[env_ids]
        translating = targets[:, :2].norm(dim=1) > self.cfg.commands.heading_hold_translation_deadband
        translation_ids = env_ids[translating]
        self.translation_heading_reference[translation_ids] = self.stair_heading_reference[translation_ids]
        self.translation_heading_active[env_ids] = translating
        self.stop_heading_reference[env_ids] = self.stair_heading_reference[env_ids]
