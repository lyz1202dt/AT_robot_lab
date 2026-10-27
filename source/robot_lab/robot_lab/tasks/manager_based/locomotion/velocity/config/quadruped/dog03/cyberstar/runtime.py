"""Stateful Dog03 task logic, independent of the simulator and importable on CPU."""

from __future__ import annotations

import json
import torch
from pathlib import Path
from types import SimpleNamespace

from .core import Dog03Flat47, Dog03Flat47Mixin, Dog03Flat282DomainRandMixin, Dog03Flat282Mixin


def namespace(data):
    if isinstance(data, dict):
        return SimpleNamespace(**{key: namespace(value) for key, value in data.items()})
    return data


def source_config():
    return json.loads(Path(__file__).with_name("config.json").read_text())


def flat_config():
    """Use the robust-flat task with dynamics randomization and pushes disabled."""
    config = source_config()["env"]
    for name in config["domain_rand"]:
        if name.startswith("randomize_") or name == "push_robots":
            config["domain_rand"][name] = False
    return config


class CyberstarRuntime(Dog03Flat282DomainRandMixin, Dog03Flat282Mixin, Dog03Flat47Mixin, Dog03Flat47):
    """Own the original tensor state; simulator I/O is injected by the Lab adapter."""

    def __init__(self, num_envs, device, dt=0.02, config=None):
        self.cfg = namespace(config or source_config()["env"])
        self.num_envs, self.device, self.dt = num_envs, device, dt
        self.num_actions = self.num_dof = self.num_dofs = 12
        self.dof_names = [
            f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")
        ]
        self.feet_indices = torch.arange(4, device=device)
        # Four ordered foot aliases precede the URDF's real bodies.
        bodies = self.cfg.asset.body_names
        self.penalised_contact_indices = torch.tensor(
            [4 + i for i, name in enumerate(bodies) if name.endswith(("_thigh", "_calf"))], device=device
        )
        self.obs_scales = self.cfg.normalization.obs_scales
        self.command_ranges = vars(self.cfg.commands.ranges)
        self.cfg.domain_rand.push_interval = int(self.cfg.domain_rand.push_interval_s / dt)
        self.ppo_iteration = 1000
        self.add_noise = self.cfg.noise.add_noise
        self.obs_buf = self.zeros(282)
        self.root_states = self.zeros(13)
        self.root_states[:, 6] = 1
        self.root_states[:, 2] = self.cfg.init_state.pos[2]
        self.base_pos = self.root_states[:, :3]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = self.zeros(3)
        self.base_lin_vel, self.base_ang_vel, self.projected_gravity = (self.zeros(3) for _ in range(3))
        self.projected_gravity[:, 2] = -1
        self.forward_vec = self.zeros(3)
        self.forward_vec[:, 0] = 1
        self.default_dof_pos = torch.tensor(
            [[getattr(self.cfg.init_state.default_joint_angles, name) for name in self.dof_names]], device=device
        )
        self.dof_pos = self.default_dof_pos.repeat(num_envs, 1)
        self.dof_vel, self.last_dof_vel, self.torques = (self.zeros(12) for _ in range(3))
        self.dof_pos_limits = torch.tensor(self.cfg.asset.joint_limits, device=device)
        mid = self.dof_pos_limits.mean(-1)
        span = (self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]) * self.cfg.rewards.soft_dof_pos_limit
        self.dof_pos_limits = torch.stack((mid - span / 2, mid + span / 2), -1)
        self.commands = self.zeros(4)
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=device
        )
        self.actions, self.last_actions = self.zeros(12), self.zeros(12)
        self.action_history = torch.zeros(num_envs, 2, 12, device=device)
        self.motor_strengths = torch.ones(num_envs, 12, device=device)
        self.kp_scales, self.kd_scales = torch.ones_like(self.motor_strengths), torch.ones_like(self.motor_strengths)
        self.control_delay = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.critic_friction = torch.ones(num_envs, 1, device=device)
        self.critic_base_mass_offset, self.critic_base_com_offset = self.zeros(1), self.zeros(3)
        self.previous_commands, self.stop_command_cue = self.zeros(3), self.zeros(3)
        for name in (
            "stopping_phase_steps",
            "stop_cue_steps",
            "stop_stable_counter",
            "straight_motion_steps",
            "episode_length_buf",
        ):
            setattr(self, name, torch.zeros(num_envs, device=device, dtype=torch.long))
        for name in ("stopping_phase", "stop_step_pending"):
            setattr(self, name, torch.zeros(num_envs, device=device, dtype=torch.bool))
        self.stand_blend = torch.ones(num_envs, device=device)
        self.feet_air_time = self.zeros(4)
        self.last_contacts = torch.zeros(num_envs, 4, device=device, dtype=torch.bool)
        self.rigid_body_states = torch.zeros(num_envs, 4 + len(bodies), 13, device=device)
        self.contact_forces = torch.zeros(num_envs, 4 + len(bodies), 3, device=device)
        self.reset_buf = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.time_out_buf = self.reset_buf.clone()
        self.rew_buf = self.zeros()
        self.reward_scales = {k: v * dt for k, v in vars(self.cfg.rewards.scales).items() if v}
        self.reward_names = [k for k in self.reward_scales if k != "termination"]
        self.reward_functions = [getattr(self, "_reward_" + k) for k in self.reward_names]
        self.episode_sums = {k: self.zeros() for k in self.reward_scales}
        self._init_buffers()
        self._height_query = lambda positions: torch.zeros_like(positions[..., 0])
        self._velocity_writer = lambda ids: None
        xx, yy = torch.meshgrid(
            torch.linspace(-0.8, 0.8, 17, device=device), torch.linspace(-0.5, 0.5, 11, device=device), indexing="ij"
        )
        self.critic_height_offsets = torch.stack((xx.flatten(), yy.flatten(), torch.zeros(187, device=device)), dim=-1)
        self.local_height_offsets = torch.tensor(
            [[0, 0, 0], [0.16, 0, 0], [-0.16, 0, 0], [0, 0.10, 0], [0, -0.10, 0]], device=device
        )
        self.reset_idx(torch.arange(num_envs, device=device))

    def zeros(self, *dims):
        return torch.zeros(self.num_envs, *dims, device=self.device)

    def _resample_commands(self, env_ids):
        if getattr(self, "external_commands", False):
            return None
        return super()._resample_commands(env_ids)

    def reset(self, ids):
        self.reset_idx(ids)
        for name in (
            "actions",
            "last_actions",
            "last_last_actions",
            "action_history",
            "last_dof_vel",
            "feet_air_time",
            "last_contacts",
        ):
            getattr(self, name)[ids] = 0
        self.episode_length_buf[ids] = 0
        self.previous_foot_velocities[ids] = 0
        self.capture_recovery_active[ids] = False
        self.capture_recovery_steps[ids] = 0
        self.capture_recovery_pose_blend[ids] = 1

    def process_actions(self, actions):
        actions = actions.clamp(-self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        if self._capture_recovery_runtime_enabled():
            actions = actions.clone()
            active = self.capture_recovery_active
            limit = self.cfg.commands.capture_recovery_action_limit
            actions[active] = actions[active].clamp(-limit, limit)
        self.action_history[:, 1] = self.action_history[:, 0]
        self.action_history[:, 0] = actions
        self.actions[:] = self.action_history[torch.arange(self.num_envs, device=self.device), self.control_delay]
        return self.actions

    def finish_step(self):
        self.last_last_actions[:] = self.last_actions
        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel

    def compute_observations(self):
        super().compute_observations()
        # The source base step clips only the returned tensors, not stored frames.
        limit = self.cfg.normalization.clip_observations
        self.obs_buf.clamp_(-limit, limit)
        self.privileged_obs_buf.clamp_(-limit, limit)

    def _write_root_velocity(self, ids):
        self._velocity_writer(ids)

    def _terrain_scan(self):
        from .math import quat_apply_yaw

        offsets = self.critic_height_offsets.unsqueeze(0).expand(self.num_envs, -1, -1)
        quat = self.base_quat[:, None, :].expand(-1, 187, -1)
        positions = self.base_pos[:, None, :] + quat_apply_yaw(quat, offsets)
        return (self.base_pos[:, 2:3] - self.cfg.terrain.critic_height_reference - self._height_query(positions)).clamp(
            -1, 1
        ) * self.obs_scales.height_measurements

    def _local_base_height(self):
        positions = self.base_pos[:, None, :] + self.local_height_offsets[None, :, :]
        return torch.median(self.base_pos[:, 2:3] - self._height_query(positions), dim=1, keepdim=True).values
