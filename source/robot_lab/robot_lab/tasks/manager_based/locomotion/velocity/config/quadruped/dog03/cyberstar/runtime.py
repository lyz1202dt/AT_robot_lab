"""Stateful Dog03 task logic, independent of the simulator and importable on CPU."""

from __future__ import annotations

import json
import math
import re
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


def joint_values(config, joint_names, device, label):
    """Resolve a common `joint` default, full-match regexes and exact joint overrides."""
    entries = vars(config) if isinstance(config, SimpleNamespace) else config
    if isinstance(entries, (int, float)):
        entries = {"joint": entries}
    for pattern in entries:
        if pattern != "joint" and not any(re.fullmatch(pattern, name) for name in joint_names):
            raise ValueError(f"{label}: pattern {pattern!r} does not match any joint")
    values = []
    for name in joint_names:
        matches = [value for pattern, value in entries.items() if pattern != "joint" and re.fullmatch(pattern, name)]
        if name in entries:
            value = entries[name]
        elif len(matches) == 1:
            value = matches[0]
        elif not matches and "joint" in entries:
            value = entries["joint"]
        else:
            raise ValueError(f"{label}: expected one value for {name}, found {len(matches)} matches")
        values.append(value)
    result = torch.tensor(values, device=device, dtype=torch.float)
    if not torch.isfinite(result).all() or (result < 0).any():
        raise ValueError(f"{label}: values must be finite and non-negative")
    return result.unsqueeze(0)


class CyberstarRuntime(Dog03Flat282DomainRandMixin, Dog03Flat282Mixin, Dog03Flat47Mixin, Dog03Flat47):
    """Own the original tensor state; simulator I/O is injected by the Lab adapter."""

    def __init__(self, num_envs, device, dt=0.02, config=None):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("The control timestep must be finite and positive")
        self.cfg = namespace(config or source_config()["env"])
        self.num_envs, self.device, self.dt = num_envs, device, dt
        self.num_actions = self.num_dof = self.num_dofs = 12
        self.dof_names = [
            f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")
        ]
        self.p_gains = joint_values(self.cfg.control.stiffness, self.dof_names, device, "control.stiffness")
        self.d_gains = joint_values(self.cfg.control.damping, self.dof_names, device, "control.damping")
        friction = getattr(self.cfg.control, "motor_friction", SimpleNamespace(coulomb=0.0, viscous=0.0))
        self.nominal_joint_coulomb_friction = joint_values(
            friction.coulomb, self.dof_names, device, "control.motor_friction.coulomb"
        )
        self.nominal_joint_viscous_friction = joint_values(
            friction.viscous, self.dof_names, device, "control.motor_friction.viscous"
        )
        self.joint_coulomb_friction = self.nominal_joint_coulomb_friction.repeat(num_envs, 1)
        self.joint_viscous_friction = self.nominal_joint_viscous_friction.repeat(num_envs, 1)
        self.feet_indices = torch.arange(4, device=device)
        # Four ordered foot aliases precede the URDF's real bodies.
        bodies = self.cfg.asset.body_names
        self.penalised_contact_indices = torch.tensor(
            [4 + i for i, name in enumerate(bodies) if name.endswith(("_thigh", "_calf"))], device=device
        )
        self.obs_scales = self.cfg.normalization.obs_scales
        self.command_ranges = vars(self.cfg.commands.ranges)
        self.cfg.domain_rand.push_interval = max(1, math.ceil(self.cfg.domain_rand.push_interval_s / dt))
        self.next_push_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
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
        delay_min, delay_max = self.cfg.domain_rand.control_delay_steps
        if not isinstance(delay_min, int) or not isinstance(delay_max, int) or not 0 <= delay_min <= delay_max:
            raise ValueError("control_delay_steps must be non-negative, ordered integers")
        self.action_history = torch.zeros(num_envs, delay_max + 1, 12, device=device)
        motor_delay_min, motor_delay_max = getattr(self.cfg.domain_rand, "motor_delay_steps", (0, 0))
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in (motor_delay_min, motor_delay_max))
            or not 0 <= motor_delay_min <= motor_delay_max
        ):
            raise ValueError("motor_delay_steps must be non-negative, ordered integers")
        self.motor_delay = torch.zeros(num_envs, 12, device=device, dtype=torch.long)
        self.motor_target_history = torch.zeros(num_envs, motor_delay_max + 1, 12, device=device)
        self.motor_history_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        self._motor_env_ids = torch.arange(num_envs, device=device).unsqueeze(1)
        self._motor_joint_ids = torch.arange(12, device=device)
        self.motor_strengths = torch.ones(num_envs, 12, device=device)
        self.default_joint_angle_offsets = self.zeros(12)
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
        self.resample_push_schedule(torch.arange(num_envs, device=device))
        self.reset_motor_delay(torch.arange(num_envs, device=device))

    def zeros(self, *dims):
        return torch.zeros(self.num_envs, *dims, device=self.device)

    def _resample_commands(self, env_ids):
        if getattr(self, "external_commands", False):
            return None
        return super()._resample_commands(env_ids)

    def randomize_default_joint_angles(self, ids):
        config = self.cfg.domain_rand
        self.default_joint_angle_offsets[ids] = (
            torch.empty_like(self.default_joint_angle_offsets[ids]).uniform_(*config.default_joint_angle_range)
            if getattr(config, "randomize_default_joint_angles", False) and not self.cfg.env.test
            else 0
        )

    def randomize_joint_friction(self, ids):
        config = self.cfg.domain_rand
        enabled = getattr(config, "randomize_joint_friction", False) and not self.cfg.env.test
        for name in ("coulomb", "viscous"):
            nominal = getattr(self, f"nominal_joint_{name}_friction")
            target = getattr(self, f"joint_{name}_friction")
            values = nominal.expand(len(ids), -1).clone()
            if enabled:
                lower, upper = getattr(config, f"joint_{name}_friction_offset_range")
                if not all(math.isfinite(value) for value in (lower, upper)) or lower > upper:
                    raise ValueError(f"joint_{name}_friction_offset_range must be finite and ordered")
                values += torch.empty_like(values).uniform_(lower, upper)
            target[ids] = values.clamp_min(0)

    def reset_motor_delay(self, ids):
        config = self.cfg.domain_rand
        enabled = getattr(config, "randomize_motor_delay", False) and not self.cfg.env.test
        lower, upper = getattr(config, "motor_delay_steps", (0, 0))
        self.motor_delay[ids] = torch.randint(lower, upper + 1, (len(ids), 12), device=self.device) if enabled else 0
        self.motor_target_history[ids] = 0
        self.motor_history_steps[ids] = 0

    def delay_motor_targets(self, targets):
        """Delay each motor's setpoint in physics steps, keeping current PD feedback."""
        if not getattr(self.cfg.domain_rand, "randomize_motor_delay", False) or self.cfg.env.test:
            return targets
        self.motor_target_history[:, 1:] = self.motor_target_history[:, :-1].clone()
        self.motor_target_history[:, 0] = targets
        delays = torch.minimum(self.motor_delay, self.motor_history_steps[:, None])
        self.motor_history_steps.add_(1).clamp_(max=self.motor_target_history.shape[1] - 1)
        return self.motor_target_history[self._motor_env_ids, delays, self._motor_joint_ids]

    def compute_motor_torques(self, targets, positions, velocities, limits):
        targets = self.delay_motor_targets(targets)
        torque = (
            self.p_gains * self.kp_scales * (targets + self.default_joint_angle_offsets - positions)
            - self.d_gains * self.kd_scales * velocities
        ) * self.motor_strengths
        self.torques[:] = torch.clamp(torque, -limits, limits)
        return self.torques

    def resample_push_schedule(self, ids):
        config = self.cfg.domain_rand
        lower, upper = getattr(config, "push_interval_offset_range_s", (0.0, 0.0))
        base = config.push_interval_s
        if not all(math.isfinite(value) for value in (base, lower, upper)) or lower > upper or base + lower <= 0:
            raise ValueError("push interval and offsets must be finite, ordered and yield positive intervals")
        offsets = torch.zeros(len(ids), device=self.device)
        if getattr(config, "randomize_push_interval", False) and not self.cfg.env.test:
            offsets.uniform_(lower, upper)
        steps = ((base + offsets) / self.dt).ceil().long().clamp_min(1)
        self.next_push_steps[ids] = self.episode_length_buf[ids] + steps

    def _scheduled_push_env_ids(self, interval):
        lengths = self.episode_length_buf
        if getattr(self.cfg.domain_rand, "randomize_push_interval", False) and not self.cfg.env.test:
            ids = ((lengths > 0) & (lengths >= self.next_push_steps)).nonzero(as_tuple=False).flatten()
            self.resample_push_schedule(ids)
            return ids
        return ((lengths > 0) & (lengths % interval == 0)).nonzero(as_tuple=False).flatten()

    def reset(self, ids):
        self.reset_idx(ids)
        self.reset_motor_delay(ids)
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
        self.resample_push_schedule(ids)
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
        self.action_history[:, 1:] = self.action_history[:, :-1].clone()
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

    def _local_base_height(self, env_ids=None):
        base_pos = self.base_pos if env_ids is None else self.base_pos[env_ids]
        positions = base_pos[:, None, :] + self.local_height_offsets[None, :, :]
        return torch.median(base_pos[:, 2:3] - self._height_query(positions), dim=1, keepdim=True).values
