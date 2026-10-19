# Copyright (c) 2025-2026 HangZhou Cyberstar Co.,Ltd.
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from Cyberstar_rl_train; see LICENSE and provenance.json.

"""Source stair commands and rewards, independent of the simulator."""
import torch

from .math import torch_rand_float, wrap_to_pi


class Csdog1Stairs282Mixin:
    def _init_buffers(self):
        super()._init_buffers()
        self.stair_start_x = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.stair_command_progress = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.stair_heading_reference = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        if hasattr(self, "friction_coeffs"):
            self.critic_friction = self.friction_coeffs.to(
                device=self.device, dtype=torch.float
            ).view(self.num_envs, 1)
        else:
            self.critic_friction = torch.ones(
                self.num_envs, 1, dtype=torch.float, device=self.device
            )
        # Keep stair touchdown detection independent of feet_air_time's
        # last_contacts buffer because reward evaluation order is not fixed.
        self.stair_previous_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
        )
        self.stair_touchdown_event = torch.zeros_like(
            self.stair_previous_contacts
        )

    def _reset_stair_landing_buffers(self, env_ids):
        if not hasattr(self, "stair_previous_contacts"):
            return
        self.stair_previous_contacts[env_ids] = False
        self.stair_touchdown_event[env_ids] = False
        self.stair_command_progress[env_ids] = 0.0

    def _post_physics_step_callback(self):
        # The robust-flat callback snapshots previous_foot_velocities into
        # stop_touchdown_pre_velocity before updating them to the current
        # frame. Reuse that pre-impact value for ordinary stair touchdowns.
        super()._post_physics_step_callback()
        contacts = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        )
        self.stair_touchdown_event[:] = (
            contacts & (~self.stair_previous_contacts)
        )
        self.stair_previous_contacts[:] = contacts

        # Track progress in the commanded body-frame direction.  The old
        # x-only distance was valid for forward-only training but would mark
        # successful left/right stair traversal as no progress.
        targets = self.command_targets[:, :2]
        target_speed = torch.norm(targets, dim=1)
        direction = targets / target_speed.clamp_min(1.0e-6).unsqueeze(1)
        speed_along_command = torch.sum(
            self.base_lin_vel[:, :2] * direction, dim=1
        )
        active_stair_translation = (
            (~self.flat_env_mask)
            & (
                target_speed
                > self.cfg.commands.heading_hold_translation_deadband
            )
        )
        self.stair_command_progress += (
            torch.clamp(speed_along_command, min=0.0)
            * self.dt
            * active_stair_translation.float()
        )

    def _resample_commands(self, env_ids):
        """Replay flat commands and sample exclusive stair translations."""
        previous_translation = self.commands[env_ids, :2].clone()
        super()._resample_commands(env_ids)
        if len(env_ids) == 0 or not hasattr(self, "flat_env_mask"):
            return

        stair_local = ~self.flat_env_mask[env_ids]
        stair_ids = env_ids[stair_local]
        count = len(stair_ids)
        if count == 0:
            return

        forward = torch.rand(count, device=self.device) < (
            self.cfg.commands.stair_forward_ratio
        )
        lateral = ~forward
        forward_speed = torch_rand_float(
            self.cfg.commands.stair_forward_speed_range[0],
            self.cfg.commands.stair_forward_speed_range[1],
            (count, 1),
            device=self.device,
        ).squeeze(1)
        lateral_speed = torch_rand_float(
            self.cfg.commands.stair_lateral_speed_range[0],
            self.cfg.commands.stair_lateral_speed_range[1],
            (count, 1),
            device=self.device,
        ).squeeze(1)
        lateral_sign = torch.where(
            torch.rand(count, device=self.device) < 0.5,
            -torch.ones(count, device=self.device),
            torch.ones(count, device=self.device),
        )

        self.command_targets[stair_ids] = 0.0
        self.command_targets[stair_ids[forward], 0] = forward_speed[forward]
        self.command_targets[stair_ids[lateral], 1] = (
            lateral_sign[lateral] * lateral_speed[lateral]
        )

        # Preserve speed only when the old and new commands use the same axis
        # and direction.  Switching forward/left/right starts the new axis at
        # zero, preventing a temporary forward+lateral mixture during ramps.
        previous = previous_translation[stair_local]
        target = self.command_targets[stair_ids, :2]
        current = torch.zeros_like(target)
        same_forward = (
            forward
            & (previous[:, 0] > 0.0)
            & (torch.abs(previous[:, 1]) < 1.0e-4)
        )
        current[same_forward, 0] = torch.clamp(
            previous[same_forward, 0],
            min=0.0,
            max=self.cfg.commands.stair_forward_speed_range[1],
        )
        same_lateral = (
            lateral
            & (torch.abs(previous[:, 0]) < 1.0e-4)
            & (previous[:, 1] * target[:, 1] > 0.0)
        )
        current[same_lateral, 1] = torch.sign(
            target[same_lateral, 1]
        ) * torch.clamp(
            torch.abs(previous[same_lateral, 1]),
            min=0.0,
            max=self.cfg.commands.stair_lateral_speed_range[1],
        )
        current[self.reset_buf[stair_ids].bool()] = 0.0

        self.commands[stair_ids, :3] = 0.0
        self.commands[stair_ids, :2] = current
        self.command_ramp_active[stair_ids] = (
            torch.norm(current - target, dim=1) > 1.0e-4
        )

        minimum_steps = max(
            1,
            int(round(
                self.cfg.commands.stair_command_duration_range[0] / self.dt
            )),
        )
        maximum_steps = max(
            minimum_steps + 1,
            int(round(
                self.cfg.commands.stair_command_duration_range[1] / self.dt
            )) + 1,
        )
        self.command_steps_remaining[stair_ids] = torch.randint(
            minimum_steps,
            maximum_steps,
            (count,),
            device=self.device,
        )
        self.translation_heading_reference[stair_ids] = (
            self.stair_heading_reference[stair_ids]
        )
        self.translation_heading_active[stair_ids] = True

    def _stair_motion_mask(self):
        return (~self.flat_env_mask) & self._gait_active_mask()

    def _stair_progress_gate(self):
        target = self.command_targets[:, :2]
        target_speed = torch.norm(target, dim=1).clamp_min(0.20)
        direction = target / target_speed.unsqueeze(1)
        speed_along_command = torch.sum(
            self.base_lin_vel[:, :2] * direction, dim=1
        )
        return torch.clamp(
            speed_along_command / target_speed, min=0.0, max=1.0
        ) * self._stair_motion_mask().float()

    def _fixed_phase_motion_mask(self):
        """Keep the flat trot prior only on flat replay environments."""
        return super()._fixed_phase_motion_mask() & self.flat_env_mask

    def _reward_feet_clearance(self):
        flat_quality = super()._reward_feet_clearance()
        foot_positions = self.rigid_body_states[:, self.feet_indices, :3]
        clearance = foot_positions[:, :, 2] - self._terrain_height_at(
            foot_positions
        )
        in_air = (
            self.contact_forces[:, self.feet_indices, 2]
            < self.cfg.rewards.clearance_contact_threshold
        )
        target = torch.clamp(
            torch.abs(self.stair_step_heights)
            + self.cfg.rewards.clearance_margin,
            self.cfg.rewards.clearance_target_min,
            self.cfg.rewards.clearance_target_max,
        ).unsqueeze(1)
        quality = torch.exp(
            -torch.square(
                (clearance - target) / self.cfg.rewards.feet_clearance_sigma
            )
        )
        airborne_count = torch.sum(in_air, dim=1)
        stair_quality = torch.sum(
            quality * in_air.float(), dim=1
        ) / airborne_count.clamp_min(1)
        stair_quality *= (airborne_count > 0).float()
        stair_quality *= self._stair_progress_gate()
        return (
            flat_quality * self.flat_env_mask.float()
            + stair_quality
        )

    def _reward_feet_air_time(self):
        """Use physical touchdown timing for stairs and flat free-phase side steps."""
        contact = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        )
        contact_filt = torch.logical_or(contact, self.last_contacts)
        first_contact = (self.feet_air_time > 0.0) & contact_filt
        self.feet_air_time += self.dt

        stair = self._stair_motion_mask()
        flat_lateral = self.flat_env_mask & self._lateral_command_mask()
        eligible = stair | flat_lateral
        target = torch.where(
            stair,
            torch.full_like(
                self.gait_phase, self.cfg.rewards.stair_air_time_target
            ),
            torch.full_like(
                self.gait_phase, self.cfg.rewards.lateral_air_time_target
            ),
        ).unsqueeze(1)
        sigma = torch.where(
            stair,
            torch.full_like(
                self.gait_phase, self.cfg.rewards.stair_air_time_sigma
            ),
            torch.full_like(
                self.gait_phase, self.cfg.rewards.lateral_air_time_sigma
            ),
        ).unsqueeze(1)
        touchdown_quality = torch.exp(
            -torch.square((self.feet_air_time - target) / sigma)
        ) * first_contact.float()
        touchdown_count = torch.sum(first_contact, dim=1)
        quality = torch.sum(touchdown_quality, dim=1) / (
            touchdown_count.clamp_min(1)
        )
        quality *= (touchdown_count > 0).float()

        flat_progress = self._translation_progress_gate()
        progress = torch.where(
            stair, self._stair_progress_gate(), flat_progress
        )
        self.feet_air_time *= ~contact_filt
        self.feet_air_time[~eligible] = 0.0
        self.last_contacts = contact
        return quality * progress * eligible.float()

    def _reward_gait_height_quality(self):
        return (
            super()._reward_gait_height_quality()
            * self.flat_env_mask.float()
        )

    def _reward_sagittal_hip_quality(self):
        return (
            super()._reward_sagittal_hip_quality()
            * self.flat_env_mask.float()
        )

    def _reward_sagittal_width_quality(self):
        return (
            super()._reward_sagittal_width_quality()
            * self.flat_env_mask.float()
        )

    def _reward_stair_forward_progress(self):
        """Reward progress along forward, left or right stair commands."""
        return self._stair_progress_gate()

    def _reward_stair_base_height(self):
        error = self._local_base_height().squeeze(1) - (
            self.cfg.rewards.stair_base_height_target
        )
        return torch.square(error) * (~self.flat_env_mask).float()

    def _reward_stair_heading_alignment(self):
        error = wrap_to_pi(
            self.rpy[:, 2] - self.stair_heading_reference
        )
        return torch.exp(
            -torch.square(error / self.cfg.rewards.stair_heading_sigma)
        ) * self._stair_progress_gate()

    def _reward_stair_support_quality(self):
        """Require support without prescribing which diagonal lands first."""
        contact_count = torch.sum(
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold,
            dim=1,
        )
        support = torch.clamp(contact_count.float() / 2.0, 0.0, 1.0)
        return support * self._stair_progress_gate()

    def _stair_descent_weight(self):
        """Apply stronger landing protection on descending patches."""
        # Robots spawn on the positive-x side and move outward. Positive
        # pyramid step heights therefore descend in commanded +x; negative
        # heights ascend toward the outer edge.
        descending = self.stair_step_heights > 0.0
        multiplier = self.cfg.rewards.stair_descent_landing_multiplier
        return 1.0 + (multiplier - 1.0) * descending.float()

    def _stair_touchdown_mask(self):
        return (
            self.stair_touchdown_event.float()
            * self._stair_motion_mask().float().unsqueeze(1)
        )

    def _reward_stair_touchdown_impact(self):
        """Penalize only excessive force on a new stair touchdown."""
        force = torch.norm(
            self.contact_forces[:, self.feet_indices, :], dim=2
        )
        soft_limit = self.cfg.rewards.stair_touchdown_force_soft_limit
        excess = torch.clamp(
            force / soft_limit - 1.0,
            min=0.0,
            max=self.cfg.rewards.stair_touchdown_force_max_excess,
        )
        cost = torch.sum(
            torch.square(excess) * self._stair_touchdown_mask(), dim=1
        )
        return cost * self._stair_descent_weight()

    def _reward_stair_touchdown_velocity(self):
        """Use the pre-contact vertical speed rather than clamped velocity."""
        soft_limit = (
            self.cfg.rewards.stair_touchdown_vertical_speed_soft_limit
        )
        downward_speed = torch.clamp(
            -self.stop_touchdown_pre_velocity[:, :, 2] - soft_limit,
            min=0.0,
        )
        excess = torch.clamp(
            downward_speed / soft_limit,
            max=self.cfg.rewards.stair_touchdown_speed_max_excess,
        )
        cost = torch.sum(
            torch.square(excess) * self._stair_touchdown_mask(), dim=1
        )
        return cost * self._stair_descent_weight()

    def _reward_stair_pitch_rate(self):
        """Reject pitch for forward motion and roll for lateral motion."""
        soft_limit = self.cfg.rewards.stair_pitch_rate_soft_limit
        targets = self.command_targets[:, :2]
        lateral = torch.abs(targets[:, 1]) > torch.abs(targets[:, 0])
        relevant_rate = torch.where(
            lateral,
            torch.abs(self.base_ang_vel[:, 0]),
            torch.abs(self.base_ang_vel[:, 1]),
        )
        excess = torch.clamp(
            relevant_rate / soft_limit - 1.0,
            min=0.0,
            max=self.cfg.rewards.stair_pitch_rate_max_excess,
        )
        return (
            torch.square(excess)
            * self._stair_motion_mask().float()
            * self._stair_descent_weight()
        )

    def _reward_lin_vel_z(self):
        """Permit necessary vertical travel while retaining a soft bound."""
        cost = super()._reward_lin_vel_z()
        weight = torch.where(
            self.flat_env_mask,
            torch.ones_like(cost),
            torch.full_like(
                cost, self.cfg.rewards.stair_vertical_velocity_weight
            ),
        )
        return cost * weight
