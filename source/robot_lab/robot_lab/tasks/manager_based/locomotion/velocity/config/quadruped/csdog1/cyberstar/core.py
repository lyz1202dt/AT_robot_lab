# Copyright (c) 2025-2026 HangZhou Cyberstar Co.,Ltd.
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from Cyberstar_rl_train; see LICENSE and provenance.json.

"""Simulator-independent source formulas (quaternions use xyzw here)."""
import math
import torch
from .math import quat_apply, quat_rotate_inverse, quat_apply_yaw, torch_rand_float, wrap_to_pi

FRAME_DIM = 47
HISTORY_LENGTH = 6
HISTORY_DIM = 282
PRIVILEGED_DIM = 297
ROBUST_PRIVILEGED_DIM = 491

class TensorBase:
    def _init_buffers(self):
        pass

    def reset_idx(self, env_ids):
        pass

    def _push_robots(self):
        """Source planar push for tasks without the advanced push curriculum.

        The Lab adapter commits the selected root velocities to the simulator.
        Leave freshly reset and non-selected environments untouched.
        """
        interval = max(1, int(self.cfg.domain_rand.push_interval))
        env_ids = ((self.episode_length_buf > 0) & (self.episode_length_buf % interval == 0)).nonzero(
            as_tuple=False
        ).flatten()
        if len(env_ids) == 0:
            return
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[env_ids, 7:9] = torch_rand_float(
            -max_vel, max_vel, (len(env_ids), 2), device=self.device
        )
        self._write_root_velocity(env_ids)

    def _update_straight_motion_steps(self):
        """Track time since a pure sagittal command began for reward weighting."""
        straight = self._straight_command_mask()
        self.straight_motion_steps[straight] += 1
        self.straight_motion_steps[~straight] = 0

    def _update_stopping_phase(self):
        """Track sudden command stops for transition-specific rewards."""
        enabled = getattr(self.cfg.commands, 'train_stop_transition', False)
        if not enabled:
            self.stopping_phase[:] = False
            self.stop_step_pending[:] = False
            self.stop_cue_steps[:] = 0
            self.stop_command_cue[:] = 0.
            self.stop_stable_counter[:] = 0
            self.stand_blend[:] = 1.
            self.previous_commands[:] = self.commands[:, :3]
            return

        previous_speed = torch.norm(self.previous_commands[:, :2], dim=1)
        previous_yaw_speed = torch.abs(self.previous_commands[:, 2])
        command_speed = torch.norm(self.commands[:, :2], dim=1)
        trigger_speed = self.cfg.commands.stop_trigger_command_speed
        trigger_yaw_speed = getattr(
            self.cfg.commands, 'stop_trigger_yaw_speed', trigger_speed
        )
        command_threshold = self.cfg.commands.stop_command_threshold
        zero_command = (command_speed < command_threshold) & (
            torch.abs(self.commands[:, 2]) < command_threshold
        )
        resume_motion = ~zero_command
        self.stopping_phase_steps[resume_motion] = 0
        self.stopping_phase[resume_motion] = False
        self.stop_step_pending[resume_motion] = False
        self.stop_cue_steps[resume_motion] = 0
        self.stop_command_cue[resume_motion] = 0.
        self.stop_stable_counter[resume_motion] = 0
        self.stand_blend[resume_motion] = 0.

        new_stops = (
            (previous_speed > trigger_speed)
            | (previous_yaw_speed > trigger_yaw_speed)
        ) & zero_command

        duration_steps = max(
            1, int(self.cfg.commands.stop_transition_duration / self.dt)
        )
        min_duration_steps = max(
            0, int(self.cfg.commands.stop_transition_min_duration / self.dt)
        )
        self.stopping_phase_steps[new_stops] = duration_steps
        self.stop_step_pending[new_stops] = True
        self.stop_stable_counter[new_stops] = 0
        self.stand_blend[new_stops] = 0.
        cue_duration_steps = max(
            1, int(self.cfg.commands.stop_cue_duration / self.dt)
        )
        self.stop_cue_steps[new_stops] = cue_duration_steps
        cue_direction = self.previous_commands[new_stops, :2] / previous_speed[
            new_stops
        ].clamp_min(1.0e-6).unsqueeze(1)
        self.stop_command_cue[new_stops, :2] = (
            cue_direction * self.cfg.commands.stop_cue_magnitude
        )
        yaw_cue_magnitude = getattr(
            self.cfg.commands,
            'stop_cue_yaw_magnitude',
            self.cfg.commands.stop_cue_magnitude,
        )
        self.stop_command_cue[new_stops, 2] = (
            torch.sign(self.previous_commands[new_stops, 2])
            * yaw_cue_magnitude
        )

        active = self.stopping_phase_steps > 0
        actual_speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
        yaw_speed = torch.abs(self.base_ang_vel[:, 2])
        body_ang_speed = torch.norm(self.base_ang_vel[:, :2], dim=1)
        tilt = torch.norm(self.projected_gravity[:, :2], dim=1)
        joint_speed = torch.mean(torch.abs(self.dof_vel), dim=1)
        contact_threshold = self.cfg.commands.stop_release_contact_force
        foot_contacts = (
            self.contact_forces[:, self.feet_indices, 2] > contact_threshold
        )
        contact_count = torch.sum(foot_contacts, dim=1)

        stable_now = (
            active
            & (actual_speed < self.cfg.commands.stop_release_speed)
            & (yaw_speed < self.cfg.commands.stop_release_yaw_speed)
            & (body_ang_speed < self.cfg.commands.stop_release_body_ang_speed)
            & (tilt < self.cfg.commands.stop_release_tilt)
            & (joint_speed < self.cfg.commands.stop_release_joint_speed)
            & (contact_count >= self.cfg.commands.stop_release_min_contacts)
        )
        self.stop_stable_counter[~stable_now] = 0
        self.stop_stable_counter[stable_now] += 1

        elapsed_steps = duration_steps - self.stopping_phase_steps
        can_release = elapsed_steps >= min_duration_steps
        stable_steps = max(
            1, int(self.cfg.commands.stop_stable_duration / self.dt)
        )
        settled = active & can_release & (
            self.stop_stable_counter >= stable_steps
        )
        self.stopping_phase_steps[settled] = 0
        self.stop_stable_counter[settled] = 0

        active = self.stopping_phase_steps > 0
        self.stopping_phase_steps[active] -= 1
        self.stopping_phase[:] = active
        self.stop_step_pending &= self.stopping_phase
        cue_active = (self.stop_cue_steps > 0) & self.stopping_phase
        self.stop_cue_steps[cue_active] -= 1
        self.stop_cue_steps[~self.stopping_phase] = 0
        self.stop_command_cue[self.stop_cue_steps <= 0] = 0.

        self.stand_blend[self.stopping_phase | resume_motion] = 0.
        recovering = zero_command & (~self.stopping_phase)
        blend_duration = max(self.cfg.commands.stand_blend_duration, self.dt)
        self.stand_blend[recovering] = torch.clamp(
            self.stand_blend[recovering] + self.dt / blend_duration,
            min=0.0,
            max=1.0,
        )
        self.previous_commands[:] = self.commands[:, :3]

    def _get_policy_commands(self):
        """Return actor commands with a short stop-transition cue."""
        policy_commands = self.commands[:, :3].clone()
        cue_active = self.stop_cue_steps > 0
        policy_commands[cue_active] = self.stop_command_cue[cue_active]
        return policy_commands

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _straight_start_multiplier(self):
        onset_steps = max(
            1, int(self.cfg.rewards.straight_start_duration / self.dt)
        )
        onset = self.straight_motion_steps <= onset_steps
        multiplier = getattr(
            self.cfg.rewards, 'straight_start_multiplier', 1.0
        )
        return 1.0 + (multiplier - 1.0) * onset.float()

    def _reward_straight_lateral_velocity(self):
        """Penalize lateral drift during commanded forward/backward motion."""
        return (
            torch.square(self.base_lin_vel[:, 1])
            * self._straight_command_mask().float()
            * self._straight_start_multiplier()
        )

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_sagittal_foot_symmetry(self):
        """Keep left/right foot trajectories mirrored during straight motion."""
        sagittal_speed = torch.abs(self.commands[:, 0])
        min_speed = getattr(
            self.cfg.rewards, 'sagittal_symmetry_min_speed', 0.2
        )
        active = (sagittal_speed > min_speed).float()

        command_sigma = getattr(
            self.cfg.rewards, 'sagittal_symmetry_command_sigma', 0.25
        )
        non_sagittal_command = (
            torch.square(self.commands[:, 1])
            + torch.square(self.commands[:, 2])
        )
        sagittal_weight = torch.exp(
            -non_sagittal_command / (command_sigma ** 2)
        )

        foot_positions = self.rigid_body_states[:, self.feet_indices, 0:3]
        relative_positions = foot_positions - self.base_pos.unsqueeze(1)
        repeated_quat = self.base_quat.unsqueeze(1).expand(
            -1, len(self.feet_indices), -1
        )
        foot_positions_body = quat_rotate_inverse(
            repeated_quat.reshape(-1, 4),
            relative_positions.reshape(-1, 3),
        ).view(self.num_envs, len(self.feet_indices), 3)

        front_lateral_asymmetry = (
            foot_positions_body[:, 0, 1] + foot_positions_body[:, 1, 1]
        )
        rear_lateral_asymmetry = (
            foot_positions_body[:, 2, 1] + foot_positions_body[:, 3, 1]
        )
        rear_lateral_width = (
            foot_positions_body[:, 2, 1] - foot_positions_body[:, 3, 1]
        )
        foot_sigma = getattr(
            self.cfg.rewards, 'sagittal_foot_symmetry_sigma', 0.03
        )
        rear_min_width = getattr(
            self.cfg.rewards, 'sagittal_rear_min_width', 0.27
        )
        rear_width_sigma = getattr(
            self.cfg.rewards, 'sagittal_rear_width_sigma', 0.02
        )
        rear_width_deficit = torch.clamp(
            rear_min_width - rear_lateral_width, min=0.0
        )
        symmetry_reward = torch.exp(
            -0.5 * (
                torch.square(front_lateral_asymmetry / foot_sigma)
                + torch.square(rear_lateral_asymmetry / foot_sigma)
                + torch.square(rear_width_deficit / rear_width_sigma)
            )
        )
        return symmetry_reward * sagittal_weight * active

class Csdog1Flat47(TensorBase):
    """A genuine 47-D policy with a two-dimensional trot clock."""

    def _init_buffers(self):
        super()._init_buffers()
        self.frame_noise_scale_vec = self._build_frame_noise_scale_vec()
        self.clean_actor_frame = torch.zeros(
            self.num_envs, FRAME_DIM, dtype=torch.float, device=self.device
        )
        self.gait_phase = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.stop_phase_settled = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.phase_recovery_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.phase_settle_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.stop_recovery_initial_swing = torch.zeros(
            self.num_envs, 4, dtype=torch.bool, device=self.device
        )
        self.stop_recovery_landed = torch.zeros(
            self.num_envs, 4, dtype=torch.bool, device=self.device
        )
        self.stop_recovery_previous_contacts = torch.zeros(
            self.num_envs, 4, dtype=torch.bool, device=self.device
        )
        self.stop_recovery_touchdown_event = torch.zeros(
            self.num_envs, 4, dtype=torch.bool, device=self.device
        )
        self.stop_recovery_touchdown_complete = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.straight_heading_reference = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        # Per-environment timers avoid synchronizing command changes with the
        # gait cycle. In particular, a stop must be seen at every gait
        # phase rather than only at phase zero.
        self.command_steps_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.command_targets = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )
        self.command_ramp_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.short_motion_step_choices = torch.tensor(
            self.cfg.commands.short_motion_steps,
            dtype=torch.long,
            device=self.device,
        )
        self.hip_indices = torch.tensor(
            [
                index
                for index, name in enumerate(self.dof_names)
                if "hip_joint" in name
            ],
            dtype=torch.long,
            device=self.device,
        )
        if len(self.hip_indices) != 4:
            raise ValueError(
                f"Expected four hip joints, got {len(self.hip_indices)}"
            )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if hasattr(self, "gait_phase"):
            self.gait_phase[env_ids] = 0.0
        if hasattr(self, "stop_phase_settled"):
            self.stop_phase_settled[env_ids] = True
            self.phase_recovery_steps[env_ids] = 0
            self.phase_settle_counter[env_ids] = 0
            self.stop_recovery_initial_swing[env_ids] = False
            self.stop_recovery_landed[env_ids] = False
            self.stop_recovery_previous_contacts[env_ids] = False
            self.stop_recovery_touchdown_event[env_ids] = False
            self.stop_recovery_touchdown_complete[env_ids] = True
        if (
            len(env_ids) > 0
            and getattr(self.cfg.commands, "initial_stand_time_s", 0.0) > 0.0
        ):
            # Every episode begins with a known zero-command posture. This
            # gives the policy an explicit initial-pose and recovery target.
            self.commands[env_ids, :] = 0.0
            if hasattr(self, "command_steps_remaining"):
                self.command_steps_remaining[env_ids] = max(
                    1,
                    int(round(
                        self.cfg.commands.initial_stand_time_s / self.dt
                    )),
                )
        if len(env_ids) > 0 and hasattr(self, "previous_commands"):
            self.command_targets[env_ids] = self.commands[env_ids, :3]
            self.command_ramp_active[env_ids] = False
            self.previous_commands[env_ids] = self.commands[env_ids, :3]
            self.stopping_phase_steps[env_ids] = 0
            self.stopping_phase[env_ids] = False
            self.stop_step_pending[env_ids] = False
            self.stop_cue_steps[env_ids] = 0
            self.stop_command_cue[env_ids] = 0.0
            self.stop_stable_counter[env_ids] = 0
            self.stand_blend[env_ids] = 1.0
            self.straight_motion_steps[env_ids] = 0

    def _gait_active_mask(self):
        return (
            (torch.norm(self.commands[:, :2], dim=1) > 0.2)
            | (torch.abs(self.commands[:, 2]) > 0.05)
        )

    def _target_gait_active_mask(self):
        """Return the intended mode even while a start command is ramping."""
        return (
            (torch.norm(self.command_targets[:, :2], dim=1) > 0.2)
            | (torch.abs(self.command_targets[:, 2]) > 0.05)
        )

    def _update_command_ramps(self):
        """Apply acceleration-limited starts without smoothing stop commands."""
        active = self.command_ramp_active
        if not torch.any(active):
            return
        max_delta = torch.tensor(
            [
                self.cfg.commands.start_acceleration_x,
                self.cfg.commands.start_acceleration_y,
                self.cfg.commands.start_acceleration_yaw,
            ],
            dtype=self.commands.dtype,
            device=self.device,
        ) * self.dt
        delta = self.command_targets[active] - self.commands[active, :3]
        step = torch.maximum(torch.minimum(delta, max_delta), -max_delta)
        self.commands[active, :3] += step
        reached_local = torch.all(
            torch.abs(delta) <= max_delta + 1.0e-6, dim=1
        )
        active_ids = active.nonzero(as_tuple=False).flatten()
        reached_ids = active_ids[reached_local]
        self.commands[reached_ids, :3] = self.command_targets[reached_ids]
        self.command_ramp_active[reached_ids] = False

    def _stop_recovery_mask(self):
        """Include the whole phase-guided landing, not only base braking state."""
        return (~self._gait_active_mask()) & (~self.stop_phase_settled)

    def _stop_swing_mask(self):
        return self._stop_recovery_mask() & (
            ~self.stop_recovery_touchdown_complete
        )

    def _stop_settle_mask(self):
        return self._stop_recovery_mask() & (
            self.stop_recovery_touchdown_complete
        )

    def _post_physics_step_callback(self):
        """Update commands, braking state, heading reference and phase clock."""
        if getattr(self.cfg.commands, "use_structured_commands", False):
            self.command_steps_remaining -= 1
            env_ids = (
                self.command_steps_remaining <= 0
            ).nonzero(as_tuple=False).flatten()
        else:
            interval = max(
                1, int(self.cfg.commands.resampling_time / self.dt)
            )
            env_ids = (
                self.episode_length_buf % interval == 0
            ).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if getattr(self.cfg.commands, "use_structured_commands", False):
            self._update_command_ramps()

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0
            )

        was_stopping = self.stopping_phase.clone()
        self._update_stopping_phase()
        new_stops = self.stopping_phase & (~was_stopping)
        self.stop_phase_settled[new_stops] = False
        self.phase_recovery_steps[new_stops] = 0
        self.phase_settle_counter[new_stops] = 0

        self._update_straight_motion_steps()
        new_straight = self.straight_motion_steps == 1
        self.straight_heading_reference[new_straight] = self.rpy[
            new_straight, 2
        ]

        active = self._gait_active_mask()
        self.stop_phase_settled[active] = False
        self.phase_recovery_steps[active] = 0
        self.phase_settle_counter[active] = 0
        contacts = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.commands.stop_release_contact_force
        )
        self.stop_recovery_touchdown_event[:] = False
        recovery = (~active) & (~self.stop_phase_settled)
        recovery_start = recovery & (self.phase_recovery_steps == 0)

        # Capture only the feet that are actually swinging when the command
        # disappears. A dog that is already in stable four-foot support does
        # not need to lift a foot merely to satisfy a timer.
        self.stop_recovery_initial_swing[recovery_start] = (
            ~contacts[recovery_start]
        )
        self.stop_recovery_landed[recovery_start] = False
        self.stop_recovery_previous_contacts[recovery_start] = contacts[
            recovery_start
        ]
        self.stop_recovery_touchdown_complete[recovery_start] = ~torch.any(
            self.stop_recovery_initial_swing[recovery_start], dim=1
        )

        new_contacts = contacts & (~self.stop_recovery_previous_contacts)
        touchdown_event = (
            recovery.unsqueeze(1)
            & self.stop_recovery_initial_swing
            & (~self.stop_recovery_landed)
            & new_contacts
        )
        self.stop_recovery_touchdown_event[:] = touchdown_event
        self.stop_recovery_landed |= touchdown_event
        completed_landing = torch.all(
            (~self.stop_recovery_initial_swing)
            | self.stop_recovery_landed,
            dim=1,
        )
        self.stop_recovery_touchdown_complete[recovery] |= completed_landing[
            recovery
        ]
        self.stop_recovery_previous_contacts[recovery] = contacts[recovery]
        self.phase_recovery_steps[recovery] += 1

        # Clear event state as soon as a new motion command takes control.
        self.stop_recovery_initial_swing[active] = False
        self.stop_recovery_landed[active] = False
        self.stop_recovery_touchdown_complete[active] = False

        contact_count = torch.sum(contacts, dim=1)
        stable_steps = max(
            1,
            int(round(
                self.cfg.commands.phase_settle_stable_duration / self.dt
            )),
        )
        stable_candidate = (
            self._stop_settle_mask()
            & (contact_count >= self.cfg.commands.phase_settle_min_contacts)
            & (
                torch.norm(self.base_lin_vel[:, :2], dim=1)
                < self.cfg.commands.phase_settle_speed
            )
            & (
                torch.abs(self.base_ang_vel[:, 2])
                < self.cfg.commands.phase_settle_yaw_speed
            )
            & (
                torch.norm(self.base_ang_vel[:, :2], dim=1)
                < self.cfg.commands.phase_settle_body_ang_speed
            )
            & (
                torch.norm(self.projected_gravity[:, :2], dim=1)
                < self.cfg.commands.phase_settle_tilt
            )
            & (
                torch.mean(torch.abs(self.dof_vel), dim=1)
                < self.cfg.commands.phase_settle_joint_speed
            )
        )
        self.phase_settle_counter[stable_candidate] += 1
        self.phase_settle_counter[
            recovery & (~stable_candidate)
        ] = 0
        safe_touchdown = self._stop_settle_mask() & (
            self.phase_settle_counter >= stable_steps
        )
        self.stop_phase_settled[safe_touchdown] = True
        self.stop_step_pending[safe_touchdown] = False
        self.phase_settle_counter[safe_touchdown] = 0

        # Keep stand shaping off until the phase-guided landing is complete.
        self.stand_blend[recovery] = 0.0

        # Continue the recovery clock only until the captured swing feet have
        # landed. Phase zero then becomes the explicit settling/stand target.
        recovery_swing = self._stop_swing_mask()
        self.gait_phase[active] = torch.remainder(
            self.gait_phase[active]
            + self.dt / self.cfg.control.gait_cycle_s,
            1.0,
        )
        self.gait_phase[recovery_swing] = torch.remainder(
            self.gait_phase[recovery_swing]
            + self.dt / self.cfg.control.recovery_gait_cycle_s,
            1.0,
        )
        self.gait_phase[
            (~active) & self.stop_recovery_touchdown_complete
        ] = 0.0

    def _resample_commands(self, env_ids):
        """Sample explicit stand, sagittal, lateral, yaw and mixed segments."""
        if len(env_ids) == 0:
            return
        if not getattr(self.cfg.commands, "use_structured_commands", False):
            self._resample_vendor_commands(env_ids)
            return

        count = len(env_ids)
        was_moving = self._target_gait_active_mask()[env_ids]
        self.commands[env_ids, :] = 0.0
        selector = torch.rand(count, device=self.device)
        stand_end = self.cfg.commands.stand_ratio
        sagittal_end = stand_end + self.cfg.commands.sagittal_ratio
        lateral_end = sagittal_end + self.cfg.commands.lateral_ratio
        yaw_end = lateral_end + self.cfg.commands.yaw_ratio
        if yaw_end > 1.0 + 1.0e-6:
            raise ValueError("structured command ratios must sum to at most 1")

        # Every moving segment is followed by a stand/recovery segment, so a
        # sampled short pulse always produces a training stop transition.
        forced_stand = was_moving
        selectable = ~forced_stand
        stand = forced_stand | (selectable & (selector < stand_end))
        sagittal = (
            selectable & (selector >= stand_end) & (selector < sagittal_end)
        )
        lateral = (
            selectable
            & (selector >= sagittal_end)
            & (selector < lateral_end)
        )
        yaw = (
            selectable & (selector >= lateral_end) & (selector < yaw_end)
        )
        mixed = selectable & (selector >= yaw_end)

        def duration_steps(duration_range):
            lower = max(1, int(round(duration_range[0] / self.dt)))
            upper = max(lower, int(round(duration_range[1] / self.dt)))
            return torch.randint(
                lower, upper + 1, (count,), device=self.device
            )

        durations = duration_steps(self.cfg.commands.motion_duration_range)
        stand_durations = duration_steps(
            self.cfg.commands.stand_duration_range
        )
        durations[stand] = stand_durations[stand]
        interrupted_recovery = forced_stand & (
            torch.rand(count, device=self.device)
            < self.cfg.commands.interrupted_recovery_ratio
        )
        interrupted_durations = duration_steps(
            self.cfg.commands.interrupted_recovery_duration_range
        )
        durations[interrupted_recovery] = interrupted_durations[
            interrupted_recovery
        ]
        moving = ~stand
        short_motion = moving & (
            torch.rand(count, device=self.device)
            < self.cfg.commands.short_motion_ratio
        )
        short_count = int(short_motion.sum().item())
        if short_count > 0:
            choice_ids = torch.randint(
                0,
                len(self.short_motion_step_choices),
                (short_count,),
                device=self.device,
            )
            durations[short_motion] = self.short_motion_step_choices[choice_ids]
        self.command_steps_remaining[env_ids] = durations

        def signed_magnitude(
            mask, minimum, maximum, negative_probability=0.5
        ):
            number = int(mask.sum().item())
            if number == 0:
                return None
            magnitude = torch_rand_float(
                minimum, maximum, (number, 1), device=self.device
            ).squeeze(1)
            sign = torch.where(
                torch.rand(number, device=self.device) < negative_probability,
                -torch.ones(number, device=self.device),
                torch.ones(number, device=self.device),
            )
            return sign * magnitude

        sagittal_values = signed_magnitude(
            sagittal,
            self.cfg.commands.min_sagittal_speed,
            max(abs(value) for value in self.command_ranges["lin_vel_x"]),
            self.cfg.commands.backward_command_ratio,
        )
        if sagittal_values is not None:
            self.commands[env_ids[sagittal], 0] = sagittal_values

        lateral_values = signed_magnitude(
            lateral,
            self.cfg.commands.min_lateral_speed,
            max(abs(value) for value in self.command_ranges["lin_vel_y"]),
        )
        if lateral_values is not None:
            self.commands[env_ids[lateral], 1] = lateral_values

        yaw_values = signed_magnitude(
            yaw,
            self.cfg.commands.min_yaw_speed,
            max(abs(value) for value in self.command_ranges["ang_vel_yaw"]),
        )
        if yaw_values is not None:
            self.commands[env_ids[yaw], 2] = yaw_values

        mixed_ids = env_ids[mixed]
        if len(mixed_ids) > 0:
            self.commands[mixed_ids, 0] = torch_rand_float(
                self.command_ranges["lin_vel_x"][0],
                self.command_ranges["lin_vel_x"][1],
                (len(mixed_ids), 1),
                device=self.device,
            ).squeeze(1)
            self.commands[mixed_ids, 1] = torch_rand_float(
                self.command_ranges["lin_vel_y"][0],
                self.command_ranges["lin_vel_y"][1],
                (len(mixed_ids), 1),
                device=self.device,
            ).squeeze(1)
            self.commands[mixed_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1],
                (len(mixed_ids), 1),
                device=self.device,
            ).squeeze(1)
            small = (
                torch.norm(self.commands[mixed_ids, :2], dim=1) < 0.2
            ) & (torch.abs(self.commands[mixed_ids, 2]) < 0.1)
            self.commands[mixed_ids[small], 0] = (
                self.cfg.commands.min_sagittal_speed
            )

        # Store the sampled target, then ramp selected starts. Stops remain
        # abrupt to match keyboard deployment and train the recovery phase.
        self.command_targets[env_ids] = self.commands[env_ids, :3]
        ramp_probability = torch.zeros(count, device=self.device)
        ramp_probability[sagittal] = self.cfg.commands.sagittal_start_ramp_ratio
        ramp_probability[lateral] = self.cfg.commands.lateral_start_ramp_ratio
        ramp_probability[yaw] = self.cfg.commands.yaw_start_ramp_ratio
        ramp_probability[mixed] = self.cfg.commands.mixed_start_ramp_ratio
        ramped = moving & (~short_motion) & (
            torch.rand(count, device=self.device) < ramp_probability
        )
        self.command_ramp_active[env_ids] = ramped
        self.commands[env_ids[ramped], :3] = 0.0

    def _phase_features(self):
        angle = 2.0 * math.pi * self.gait_phase
        return torch.stack((torch.sin(angle), torch.cos(angle)), dim=1)

    def _build_clean_actor_frame(self):
        return torch.cat((
            self._phase_features(),
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self._get_policy_commands() * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=1)

    def _build_frame_noise_scale_vec(self):
        """Build noise in the same semantic order as the 47-D observation."""
        noise = torch.zeros(FRAME_DIM, device=self.device)
        scales = self.cfg.noise.noise_scales
        level = self.cfg.noise.noise_level

        # [phase2, angular velocity3, gravity3, commands3,
        #  joint position12, joint velocity12, previous action12]
        noise[0:2] = 0.0
        noise[2:5] = (
            scales.ang_vel * level * self.obs_scales.ang_vel
        )
        noise[5:8] = scales.gravity * level
        noise[8:11] = 0.0
        noise[11:23] = (
            scales.dof_pos * level * self.obs_scales.dof_pos
        )
        noise[23:35] = (
            scales.dof_vel * level * self.obs_scales.dof_vel
        )
        noise[35:47] = 0.0
        return noise

    def _current_actor_frame(self):
        self.clean_actor_frame[:] = self._build_clean_actor_frame()
        actor_frame = self.clean_actor_frame.clone()
        if self.add_noise:
            # Phase remains deterministic. Draw only for the remaining 45
            # entries so phase never receives accidental numerical noise.
            vendor_frame = actor_frame[:, 2:]
            vendor_frame += (
                2.0 * torch.rand_like(vendor_frame) - 1.0
            ) * self.frame_noise_scale_vec[2:]
        return actor_frame

    def _write_flat_critic_observation(self):
        """Write clean current state for the later asymmetric history critic."""
        if self.privileged_obs_buf is None:
            return
        # PPO retains this reference across env.step(), so replace the tensor.
        self.privileged_obs_buf = torch.cat((
            self.clean_actor_frame,
            self.base_lin_vel * self.obs_scales.lin_vel,
        ), dim=1)

    def compute_observations(self):
        # PPO keeps a reference to obs_buf until process_env_step(). Replacing
        # the tensor is essential: an in-place write here would mutate the
        # previous observation into the next one before RolloutStorage copies
        # it, pairing action/value(t) with observation(t+1).
        self.obs_buf = self._current_actor_frame()
        self._write_flat_critic_observation()

    def _reward_tracking_lin_vel(self):
        error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]
        ), dim=1)
        return torch.exp(-error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_active(self):
        """Track commanded translation without rewarding stand samples."""
        return self._reward_tracking_lin_vel() * (
            (torch.norm(self.commands[:, :2], dim=1) > 0.2).float()
        )

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 0.1
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        reward = torch.sum(
            (self.feet_air_time - 0.5) * first_contact, dim=1
        )
        reward *= torch.norm(self.commands[:, :2], dim=1) > 0.2
        self.feet_air_time *= ~contact_filt
        return reward

    def _reward_stand_still(self):
        deviation = torch.abs(self.dof_pos - self.default_dof_pos)
        return torch.mean(deviation, dim=1) * self._stand_reward_weight()

    def _stand_command_mask(self):
        return (
            torch.norm(self.commands[:, :3], dim=1)
            < self.cfg.rewards.stand_command_threshold
        )

    def _stand_reward_weight(self):
        """Blend stand shaping in only after the braking phase settles."""
        return self._stand_command_mask().float() * self.stand_blend

    def _sagittal_command_mask(self):
        """Select pure forward/backward commands for posture shaping."""
        return (
            (torch.abs(self.commands[:, 0]) > 0.2)
            & (torch.abs(self.commands[:, 1]) < 0.05)
            & (torch.abs(self.commands[:, 2]) < 0.05)
        )

    def _foot_positions_body(self):
        foot_positions = self.rigid_body_states[:, self.feet_indices, :3]
        relative_positions = foot_positions - self.base_pos.unsqueeze(1)
        repeated_quat = self.base_quat.unsqueeze(1).expand(
            -1, len(self.feet_indices), -1
        )
        return quat_rotate_inverse(
            repeated_quat.reshape(-1, 4),
            relative_positions.reshape(-1, 3),
        ).view(self.num_envs, len(self.feet_indices), 3)

    def _reward_stand_velocity(self):
        """Penalize residual translation and rotation after command removal."""
        linear_error = torch.sum(torch.square(self.base_lin_vel), dim=1)
        angular_error = torch.sum(torch.square(self.base_ang_vel), dim=1)
        return (
            linear_error
            + self.cfg.rewards.stand_angular_velocity_weight * angular_error
        ) * self._stand_reward_weight()

    def _reward_stand_orientation(self):
        """Penalize leaning while the commanded velocity is zero."""
        tilt_error = torch.sum(
            torch.square(self.projected_gravity[:, :2]), dim=1
        )
        return tilt_error * self._stand_reward_weight()

    def _reward_stand_stability(self):
        """Positive bounded reward for a stopped, upright default posture."""
        velocity_error = (
            torch.sum(torch.square(self.base_lin_vel), dim=1)
            + self.cfg.rewards.stand_angular_velocity_weight
            * torch.sum(torch.square(self.base_ang_vel), dim=1)
        )
        tilt_error = torch.sum(
            torch.square(self.projected_gravity[:, :2]), dim=1
        )
        joint_error = torch.mean(
            torch.square(self.dof_pos - self.default_dof_pos), dim=1
        )
        quality = torch.exp(
            -velocity_error / self.cfg.rewards.stand_velocity_sigma
            -tilt_error / self.cfg.rewards.stand_orientation_sigma
            -joint_error / self.cfg.rewards.stand_joint_sigma
        )
        return quality * self._stand_reward_weight()

    def _reward_stand_pose_quality(self):
        """Reward the nominal joint pose without an exponential dead zone."""
        joint_error = torch.mean(
            torch.square(self.dof_pos - self.default_dof_pos), dim=1
        )
        quality = 1.0 / (
            1.0 + joint_error / self.cfg.rewards.stand_pose_sigma
        )
        return quality * self._stand_reward_weight()

    def _reward_stand_motion_quality(self):
        """Reward motionless joints and feet under a zero command."""
        joint_motion = torch.mean(torch.square(self.dof_vel), dim=1)
        foot_velocities = self.rigid_body_states[
            :, self.feet_indices, 7:10
        ]
        foot_motion = torch.mean(torch.square(foot_velocities), dim=(1, 2))
        joint_quality = 1.0 / (
            1.0
            + joint_motion / self.cfg.rewards.stand_joint_velocity_sigma
        )
        foot_quality = 1.0 / (
            1.0
            + foot_motion / self.cfg.rewards.stand_foot_velocity_sigma
        )
        return 0.5 * (joint_quality + foot_quality) * self._stand_reward_weight()

    def _reward_stand_contact_quality(self):
        """Prefer a quiet four-foot support polygon while standing."""
        contacts = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.stand_contact_threshold
        )
        contact_fraction = torch.mean(contacts.float(), dim=1)
        all_feet = torch.all(contacts, dim=1).float()
        quality = 0.5 * contact_fraction + 0.5 * all_feet
        return quality * self._stand_reward_weight()

    def _reward_stand_height_quality(self):
        """Prevent the rear-leg crouch that lowers the base while stopped."""
        height_error = torch.square(
            self.root_states[:, 2] - self.cfg.rewards.stand_base_height_target
        )
        quality = 1.0 / (
            1.0 + height_error / self.cfg.rewards.stand_height_sigma
        )
        return quality * self._stand_reward_weight()

    def _reward_stand_level_quality(self):
        """Supply a broad positive signal for a level body at zero command."""
        tilt_error = torch.sum(
            torch.square(self.projected_gravity[:, :2]), dim=1
        )
        quality = 1.0 / (
            1.0 + tilt_error / self.cfg.rewards.stand_level_sigma
        )
        return quality * self._stand_reward_weight()

    def _reward_stand_action_quality(self):
        """Suppress residual gait actions while allowing small load offsets."""
        action_error = torch.mean(torch.square(self.actions), dim=1)
        quality = 1.0 / (
            1.0 + action_error / self.cfg.rewards.stand_action_sigma
        )
        return quality * self._stand_reward_weight()

    def _reward_action_magnitude(self):
        """Soft-limit large targets on the direct vendor action path."""
        excess = torch.clamp(
            torch.abs(self.actions) - self.cfg.rewards.action_soft_limit,
            min=0.0,
        )
        return torch.sum(torch.square(excess), dim=1)

    def _reward_stop_post_step_motion(self):
        joint_motion = torch.mean(torch.square(self.dof_vel), dim=1)
        stable = torch.exp(
            -joint_motion / self.cfg.rewards.stop_post_step_motion_sigma
        )
        return stable * self._stop_settle_mask().float()

    def _reward_stop_post_step_contact(self):
        contacts = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.stop_post_step_contact_force
        )
        return (
            torch.mean(contacts.float(), dim=1)
            * self._stop_settle_mask().float()
        )

    def _reward_stop_velocity(self):
        current_speed = torch.norm(self.root_states[:, 7:9], dim=1)
        stable_velocity = torch.exp(
            -torch.square(current_speed) / self.cfg.rewards.stop_velocity_sigma
        )
        return stable_velocity * self._stop_settle_mask().float()

    def _reward_stop_path_velocity(self):
        """This term integrates to travelled path length during braking."""
        speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
        return speed * self._stop_recovery_mask().float()

    def _reward_stop_tilt(self):
        tilt = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return tilt * self._stop_recovery_mask().float()

    def _reward_stop_yaw_rate(self):
        yaw_rate = torch.square(self.base_ang_vel[:, 2])
        return yaw_rate * self._stop_recovery_mask().float()

    def _reward_stop_phase_completion(self):
        """Match the trot schedule only while captured swing feet are airborne."""
        contacts = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        )
        desired_stance = self._desired_stance_mask()
        phase_agreement = (contacts == desired_stance).float().mean(dim=1)
        return phase_agreement * self._stop_swing_mask().float()

    def _reward_stop_touchdown(self):
        """Sparse reward for the real contact edge of captured swing feet."""
        return torch.sum(
            self.stop_recovery_touchdown_event.float(), dim=1
        )

    def _reward_stop_feet_slip(self):
        contacts = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        )
        planar_velocity = self.rigid_body_states[
            :, self.feet_indices, 7:9
        ]
        slip = torch.sum(
            torch.square(planar_velocity) * contacts.unsqueeze(-1).float(),
            dim=(1, 2),
        )
        return slip * self._stop_recovery_mask().float()

    def _reward_straight_heading(self):
        heading_error = wrap_to_pi(
            self.rpy[:, 2] - self.straight_heading_reference
        )
        return (
            torch.square(heading_error)
            * self._straight_command_mask().float()
        )

    def _lateral_command_mask(self):
        return (
            (torch.abs(self.commands[:, 1]) > 0.2)
            & (torch.abs(self.commands[:, 0]) < 0.05)
            & (torch.abs(self.commands[:, 2]) < 0.05)
        )

    def _reward_lateral_cross_velocity(self):
        return (
            torch.square(self.base_lin_vel[:, 0])
            * self._lateral_command_mask().float()
        )

    def _reward_lateral_yaw_rate(self):
        return (
            torch.square(self.base_ang_vel[:, 2])
            * self._lateral_command_mask().float()
        )

    def _reward_lateral_direction_quality(self):
        """Reward pure side motion without longitudinal or yaw leakage."""
        quality = torch.exp(
            -torch.square(self.base_lin_vel[:, 0])
            / self.cfg.rewards.lateral_cross_sigma
            -torch.square(self.base_ang_vel[:, 2])
            / self.cfg.rewards.lateral_yaw_sigma
        )
        return quality * self._lateral_command_mask().float()

    def _reward_lateral_hip_excess(self):
        excess = torch.clamp(
            torch.abs(self.dof_pos[:, self.hip_indices])
            - self.cfg.rewards.lateral_hip_soft_limit,
            min=0.0,
        )
        return (
            torch.sum(torch.square(excess), dim=1)
            * self._lateral_command_mask().float()
        )

    def _reward_lateral_width_balance(self):
        foot_positions_body = self._foot_positions_body()
        front_width = torch.abs(
            foot_positions_body[:, 0, 1] - foot_positions_body[:, 1, 1]
        )
        rear_width = torch.abs(
            foot_positions_body[:, 2, 1] - foot_positions_body[:, 3, 1]
        )
        minimum = self.cfg.rewards.lateral_width_min
        maximum = self.cfg.rewards.lateral_width_max
        below = torch.square(torch.clamp(minimum - front_width, min=0.0))
        below += torch.square(torch.clamp(minimum - rear_width, min=0.0))
        above = torch.square(torch.clamp(front_width - maximum, min=0.0))
        above += torch.square(torch.clamp(rear_width - maximum, min=0.0))
        asymmetry = torch.square(torch.clamp(
            torch.abs(front_width - rear_width)
            - self.cfg.rewards.lateral_width_asymmetry_soft_limit,
            min=0.0,
        ))
        return (
            (below + above + asymmetry)
            * self._lateral_command_mask().float()
        )

    def _turn_command_mask(self):
        return (
            (torch.abs(self.commands[:, 2]) > 0.2)
            & (torch.norm(self.commands[:, :2], dim=1) < 0.05)
        )

    def _reward_turn_translation(self):
        return (
            torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
            * self._turn_command_mask().float()
        )

    def _reward_gait_height_quality(self):
        """Prevent the policy from lowering its body during locomotion."""
        height_error = torch.square(
            self.root_states[:, 2] - self.cfg.rewards.gait_base_height_target
        )
        quality = 1.0 / (
            1.0 + height_error / self.cfg.rewards.gait_height_sigma
        )
        return quality * self._gait_active_mask().float()

    def _reward_sagittal_hip_quality(self):
        """Keep abduction joints near neutral in straight locomotion."""
        hip_error = torch.mean(
            torch.square(self.dof_pos[:, self.hip_indices]), dim=1
        )
        quality = 1.0 / (
            1.0 + hip_error / self.cfg.rewards.sagittal_hip_sigma
        )
        return quality * self._sagittal_command_mask().float()

    def _reward_sagittal_width_quality(self):
        """Discourage the excessively wide support polygon seen in playback."""
        foot_positions_body = self._foot_positions_body()
        front_width = torch.abs(
            foot_positions_body[:, 0, 1] - foot_positions_body[:, 1, 1]
        )
        rear_width = torch.abs(
            foot_positions_body[:, 2, 1] - foot_positions_body[:, 3, 1]
        )
        width_error = 0.5 * (
            torch.square(
                front_width - self.cfg.rewards.sagittal_front_width_target
            )
            + torch.square(
                rear_width - self.cfg.rewards.sagittal_rear_width_target
            )
        )
        quality = 1.0 / (
            1.0 + width_error / self.cfg.rewards.sagittal_width_sigma
        )
        return quality * self._sagittal_command_mask().float()

    def _reward_joint_deviation(self):
        error = torch.sum(torch.square(
            self.dof_pos - self.default_dof_pos
        ), dim=1)
        return torch.exp(-error / 0.6)

    def _reward_feet_clearance(self):
        foot_positions = self.rigid_body_states[:, self.feet_indices, :3]
        foot_heights = foot_positions[:, :, 2]
        contact_forces = self.contact_forces[:, self.feet_indices, 2]
        in_air = contact_forces < self.cfg.rewards.clearance_contact_threshold
        height_error = torch.abs(
            foot_heights - self.cfg.rewards.feet_clearance_target
        )
        height_reward = torch.exp(
            -height_error / self.cfg.rewards.feet_clearance_sigma
        )
        clearance = torch.sum(
            height_reward * in_air.float(), dim=1
        ) / (torch.sum(in_air.float(), dim=1) + 1.0e-6)
        speed_factor = (
            torch.norm(self.commands[:, :2], dim=1) >= 0.4
        ).float()
        return clearance * speed_factor

    def _diagonal_leg_phase(self):
        """Return phases for feet ordered [FL, FR, RL, RR].

        PND offsets its two legs by half a cycle. For a quadruped trot the same
        construction is applied to diagonal pairs: FL+RR and FR+RL.
        """
        phase_a = self.gait_phase
        phase_b = torch.remainder(
            self.gait_phase + self.cfg.rewards.phase_offset, 1.0
        )
        return torch.stack(
            (phase_a, phase_b, phase_b, phase_a), dim=1
        )

    def _desired_stance_mask(self):
        return (
            self._diagonal_leg_phase()
            < self.cfg.rewards.phase_stance_ratio
        )

    def _reward_phase_contact(self):
        """PND-style agreement between desired phase and actual contact."""
        desired_stance = self._desired_stance_mask()
        actual_contact = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        )
        agreement = (actual_contact == desired_stance).float().mean(dim=1)
        return agreement * self._gait_active_mask().float()

    def _reward_feet_swing_height(self):
        """PND-style squared height error for feet that are in the air."""
        foot_states = self.rigid_body_states[:, self.feet_indices, :]
        foot_heights = foot_states[:, :, 2]
        contact = (
            torch.norm(
                self.contact_forces[:, self.feet_indices, :3], dim=2
            ) > self.cfg.rewards.phase_contact_threshold
        )
        height_error = torch.square(
            foot_heights - self.cfg.rewards.phase_swing_height_target
        )
        return torch.sum(height_error * (~contact).float(), dim=1) * (
            self._gait_active_mask().float()
        )

    def _reward_contact_no_vel(self):
        """PND-style penalty for feet sliding while in contact."""
        foot_states = self.rigid_body_states[:, self.feet_indices, :]
        foot_velocities = foot_states[:, :, 7:10]
        contact = (
            torch.norm(
                self.contact_forces[:, self.feet_indices, :3], dim=2
            ) > self.cfg.rewards.phase_contact_threshold
        )
        contact_velocity = foot_velocities * contact.unsqueeze(-1).float()
        return torch.sum(
            torch.square(contact_velocity), dim=(1, 2)
        ) * self._gait_active_mask().float()

class Csdog1Flat47Mixin:
    """Validated 47-D behavior layered cooperatively over the common base."""

    def _init_buffers(self):
        super()._init_buffers()
        self.last_motion_command = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )
        self.stop_entry_direction = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )
        self.zero_command_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.contact_duty_ema = torch.full(
            (self.num_envs, 4),
            0.5,
            dtype=torch.float,
            device=self.device,
        )
        self.finetune_short_step_choices = torch.tensor(
            self.cfg.commands.finetune_short_motion_steps,
            dtype=torch.long,
            device=self.device,
        )
        self.stop_adjustment_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.stop_adjustment_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.stop_adjustment_contact_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.stop_adjustment_start_phase = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.stop_adjustment_lifted = torch.zeros(
            self.num_envs, 4, dtype=torch.bool, device=self.device
        )
        self.stop_adjustment_landed = torch.zeros(
            self.num_envs, 4, dtype=torch.bool, device=self.device
        )
        self.stop_adjustment_previous_contacts = torch.zeros(
            self.num_envs, 4, dtype=torch.bool, device=self.device
        )
        self.stop_adjustment_touchdown_event = torch.zeros(
            self.num_envs, 4, dtype=torch.bool, device=self.device
        )
        self.stop_adjustment_supported_pairs = torch.zeros(
            self.num_envs, 2, dtype=torch.bool, device=self.device
        )
        self.stop_adjustment_unsafe_contact_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.pose_recovery_blend = torch.ones(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.stop_foot_targets = torch.tensor(
            self.cfg.rewards.stop_foot_targets,
            dtype=torch.float,
            device=self.device,
        )
        if self.stop_foot_targets.shape != (4, 2):
            raise ValueError(
                "stop_foot_targets must have shape [4, 2] in "
                "[FL, FR, RL, RR] order"
            )
        self.thigh_indices = torch.tensor(
            [
                index
                for index, name in enumerate(self.dof_names)
                if "thigh_joint" in name
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.calf_indices = torch.tensor(
            [
                index
                for index, name in enumerate(self.dof_names)
                if "calf_joint" in name
            ],
            dtype=torch.long,
            device=self.device,
        )
        if len(self.thigh_indices) != 4 or len(self.calf_indices) != 4:
            raise ValueError(
                "Expected four thigh and four calf joints, got "
                f"{len(self.thigh_indices)} and {len(self.calf_indices)}"
            )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0 or not hasattr(self, "last_motion_command"):
            return
        self.last_motion_command[env_ids] = 0.0
        self.stop_entry_direction[env_ids] = 0.0
        self.zero_command_steps[env_ids] = 0
        self.contact_duty_ema[env_ids] = 0.5
        self.stop_adjustment_active[env_ids] = False
        self.stop_adjustment_steps[env_ids] = 0
        self.stop_adjustment_contact_counter[env_ids] = 0
        self.stop_adjustment_start_phase[env_ids] = 0.0
        self.stop_adjustment_lifted[env_ids] = False
        self.stop_adjustment_landed[env_ids] = False
        self.stop_adjustment_previous_contacts[env_ids] = False
        self.stop_adjustment_touchdown_event[env_ids] = False
        self.stop_adjustment_supported_pairs[env_ids] = False
        self.stop_adjustment_unsafe_contact_steps[env_ids] = 0
        self.pose_recovery_blend[env_ids] = 1.0

    def _resample_commands(self, env_ids):
        """Bias long sagittal samples high while retaining keyboard taps."""
        if len(env_ids) == 0:
            return

        was_moving = self._target_gait_active_mask()[env_ids].clone()
        previous_motion = self.last_motion_command[env_ids].clone()
        previous_zero_steps = self.zero_command_steps[env_ids].clone()
        super()._resample_commands(env_ids)

        if not getattr(self.cfg.commands, "use_structured_commands", False):
            return

        # Uniform [0.25, 1.0] sampling left relatively few long high-speed
        # sagittal frames. Replace half of newly sampled pure sagittal targets
        # by |vx| in [0.8, 1.0]. Preserve the sign, the explicit 50/50
        # forward/backward mix, and any start ramp selected by the base task.
        sampled_targets = self.command_targets[env_ids]
        pure_sagittal = (
            (~was_moving)
            & (torch.abs(sampled_targets[:, 0]) > 0.2)
            & (torch.abs(sampled_targets[:, 1]) < 0.05)
            & (torch.abs(sampled_targets[:, 2]) < 0.05)
        )
        high_speed = pure_sagittal & (
            torch.rand(len(env_ids), device=self.device)
            < self.cfg.commands.finetune_high_speed_sagittal_ratio
        )
        high_speed_count = int(high_speed.sum().item())
        if high_speed_count > 0:
            high_speed_ids = env_ids[high_speed]
            direction = torch.sign(sampled_targets[high_speed, 0])
            magnitude = (
                self.cfg.commands.finetune_high_speed_min
                + (
                    self.cfg.commands.finetune_high_speed_max
                    - self.cfg.commands.finetune_high_speed_min
                )
                * torch.rand(high_speed_count, device=self.device)
            )
            self.command_targets[high_speed_ids, 0] = direction * magnitude
            not_ramped = ~self.command_ramp_active[high_speed_ids]
            immediate_ids = high_speed_ids[not_ramped]
            self.commands[immediate_ids, 0] = self.command_targets[
                immediate_ids, 0
            ]

        # Convert a subset of newly sampled moving segments into exact
        # keyboard-like taps.  Their signs are sampled with the same explicit
        # 50% backward ratio as the ordinary sagittal command distribution.
        target_is_moving = self._target_gait_active_mask()[env_ids]
        new_motion = (~was_moving) & target_is_moving
        targeted_tap = new_motion & (
            torch.rand(len(env_ids), device=self.device)
            < self.cfg.commands.finetune_short_motion_ratio
        )

        # Also recreate rapid repeated taps after a short zero-command gap.
        # Reusing the previous direction is sign-balanced because the source
        # sagittal taps themselves are sampled 50/50 forward/backward.
        min_gap_steps = max(
            1,
            int(round(
                self.cfg.commands.finetune_repeat_stop_range[0] / self.dt
            )),
        )
        max_gap_steps = max(
            min_gap_steps,
            int(round(
                self.cfg.commands.finetune_repeat_stop_range[1] / self.dt
            )),
        )
        previous_was_sagittal = (
            (torch.abs(previous_motion[:, 0]) > 0.2)
            & (torch.abs(previous_motion[:, 1]) < 0.05)
            & (torch.abs(previous_motion[:, 2]) < 0.05)
        )
        repeat_tap = (
            (~was_moving)
            & previous_was_sagittal
            & (previous_zero_steps >= min_gap_steps)
            & (previous_zero_steps <= max_gap_steps)
            & (
                torch.rand(len(env_ids), device=self.device)
                < self.cfg.commands.finetune_repeat_motion_ratio
            )
        )

        selected = targeted_tap | repeat_tap
        selected_count = int(selected.sum().item())
        if selected_count == 0:
            return

        selected_ids = env_ids[selected]
        selected_local = selected.nonzero(as_tuple=False).flatten()
        directions = torch.where(
            torch.rand(selected_count, device=self.device)
            < self.cfg.commands.backward_command_ratio,
            -torch.ones(selected_count, device=self.device),
            torch.ones(selected_count, device=self.device),
        )
        repeat_selected = repeat_tap[selected]
        directions[repeat_selected] = torch.sign(
            previous_motion[selected_local[repeat_selected], 0]
        )

        self.commands[selected_ids, :] = 0.0
        self.commands[selected_ids, 0] = (
            directions * self.cfg.commands.finetune_short_motion_speed
        )
        self.command_targets[selected_ids] = self.commands[selected_ids, :3]
        self.command_ramp_active[selected_ids] = False

        choice_ids = torch.randint(
            0,
            len(self.finetune_short_step_choices),
            (selected_count,),
            device=self.device,
        )
        self.command_steps_remaining[selected_ids] = (
            self.finetune_short_step_choices[choice_ids]
        )

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()

        active = self._gait_active_mask()
        self.zero_command_steps[active] = 0
        self.zero_command_steps[~active] += 1
        self.last_motion_command[active] = self.commands[active, :3]

        # The first recovery step still knows the command direction that just
        # disappeared. Preserve it for a direction-aware rebound penalty.
        recovery_start = (
            (~active)
            & (~self.stop_phase_settled)
            & (self.phase_recovery_steps == 1)
        )
        previous_xy = self.last_motion_command[recovery_start, :2]
        if previous_xy.numel() > 0:
            self.stop_entry_direction[recovery_start] = previous_xy / torch.norm(
                previous_xy, dim=1, keepdim=True
            ).clamp_min(1.0e-6)

        # A command release starts one complete, smooth gait adjustment. The
        # phase continues from its current value; it is never reset in the
        # middle of a swing. Completion requires a short all-feet contact
        # window (or three feet at the safety timeout), so the policy can
        # reposition its feet before nominal-pose shaping takes over.
        contacts_bool = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.commands.stop_release_contact_force
        )
        contact_count = torch.sum(contacts_bool, dim=1)
        new_stops = recovery_start
        self.stop_adjustment_active[new_stops] = True
        self.stop_adjustment_steps[new_stops] = 0
        self.stop_adjustment_contact_counter[new_stops] = 0
        self.stop_adjustment_start_phase[new_stops] = self.gait_phase[
            new_stops
        ]
        self.stop_adjustment_lifted[new_stops] = False
        self.stop_adjustment_landed[new_stops] = False
        self.stop_adjustment_previous_contacts[new_stops] = contacts_bool[
            new_stops
        ]
        self.stop_adjustment_touchdown_event[new_stops] = False
        self.stop_adjustment_supported_pairs[new_stops] = False
        self.stop_adjustment_unsafe_contact_steps[new_stops] = 0
        self.pose_recovery_blend[new_stops] = 0.0

        self.stop_adjustment_active[active] = False
        self.stop_adjustment_steps[active] = 0
        self.stop_adjustment_contact_counter[active] = 0
        self.stop_adjustment_lifted[active] = False
        self.stop_adjustment_landed[active] = False
        self.stop_adjustment_previous_contacts[active] = False
        self.stop_adjustment_supported_pairs[active] = False
        self.stop_adjustment_unsafe_contact_steps[active] = 0
        self.pose_recovery_blend[active] = 0.0

        self.stop_adjustment_touchdown_event[:] = False
        adjusting = self.stop_adjustment_active & (~active)
        self.stop_adjustment_steps[adjusting] += 1
        adjusting_feet = adjusting.unsqueeze(1)
        self.stop_adjustment_lifted |= adjusting_feet & (~contacts_bool)
        adjustment_touchdown = (
            adjusting_feet
            & self.stop_adjustment_lifted
            & contacts_bool
            & (~self.stop_adjustment_previous_contacts)
        )
        self.stop_adjustment_touchdown_event[:] = adjustment_touchdown
        self.stop_adjustment_landed |= adjustment_touchdown
        self.stop_adjustment_previous_contacts[adjusting] = contacts_bool[
            adjusting
        ]
        # A valid trot adjustment must retain a diagonal support pair. Count
        # unsafe frames and remember that both diagonal pairs have supported
        # the body during the cycle. This prevents a four-foot hop from
        # satisfying the same completion condition as a real adjustment step.
        diagonal_support = torch.stack((
            contacts_bool[:, 0] & contacts_bool[:, 3],
            contacts_bool[:, 1] & contacts_bool[:, 2],
        ), dim=1)
        self.stop_adjustment_supported_pairs |= (
            adjusting_feet & diagonal_support
        )
        unsafe_support = adjusting & (
            contact_count
            < self.cfg.commands.stop_adjustment_min_support_contacts
        )
        self.stop_adjustment_unsafe_contact_steps[unsafe_support] += 1
        minimum_steps = max(
            1,
            int(round(
                self.cfg.commands.stop_adjustment_min_duration / self.dt
            )),
        )
        soft_maximum_steps = max(
            minimum_steps,
            int(round(
                self.cfg.commands.stop_adjustment_max_duration / self.dt
            )),
        )
        hard_maximum_steps = max(
            soft_maximum_steps,
            int(round(
                self.cfg.commands.stop_adjustment_hard_max_duration / self.dt
            )),
        )
        eligible = adjusting & (
            self.stop_adjustment_steps >= minimum_steps
        )
        lifted_count = torch.sum(
            self.stop_adjustment_lifted, dim=1
        )
        lifted_feet_landed = torch.all(
            (~self.stop_adjustment_lifted)
            | self.stop_adjustment_landed,
            dim=1,
        )
        support_sequence_complete = torch.all(
            self.stop_adjustment_supported_pairs, dim=1
        )
        support_remained_safe = (
            self.stop_adjustment_unsafe_contact_steps
            <= self.cfg.commands.stop_adjustment_max_unsafe_contact_steps
        )
        completed_real_step = (
            lifted_count
            >= self.cfg.commands.stop_adjustment_required_lifted_feet
        ) & lifted_feet_landed & support_sequence_complete & support_remained_safe
        stable_contacts = (
            eligible
            & completed_real_step
            & (
                contact_count
                >= self.cfg.commands.stop_adjustment_completion_contacts
            )
        )
        self.stop_adjustment_contact_counter[stable_contacts] += 1
        self.stop_adjustment_contact_counter[
            adjusting & (~stable_contacts)
        ] = 0
        required_contact_steps = max(
            1,
            int(round(
                self.cfg.commands.stop_adjustment_contact_duration / self.dt
            )),
        )
        normal_complete = adjusting & (
            self.stop_adjustment_contact_counter
            >= required_contact_steps
        )
        safety_complete = (
            adjusting
            & (self.stop_adjustment_steps >= soft_maximum_steps)
            & completed_real_step
            & (contact_count >= 3)
        )
        hard_complete = (
            adjusting
            & (self.stop_adjustment_steps >= hard_maximum_steps)
            & (contact_count >= 3)
        )
        adjustment_complete = (
            normal_complete | safety_complete | hard_complete
        )
        ongoing_adjustment = adjusting & (~adjustment_complete)

        adjustment_angle = (
            self.stop_adjustment_steps.float()
            * self.dt
            / self.cfg.commands.stop_adjustment_cycle_s
        )
        self.gait_phase[ongoing_adjustment] = torch.remainder(
            self.stop_adjustment_start_phase[ongoing_adjustment]
            + adjustment_angle[ongoing_adjustment],
            1.0,
        )
        self.stop_phase_settled[ongoing_adjustment] = False
        self.stop_recovery_touchdown_complete[ongoing_adjustment] = False
        self.phase_settle_counter[ongoing_adjustment] = 0
        self.stand_blend[ongoing_adjustment] = 0.0

        self.stop_adjustment_active[adjustment_complete] = False
        self.stop_recovery_touchdown_complete[adjustment_complete] = True
        self.stop_phase_settled[adjustment_complete] = False
        self.phase_settle_counter[adjustment_complete] = 0
        self.gait_phase[adjustment_complete] = 0.0

        pose_ready = (
            (~active)
            & (~self.stop_adjustment_active)
            & self.stop_recovery_touchdown_complete
        )
        pose_blend_duration = max(
            self.cfg.commands.stop_pose_blend_duration, self.dt
        )
        self.pose_recovery_blend[pose_ready] = torch.clamp(
            self.pose_recovery_blend[pose_ready]
            + self.dt / pose_blend_duration,
            min=0.0,
            max=1.0,
        )

        contacts = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        ).float()
        alpha = math.exp(
            -self.dt / self.cfg.rewards.contact_duty_ema_time_constant
        )
        self.contact_duty_ema.mul_(alpha).add_(contacts, alpha=1.0 - alpha)
        # Do not carry an old directional gait's duty bias through a stand or
        # lateral/turn segment into the next straight segment.
        sagittal = self._sagittal_command_mask()
        self.contact_duty_ema[~sagittal] = 0.5

    def _stand_reward_weight(self):
        """Start nominal-pose shaping smoothly after adjustment touchdown."""
        blend = torch.maximum(
            self.stand_blend, self.pose_recovery_blend
        )
        return self._stand_command_mask().float() * blend

    def _persistent_stop_mask(self):
        """Cover touchdown, braking, and the complete stand-blend interval."""
        zero_command = ~self._gait_active_mask()
        return zero_command & (
            (~self.stop_phase_settled)
            | self.stopping_phase
            | (self.stand_blend < 1.0)
        )

    def _persistent_stop_settle_mask(self):
        return self._persistent_stop_mask() & (
            self.stop_recovery_touchdown_complete
        )

    def _reward_stop_velocity(self):
        speed = torch.norm(self.root_states[:, 7:9], dim=1)
        quality = torch.exp(
            -torch.square(speed) / self.cfg.rewards.stop_velocity_sigma
        )
        return quality * self._persistent_stop_settle_mask().float()

    def _reward_stop_path_velocity(self):
        speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
        return speed * self._persistent_stop_mask().float()

    def _reward_stop_tilt(self):
        tilt = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return tilt * self._persistent_stop_mask().float()

    def _reward_stop_yaw_rate(self):
        return (
            torch.square(self.base_ang_vel[:, 2])
            * self._persistent_stop_mask().float()
        )

    def _reward_stop_feet_slip(self):
        contacts = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        )
        planar_velocity = self.rigid_body_states[:, self.feet_indices, 7:9]
        slip = torch.sum(
            torch.square(planar_velocity) * contacts.unsqueeze(-1).float(),
            dim=(1, 2),
        )
        return slip * self._persistent_stop_mask().float()

    def _reward_stop_post_step_motion(self):
        joint_motion = torch.mean(torch.square(self.dof_vel), dim=1)
        quality = torch.exp(
            -joint_motion / self.cfg.rewards.stop_post_step_motion_sigma
        )
        return quality * self._persistent_stop_settle_mask().float()

    def _reward_stop_post_step_contact(self):
        contacts = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.stop_post_step_contact_force
        )
        return (
            torch.mean(contacts.float(), dim=1)
            * self._persistent_stop_settle_mask().float()
        )

    def _reward_stop_rebound(self):
        """Penalize motion opposite to the command that was just released."""
        along_entry = torch.sum(
            self.base_lin_vel[:, :2] * self.stop_entry_direction, dim=1
        )
        reverse_speed = torch.clamp(-along_entry, min=0.0)
        return (
            torch.square(reverse_speed)
            * self._persistent_stop_mask().float()
        )

    def _reward_stop_action_rate(self):
        action_delta = self.actions - self.last_actions
        return (
            torch.mean(torch.square(action_delta), dim=1)
            * self.stop_adjustment_active.float()
        )

    def _phase_swing_clearance_quality(self, target_height, sigma):
        """Reward every phase-designated swing foot, even if it stays planted."""
        desired_swing = ~self._desired_stance_mask()
        foot_heights = self.rigid_body_states[:, self.feet_indices, 2]
        contacts = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        )
        height_quality = torch.exp(
            -torch.square((foot_heights - target_height) / sigma)
        )
        foot_quality = (
            0.65 * height_quality + 0.35 * (~contacts).float()
        )
        swing_count = torch.sum(desired_swing.float(), dim=1).clamp_min(1.0)
        mean_quality = torch.sum(
            foot_quality * desired_swing.float(), dim=1
        ) / swing_count
        weakest_quality = torch.min(
            torch.where(
                desired_swing,
                foot_quality,
                torch.ones_like(foot_quality),
            ),
            dim=1,
        ).values
        # The weakest-foot term prevents the policy from sacrificing one leg,
        # which is exactly what the negative-yaw RR trace showed.
        return 0.5 * (mean_quality + weakest_quality)

    def _reward_stop_swing_clearance_quality(self):
        quality = self._phase_swing_clearance_quality(
            self.cfg.rewards.stop_adjustment_swing_height,
            self.cfg.rewards.stop_adjustment_swing_sigma,
        )
        return quality * self.stop_adjustment_active.float()

    def _reward_stop_adjustment_touchdown(self):
        return torch.sum(
            self.stop_adjustment_touchdown_event.float(), dim=1
        )

    def _reward_stop_support_quality(self):
        """Reward the phase-designated diagonal pair supporting the body."""
        contacts = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        )
        desired_stance = self._desired_stance_mask()
        desired_support = contacts & desired_stance
        support_fraction = torch.sum(
            desired_support.float(), dim=1
        ) / torch.sum(desired_stance.float(), dim=1).clamp_min(1.0)
        complete_pair = torch.all(
            contacts | (~desired_stance), dim=1
        ).float()
        quality = 0.5 * (support_fraction + complete_pair)
        return quality * self.stop_adjustment_active.float()

    def _reward_stop_support_loss(self):
        """Strongly reject one-foot support and simultaneous flight."""
        contacts = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        )
        desired_stance = self._desired_stance_mask()
        missing_fraction = torch.sum(
            (desired_stance & (~contacts)).float(), dim=1
        ) / torch.sum(desired_stance.float(), dim=1).clamp_min(1.0)
        contact_count = torch.sum(contacts, dim=1)
        support_deficit = torch.clamp(
            self.cfg.commands.stop_adjustment_min_support_contacts
            - contact_count,
            min=0,
        ).float() / self.cfg.commands.stop_adjustment_min_support_contacts
        loss = missing_fraction + 2.0 * support_deficit
        return loss * self.stop_adjustment_active.float()

    def _reward_stop_vertical_motion(self):
        """Penalize body hopping while leaving normal foot swing available."""
        vertical_speed = self.root_states[:, 9]
        height_error = (
            self.root_states[:, 2]
            - self.cfg.rewards.stop_adjustment_base_height_target
        )
        motion = (
            torch.square(vertical_speed)
            + self.cfg.rewards.stop_adjustment_height_weight
            * torch.square(height_error)
        )
        return motion * self.stop_adjustment_active.float()

    def _reward_stop_foot_placement_quality(self):
        """Place stance feet without rewarding swing feet for staying planted."""
        feet_xy = self._foot_positions_body()[:, :, :2]
        foot_error = torch.sum(
            torch.square(feet_xy - self.stop_foot_targets.unsqueeze(0)),
            dim=2,
        )
        foot_quality = 1.0 / (
            1.0
            + foot_error / self.cfg.rewards.stop_foot_placement_sigma
        )
        desired_stance = self._desired_stance_mask().float()
        quality = torch.sum(
            foot_quality * desired_stance, dim=1
        ) / torch.sum(desired_stance, dim=1).clamp_min(1.0)
        return quality * self.stop_adjustment_active.float()

    def _reward_stop_pose_transition_quality(self):
        """Gradually introduce the default pose during the adjustment step."""
        pose_error = torch.mean(
            torch.square(self.dof_pos - self.default_dof_pos), dim=1
        )
        quality = 1.0 / (
            1.0 + pose_error / self.cfg.rewards.stop_pose_sigma
        )
        minimum_steps = max(
            1,
            int(round(
                self.cfg.commands.stop_adjustment_min_duration / self.dt
            )),
        )
        raw_progress = torch.clamp(
            self.stop_adjustment_steps.float() / minimum_steps,
            min=0.0,
            max=1.0,
        )
        # Leave the first half-cycle free for swing, then introduce nominal
        # pose shaping continuously instead of locking the joints at release.
        progress = torch.clamp(
            2.0 * raw_progress - 1.0, min=0.0, max=1.0
        )
        return (
            quality
            * progress
            * self.stop_adjustment_active.float()
        )

    def _reward_stand_joint_symmetry_quality(self):
        """Prefer the nominal four-leg pose, not merely a motionless pose."""
        hip_error = torch.mean(
            torch.square(self.dof_pos[:, self.hip_indices]), dim=1
        )
        thigh = self.dof_pos[:, self.thigh_indices]
        calf = self.dof_pos[:, self.calf_indices]
        thigh_error = torch.mean(
            torch.square(thigh - torch.mean(thigh, dim=1, keepdim=True)),
            dim=1,
        )
        calf_error = torch.mean(
            torch.square(calf - torch.mean(calf, dim=1, keepdim=True)),
            dim=1,
        )
        error = hip_error + thigh_error + calf_error
        quality = 1.0 / (
            1.0 + error / self.cfg.rewards.stand_symmetry_sigma
        )
        return quality * self._stand_reward_weight()

    def _reward_stand_foot_symmetry_quality(self):
        foot_positions = self._foot_positions_body()
        front_lateral_sum = foot_positions[:, 0, 1] + foot_positions[:, 1, 1]
        rear_lateral_sum = foot_positions[:, 2, 1] + foot_positions[:, 3, 1]
        front_x_difference = foot_positions[:, 0, 0] - foot_positions[:, 1, 0]
        rear_x_difference = foot_positions[:, 2, 0] - foot_positions[:, 3, 0]
        error = 0.25 * (
            torch.square(front_lateral_sum)
            + torch.square(rear_lateral_sum)
            + torch.square(front_x_difference)
            + torch.square(rear_x_difference)
        )
        quality = 1.0 / (
            1.0 + error / self.cfg.rewards.stand_foot_symmetry_sigma
        )
        return quality * self._stand_reward_weight()

    def _reward_straight_yaw_rate(self):
        """Directly penalize the yaw-rate signal available to the Actor."""
        backward = torch.where(
            self.commands[:, 0] < -self.cfg.rewards.straight_min_speed,
            torch.full_like(
                self.base_ang_vel[:, 2],
                self.cfg.rewards.backward_yaw_stability_multiplier,
            ),
            torch.ones_like(self.base_ang_vel[:, 2]),
        )
        return (
            torch.square(self.base_ang_vel[:, 2])
            * self._straight_command_mask().float()
            * self._straight_start_multiplier()
            * backward
        )

    def _reward_tracking_lin_vel_active(self):
        """Track motion without letting the generic term reward standing.

        The scalar scale preserves the old total weight for lateral motion;
        the multiplier raises only pure forward/backward tracking from 2 to 3.
        """
        active_translation = (
            torch.norm(self.commands[:, :2], dim=1) > 0.2
        ).float()
        sagittal_multiplier = torch.where(
            self._sagittal_command_mask(),
            torch.full_like(
                active_translation,
                self.cfg.rewards.sagittal_tracking_multiplier,
            ),
            torch.ones_like(active_translation),
        )
        return (
            self._reward_tracking_lin_vel()
            * active_translation
            * sagittal_multiplier
        )

    def _reward_sagittal_speed_shortfall(self):
        """Penalize under-speed, but do not punish harmless overshoot."""
        direction = torch.sign(self.commands[:, 0])
        velocity_along_command = direction * self.base_lin_vel[:, 0]
        shortfall = torch.clamp(
            torch.abs(self.commands[:, 0]) - velocity_along_command,
            min=0.0,
        )
        return (
            torch.square(shortfall)
            * self._sagittal_command_mask().float()
        )

    def _reward_straight_direction_quality(self):
        direction_quality = torch.exp(
            -torch.square(self.base_lin_vel[:, 1])
            / self.cfg.rewards.straight_lateral_velocity_sigma
            -torch.square(self.base_ang_vel[:, 2])
            / self.cfg.rewards.straight_direction_yaw_sigma
        )
        speed_quality = torch.exp(
            -torch.square(
                self.commands[:, 0] - self.base_lin_vel[:, 0]
            ) / self.cfg.rewards.straight_direction_speed_sigma
        )
        return (
            direction_quality
            * speed_quality
            * self._straight_command_mask().float()
            * self._straight_start_multiplier()
        )

    def _reward_turn_swing_clearance_quality(self):
        quality = self._phase_swing_clearance_quality(
            self.cfg.rewards.turn_swing_height_target,
            self.cfg.rewards.turn_swing_height_sigma,
        )
        return quality * self._turn_command_mask().float()

    def _reward_sagittal_contact_duty_asymmetry(self):
        """Discourage the measured unequal stance duty between the four feet."""
        mean_duty = torch.mean(self.contact_duty_ema, dim=1, keepdim=True)
        variance = torch.mean(
            torch.square(self.contact_duty_ema - mean_duty), dim=1
        )
        return variance * self._sagittal_command_mask().float()

    def _reward_sagittal_slip_asymmetry(self):
        """Suppress one-sided stance slip without penalizing useful swing."""
        contacts = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        ).float()
        planar_velocity = self.rigid_body_states[:, self.feet_indices, 7:9]
        foot_slip = torch.sum(torch.square(planar_velocity), dim=2) * contacts
        left = 0.5 * (foot_slip[:, 0] + foot_slip[:, 2])
        right = 0.5 * (foot_slip[:, 1] + foot_slip[:, 3])
        return (
            torch.square(left - right)
            * self._sagittal_command_mask().float()
        )

class Csdog1Flat282Mixin:
    """282-D HIM/WTW behavior layered over the shared 47-D behavior."""

    def _init_buffers(self):
        super()._init_buffers()
        self.observation_history = torch.zeros(
            self.num_envs,
            HISTORY_LENGTH,
            FRAME_DIM,
            dtype=torch.float,
            device=self.device,
        )
        self.clean_observation_history = torch.zeros_like(
            self.observation_history
        )
        self.history_needs_reset = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.last_last_actions = torch.zeros_like(self.actions)
        self.stop_heading_reference = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.translation_heading_reference = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.translation_heading_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        if hasattr(self, "friction_coeffs"):
            self.critic_friction = self.friction_coeffs.to(
                device=self.device, dtype=torch.float
            ).view(self.num_envs, 1)
        else:
            self.critic_friction = torch.ones(
                self.num_envs, 1, dtype=torch.float, device=self.device
            )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if hasattr(self, "history_needs_reset"):
            self.history_needs_reset[env_ids] = True
            self.last_last_actions[env_ids] = 0.0
            self.stop_heading_reference[env_ids] = self.rpy[env_ids, 2]
            targets = self._command_mode_targets()[env_ids]
            pure_translation = (
                (torch.norm(targets[:, :2], dim=1)
                 > self.cfg.commands.heading_hold_translation_deadband)
                & (torch.abs(targets[:, 2])
                   < self.cfg.commands.heading_hold_manual_yaw_deadband)
            )
            self.translation_heading_active[env_ids] = pure_translation
            no_translation = env_ids[~pure_translation]
            self.translation_heading_reference[no_translation] = self.rpy[
                no_translation, 2
            ]

    def _resample_commands(self, env_ids):
        """Give exact keyboard-speed turns enough samples from scratch."""
        super()._resample_commands(env_ids)
        if len(env_ids) == 0:
            return
        if getattr(self.cfg.commands, "use_structured_commands", False):
            targets = self.command_targets[env_ids]
            pure_turn = (
                (torch.abs(targets[:, 2]) > 0.2)
                & (torch.norm(targets[:, :2], dim=1) < 0.05)
            )
            high_speed = pure_turn & (
                torch.rand(len(env_ids), device=self.device)
                < self.cfg.commands.high_speed_turn_ratio
            )
            count = int(high_speed.sum().item())
            if count > 0:
                selected_ids = env_ids[high_speed]
                direction = torch.sign(targets[high_speed, 2])
                magnitude = torch_rand_float(
                    self.cfg.commands.high_speed_turn_min,
                    self.cfg.commands.high_speed_turn_max,
                    (count, 1),
                    device=self.device,
                ).squeeze(1)
                self.command_targets[selected_ids, 2] = direction * magnitude
                immediate = selected_ids[
                    ~self.command_ramp_active[selected_ids]
                ]
                self.commands[immediate, 2] = self.command_targets[
                    immediate, 2
                ]

            # Pure lateral commands need their own high-speed tail; otherwise
            # increasing only the configured range leaves too few near-limit
            # samples for reliable 1.5-m/s tracking.
            targets = self.command_targets[env_ids]
            pure_lateral = (
                (torch.abs(targets[:, 1]) > 0.2)
                & (torch.abs(targets[:, 0]) < 0.05)
                & (torch.abs(targets[:, 2]) < 0.05)
            )
            high_speed_lateral = pure_lateral & (
                torch.rand(len(env_ids), device=self.device)
                < self.cfg.commands.high_speed_lateral_ratio
            )
            lateral_count = int(high_speed_lateral.sum().item())
            if lateral_count > 0:
                lateral_ids = env_ids[high_speed_lateral]
                direction = torch.sign(targets[high_speed_lateral, 1])
                magnitude = torch_rand_float(
                    self.cfg.commands.high_speed_lateral_min,
                    self.cfg.commands.high_speed_lateral_max,
                    (lateral_count, 1),
                    device=self.device,
                ).squeeze(1)
                self.command_targets[lateral_ids, 1] = direction * magnitude
                immediate = lateral_ids[
                    ~self.command_ramp_active[lateral_ids]
                ]
                self.commands[immediate, 1] = self.command_targets[
                    immediate, 1
                ]

            # The three axes may each reach 1.5 in a pure primitive, but a
            # mixed sample must not request sqrt(1.5^2 + 1.5^2) translation
            # together with a full-rate turn.
            targets = self.command_targets[env_ids]
            mixed = torch.sum(torch.abs(targets) > 1.0e-6, dim=1) >= 2
            mixed_ids = env_ids[mixed]
            if len(mixed_ids) > 0:
                translation = self.command_targets[mixed_ids, :2]
                translation_norm = torch.norm(
                    translation, dim=1, keepdim=True
                ).clamp_min(1.0e-6)
                translation_scale = torch.clamp(
                    self.cfg.commands.mixed_translation_max
                    / translation_norm,
                    max=1.0,
                )
                self.command_targets[mixed_ids, :2] = (
                    translation * translation_scale
                )
                self.command_targets[mixed_ids, 2] = torch.clamp(
                    self.command_targets[mixed_ids, 2],
                    -self.cfg.commands.mixed_yaw_max,
                    self.cfg.commands.mixed_yaw_max,
                )
                immediate = mixed_ids[
                    ~self.command_ramp_active[mixed_ids]
                ]
                self.commands[immediate, :3] = self.command_targets[
                    immediate, :3
                ]

        if hasattr(self, "translation_heading_reference"):
            targets = self._command_mode_targets()[env_ids]
            pure_translation = (
                (torch.norm(targets[:, :2], dim=1)
                 > self.cfg.commands.heading_hold_translation_deadband)
                & (torch.abs(targets[:, 2])
                   < self.cfg.commands.heading_hold_manual_yaw_deadband)
            )
            self.translation_heading_active[env_ids] = pure_translation
            selected_ids = env_ids[pure_translation]
            self.translation_heading_reference[selected_ids] = self.rpy[
                selected_ids, 2
            ]

            # A reactive controller produces only small corrections after the
            # policy has already drifted.  Explicitly expose a subset of pure
            # lateral samples to the counter-yaw combinations needed at test
            # time.  Raw command targets remain pure lateral for reward masks.
            lateral = (
                pure_translation
                & (torch.abs(targets[:, 1]) > 0.2)
                & (torch.abs(targets[:, 0]) < 0.05)
            )
            perturb = lateral & (
                torch.rand(len(env_ids), device=self.device)
                < self.cfg.commands.lateral_heading_perturb_ratio
            )
            perturb_count = int(perturb.sum().item())
            if perturb_count > 0:
                perturb_ids = env_ids[perturb]
                magnitude = torch_rand_float(
                    self.cfg.commands.lateral_heading_correction_min,
                    self.cfg.commands.lateral_heading_correction_max,
                    (perturb_count, 1),
                    device=self.device,
                ).squeeze(1)
                oppose_lateral = (
                    torch.rand(perturb_count, device=self.device)
                    < self.cfg.commands.lateral_heading_opposite_ratio
                )
                random_sign = torch.where(
                    torch.rand(perturb_count, device=self.device) < 0.5,
                    -torch.ones(perturb_count, device=self.device),
                    torch.ones(perturb_count, device=self.device),
                )
                lateral_sign = torch.sign(targets[perturb, 1])
                correction_sign = torch.where(
                    oppose_lateral, -lateral_sign, random_sign
                )
                initial_correction = correction_sign * magnitude
                self.translation_heading_reference[perturb_ids] += (
                    initial_correction / self.cfg.commands.heading_hold_kp
                )

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._apply_translation_heading_hold()
        # The first adjustment frame is the pose at command release. Preserve
        # its yaw so the complete landing step is rewarded for keeping the
        # original heading instead of only ending with zero yaw rate.
        new_stops = self.stop_adjustment_active & (
            self.stop_adjustment_steps == 1
        )
        self._align_lateral_stop_phase(new_stops)
        self.stop_heading_reference[new_stops] = self.rpy[new_stops, 2]

    def _align_lateral_stop_phase(self, new_stops):
        """Resume the stop step from the diagonal that is actually loaded.

        Pure lateral motion deliberately has no desired diagonal contact
        sequence. At command release the common stop controller needs one,
        so select the phase whose stance pair best matches measured contacts.
        Four-foot contact and other ties keep the uninterrupted clock.
        """
        lateral_stop = new_stops & (
            torch.abs(self.stop_entry_direction[:, 1])
            > torch.abs(self.stop_entry_direction[:, 0])
        )
        contact = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.commands.stop_release_contact_force
        )
        support_a = contact[:, 0].int() + contact[:, 3].int()
        support_b = contact[:, 1].int() + contact[:, 2].int()
        align_a = lateral_stop & (support_a > support_b)
        align_b = lateral_stop & (support_b > support_a)

        # phase 0.25: FL+RR stance; phase 0.75: FR+RL stance.
        aligned_phase = self.gait_phase.clone()
        aligned_phase[align_a] = 0.25
        aligned_phase[align_b] = 0.75
        aligned = align_a | align_b
        elapsed_phase = (
            self.stop_adjustment_steps.float()
            * self.dt
            / self.cfg.commands.stop_adjustment_cycle_s
        )
        self.gait_phase[aligned] = aligned_phase[aligned]
        self.stop_adjustment_start_phase[aligned] = torch.remainder(
            aligned_phase[aligned] - elapsed_phase[aligned], 1.0
        )

    def _command_mode_targets(self):
        """Return raw operator targets, before the outer heading controller."""
        if (
            hasattr(self, "command_targets")
            and getattr(self.cfg.commands, "use_structured_commands", False)
        ):
            return self.command_targets
        return self.commands[:, :3]

    def _apply_translation_heading_hold(self):
        """Convert translation heading error to a small yaw-rate command.

        This matches a deployable outer loop: the actor still receives only
        phase, IMU, velocity command, joints and action history.  Manual yaw
        commands pass through unchanged and a zero command remains exactly
        zero, so the phase-guided stop state is not bypassed.
        """
        if (
            not self.cfg.commands.translation_heading_hold
            or self.cfg.commands.heading_command
        ):
            return
        targets = self._command_mode_targets()
        pure_translation = (
            (torch.norm(targets[:, :2], dim=1)
             > self.cfg.commands.heading_hold_translation_deadband)
            & (torch.abs(targets[:, 2])
               < self.cfg.commands.heading_hold_manual_yaw_deadband)
        )
        current_translation = (
            torch.norm(self.commands[:, :2], dim=1)
            > self.cfg.commands.heading_hold_translation_deadband
        )
        active = (
            pure_translation
            & self.translation_heading_active
            & current_translation
        )
        error = wrap_to_pi(
            self.translation_heading_reference - self.rpy[:, 2]
        )
        correction = torch.clamp(
            self.cfg.commands.heading_hold_kp * error,
            -self.cfg.commands.heading_hold_yaw_limit,
            self.cfg.commands.heading_hold_yaw_limit,
        )
        self.commands[active, 2] = correction[active]
        self.translation_heading_active[~pure_translation] = False

    def _straight_command_mask(self):
        targets = self._command_mode_targets()
        return (
            (torch.abs(targets[:, 0]) > self.cfg.rewards.straight_min_speed)
            & (torch.abs(targets[:, 1])
               < self.cfg.rewards.straight_max_lateral_command)
            & (torch.abs(targets[:, 2])
               < self.cfg.rewards.straight_max_yaw_command)
        )

    def _sagittal_command_mask(self):
        targets = self._command_mode_targets()
        return (
            (torch.abs(targets[:, 0]) > 0.2)
            & (torch.abs(targets[:, 1]) < 0.05)
            & (torch.abs(targets[:, 2]) < 0.05)
        )

    def _lateral_command_mask(self):
        targets = self._command_mode_targets()
        return (
            (torch.abs(targets[:, 1]) > 0.2)
            & (torch.abs(targets[:, 0]) < 0.05)
            & (torch.abs(targets[:, 2]) < 0.05)
        )

    def _turn_command_mask(self):
        targets = self._command_mode_targets()
        return (
            (torch.abs(targets[:, 2]) > 0.2)
            & (torch.norm(targets[:, :2], dim=1) < 0.05)
        )

    def _fixed_phase_motion_mask(self):
        """Modes for which the two-dimensional trot clock is authoritative."""
        return self._sagittal_command_mask() | self._turn_command_mask()

    def _stop_heading_error(self):
        return wrap_to_pi(self.rpy[:, 2] - self.stop_heading_reference)

    def _roll_history(self, actor_frame, clean_frame):
        reset_ids = self.history_needs_reset.nonzero(
            as_tuple=False
        ).flatten()
        if len(reset_ids) > 0:
            # Deployment must repeat the first valid frame too; zero padding
            # would create a reset transient absent from steady-state data.
            self.observation_history[reset_ids] = actor_frame[
                reset_ids
            ].unsqueeze(1).expand(-1, HISTORY_LENGTH, -1)
            self.clean_observation_history[reset_ids] = clean_frame[
                reset_ids
            ].unsqueeze(1).expand(-1, HISTORY_LENGTH, -1)
            self.history_needs_reset[reset_ids] = False

        keep = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        keep[reset_ids] = False
        if torch.any(keep):
            self.observation_history[keep, :-1] = self.observation_history[
                keep, 1:
            ].clone()
            self.observation_history[keep, -1] = actor_frame[keep]
            self.clean_observation_history[
                keep, :-1
            ] = self.clean_observation_history[keep, 1:].clone()
            self.clean_observation_history[keep, -1] = clean_frame[keep]

    def compute_observations(self):
        actor_frame = self._current_actor_frame()
        clean_frame = self.clean_actor_frame.clone()
        self._roll_history(actor_frame, clean_frame)
        self.obs_buf = self.observation_history.reshape(
            self.num_envs, HISTORY_DIM
        ).clone()
        self._write_flat_critic_observation()

    @staticmethod
    def _normalize_range(value, lower, upper):
        if upper <= lower:
            return torch.zeros_like(value)
        return 2.0 * (value - lower) / (upper - lower) - 1.0

    def _write_flat_critic_observation(self):
        clean_history = self.clean_observation_history.reshape(
            self.num_envs, HISTORY_DIM
        ).clone()
        foot_forces = torch.clamp(
            self.contact_forces[:, self.feet_indices, 2]
            / self.cfg.rewards.max_contact_force,
            0.0,
            2.0,
        )
        friction = self._normalize_range(
            self.critic_friction,
            self.cfg.domain_rand.friction_range[0],
            self.cfg.domain_rand.friction_range[1],
        )
        motor_strength = torch.mean(self.motor_strengths, dim=1, keepdim=True)
        motor_strength = self._normalize_range(
            motor_strength,
            self.cfg.domain_rand.motor_strength_range[0],
            self.cfg.domain_rand.motor_strength_range[1],
        )
        maximum_delay = max(1, self.cfg.domain_rand.control_delay_steps[1])
        delay = self.control_delay.float().unsqueeze(1) / maximum_delay
        stop_state = torch.stack(
            (
                self.stop_adjustment_active.float(),
                self.pose_recovery_blend,
            ),
            dim=1,
        )
        stop_heading_error = self._stop_heading_error().unsqueeze(1)
        self.privileged_obs_buf = torch.cat(
            (
                clean_history,
                self.base_lin_vel * self.obs_scales.lin_vel,
                foot_forces,
                friction,
                motor_strength,
                delay,
                stop_state,
                stop_heading_error,
                self.stop_entry_direction,
            ),
            dim=1,
        )
        if self.privileged_obs_buf.shape[1] != PRIVILEGED_DIM:
            raise RuntimeError(
                f"Critic observation is {self.privileged_obs_buf.shape[1]}D, "
                f"expected {PRIVILEGED_DIM}D"
            )

    def compute_reward(self):
        """Ji22: positive task reward multiplicatively shaped by costs."""
        positive = torch.zeros_like(self.rew_buf)
        negative = torch.zeros_like(self.rew_buf)
        for name, reward_function in zip(
            self.reward_names, self.reward_functions
        ):
            reward = reward_function() * self.reward_scales[name]
            if self.reward_scales[name] > 0.0:
                positive += reward
            else:
                negative += reward
            self.episode_sums[name] += reward

        exponent = torch.clamp(
            negative / self.cfg.rewards.sigma_rew_neg,
            min=-20.0,
            max=0.0,
        )
        self.rew_buf[:] = positive * torch.exp(exponent)
        if "termination" in self.reward_scales:
            reward = (
                self._reward_termination()
                * self.reward_scales["termination"]
            )
            self.rew_buf += reward
            self.episode_sums["termination"] += reward

    def _yaw_tracking_quality(self):
        error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2]
        )
        translating = torch.norm(self.commands[:, :2], dim=1) > 0.2
        sigma = torch.where(
            translating,
            torch.full_like(
                error,
                self.cfg.rewards.translation_yaw_cauchy_sigma,
            ),
            torch.full_like(
                error,
                self.cfg.rewards.tracking_ang_vel_cauchy_sigma,
            ),
        )
        return 1.0 / (
            1.0 + error / sigma
        )

    def _translation_progress_gate(self):
        """Return signed progress along the commanded planar direction."""
        command_xy = self.commands[:, :2]
        command_norm_squared = torch.sum(
            torch.square(command_xy), dim=1
        ).clamp_min(0.04)
        aligned_velocity = torch.sum(
            command_xy * self.base_lin_vel[:, :2], dim=1
        )
        return torch.clamp(
            aligned_velocity / command_norm_squared, min=0.0, max=1.0
        )

    def _turn_direction_progress(self):
        command = self.commands[:, 2]
        aligned_yaw_rate = torch.sign(command) * self.base_ang_vel[:, 2]
        return torch.clamp(
            aligned_yaw_rate / torch.abs(command).clamp_min(0.2),
            min=0.0,
            max=1.0,
        )

    def _reward_tracking_ang_vel_active(self):
        """Track yaw for every moving command, including a zero-yaw target.

        Baseline rewards yaw tracking independently of planar-speed tracking.
        Keep that full signal for every translation so an untrained lateral
        policy cannot lose the very reward that teaches it not to turn.  Only
        pure turns retain the signed progress gate that prevents standing from
        collecting turn reward.
        """
        pure_turn = self._turn_command_mask()
        progress = torch.where(
            pure_turn,
            self._turn_direction_progress(),
            torch.ones_like(self.commands[:, 2]),
        )
        return (
            self._yaw_tracking_quality()
            * progress
            * self._gait_active_mask().float()
        )

    def _reward_lateral_velocity_quality(self):
        """Supply a non-vanishing signed side-speed learning signal."""
        error = torch.square(
            self.commands[:, 1] - self.base_lin_vel[:, 1]
        )
        quality = 1.0 / (
            1.0
            + error / self.cfg.rewards.lateral_velocity_cauchy_sigma
        )
        return quality * self._lateral_command_mask().float()

    def _translation_yaw_error(self):
        """Yaw-rate error relative to the command visible to the actor."""
        return self.base_ang_vel[:, 2] - self.commands[:, 2]

    def _reward_lateral_yaw_rate(self):
        """Permit the nonzero counter-yaw requested by heading recovery."""
        return (
            torch.square(self._translation_yaw_error())
            * self._lateral_command_mask().float()
        )

    def _reward_straight_yaw_rate(self):
        """Use corrected yaw-rate tracking rather than an incompatible zero."""
        targets = self._command_mode_targets()
        backward = torch.where(
            targets[:, 0] < -self.cfg.rewards.straight_min_speed,
            torch.full_like(
                self.base_ang_vel[:, 2],
                self.cfg.rewards.backward_yaw_stability_multiplier,
            ),
            torch.ones_like(self.base_ang_vel[:, 2]),
        )
        return (
            torch.square(self._translation_yaw_error())
            * self._straight_command_mask().float()
            * self._straight_start_multiplier()
            * backward
        )

    def _reward_straight_yaw_stability(self):
        sigma = self.cfg.rewards.straight_yaw_rate_sigma
        stability = torch.exp(
            -torch.square(self._translation_yaw_error() / sigma)
        )
        targets = self._command_mode_targets()
        direction_multiplier = torch.where(
            targets[:, 0] < -self.cfg.rewards.straight_min_speed,
            torch.full_like(
                stability,
                self.cfg.rewards.backward_yaw_stability_multiplier,
            ),
            torch.ones_like(stability),
        )
        return (
            stability
            * self._straight_command_mask().float()
            * self._straight_start_multiplier()
            * direction_multiplier
        )

    def _reward_straight_direction_quality(self):
        direction_quality = torch.exp(
            -torch.square(self.base_lin_vel[:, 1])
            / self.cfg.rewards.straight_lateral_velocity_sigma
            -torch.square(self._translation_yaw_error())
            / self.cfg.rewards.straight_direction_yaw_sigma
        )
        speed_quality = torch.exp(
            -torch.square(
                self.commands[:, 0] - self.base_lin_vel[:, 0]
            ) / self.cfg.rewards.straight_direction_speed_sigma
        )
        return (
            direction_quality
            * speed_quality
            * self._straight_command_mask().float()
            * self._straight_start_multiplier()
        )

    def _reward_lateral_width_quality(self):
        """Reward a symmetric footprint only after useful side progress.

        A positive bounded quality avoids the Ji22 dead zone created by the
        earlier large negative width cost.  Progress gating prevents standing
        under a lateral command from collecting the posture reward.
        """
        feet = self._foot_positions_body()
        desired_contact = self._smooth_desired_contact_states()
        actual_contact = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        ).float()

        # A trot supports diagonal pairs, not front/rear pairs.  Evaluate only
        # pairs that should be in stance and are actually carrying load.  This
        # leaves airborne feet free to cross laterally during a valid side step.
        pair_indices = ((0, 3), (1, 2))
        weighted_quality = torch.zeros(self.num_envs, device=self.device)
        total_weight = torch.zeros_like(weighted_quality)
        for first, second in pair_indices:
            pair_weight = torch.minimum(
                desired_contact[:, first], desired_contact[:, second]
            ) * actual_contact[:, first] * actual_contact[:, second]
            width = torch.abs(
                feet[:, first, 1] - feet[:, second, 1]
            )
            lateral_center = 0.5 * (
                feet[:, first, 1] + feet[:, second, 1]
            )
            fore_aft_span = torch.abs(
                feet[:, first, 0] - feet[:, second, 0]
            )
            error = (
                torch.square(
                    width - self.cfg.rewards.lateral_width_quality_target
                )
                + torch.square(lateral_center)
                + self.cfg.rewards.lateral_support_length_weight
                * torch.square(
                    fore_aft_span
                    - self.cfg.rewards.lateral_support_length_target
                )
            )
            pair_quality = 1.0 / (
                1.0
                + error / self.cfg.rewards.lateral_width_quality_sigma
            )
            weighted_quality += pair_quality * pair_weight
            total_weight += pair_weight
        quality = weighted_quality / total_weight.clamp_min(0.1)
        quality *= (total_weight > 0.1).float()
        return (
            quality
            * self._translation_progress_gate()
            * self._lateral_command_mask().float()
        )

    def _reward_lateral_direction_quality(self):
        """Reward side motion while following the current correction target."""
        quality = torch.exp(
            -torch.square(self.base_lin_vel[:, 0])
            / self.cfg.rewards.lateral_cross_sigma
            -torch.square(self._translation_yaw_error())
            / self.cfg.rewards.lateral_yaw_sigma
        )
        return (
            quality
            * self._translation_progress_gate()
            * self._lateral_command_mask().float()
        )

    def _reward_lateral_contact_quality(self):
        """Include the weakest foot so a rear foot cannot be sacrificed."""
        desired_contact = self._smooth_desired_contact_states()
        actual_contact = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        ).float()
        agreement = (
            actual_contact * desired_contact
            + (1.0 - actual_contact) * (1.0 - desired_contact)
        )
        quality = 0.5 * (
            torch.mean(agreement, dim=1)
            + torch.min(agreement, dim=1).values
        )
        return (
            quality
            * self._translation_progress_gate()
            * self._lateral_command_mask().float()
        )

    def _reward_lateral_foot_slip_max(self):
        """Penalize the worst loaded foot instead of hiding it in a mean."""
        contact = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        ).float()
        planar_velocity = self.rigid_body_states[
            :, self.feet_indices, 7:9
        ]
        slip_squared = torch.sum(
            torch.square(planar_velocity), dim=2
        ) * contact
        return (
            torch.max(slip_squared, dim=1).values
            * self._translation_progress_gate()
            * self._lateral_command_mask().float()
        )

    def _reward_lateral_width_balance(self):
        """Normalize metre-squared footprint error before Ji22 shaping."""
        return (
            super()._reward_lateral_width_balance()
            / self.cfg.rewards.lateral_width_balance_sigma
        )

    def _reward_lateral_swing_height_asymmetry(self):
        """Match the two feet belonging to each diagonal swing pair."""
        desired_swing = 1.0 - self._smooth_desired_contact_states()
        foot_height = self.rigid_body_states[:, self.feet_indices, 2]
        pair_a_weight = torch.minimum(
            desired_swing[:, 0], desired_swing[:, 3]
        )
        pair_b_weight = torch.minimum(
            desired_swing[:, 1], desired_swing[:, 2]
        )
        pair_error = (
            torch.square(foot_height[:, 0] - foot_height[:, 3])
            * pair_a_weight
            + torch.square(foot_height[:, 1] - foot_height[:, 2])
            * pair_b_weight
        )
        swing_weight = pair_a_weight + pair_b_weight
        normalized_error = pair_error / swing_weight.clamp_min(0.1)
        normalized_error *= (swing_weight > 0.1).float()
        return (
            normalized_error
            / self.cfg.rewards.lateral_swing_asymmetry_sigma
            * self._lateral_command_mask().float()
        )

    def _turn_progress_gate(self):
        """Open positive turn-shaping only after turning the right way."""
        command = self.commands[:, 2]
        aligned_yaw_rate = torch.sign(command) * self.base_ang_vel[:, 2]
        full_gate_rate = (
            torch.abs(command).clamp_min(0.2)
            * self.cfg.rewards.turn_gate_full_ratio
        )
        progress = torch.clamp(
            aligned_yaw_rate / full_gate_rate, min=0.0, max=1.0
        )
        return (
            progress
            * self._yaw_tracking_quality()
            * self._turn_command_mask().float()
        )

    def _reward_turn_direction_quality(self):
        """Reward signed yaw progress; wrong-way rotation receives zero."""
        return (
            self._turn_direction_progress()
            * self._yaw_tracking_quality()
            * self._turn_command_mask().float()
        )

    def _smooth_desired_contact_states(self):
        phase = self._diagonal_leg_phase()
        duty = self.cfg.rewards.phase_stance_ratio
        # A periodic cosine threshold represents stance on [0, duty].
        score = torch.cos(2.0 * math.pi * (phase - 0.5 * duty))
        threshold = math.cos(math.pi * duty)
        return torch.sigmoid(
            (score - threshold) / self.cfg.rewards.gait_contact_smoothing
        )

    def _reward_tracking_contacts_shaped_force(self):
        desired_contact = self._smooth_desired_contact_states()
        foot_force = torch.norm(
            self.contact_forces[:, self.feet_indices, :], dim=2
        )
        swing_collision = (1.0 - desired_contact) * (
            1.0
            - torch.exp(
                -torch.square(foot_force)
                / self.cfg.rewards.gait_force_sigma
            )
        )
        return (
            torch.mean(swing_collision, dim=1)
            * self._fixed_phase_motion_mask().float()
        )

    def _reward_tracking_contacts_shaped_vel(self):
        desired_contact = self._smooth_desired_contact_states()
        foot_velocity = self.rigid_body_states[
            :, self.feet_indices, 7:10
        ]
        speed_squared = torch.sum(torch.square(foot_velocity), dim=2)
        stance_motion = desired_contact * (
            1.0
            - torch.exp(
                -speed_squared / self.cfg.rewards.gait_velocity_sigma
            )
        )
        return (
            torch.mean(stance_motion, dim=1)
            * self._fixed_phase_motion_mask().float()
        )

    def _reward_phase_swing_clearance(self):
        phase = self._diagonal_leg_phase()
        duty = self.cfg.rewards.phase_stance_ratio
        swing_phase = torch.clamp(
            (phase - duty) / (1.0 - duty), min=0.0, max=1.0
        )
        swing_profile = torch.sin(math.pi * swing_phase)
        target = (
            self.cfg.rewards.swing_foot_radius
            + self.cfg.rewards.swing_height_amplitude * swing_profile
        )
        turn = self._turn_command_mask().float().unsqueeze(1)
        target += turn * self.cfg.rewards.turn_height_bonus * swing_profile
        foot_height = self.rigid_body_states[:, self.feet_indices, 2]
        desired_swing = 1.0 - self._smooth_desired_contact_states()
        error = torch.square(foot_height - target) * desired_swing
        return (
            torch.sum(error, dim=1)
            / desired_swing.sum(dim=1).clamp_min(1.0)
            * self._fixed_phase_motion_mask().float()
        )

    def _reward_raibert_heuristic(self):
        feet_body = self._foot_positions_body()[:, :, :2]
        nominal = self.stop_foot_targets.unsqueeze(0)
        phase = self._diagonal_leg_phase()
        phase_offset = torch.abs(1.0 - 2.0 * phase) - 0.5
        frequency = 1.0 / self.cfg.control.gait_cycle_s

        command_xy = self.commands[:, :2].unsqueeze(1).expand(-1, 4, -1)
        yaw = self.commands[:, 2].view(-1, 1, 1)
        yaw_velocity = torch.stack(
            (-nominal[..., 1], nominal[..., 0]), dim=2
        ).expand(self.num_envs, -1, -1) * yaw
        desired_velocity = command_xy + yaw_velocity
        offset = phase_offset.unsqueeze(2) * desired_velocity / (
            2.0 * frequency
        )
        target = nominal + offset
        error = torch.mean(torch.square(feet_body - target), dim=(1, 2))
        return error * self._fixed_phase_motion_mask().float()

    def _reward_feet_clearance(self):
        """Reward lateral swing clearance using measured contact only.

        No desired swing pair is consulted, so the policy may discover the
        lateral footfall ordering that best preserves heading and balance.
        """
        foot_height = self.rigid_body_states[:, self.feet_indices, 2]
        contact = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.clearance_contact_threshold
        )
        in_air = ~contact
        height_quality = torch.exp(
            -torch.square(
                (
                    foot_height
                    - self.cfg.rewards.lateral_clearance_target
                )
                / self.cfg.rewards.lateral_clearance_sigma
            )
        )
        airborne_count = torch.sum(in_air, dim=1)
        quality = torch.sum(
            height_quality * in_air.float(), dim=1
        ) / airborne_count.clamp_min(1)
        quality *= (airborne_count > 0).float()
        return (
            quality
            * self._translation_progress_gate()
            * self._lateral_command_mask().float()
        )

    def _reward_feet_air_time(self):
        """Bounded lateral touchdown timing based on actual contacts."""
        contact = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        )
        contact_filt = torch.logical_or(contact, self.last_contacts)
        first_contact = (self.feet_air_time > 0.0) & contact_filt
        self.feet_air_time += self.dt

        touchdown_quality = torch.exp(
            -torch.square(
                (
                    self.feet_air_time
                    - self.cfg.rewards.lateral_air_time_target
                )
                / self.cfg.rewards.lateral_air_time_sigma
            )
        ) * first_contact.float()
        touchdown_count = torch.sum(first_contact, dim=1)
        quality = torch.sum(touchdown_quality, dim=1) / (
            touchdown_count.clamp_min(1)
        )
        quality *= (touchdown_count > 0).float()

        lateral = self._lateral_command_mask()
        self.feet_air_time *= ~contact_filt
        self.feet_air_time[~lateral] = 0.0
        self.last_contacts = contact
        return (
            quality
            * self._translation_progress_gate()
            * lateral.float()
        )

    def _reward_action_smoothness_1(self):
        return torch.sum(
            torch.square(self.actions - self.last_actions), dim=1
        )

    def _reward_action_smoothness_2(self):
        return torch.sum(
            torch.square(
                self.actions
                - 2.0 * self.last_actions
                + self.last_last_actions
            ),
            dim=1,
        )

    def _reward_feet_slip_wtw(self):
        contact = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        ).float()
        planar_velocity = self.rigid_body_states[
            :, self.feet_indices, 7:9
        ]
        return torch.sum(
            torch.sum(torch.square(planar_velocity), dim=2) * contact,
            dim=1,
        )

    def _reward_joint_power(self):
        return torch.sum(
            torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1
        )

    def _reward_stop_heading_quality(self):
        quality = torch.exp(
            -torch.square(self._stop_heading_error())
            / self.cfg.rewards.stop_heading_sigma
        )
        return quality * self._persistent_stop_mask().float()

    def _reward_stop_cross_velocity(self):
        """Reject the sideways kick that path speed alone cannot identify."""
        cross_direction = torch.stack(
            (-self.stop_entry_direction[:, 1],
             self.stop_entry_direction[:, 0]),
            dim=1,
        )
        cross_speed = torch.sum(
            self.base_lin_vel[:, :2] * cross_direction, dim=1
        )
        return (
            torch.square(cross_speed)
            * self._persistent_stop_mask().float()
        )

    def _reward_turn_stationary_quality(self):
        planar_speed_squared = torch.sum(
            torch.square(self.base_lin_vel[:, :2]), dim=1
        )
        quality = torch.exp(
            -planar_speed_squared / self.cfg.rewards.turn_translation_sigma
        )
        return quality * self._turn_progress_gate()

    def _reward_turn_level_quality(self):
        tilt = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        quality = torch.exp(-tilt / self.cfg.rewards.turn_level_sigma)
        return quality * self._turn_progress_gate()

    def _reward_turn_hip_quality(self):
        hip_error = torch.mean(
            torch.square(self.dof_pos[:, self.hip_indices]), dim=1
        )
        quality = 1.0 / (
            1.0 + hip_error / self.cfg.rewards.turn_hip_sigma
        )
        return quality * self._turn_progress_gate()

    def _reward_turn_foot_width_quality(self):
        feet = self._foot_positions_body()
        front_width = torch.abs(feet[:, 0, 1] - feet[:, 1, 1])
        rear_width = torch.abs(feet[:, 2, 1] - feet[:, 3, 1])
        width_error = 0.5 * (
            torch.square(front_width - self.cfg.rewards.turn_foot_width_target)
            + torch.square(rear_width - self.cfg.rewards.turn_foot_width_target)
        )
        quality = 1.0 / (
            1.0 + width_error / self.cfg.rewards.turn_foot_width_sigma
        )
        return quality * self._turn_progress_gate()

    def _reward_turn_contact_phase_quality(self):
        desired_contact = self._smooth_desired_contact_states()
        actual_contact = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        ).float()
        agreement = (
            actual_contact * desired_contact
            + (1.0 - actual_contact) * (1.0 - desired_contact)
        )
        # Include the weakest foot so sacrificing one rear leg cannot be
        # hidden by the mean contact score of the other three.
        quality = 0.5 * (
            torch.mean(agreement, dim=1)
            + torch.min(agreement, dim=1).values
        )
        return quality * self._turn_progress_gate()

    def _reward_turn_swing_clearance_quality(self):
        """Do not pay swing-form reward for standing under a turn command."""
        quality = self._phase_swing_clearance_quality(
            self.cfg.rewards.turn_swing_height_target,
            self.cfg.rewards.turn_swing_height_sigma,
        )
        return quality * self._turn_progress_gate()

class Csdog1Flat282DomainRandMixin:
    """Terrain and dynamics privileges used only during robust fine-tuning."""

    def _push_recovery_enabled(self):
        """Return whether disturbance-triggered stepping recovery is active."""
        return bool(getattr(
            self.cfg.commands, "enable_push_recovery", True
        ))

    def _capture_recovery_enabled(self):
        """Return whether learned, free-foot capture stepping is enabled."""
        return bool(getattr(
            self.cfg.commands, "enable_capture_recovery", False
        ))

    def _capture_recovery_runtime_enabled(self):
        """Keep capture CUDA work out of the rollout before its curriculum.

        This deliberately uses only Python scalars.  In particular, callers
        can use it before touching ``capture_recovery_active`` and therefore
        avoid launching/synchronizing recovery kernels during the model_1000
        warm-up period.
        """
        if not self._capture_recovery_enabled():
            return False
        iteration = int(getattr(self, "ppo_iteration", 0))
        start = int(self.cfg.commands.capture_recovery_start_iteration)
        return iteration > start

    def _init_buffers(self):
        super()._init_buffers()
        # Isaac Gym reports the post-contact foot velocity on the touchdown
        # frame.  Keep the previous frame so the reward can penalize the
        # actual approach speed instead of an already-clamped value.
        self.previous_foot_velocities = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            3,
            dtype=torch.float,
            device=self.device,
        )
        self.stop_touchdown_pre_velocity = torch.zeros_like(
            self.previous_foot_velocities
        )
        self.push_recovery_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.push_recovery_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.push_recovery_stable_steps = torch.zeros_like(
            self.push_recovery_steps
        )
        self.push_recovery_pose_blend = torch.ones(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.push_recovery_lifted = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
        )
        # Capture recovery deliberately has its own compact state.  It does
        # not reuse stop_adjustment_active because that controller prescribes
        # diagonal swing/stance pairs and fixed nominal footholds.
        self.capture_recovery_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.capture_recovery_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.capture_recovery_stable_steps = torch.zeros_like(
            self.capture_recovery_steps
        )
        self.capture_recovery_pose_blend = torch.ones(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.capture_recovery_previous_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
        )
        self.capture_recovery_lifted = torch.zeros_like(
            self.capture_recovery_previous_contacts
        )
        self.capture_recovery_landed = torch.zeros_like(
            self.capture_recovery_previous_contacts
        )
        self.capture_recovery_touchdown_event = torch.zeros_like(
            self.capture_recovery_previous_contacts
        )
        # Registered rewards are still called by the generic dispatcher.
        # Return this buffer before the capture curriculum instead of
        # allocating or deriving a new all-zero CUDA tensor every step.
        self.capture_recovery_reward_zero = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0 or not hasattr(
            self, "previous_foot_velocities"
        ):
            return
        self.previous_foot_velocities[env_ids] = 0.0
        self.stop_touchdown_pre_velocity[env_ids] = 0.0
        if self._push_recovery_enabled():
            self.push_recovery_active[env_ids] = False
            self.push_recovery_steps[env_ids] = 0
            self.push_recovery_stable_steps[env_ids] = 0
            self.push_recovery_pose_blend[env_ids] = 1.0
            self.push_recovery_lifted[env_ids] = False
        if self._capture_recovery_runtime_enabled():
            self.capture_recovery_active[env_ids] = False
            self.capture_recovery_steps[env_ids] = 0
            self.capture_recovery_stable_steps[env_ids] = 0
            self.capture_recovery_pose_blend[env_ids] = 1.0
            self.capture_recovery_previous_contacts[env_ids] = False
            self.capture_recovery_lifted[env_ids] = False
            self.capture_recovery_landed[env_ids] = False
            self.capture_recovery_touchdown_event[env_ids] = False

    def _capture_recovery_curriculum_scale(self):
        """Ramp only capture-step activation, never disturbance magnitude."""
        iteration = float(getattr(self, "ppo_iteration", 0))
        start = float(
            self.cfg.commands.capture_recovery_start_iteration
        )
        end = float(self.cfg.commands.capture_recovery_end_iteration)
        if end <= start:
            return 1.0 if iteration >= end else 0.0
        return min(max((iteration - start) / (end - start), 0.0), 1.0)

    def _start_capture_recovery(self, env_ids):
        """Start a free-foot capture window for settled standing robots."""
        if not self._capture_recovery_runtime_enabled() or len(env_ids) == 0:
            return

        probability = self._capture_recovery_curriculum_scale()
        if probability <= 0.0:
            return
        if probability < 1.0:
            selected = torch.rand(len(env_ids), device=self.device) < probability
            env_ids = env_ids[selected]
            if len(env_ids) == 0:
                return

        contacts = (
            self.contact_forces[env_ids][:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        )
        self.capture_recovery_active[env_ids] = True
        self.capture_recovery_steps[env_ids] = 0
        self.capture_recovery_stable_steps[env_ids] = 0
        self.capture_recovery_pose_blend[env_ids] = 0.0
        self.capture_recovery_previous_contacts[env_ids] = contacts
        self.capture_recovery_lifted[env_ids] = False
        self.capture_recovery_landed[env_ids] = False
        self.capture_recovery_touchdown_event[env_ids] = False

    def _capture_foothold_targets(self):
        """Return body-frame capture targets for [FL, FR, RL, RR].

        Translation moves the support region in the measured direction of
        motion/fall.  Yaw adds a tangential displacement around the nominal
        footprint.  These simulator-only targets shape training and do not
        add any Actor observation or deployment input.
        """
        cfg = self.cfg.rewards
        translation = (
            cfg.capture_velocity_gain * self.base_lin_vel[:, :2]
            - cfg.capture_tilt_gain * self.projected_gravity[:, :2]
        )
        translation[:, 0] = torch.clamp(
            translation[:, 0],
            -cfg.capture_max_offset_x,
            cfg.capture_max_offset_x,
        )
        translation[:, 1] = torch.clamp(
            translation[:, 1],
            -cfg.capture_max_offset_y,
            cfg.capture_max_offset_y,
        )

        nominal = self.stop_foot_targets.unsqueeze(0).expand(
            self.num_envs, -1, -1
        )
        tangent = torch.stack(
            (-nominal[:, :, 1], nominal[:, :, 0]), dim=2
        )
        yaw_offset = (
            cfg.capture_yaw_gain
            * self.base_ang_vel[:, 2].view(-1, 1, 1)
            * tangent
        )
        yaw_norm = torch.norm(yaw_offset, dim=2, keepdim=True).clamp_min(
            1.0e-6
        )
        yaw_offset = yaw_offset * torch.clamp(
            cfg.capture_max_yaw_offset / yaw_norm, max=1.0
        )
        return nominal + translation.unsqueeze(1) + yaw_offset

    def _update_capture_recovery(self):
        """Track learned touchdowns, settling and smooth pose restoration."""
        # The gate must precede every recovery-tensor operation.  This is the
        # important A/B change for the legacy Isaac Gym GPU pipeline.
        if not self._capture_recovery_runtime_enabled():
            return

        stand = self._stand_command_mask()
        commanded = self.capture_recovery_active & (~stand)
        if torch.any(commanded):
            self.capture_recovery_active[commanded] = False
            self.capture_recovery_pose_blend[commanded] = 1.0
            self.capture_recovery_steps[commanded] = 0
            self.capture_recovery_stable_steps[commanded] = 0
            self.gait_phase[commanded] = 0.0

        active = self.capture_recovery_active & stand
        self.capture_recovery_touchdown_event[:] = False
        if torch.any(active):
            self.capture_recovery_steps[active] += 1
            contacts = (
                self.contact_forces[:, self.feet_indices, 2]
                > self.cfg.rewards.phase_contact_threshold
            )
            active_feet = active.unsqueeze(1)
            self.capture_recovery_lifted |= active_feet & (~contacts)
            touchdown = (
                active_feet
                & self.capture_recovery_lifted
                & contacts
                & (~self.capture_recovery_previous_contacts)
            )
            self.capture_recovery_touchdown_event[:] = touchdown
            self.capture_recovery_landed |= touchdown
            self.capture_recovery_previous_contacts[active] = contacts[active]

            contact_count = torch.sum(contacts, dim=1)
            stable = (
                active
                & (
                    torch.norm(self.base_lin_vel[:, :2], dim=1)
                    < self.cfg.commands.capture_recovery_settle_speed
                )
                & (
                    torch.norm(self.base_ang_vel, dim=1)
                    < self.cfg.commands.capture_recovery_settle_ang_speed
                )
                & (
                    torch.norm(self.projected_gravity[:, :2], dim=1)
                    < self.cfg.commands.capture_recovery_settle_tilt
                )
                & (
                    contact_count
                    >= self.cfg.commands.capture_recovery_settle_contacts
                )
            )
            self.capture_recovery_stable_steps[stable] += 1
            self.capture_recovery_stable_steps[active & (~stable)] = 0

            minimum_steps = max(
                1,
                int(round(
                    self.cfg.commands.capture_recovery_min_duration_s
                    / self.dt
                )),
            )
            stable_steps = max(
                1,
                int(round(
                    self.cfg.commands.capture_recovery_stable_duration_s
                    / self.dt
                )),
            )
            maximum_steps = max(
                minimum_steps,
                int(round(
                    self.cfg.commands.capture_recovery_max_duration_s
                    / self.dt
                )),
            )
            made_capture_step = torch.any(
                self.capture_recovery_landed, dim=1
            )
            recovered = (
                active
                & (self.capture_recovery_steps >= minimum_steps)
                & (self.capture_recovery_stable_steps >= stable_steps)
                & made_capture_step
            )
            timed_out = active & (
                self.capture_recovery_steps >= maximum_steps
            )
            completed = recovered | timed_out

            ongoing = active & (~completed)
            # Preserve the familiar stationary phase [sin, cos] = [0, 1].
            # Recovery rewards never call _desired_stance_mask(), so the
            # policy remains free to choose a foot from proprioceptive state.
            self.gait_phase[ongoing] = 0.0

            if torch.any(completed):
                self.capture_recovery_active[completed] = False
                self.capture_recovery_steps[completed] = 0
                self.capture_recovery_stable_steps[completed] = 0
                self.capture_recovery_pose_blend[completed] = 0.0
                self.gait_phase[completed] = 0.0

        blending = (
            (~self.capture_recovery_active)
            & stand
            & (self.capture_recovery_pose_blend < 1.0)
        )
        if torch.any(blending):
            duration = max(
                self.cfg.commands.capture_recovery_pose_blend_duration_s,
                self.dt,
            )
            self.capture_recovery_pose_blend[blending] = torch.clamp(
                self.capture_recovery_pose_blend[blending]
                + self.dt / duration,
                min=0.0,
                max=1.0,
            )

    def _start_push_recovery(self, env_ids):
        """Open a temporary stepping window after a large settled-stand push."""
        if not self._push_recovery_enabled() or len(env_ids) == 0:
            return
        self.push_recovery_active[env_ids] = True
        self.push_recovery_steps[env_ids] = 0
        self.push_recovery_stable_steps[env_ids] = 0
        self.push_recovery_pose_blend[env_ids] = 0.0
        self.push_recovery_lifted[env_ids] = False

    def _update_push_recovery(self):
        """Finish recovery after sustained stability or a bounded timeout."""
        if not self._push_recovery_enabled():
            # Keep the ablation exact even if this switch is changed after an
            # environment has already entered recovery.
            self.push_recovery_active[:] = False
            self.push_recovery_steps[:] = 0
            self.push_recovery_stable_steps[:] = 0
            self.push_recovery_pose_blend[:] = 1.0
            self.push_recovery_lifted[:] = False
            return

        active = self.push_recovery_active
        if not torch.any(active):
            # A motion command owns posture shaping. Do not carry a stale
            # post-recovery blend into its later stop transition.
            moving = ~self._stand_command_mask()
            self.push_recovery_pose_blend[moving] = 1.0
            return

        self.push_recovery_steps[active] += 1
        contacts = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        )
        self.push_recovery_lifted |= active.unsqueeze(1) & (~contacts)

        stand = self._stand_command_mask()
        command_interrupted = active & (~stand)
        contact_count = torch.sum(contacts, dim=1)
        stable = (
            active
            & stand
            & (contact_count >= self.cfg.commands.push_recovery_min_contacts)
            & (
                torch.norm(self.base_lin_vel[:, :2], dim=1)
                < self.cfg.commands.push_recovery_settle_speed
            )
            & (
                torch.norm(self.base_ang_vel, dim=1)
                < self.cfg.commands.push_recovery_settle_ang_speed
            )
            & (
                torch.norm(self.projected_gravity[:, :2], dim=1)
                < self.cfg.commands.push_recovery_settle_tilt
            )
            & (
                torch.mean(torch.abs(self.dof_vel), dim=1)
                < self.cfg.commands.push_recovery_settle_joint_speed
            )
        )
        self.push_recovery_stable_steps[stable] += 1
        self.push_recovery_stable_steps[active & (~stable)] = 0

        minimum_steps = max(
            1,
            int(round(
                self.cfg.commands.push_recovery_min_duration_s / self.dt
            )),
        )
        stable_steps = max(
            1,
            int(round(
                self.cfg.commands.push_recovery_stable_duration_s / self.dt
            )),
        )
        maximum_steps = max(
            minimum_steps,
            int(round(
                self.cfg.commands.push_recovery_max_duration_s / self.dt
            )),
        )
        adjusted = torch.any(self.push_recovery_lifted, dim=1)
        recovered = (
            active
            & (self.push_recovery_steps >= minimum_steps)
            & (self.push_recovery_stable_steps >= stable_steps)
            & adjusted
        )
        timed_out = active & (self.push_recovery_steps >= maximum_steps)
        completed = recovered | timed_out | command_interrupted
        completed_stand = completed & stand
        self.push_recovery_active[completed] = False
        self.push_recovery_steps[completed] = 0
        self.push_recovery_stable_steps[completed] = 0
        # Restore the nominal pose gradually after recovery. A new locomotion
        # command bypasses this blend because it has its own gait objectives.
        self.push_recovery_pose_blend[completed_stand] = 0.0
        self.push_recovery_pose_blend[command_interrupted] = 1.0

        blending = (
            (~self.push_recovery_active)
            & self._stand_command_mask()
            & (self.push_recovery_pose_blend < 1.0)
        )
        blend_duration = max(
            self.cfg.commands.push_recovery_pose_blend_duration_s,
            self.dt,
        )
        self.push_recovery_pose_blend[blending] = torch.clamp(
            self.push_recovery_pose_blend[blending]
            + self.dt / blend_duration,
            min=0.0,
            max=1.0,
        )

    def _push_robots(self):
        """Apply one safely gated disturbance mode to each selected robot."""
        domain_rand = self.cfg.domain_rand
        if not hasattr(domain_rand, "advanced_push_start_iteration"):
            return super()._push_robots()
        capture_runtime = self._capture_recovery_runtime_enabled()

        env_ids = torch.arange(self.num_envs, device=self.device)
        interval = int(domain_rand.push_interval)
        # Do not immediately push an environment that has just been reset.
        push_env_ids = env_ids[
            (self.episode_length_buf[env_ids] > 0)
            & (self.episode_length_buf[env_ids] % interval == 0)
        ]
        if len(push_env_ids) == 0:
            return

        # A velocity impulse while the body is already falling or while the
        # phase controller is completing its final landing can create an
        # unrealistically deep contact in a single PhysX step.  Standing and
        # ordinary locomotion remain eligible, so recovery is still trained.
        root_state = self.root_states[push_env_ids]
        root_is_finite = torch.isfinite(root_state[:, :13]).all(dim=1)
        gravity_is_finite = torch.isfinite(
            self.projected_gravity[push_env_ids]
        ).all(dim=1)
        tilt = torch.norm(
            self.projected_gravity[push_env_ids, :2], dim=1
        )
        safe = (
            root_is_finite
            & gravity_is_finite
            & (tilt < float(domain_rand.push_max_safe_tilt))
            & (
                torch.abs(root_state[:, 9])
                < float(domain_rand.push_max_safe_vertical_speed)
            )
            & (
                torch.norm(root_state[:, 7:9], dim=1)
                < float(domain_rand.push_max_safe_planar_speed)
            )
            & (
                torch.norm(root_state[:, 10:13], dim=1)
                < float(domain_rand.push_max_safe_angular_speed)
            )
            & (
                root_state[:, 2]
                > float(domain_rand.push_min_safe_base_height)
            )
            & (
                root_state[:, 2]
                < float(domain_rand.push_max_safe_base_height)
            )
        )
        # Diagnostic A/B switch: the known-good pre-recovery implementation
        # did not condition a velocity impulse on an already-active foot
        # contact manifold.  Keep the newer gate available, but disable it for
        # this run while leaving every disturbance value and mode unchanged.
        if getattr(domain_rand, "push_require_contact_gate", True):
            contacts = (
                self.contact_forces[push_env_ids][
                    :, self.feet_indices, 2
                ]
                > self.cfg.rewards.phase_contact_threshold
            )
            safe &= torch.sum(contacts, dim=1) >= int(
                domain_rand.push_min_safe_contacts
            )
        if hasattr(self, "stop_adjustment_active"):
            safe &= ~self.stop_adjustment_active[push_env_ids]
        if hasattr(self, "push_recovery_active"):
            safe &= ~self.push_recovery_active[push_env_ids]
        if capture_runtime and hasattr(self, "capture_recovery_active"):
            safe &= ~self.capture_recovery_active[push_env_ids]
        push_env_ids = push_env_ids[safe]
        if len(push_env_ids) == 0:
            return

        iteration = float(getattr(self, "ppo_iteration", 0))
        planar_start = float(domain_rand.planar_push_start_iteration)
        planar_end = float(domain_rand.planar_push_end_iteration)
        if planar_end <= planar_start:
            planar_scale = 1.0 if iteration >= planar_end else 0.0
        else:
            planar_scale = min(max(
                (iteration - planar_start) / (planar_end - planar_start),
                0.0,
            ), 1.0)
        angular_start = float(domain_rand.advanced_push_start_iteration)
        angular_end = float(domain_rand.advanced_push_end_iteration)
        if angular_end <= angular_start:
            angular_scale = 1.0 if iteration >= angular_end else 0.0
        else:
            angular_scale = min(max(
                (iteration - angular_start) / (angular_end - angular_start),
                0.0,
            ), 1.0)
        self.planar_push_curriculum_scale = planar_scale
        self.advanced_push_curriculum_scale = angular_scale

        count = len(push_env_ids)

        # At angular_scale=0 every event is planar. The angular-mode
        # probability grows continuously and reaches the configured 50/30/20
        # planar/roll-pitch/yaw split at angular_scale=1. A robot never
        # receives a linear and angular impulse in the same simulation frame.
        roll_pitch_probability = (
            angular_scale * float(domain_rand.roll_pitch_push_probability)
        )
        yaw_probability = (
            angular_scale * float(domain_rand.yaw_push_probability)
        )
        mode_sample = torch.rand(count, device=self.device)
        roll_pitch_mask = mode_sample < roll_pitch_probability
        yaw_mask = (
            (mode_sample >= roll_pitch_probability)
            & (mode_sample < roll_pitch_probability + yaw_probability)
        )
        planar_mask = ~(roll_pitch_mask | yaw_mask)
        large_disturbance = None
        if capture_runtime:
            large_disturbance = torch.zeros(
                count, dtype=torch.bool, device=self.device
            )

        # Build the complete post-push velocity in a private tensor.  Nothing
        # below modifies Isaac Gym's shared root-state tensor until every
        # random sample, norm, threshold and index has already been computed.
        new_velocities = self.root_states[push_env_ids, 7:13].clone()

        planar_local_ids = torch.nonzero(
            planar_mask, as_tuple=False
        ).flatten()
        if len(planar_local_ids) > 0:
            max_linear = (
                float(domain_rand.planar_push_start_vel_xy)
                + planar_scale
                * (
                    float(domain_rand.max_push_vel_xy)
                    - float(domain_rand.planar_push_start_vel_xy)
                )
            )
            sampled_linear = torch_rand_float(
                -max_linear,
                max_linear,
                (len(planar_local_ids), 2),
                device=self.device,
            )
            # Interpret max_push_vel_xy as the total planar-vector magnitude,
            # not a separate limit on x and y (which would be sqrt(2) larger).
            sampled_norm = torch.norm(
                sampled_linear, dim=1, keepdim=True
            ).clamp_min(1.0e-6)
            sampled_linear *= torch.clamp(
                max_linear / sampled_norm, max=1.0
            )
            # Change only the sampled disturbance limit over the planar
            # course.  Do not add the robot's existing velocity: that hidden
            # semantic change could drive the submitted root speed close to
            # the 1.5-m/s post-push cap well before capture recovery starts.
            linear = sampled_linear
            linear_norm = torch.norm(
                linear, dim=1, keepdim=True
            ).clamp_min(1.0e-6)
            linear *= torch.clamp(
                float(domain_rand.max_post_push_vel_xy) / linear_norm,
                max=1.0,
            )
            new_velocities[planar_local_ids, 0:2] = linear
            if capture_runtime:
                large_disturbance[planar_local_ids] = (
                    torch.norm(sampled_linear, dim=1)
                    >= float(domain_rand.push_recovery_linear_threshold)
                )

        roll_pitch_local_ids = torch.nonzero(
            roll_pitch_mask, as_tuple=False
        ).flatten()
        if len(roll_pitch_local_ids) > 0:
            angular_value = angular_scale * torch_rand_float(
                -float(domain_rand.max_push_ang_vel_xy),
                float(domain_rand.max_push_ang_vel_xy),
                (len(roll_pitch_local_ids), 1),
                device=self.device,
            ).squeeze(1)
            angular_xy_delta = torch.zeros(
                len(roll_pitch_local_ids),
                2,
                dtype=self.root_states.dtype,
                device=self.device,
            )
            # A roll/pitch event chooses one axis. Sampling both at their
            # per-axis maximum created a hidden sqrt(2) angular disturbance.
            angular_axis = torch.randint(
                0, 2, (len(roll_pitch_local_ids),), device=self.device
            )
            angular_xy_delta[
                torch.arange(
                    len(roll_pitch_local_ids), device=self.device
                ),
                angular_axis,
            ] = angular_value
            angular_xy = (
                new_velocities[roll_pitch_local_ids, 3:5]
                + angular_xy_delta
            )
            new_velocities[roll_pitch_local_ids, 3:5] = torch.clamp(
                angular_xy,
                -float(domain_rand.max_post_push_ang_vel),
                float(domain_rand.max_post_push_ang_vel),
            )
            if capture_runtime:
                large_disturbance[roll_pitch_local_ids] = (
                    torch.norm(angular_xy_delta, dim=1)
                    >= float(domain_rand.push_recovery_roll_pitch_threshold)
                )

        yaw_local_ids = torch.nonzero(yaw_mask, as_tuple=False).flatten()
        if len(yaw_local_ids) > 0:
            yaw_delta = angular_scale * torch_rand_float(
                -float(domain_rand.max_push_yaw_vel),
                float(domain_rand.max_push_yaw_vel),
                (len(yaw_local_ids), 1),
                device=self.device,
            )
            yaw = new_velocities[yaw_local_ids, 5:6] + yaw_delta
            new_velocities[yaw_local_ids, 5:6] = torch.clamp(
                yaw,
                -float(domain_rand.max_post_push_ang_vel),
                float(domain_rand.max_post_push_ang_vel),
            )
            if capture_runtime:
                large_disturbance[yaw_local_ids] = (
                    torch.abs(yaw_delta.squeeze(1))
                    >= float(domain_rand.push_recovery_yaw_threshold)
                )

        # This is the only write to the PhysX-owned root tensor. Submit it
        # immediately: no thresholding, indexing or allocation is allowed
        # between this assignment and the Isaac Gym transaction.
        self.root_states[push_env_ids, 7:13] = new_velocities
        self._write_root_velocity(push_env_ids)

        # Arm recovery only after PhysX has received the velocity update.  It
        # first affects the policy on the next control step, never during the
        # root-state transaction above.
        if capture_runtime:
            settled_stand = (
                self._stand_command_mask()[push_env_ids]
                & self.stop_phase_settled[push_env_ids]
                & (~self.stop_adjustment_active[push_env_ids])
                & (self.pose_recovery_blend[push_env_ids] >= 0.999)
                & (self.capture_recovery_pose_blend[push_env_ids] >= 0.999)
            )
            self._start_capture_recovery(
                push_env_ids[large_disturbance & settled_stand]
            )

    def _post_physics_step_callback(self):
        # Preserve the pre-contact velocity before the shared callback updates
        # stop_adjustment_touchdown_event for the current policy step.
        self.stop_touchdown_pre_velocity[:] = (
            self.previous_foot_velocities
        )
        super()._post_physics_step_callback()
        self.previous_foot_velocities[:] = self.rigid_body_states[
            :, self.feet_indices, 7:10
        ]
        # Keep the failed legacy recovery path entirely off CUDA.  The new
        # capture path is independent and uses no extra PhysX state writes.
        if self._push_recovery_enabled():
            self._update_push_recovery()
        if self._capture_recovery_runtime_enabled():
            self._update_capture_recovery()

    def _stand_reward_weight(self):
        """Release four-foot pose locking during a large-push recovery."""
        inherited = super()._stand_reward_weight()
        if self._push_recovery_enabled():
            inherited = (
                inherited
                * (~self.push_recovery_active).float()
                * self.push_recovery_pose_blend
            )
        if self._capture_recovery_runtime_enabled():
            recovery_weight = torch.where(
                self.capture_recovery_active,
                torch.full_like(
                    self.capture_recovery_pose_blend,
                    self.cfg.commands.capture_recovery_stand_reward_floor,
                ),
                self.capture_recovery_pose_blend,
            )
            inherited = inherited * recovery_weight
        return inherited

    def _reward_push_recovery_upright_quality(self):
        tilt = torch.sum(
            torch.square(self.projected_gravity[:, :2]), dim=1
        )
        quality = 1.0 / (
            1.0 + tilt / self.cfg.rewards.push_recovery_tilt_sigma
        )
        return quality * self.push_recovery_active.float()

    def _reward_push_recovery_motion_quality(self):
        motion = (
            torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
            + self.cfg.rewards.push_recovery_angular_weight
            * torch.sum(torch.square(self.base_ang_vel), dim=1)
        )
        quality = 1.0 / (
            1.0 + motion / self.cfg.rewards.push_recovery_motion_sigma
        )
        return quality * self.push_recovery_active.float()

    def _reward_push_recovery_foot_adjustment(self):
        """Reward one/two-foot adjustment, while rejecting a four-foot hop."""
        lifted_count = torch.sum(self.push_recovery_lifted, dim=1)
        adjusted = (lifted_count >= 1) & (lifted_count <= 2)
        return adjusted.float() * self.push_recovery_active.float()

    def _reward_push_recovery_support_loss(self):
        contacts = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        )
        contact_count = torch.sum(contacts, dim=1)
        minimum_contacts = getattr(
            self.cfg.commands, "push_recovery_min_contacts", 2
        )
        deficit = torch.clamp(
            minimum_contacts - contact_count,
            min=0,
        ).float() / max(
            1, minimum_contacts
        )
        return deficit * self.push_recovery_active.float()

    def _capture_recovery_mask(self):
        return self.capture_recovery_active.float()

    def _reward_capture_recovery_quality(self):
        """Combined positive capture-step objective with a hard warm-up gate.

        Keeping these components in one function avoids dispatching six
        separate full-environment CUDA reward pipelines on every control
        step.  The coefficients preserve the previously configured relative
        reward strengths.
        """
        if not self._capture_recovery_runtime_enabled():
            return self.capture_recovery_reward_zero
        active = self.capture_recovery_active
        if not torch.any(active):
            return self.capture_recovery_reward_zero

        cfg = self.cfg.rewards
        active_float = active.float()
        tilt = torch.sum(
            torch.square(self.projected_gravity[:, :2]), dim=1
        )
        upright_quality = 1.0 / (1.0 + tilt / cfg.capture_tilt_sigma)
        motion = (
            torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
            + cfg.capture_angular_weight
            * torch.sum(torch.square(self.base_ang_vel), dim=1)
        )
        velocity_quality = 1.0 / (
            1.0 + motion / cfg.capture_motion_sigma
        )

        contacts_bool = (
            self.contact_forces[:, self.feet_indices, 2]
            > cfg.phase_contact_threshold
        )
        contacts = contacts_bool.float()
        contact_count = torch.sum(contacts, dim=1)
        support_quality = torch.clamp(
            contact_count / 3.0, min=0.0, max=1.0
        )

        feet_xy = self._foot_positions_body()[:, :, :2]
        targets = self._capture_foothold_targets()
        target_error = torch.sum(torch.square(feet_xy - targets), dim=2)
        touchdown_quality = torch.sum(
            (
                1.0
                / (1.0 + target_error / cfg.capture_foothold_sigma)
            )
            * self.capture_recovery_touchdown_event.float(),
            dim=1,
        )

        swing = (
            active.unsqueeze(1)
            & self.capture_recovery_lifted
            & (~self.capture_recovery_landed)
            & (~contacts_bool)
        )
        swing_float = swing.float()
        swing_count = torch.sum(swing_float, dim=1).clamp_min(1.0)
        swing_quality = (
            torch.sum(
                (
                    1.0
                    / (1.0 + target_error / cfg.capture_swing_sigma)
                )
                * swing_float,
                dim=1,
            )
            / swing_count
        )

        support_centroid = torch.sum(
            feet_xy * contacts.unsqueeze(2), dim=1
        ) / contact_count.unsqueeze(1).clamp_min(1.0)
        target_centroid = torch.mean(targets, dim=1)
        centroid_error = torch.sum(
            torch.square(support_centroid - target_centroid), dim=1
        )
        centroid_quality = (
            1.0 / (1.0 + centroid_error / cfg.capture_centroid_sigma)
        ) * (contact_count >= 2.0).float()

        return active_float * (
            upright_quality
            + 1.50 * velocity_quality
            + 0.50 * touchdown_quality
            + 0.25 * swing_quality
            + 0.75 * centroid_quality
            + 0.50 * support_quality
        )

    def _reward_capture_recovery_cost(self):
        """Combined support/slip/extra-step cost for active recovery only."""
        if not self._capture_recovery_runtime_enabled():
            return self.capture_recovery_reward_zero
        active = self.capture_recovery_active
        if not torch.any(active):
            return self.capture_recovery_reward_zero

        contacts = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        )
        contact_count = torch.sum(contacts, dim=1)
        support_loss = torch.clamp(
            self.cfg.commands.capture_recovery_min_support_contacts
            - contact_count,
            min=0,
        ).float()
        foot_velocity = self.rigid_body_states[
            :, self.feet_indices, 7:9
        ]
        slip = torch.sum(
            torch.square(foot_velocity) * contacts.unsqueeze(2).float(),
            dim=(1, 2),
        )
        lifted_count = torch.sum(self.capture_recovery_lifted, dim=1)
        excess_steps = torch.clamp(lifted_count - 2, min=0).float()
        return self._capture_recovery_mask() * (
            0.50 * support_loss
            + 0.05 * slip
            + 0.10 * excess_steps
        )

    def _reward_capture_recovery_upright_quality(self):
        tilt = torch.sum(
            torch.square(self.projected_gravity[:, :2]), dim=1
        )
        quality = 1.0 / (
            1.0 + tilt / self.cfg.rewards.capture_tilt_sigma
        )
        return quality * self._capture_recovery_mask()

    def _reward_capture_recovery_velocity_quality(self):
        motion = (
            torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
            + self.cfg.rewards.capture_angular_weight
            * torch.sum(torch.square(self.base_ang_vel), dim=1)
        )
        quality = 1.0 / (
            1.0 + motion / self.cfg.rewards.capture_motion_sigma
        )
        return quality * self._capture_recovery_mask()

    def _reward_capture_touchdown_quality(self):
        """Reward only a learned lift->land event at its capture target."""
        feet_xy = self._foot_positions_body()[:, :, :2]
        targets = self._capture_foothold_targets()
        error = torch.sum(torch.square(feet_xy - targets), dim=2)
        quality = 1.0 / (
            1.0 + error / self.cfg.rewards.capture_foothold_sigma
        )
        touchdown = self.capture_recovery_touchdown_event.float()
        return torch.sum(quality * touchdown, dim=1)

    def _reward_capture_swing_target_quality(self):
        """Provide dense target guidance only after a foot has actually lifted."""
        contacts = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        )
        swing = (
            self.capture_recovery_active.unsqueeze(1)
            & self.capture_recovery_lifted
            & (~self.capture_recovery_landed)
            & (~contacts)
        )
        feet_xy = self._foot_positions_body()[:, :, :2]
        targets = self._capture_foothold_targets()
        error = torch.sum(torch.square(feet_xy - targets), dim=2)
        quality = 1.0 / (
            1.0 + error / self.cfg.rewards.capture_swing_sigma
        )
        swing_count = torch.sum(swing.float(), dim=1).clamp_min(1.0)
        return (
            torch.sum(quality * swing.float(), dim=1)
            / swing_count
            * self._capture_recovery_mask()
        )

    def _reward_capture_support_centroid_quality(self):
        """Move the measured support region toward the capture point."""
        contacts = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        ).float()
        contact_count = torch.sum(contacts, dim=1, keepdim=True)
        feet_xy = self._foot_positions_body()[:, :, :2]
        support_centroid = torch.sum(
            feet_xy * contacts.unsqueeze(2), dim=1
        ) / contact_count.clamp_min(1.0)
        target_centroid = torch.mean(
            self._capture_foothold_targets(), dim=1
        )
        error = torch.sum(
            torch.square(support_centroid - target_centroid), dim=1
        )
        quality = 1.0 / (
            1.0 + error / self.cfg.rewards.capture_centroid_sigma
        )
        valid_support = contact_count.squeeze(1) >= 2.0
        return (
            quality
            * valid_support.float()
            * self._capture_recovery_mask()
        )

    def _reward_capture_support_quality(self):
        contacts = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        )
        # Three contacts already constitute full support during a one-foot
        # capture step; this does not reward keeping every foot locked down.
        quality = torch.clamp(
            torch.sum(contacts.float(), dim=1) / 3.0,
            min=0.0,
            max=1.0,
        )
        return quality * self._capture_recovery_mask()

    def _reward_capture_support_loss(self):
        contacts = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        )
        deficit = torch.clamp(
            self.cfg.commands.capture_recovery_min_support_contacts
            - torch.sum(contacts, dim=1),
            min=0,
        ).float()
        return deficit * self._capture_recovery_mask()

    def _reward_capture_foot_slip(self):
        contacts = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.phase_contact_threshold
        )
        foot_velocity = self.rigid_body_states[
            :, self.feet_indices, 7:9
        ]
        slip = torch.sum(
            torch.square(foot_velocity)
            * contacts.unsqueeze(2).float(),
            dim=(1, 2),
        )
        return slip * self._capture_recovery_mask()

    def _reward_capture_excess_steps(self):
        lifted_count = torch.sum(
            self.capture_recovery_lifted, dim=1
        )
        excess = torch.clamp(lifted_count - 2, min=0).float()
        return excess * self._capture_recovery_mask()

    def _reward_capture_no_step(self):
        """Discourage passive four-foot bracing while still dynamically unsafe."""
        grace_steps = max(
            1,
            int(round(
                self.cfg.commands.capture_recovery_step_grace_s / self.dt
            )),
        )
        unstable = (
            (
                torch.norm(self.base_lin_vel[:, :2], dim=1)
                > self.cfg.commands.capture_recovery_no_step_speed
            )
            | (
                torch.norm(self.base_ang_vel, dim=1)
                > self.cfg.commands.capture_recovery_no_step_ang_speed
            )
            | (
                torch.norm(self.projected_gravity[:, :2], dim=1)
                > self.cfg.commands.capture_recovery_no_step_tilt
            )
        )
        no_lift = ~torch.any(self.capture_recovery_lifted, dim=1)
        late = self.capture_recovery_steps >= grace_steps
        return (
            unstable & no_lift & late & self.capture_recovery_active
        ).float()

    def _stop_landing_weight(self):
        """Apply a little more pressure to the measured turn-stop failure."""
        pure_turn_entry = (
            (torch.abs(self.last_motion_command[:, 2]) > 0.05)
            & (torch.norm(self.last_motion_command[:, :2], dim=1) < 0.20)
        )
        multiplier = self.cfg.rewards.turn_stop_landing_multiplier
        return 1.0 + (multiplier - 1.0) * pure_turn_entry.float()

    def _reward_stop_touchdown_impact(self):
        """Penalize force above a soft limit only on a new stop touchdown."""
        touchdown = self.stop_adjustment_touchdown_event.float()
        force = torch.norm(
            self.contact_forces[:, self.feet_indices, :], dim=2
        )
        soft_limit = self.cfg.rewards.stop_touchdown_force_soft_limit
        normalized_excess = torch.clamp(
            force / soft_limit - 1.0,
            min=0.0,
            max=self.cfg.rewards.stop_touchdown_force_max_excess,
        )
        cost = torch.sum(
            torch.square(normalized_excess) * touchdown, dim=1
        )
        return cost * self._stop_landing_weight()

    def _reward_stop_touchdown_velocity(self):
        """Penalize downward approach speed, not ordinary swing velocity."""
        touchdown = self.stop_adjustment_touchdown_event.float()
        downward_speed = torch.clamp(
            -self.stop_touchdown_pre_velocity[:, :, 2]
            - self.cfg.rewards.stop_touchdown_vertical_speed_soft_limit,
            min=0.0,
        )
        scale = self.cfg.rewards.stop_touchdown_vertical_speed_soft_limit
        cost = torch.sum(
            torch.square(downward_speed / scale) * touchdown, dim=1
        )
        return cost * self._stop_landing_weight()

    def _reward_stop_calf_velocity_excess(self):
        """Reject fast terminal extension without suppressing normal lift."""
        # All four calf joints extend in the positive joint direction.  Gate
        # the cost to feet that have lifted and are still waiting to land (or
        # touch down on this frame), otherwise an absolute-speed cost also
        # discourages the intended recovery swing.
        landing_feet = (
            self.stop_adjustment_lifted
            & (~self.stop_adjustment_landed)
        ) | self.stop_adjustment_touchdown_event
        calf_speed = torch.clamp(
            self.dof_vel[:, self.calf_indices], min=0.0
        )
        soft_limit = self.cfg.rewards.stop_calf_velocity_soft_limit
        normalized_excess = torch.clamp(
            calf_speed / soft_limit - 1.0,
            min=0.0,
            max=self.cfg.rewards.stop_calf_velocity_max_excess,
        )
        # Sum per-foot costs instead of averaging them.  Averaging diluted the
        # isolated RR event measured on hardware by a factor of four.
        cost = torch.sum(
            torch.square(normalized_excess) * landing_feet.float(),
            dim=1,
        )
        return (
            cost
            * self.stop_adjustment_active.float()
            * self._stop_landing_weight()
        )

    def _reward_stop_calf_extension_gap(self):
        """Limit aggressive PD extension targets before the final landing."""
        landing_feet = (
            self.stop_adjustment_lifted
            & (~self.stop_adjustment_landed)
        ) | self.stop_adjustment_touchdown_event
        calf_target = (
            self.default_dof_pos[:, self.calf_indices]
            + self.cfg.control.action_scale
            * self.actions[:, self.calf_indices]
        )
        # Hardware logs showed target-current gaps of 0.30--0.55 rad during
        # the final FL/RR landing.  Only positive gaps extend the calf toward
        # the ground; knee flexion used to lift the foot remains unpenalized.
        soft_limit = self.cfg.rewards.stop_calf_extension_gap_soft_limit
        normalized_excess = torch.clamp(
            (calf_target - self.dof_pos[:, self.calf_indices])
            / soft_limit
            - 1.0,
            min=0.0,
            max=self.cfg.rewards.stop_calf_extension_gap_max_excess,
        )
        cost = torch.sum(
            torch.square(normalized_excess) * landing_feet.float(),
            dim=1,
        )
        return (
            cost
            * self.stop_adjustment_active.float()
            * self._stop_landing_weight()
        )

    def _resample_commands(self, env_ids):
        """Replace the generic mixed tail with two deployable combinations.

        The inherited structured sampler selects the final 20% of segments as
        mixed motion.  It normally randomizes vx, vy and yaw simultaneously;
        here that tail is made explicit: 75% sagittal+yaw and 25%
        sagittal+lateral, corresponding to 15% and 5% of all segments.
        """
        super()._resample_commands(env_ids)
        if (
            len(env_ids) == 0
            or not getattr(
                self.cfg.commands, "use_structured_commands", False
            )
        ):
            return

        targets = self.command_targets[env_ids, :3]
        # The inherited mixed branch samples all three continuous axes. Pure
        # modes (including heading-hold perturbations) retain one raw target.
        mixed = torch.sum(torch.abs(targets) > 1.0e-6, dim=1) >= 2
        mixed_ids = env_ids[mixed]
        count = len(mixed_ids)
        if count == 0:
            return

        def signed_uniform(lower, upper, negative_probability=0.5):
            magnitude = lower + (upper - lower) * torch.rand(
                count, device=self.device
            )
            negative = (
                torch.rand(count, device=self.device)
                < negative_probability
            )
            return torch.where(negative, -magnitude, magnitude)

        self.command_targets[mixed_ids, :3] = 0.0
        max_sagittal = max(
            abs(value) for value in self.command_ranges["lin_vel_x"]
        )
        self.command_targets[mixed_ids, 0] = signed_uniform(
            self.cfg.commands.min_sagittal_speed,
            max_sagittal,
            self.cfg.commands.backward_command_ratio,
        )

        sagittal_yaw = (
            torch.rand(count, device=self.device)
            < self.cfg.commands.mixed_sagittal_yaw_fraction
        )
        sagittal_lateral = ~sagittal_yaw

        yaw_count = int(sagittal_yaw.sum().item())
        if yaw_count > 0:
            yaw_magnitude = (
                self.cfg.commands.mixed_yaw_min
                + (
                    self.cfg.commands.mixed_yaw_max
                    - self.cfg.commands.mixed_yaw_min
                )
                * torch.rand(yaw_count, device=self.device)
            )
            yaw_sign = torch.where(
                torch.rand(yaw_count, device=self.device) < 0.5,
                -torch.ones(yaw_count, device=self.device),
                torch.ones(yaw_count, device=self.device),
            )
            self.command_targets[mixed_ids[sagittal_yaw], 2] = (
                yaw_sign * yaw_magnitude
            )

        lateral_count = int(sagittal_lateral.sum().item())
        if lateral_count > 0:
            lateral_magnitude = (
                self.cfg.commands.mixed_lateral_min
                + (
                    self.cfg.commands.mixed_lateral_max
                    - self.cfg.commands.mixed_lateral_min
                )
                * torch.rand(lateral_count, device=self.device)
            )
            lateral_sign = torch.where(
                torch.rand(lateral_count, device=self.device) < 0.5,
                -torch.ones(lateral_count, device=self.device),
                torch.ones(lateral_count, device=self.device),
            )
            lateral_ids = mixed_ids[sagittal_lateral]
            self.command_targets[lateral_ids, 1] = (
                lateral_sign * lateral_magnitude
            )
            translation = self.command_targets[lateral_ids, :2]
            translation_norm = torch.norm(
                translation, dim=1, keepdim=True
            ).clamp_min(1.0e-6)
            translation_scale = torch.clamp(
                self.cfg.commands.mixed_translation_max
                / translation_norm,
                max=1.0,
            )
            self.command_targets[lateral_ids, :2] = (
                translation * translation_scale
            )

            # A zero-yaw diagonal translation uses the existing outer heading
            # hold just like pure forward and lateral commands.
            if hasattr(self, "translation_heading_active"):
                self.translation_heading_active[lateral_ids] = True
                self.translation_heading_reference[lateral_ids] = self.rpy[
                    lateral_ids, 2
                ]

        if hasattr(self, "translation_heading_active"):
            yaw_ids = mixed_ids[sagittal_yaw]
            self.translation_heading_active[yaw_ids] = False
            self.translation_heading_reference[yaw_ids] = self.rpy[yaw_ids, 2]

        # Preserve zero as the start of inherited ramped commands; immediate
        # segments receive the new target now.
        self.commands[mixed_ids, :3] = 0.0
        immediate_ids = mixed_ids[~self.command_ramp_active[mixed_ids]]
        self.commands[immediate_ids, :3] = self.command_targets[
            immediate_ids, :3
        ]

    def _write_flat_critic_observation(self):
        # Preserve the complete 297-D flat Critic prefix so checkpoint columns
        # and HIM's velocity target keep exactly the same meaning.
        super()._write_flat_critic_observation()

        mass = self._normalize_range(
            self.critic_base_mass_offset,
            self.cfg.domain_rand.added_mass_range[0],
            self.cfg.domain_rand.added_mass_range[1],
        )
        com_x = self._normalize_range(
            self.critic_base_com_offset[:, 0:1],
            self.cfg.domain_rand.com_range_x[0],
            self.cfg.domain_rand.com_range_x[1],
        )
        com_y = self._normalize_range(
            self.critic_base_com_offset[:, 1:2],
            self.cfg.domain_rand.com_range_y[0],
            self.cfg.domain_rand.com_range_y[1],
        )
        com_z = self._normalize_range(
            self.critic_base_com_offset[:, 2:3],
            self.cfg.domain_rand.com_range_z[0],
            self.cfg.domain_rand.com_range_z[1],
        )
        mean_kp = torch.mean(self.kp_scales, dim=1, keepdim=True)
        mean_kd = torch.mean(self.kd_scales, dim=1, keepdim=True)
        mean_kp = self._normalize_range(
            mean_kp,
            self.cfg.domain_rand.kp_scale_range[0],
            self.cfg.domain_rand.kp_scale_range[1],
        )
        mean_kd = self._normalize_range(
            mean_kd,
            self.cfg.domain_rand.kd_scale_range[0],
            self.cfg.domain_rand.kd_scale_range[1],
        )
        self.privileged_obs_buf = torch.cat(
            (
                self.privileged_obs_buf,
                self._terrain_scan(),
                self._local_base_height(),
                mass,
                com_x,
                com_y,
                com_z,
                mean_kp,
                mean_kd,
            ),
            dim=1,
        )
        if self.privileged_obs_buf.shape[1] != ROBUST_PRIVILEGED_DIM:
            raise RuntimeError(
                f"Robust Critic observation is "
                f"{self.privileged_obs_buf.shape[1]}D, expected "
                f"{ROBUST_PRIVILEGED_DIM}D"
            )

