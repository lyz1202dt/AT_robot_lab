# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Observation helpers matching Cyberstar's deployed policy input."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.managers import ManagerTermBase, ObservationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class Dog03Phase(ManagerTermBase):
    """Stateful phase clock matching ``mnn_model.cpp::updatePhase``.

    A released command keeps the clock running for one stop-adjustment cycle,
    then holds phase zero. Deployment's debounce and measured settling window
    affect recovery diagnostics/actuator limiting, but not this phase trajectory.
    Each observation term owns its clock; repeated reads in one control step
    must not advance it (including reads by the HIM wrapper).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._phase = torch.zeros(env.num_envs, device=env.device)
        self._stop_elapsed = torch.zeros_like(self._phase)
        self._was_moving = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._stopping = torch.zeros_like(self._was_moving)
        self._last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = slice(None) if env_ids is None else env_ids
        self._phase[ids] = 0.0
        self._stop_elapsed[ids] = 0.0
        self._was_moving[ids] = False
        self._stopping[ids] = False
        self._last_step[ids] = -1

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        cycle_time: float,
        stop_cycle_time: float,
        stop_duration: float,
        translation_threshold: float,
        yaw_threshold: float,
    ) -> torch.Tensor:
        command = env.command_manager.get_command(command_name)
        moving = ~(
            (torch.linalg.vector_norm(command[:, :2], dim=-1) < translation_threshold)
            & (command[:, 2].abs() < yaw_threshold)
        )
        update = self._last_step != env.common_step_counter
        starting_stop = update & ~moving & self._was_moving
        self._stopping[starting_stop] = True
        self._stop_elapsed[starting_stop] = 0.0

        running = update & moving
        stopping = update & ~moving & self._stopping
        self._phase[running] = (self._phase[running] + env.step_dt / cycle_time) % 1.0
        self._phase[stopping] = (self._phase[stopping] + env.step_dt / stop_cycle_time) % 1.0
        self._stop_elapsed[stopping] += env.step_dt
        finished = stopping & (self._stop_elapsed >= stop_duration)
        self._stopping[finished | running] = False
        self._stop_elapsed[running] = 0.0
        self._phase[update & ~moving & ~self._stopping] = 0.0
        self._was_moving[update] = moving[update]
        self._last_step[update] = env.common_step_counter

        angle = 2 * torch.pi * self._phase
        return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


def previous_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Previous network output after deployment's action clipping."""
    return env.action_manager.action.clamp(-100.0, 100.0)


def velocity_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Pre-scale limits equivalent to clipping scaled commands to +/-100."""
    command = env.command_manager.get_command(command_name)
    limits = command.new_tensor((100.0, 100.0, 400.0))
    return torch.clamp(command, min=-limits, max=limits)


class Dog03JointPositionAction(JointPositionAction):
    """Clip network actions before scaling/offset, as deployment does."""

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions.clamp(-100.0, 100.0))
