"""CPU checks for deployment observations, without launching Isaac Sim.

Run with ``python3 -m unittest discover -s tests -p 'test_csdog1_*.py'``.
Only the Isaac manager/action base classes are replaced with minimal doubles.
"""

import importlib.util
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch


class ManagerTerm:
    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env


class JointAction:
    def process_actions(self, actions):
        self.received_actions = actions


def load_helpers():
    managers = ModuleType("isaaclab.managers")
    managers.ManagerTermBase = ManagerTerm
    managers.ObservationTermCfg = object
    actions = ModuleType("isaaclab.envs.mdp.actions")
    actions.JointPositionAction = JointAction
    path = Path(__file__).resolve().parents[1] / (
        "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/"
        "config/quadruped/csdog1/observations.py"
    )
    spec = importlib.util.spec_from_file_location("csdog1_observation_helpers", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"isaaclab.managers": managers, "isaaclab.envs.mdp.actions": actions}):
        spec.loader.exec_module(module)
    return module


helpers = load_helpers()


class PhaseTest(unittest.TestCase):
    def setUp(self):
        self.command = torch.zeros(3, 3)
        self.env = SimpleNamespace(
            num_envs=3, device="cpu", common_step_counter=0, step_dt=0.02,
            command_manager=SimpleNamespace(get_command=lambda _: self.command),
        )
        self.phase = helpers.CSDog1Phase(None, self.env)
        self.params = dict(
            command_name="base_velocity", cycle_time=0.5, stop_cycle_time=0.5,
            stop_duration=0.5, translation_threshold=0.01, yaw_threshold=0.01,
        )

    def read(self):
        return self.phase(self.env, **self.params)

    def step(self):
        self.env.common_step_counter += 1
        return self.read()

    def test_stationary_and_turning_thresholds(self):
        self.command[1, 2] = 0.05  # Strict '<': boundary is locomotion.
        self.command[2, 0] = 0.05
        result = self.read()
        torch.testing.assert_close(result[0], torch.tensor([0., 1.]))
        angle = torch.tensor(2 * torch.pi * 0.04)
        expected = torch.stack((angle.sin(), angle.cos()))
        torch.testing.assert_close(result[1], expected)
        torch.testing.assert_close(result[2], expected)

    def test_repeated_reads_do_not_advance(self):
        self.command[:, 0] = 0.5
        first = self.read().clone()
        torch.testing.assert_close(self.read(), first)
        self.assertFalse(torch.equal(self.step(), first))

    def test_stop_cycle_and_restart(self):
        self.command[:, 0] = 0.5
        self.read()
        self.command.zero_()
        for _ in range(24):
            self.step()
        self.assertTrue((self.phase._stopping).all())
        result = self.step()  # Exactly 0.50 s after command release.
        torch.testing.assert_close(result, torch.tensor([[0., 1.]]).expand(3, -1))
        for _ in range(30):
            torch.testing.assert_close(self.step(), result)
        self.command[:, 2] = 0.1  # Pure yaw also restarts the clock.
        self.step()
        torch.testing.assert_close(self.phase._phase, torch.full((3,), 0.04))

    def test_short_zero_crossing_resumes_without_reset(self):
        self.command[:, 0] = 0.5
        self.read()
        self.command.zero_()
        self.step()
        self.step()
        self.command[:, 0] = -0.5
        self.step()
        torch.testing.assert_close(self.phase._phase, torch.full((3,), 0.16))

    def test_partial_reset_same_step(self):
        self.command[:, 0] = 0.5
        before = self.read().clone()
        self.command[1].zero_()
        self.phase.reset([1])
        after = self.read()
        torch.testing.assert_close(after[[0, 2]], before[[0, 2]])
        torch.testing.assert_close(after[1], torch.tensor([0., 1.]))
        self.phase.reset()
        self.command.zero_()
        torch.testing.assert_close(self.read(), torch.tensor([[0., 1.]]).expand(3, -1))

    def test_actions_clipped_before_scaling_and_feedback(self):
        action = torch.tensor([[-150., -2., 0., 3., 150.]])
        term = helpers.CSDog1JointPositionAction()
        term.process_actions(action)
        expected = torch.tensor([[-100., -2., 0., 3., 100.]])
        torch.testing.assert_close(term.received_actions, expected)
        self.env.action_manager = SimpleNamespace(action=action)
        torch.testing.assert_close(helpers.previous_action(self.env), expected)

    def test_scaled_command_clipping_matches_deployment(self):
        self.command[:] = torch.tensor([[200., -150., 800.], [0.3, -0.2, 0.4], [-200., 150., -800.]])
        scales = torch.tensor([1., 1., 0.25])
        actual = helpers.velocity_commands(self.env, "base_velocity") * scales
        expected = (self.command * scales).clamp(-100., 100.)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
