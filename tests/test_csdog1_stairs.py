"""Stair runtime, source checkpoint transfer and Adam regression checks on CPU."""
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/csdog1/cyberstar'
name = 'csdog1_stairs_tests'
spec = importlib.util.spec_from_file_location(name, PACKAGE / '__init__.py', submodule_search_locations=[str(PACKAGE)])
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
from csdog1_stairs_tests.runtime import CyberstarRuntime
from csdog1_stairs_tests.stairs_geometry import pyramid_stair_heights, stair_patch_specs
from csdog1_stairs_tests.stairs_runtime import CyberstarStairsRuntime, stairs_config


def load_him():
    path = ROOT / 'source/robot_lab/robot_lab/him'
    package = ModuleType('csdog1_him_tests')
    package.__path__ = [str(path)]
    sys.modules[package.__name__] = package
    # These tests do not create a logger; tensorboard is optional on CPU hosts.
    tensorboard = ModuleType('torch.utils.tensorboard')
    tensorboard.SummaryWriter = object
    with patch.dict(sys.modules, {'torch.utils.tensorboard': tensorboard}):
        from csdog1_him_tests.him_actor_critic import HIMActorCritic
        from csdog1_him_tests.him_on_policy_runner import HIMOnPolicyRunner
        from csdog1_him_tests.him_ppo import HIMPPO
    return HIMActorCritic, HIMOnPolicyRunner, HIMPPO


ActorCritic, Runner, PPO = load_him()


def make_model(critic_dim=491):
    return ActorCritic(
        282, critic_dim, 47, 12, actor_hidden_dims=[32, 16], critic_hidden_dims=[32, 16],
        estimator_vel_slice=(282, 285) if critic_dim == 491 else (47, 50),
        estimator_target_slice=tuple(range(237, 243)) + tuple(range(246, 285)) if critic_dim == 491 else tuple(range(2, 8)) + tuple(range(11, 50)),
        actor_current_frame='last',
    )


class StairRuntimeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.runtime = CyberstarStairsRuntime(32, 'cpu')
        self.runtime.flat_env_mask[8:] = False
        self.runtime.stair_step_heights[8:20] = -.05
        self.runtime.stair_step_heights[20:] = .05

    def test_critic_matches_robust_flat_layout_and_history(self):
        flat = CyberstarRuntime(32, 'cpu')
        stair = self.runtime
        for runtime in (flat, stair):
            runtime.add_noise = False
            runtime.commands[:] = torch.tensor([.5, -.1, .2, 0])
            runtime.base_lin_vel[:] = torch.tensor([.2, -.1, .05])
            runtime.compute_observations()
            self.assertEqual(runtime.obs_buf.shape, (32, 282))
            self.assertEqual(runtime.privileged_obs_buf.shape, (32, 491))
        torch.testing.assert_close(stair.obs_buf, flat.obs_buf)
        torch.testing.assert_close(stair.privileged_obs_buf, flat.privileged_obs_buf)

    def test_stair_commands_remain_exclusive_during_resampling(self):
        runtime = self.runtime
        for _ in range(10):
            runtime._resample_commands(torch.arange(32))
            for _ in range(5):
                runtime._post_physics_step_callback()
                current = runtime.commands[8:, :2]
                target = runtime.command_targets[8:, :2]
                self.assertFalse(((current[:, 0].abs() > 1e-5) & (current[:, 1].abs() > 1e-5)).any())
                self.assertFalse(((target[:, 0].abs() > 1e-5) & (target[:, 1].abs() > 1e-5)).any())
        self.assertTrue((runtime.command_targets[8:, 0] > 0).any())
        self.assertTrue((runtime.command_targets[8:, 1].abs() > 0).any())

    def test_basic_push_sparse_and_initial_step(self):
        runtime = self.runtime
        self.assertFalse(hasattr(runtime.cfg.domain_rand, 'advanced_push_start_iteration'))
        calls = []
        runtime._velocity_writer = lambda ids: calls.append(ids.clone())
        runtime._push_robots()  # This previously raised AttributeError even before any push was due.
        self.assertEqual(calls, [])
        runtime.root_states[:, 7:13] = 1.
        runtime.episode_length_buf[:] = 1
        runtime.episode_length_buf[2] = 0
        runtime.episode_length_buf[3] = runtime.cfg.domain_rand.push_interval
        runtime._push_robots()
        torch.testing.assert_close(calls[0], torch.tensor([3]))
        self.assertLessEqual(runtime.root_states[3, 7:9].abs().max(), .5)
        torch.testing.assert_close(runtime.root_states[[0, 1, 2, 4], 7:13], torch.ones(4, 6))
        torch.testing.assert_close(runtime.root_states[3, 9:13], torch.ones(4))

    def test_rewards_finite_touchdown_descent_and_partial_reset(self):
        runtime = self.runtime
        runtime.command_targets[:, 0] = .5
        runtime.commands[:, 0] = .5
        runtime.base_lin_vel[:, 0] = .4
        runtime._post_physics_step_callback()
        runtime.contact_forces[:, :4, 2] = 120
        runtime.previous_foot_velocities[:, :, 2] = -1.
        runtime._post_physics_step_callback()
        self.assertTrue(runtime.stair_touchdown_event.all())
        self.assertEqual(len(runtime.reward_names), 85)
        for name, function in zip(runtime.reward_names, runtime.reward_functions):
            self.assertTrue(torch.isfinite(function()).all(), name)
        impact = runtime._reward_stair_touchdown_impact()
        self.assertEqual(impact[:8].count_nonzero(), 0)
        torch.testing.assert_close(impact[20:], impact[8:20] * 1.5)
        runtime.episode_length_buf[:] = 10
        runtime.stair_command_progress[8:10] = 2.
        runtime.reset(torch.tensor([8, 9]))
        self.assertEqual(runtime.stair_success, 1.)
        self.assertFalse(runtime.stair_touchdown_event[8:10].any())
        self.assertTrue(runtime.stair_touchdown_event[10:].all())

    def test_him_ppo_updates_on_stair_observations(self):
        runtime = self.runtime
        algorithm = PPO(make_model(), num_learning_epochs=1, num_mini_batches=1)
        algorithm.init_storage(32, 4, [282], [491], [12])
        runtime.compute_observations()
        before = algorithm.actor_critic.critic[0].weight.detach().clone()
        with torch.inference_mode():
            for _ in range(4):
                actions = algorithm.act(runtime.obs_buf, runtime.privileged_obs_buf)
                runtime.process_actions(actions)
                runtime._post_physics_step_callback()
                runtime.compute_reward()
                runtime.compute_observations()
                algorithm.process_env_step(runtime.rew_buf, runtime.reset_buf, {}, runtime.privileged_obs_buf)
                runtime.finish_step()
            algorithm.compute_returns(runtime.privileged_obs_buf)
        losses = algorithm.update()
        self.assertTrue(all(math.isfinite(loss) for loss in losses))
        self.assertFalse(torch.equal(before, algorithm.actor_critic.critic[0].weight))

    def test_blocked_clearance_has_signal_but_zero_command_does_not(self):
        runtime = self.runtime
        runtime.commands[:] = 0.
        runtime.command_targets[:] = 0.
        runtime.commands[:, 0] = .5
        runtime.command_targets[:, 0] = .5
        runtime.base_lin_vel[:] = 0.
        target = (runtime.stair_step_heights.abs() + runtime.cfg.rewards.clearance_margin).clamp(
            runtime.cfg.rewards.clearance_target_min, runtime.cfg.rewards.clearance_target_max
        )
        runtime.rigid_body_states[:, runtime.feet_indices, 2] = target[:, None]
        runtime.contact_forces[:] = 0.
        blocked = runtime._reward_feet_clearance()
        self.assertTrue((blocked[8:] > 0).all())
        runtime.base_lin_vel[:, 0] = .5
        self.assertTrue((runtime._reward_feet_clearance()[8:] > blocked[8:]).all())
        runtime.commands[:] = 0.
        runtime.command_targets[:] = 0.
        self.assertEqual(runtime._reward_feet_clearance().count_nonzero(), 0)

    def test_stand_height_quality_is_invariant_to_terrain_elevation(self):
        runtime = self.runtime
        runtime.commands[:] = 0.
        runtime.stand_blend[:] = 1.
        runtime.root_states[:, 2] = runtime.cfg.rewards.stand_base_height_target
        initial = runtime._reward_stand_height_quality().clone()
        runtime.root_states[:, 2] += .4
        runtime._height_query = lambda positions: torch.full_like(positions[..., 0], .4)
        torch.testing.assert_close(runtime._reward_stand_height_quality(), initial)


class StairGeometryTest(unittest.TestCase):
    def test_each_stage_controls_generated_heights(self):
        for stage in range(5):
            with self.subTest(stage=stage), patch.dict(os.environ, {'CSDOG1_STAIR_STAGE': str(stage)}):
                config = stairs_config()
                reference = config['stages'][str(stage)]
                terrain = config['env']['terrain']
                self.assertEqual(config['stage'], stage)
                self.assertEqual(terrain['stair_height_bands'], reference['height_bands'])
                self.assertEqual(terrain['stair_step_height_min'], reference['min_height'])
                self.assertEqual(terrain['stair_step_height_max'], reference['max_height'])
                specs = stair_patch_specs(256, terrain['flat_terrain_ratio'], terrain['stair_up_ratio'],
                                         terrain['stair_height_bands'], np.random.default_rng(1))
                heights = [abs(height) for height, _ in specs if height]
                self.assertGreaterEqual(min(heights), reference['min_height'])
                self.assertLessEqual(max(heights), reference['max_height'])
                self.assertEqual(sum(height == 0 for height, _ in specs), 77)

    def test_default_stage_and_invalid_override(self):
        with patch.dict(os.environ):
            os.environ.pop('CSDOG1_STAIR_STAGE', None)
            raw = json.loads((PACKAGE / 'stairs_config.json').read_text())
            self.assertEqual(stairs_config()['stage'], raw['stage'])
        for invalid in ('-1', '5', 'abc', ''):
            with self.subTest(invalid=invalid), patch.dict(os.environ, {'CSDOG1_STAIR_STAGE': invalid}):
                with self.assertRaisesRegex(ValueError, 'CSDOG1_STAIR_STAGE'):
                    stairs_config()

    def test_distribution_and_ascent_descent_sign(self):
        config = stairs_config()['env']['terrain']
        specs = stair_patch_specs(256, .3, .65, config['stair_height_bands'], np.random.default_rng(1))
        self.assertEqual(sum(height == 0 for height, _ in specs), 77)
        self.assertLessEqual(abs(sum(height < 0 for height, _ in specs) / 179 - .65), .01)
        for height, width in specs:
            heights = pyramid_stair_heights((8., 8.), .1, .005, height, width or .35, 3.)
            self.assertEqual(heights.shape, (80, 80))
            self.assertEqual(np.sign(heights[40, 40]), np.sign(height))
            self.assertEqual(heights[0, 0], 0)


class OptimizerCheckpointTest(unittest.TestCase):
    def test_incompatible_adam_moments_rejected_before_step(self):
        old = torch.nn.Parameter(torch.randn(8, 491))
        old_optimizer = torch.optim.Adam([old], lr=.01)
        old.square().mean().backward()
        old_optimizer.step()
        current = torch.nn.Parameter(torch.randn(8, 50))
        optimizer = torch.optim.Adam([current], lr=.00005)
        Runner._load_optimizer_state(optimizer, old_optimizer.state_dict(), 'test')
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(optimizer.param_groups[0]['lr'], .00005)
        current.square().mean().backward()
        optimizer.step()
        self.assertEqual(optimizer.state[current]['exp_avg'].shape, current.shape)

    def test_matching_adam_state_restored(self):
        param = torch.nn.Parameter(torch.randn(8, 50))
        old = torch.optim.Adam([param], lr=.01)
        param.square().mean().backward()
        old.step()
        optimizer = torch.optim.Adam([param], lr=.00005)
        Runner._load_optimizer_state(optimizer, old.state_dict(), 'test')
        torch.testing.assert_close(optimizer.state[param]['exp_avg'], old.state[param]['exp_avg'])
        self.assertEqual(optimizer.param_groups[0]['lr'], .01)

    def test_491_checkpoint_transfers_all_model_weights(self):
        source = make_model()
        runner = Runner.__new__(Runner)
        runner.alg = PPO(make_model())
        runner.env = SimpleNamespace(unwrapped=SimpleNamespace(cyberstar=True))
        runner.cfg = {'load_optimizer': False}
        runner.device = 'cpu'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save({'model_state_dict': source.state_dict(), 'iter': 10000}, path)
            runner.load(str(path))
        for name, value in source.state_dict().items():
            torch.testing.assert_close(runner.alg.actor_critic.state_dict()[name], value)
        self.assertEqual(runner.current_learning_iteration, 10000)
        self.assertEqual(len(runner.alg.optimizer.state), 0)


if __name__ == '__main__':
    unittest.main()
