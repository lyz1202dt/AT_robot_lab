"""CPU integration checks for the full Cyberstar task (no Isaac Sim imports)."""

import copy
import importlib.util
import sys
import torch
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/dog03/cyberstar"
spec = importlib.util.spec_from_file_location(
    "dog03_cyberstar_tests", PACKAGE / "__init__.py", submodule_search_locations=[str(PACKAGE)]
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from dog03_cyberstar_tests.runtime import CyberstarRuntime, flat_config, source_config


class Dog03CyberstarTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.runtime = CyberstarRuntime(16, "cpu")

    def test_flat_config_preserves_network_contract_and_source(self):
        config = flat_config()
        original = source_config()["env"]
        flags = [name for name in original["domain_rand"] if name.startswith("randomize_") or name == "push_robots"]
        self.assertTrue(flags)
        for name in flags:
            self.assertFalse(config["domain_rand"][name])
            self.assertTrue(original["domain_rand"][name])
        expected = copy.deepcopy(original)
        for name in flags:
            expected["domain_rand"][name] = False
        self.assertEqual(config, expected)
        runtime = CyberstarRuntime(16, "cpu", config=config)
        runtime.compute_observations()
        self.assertEqual(runtime.obs_buf.shape, (16, 282))
        self.assertEqual(runtime.privileged_obs_buf.shape, (16, 491))

    def test_startup_respects_randomization_flags(self):
        # Exercise the real simulator adapter method against a CPU PhysX view.
        import ast
        from types import SimpleNamespace

        tree = ast.parse((PACKAGE / "lab.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Dog03CyberstarEnv")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_randomize_startup")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(PACKAGE / "lab.py"), "exec"), namespace)
        for enabled in (False, True):
            runtime = CyberstarRuntime(16, "cpu", config=source_config()["env"] if enabled else flat_config())
            masses = torch.ones(16, 1)
            coms = torch.zeros(16, 1, 7)
            materials = torch.ones(16, 1, 3)
            view = SimpleNamespace(
                get_masses=lambda: masses, get_coms=lambda: coms, get_material_properties=lambda: materials,
                set_masses=lambda *args: None, set_coms=lambda *args: None,
                set_material_properties=lambda *args: None,
            )
            env = SimpleNamespace(
                cyberstar=runtime, num_envs=16, device="cpu",
                scene={"robot": SimpleNamespace(root_physx_view=view, find_bodies=lambda name: ([0], [name]))},
            )
            namespace["_randomize_startup"](env)
            if enabled:
                self.assertGreater(runtime.critic_base_mass_offset.abs().sum().item(), 0)
                self.assertGreater(runtime.critic_base_com_offset.abs().sum().item(), 0)
                self.assertGreater((runtime.critic_friction - 1).abs().sum().item(), 0)
            else:
                torch.testing.assert_close(masses, torch.ones_like(masses))
                torch.testing.assert_close(coms, torch.zeros_like(coms))
                torch.testing.assert_close(materials[:, :, :2], torch.ones_like(materials[:, :, :2]))
                torch.testing.assert_close(runtime.critic_base_mass_offset, torch.zeros(16, 1))
                torch.testing.assert_close(runtime.critic_base_com_offset, torch.zeros(16, 3))
                torch.testing.assert_close(runtime.critic_friction, torch.ones(16, 1))

    def test_history_reset_and_critic_layout(self):
        r = self.runtime
        r.add_noise = False
        r.compute_observations()
        self.assertEqual(r.obs_buf.shape, (16, 282))
        self.assertEqual(r.privileged_obs_buf.shape, (16, 491))
        torch.testing.assert_close(r.obs_buf.reshape(16, 6, 47), r.clean_actor_frame[:, None].expand(-1, 6, -1))
        r.base_lin_vel[:] = torch.tensor([0.5, -0.2, 0.1])
        r.commands[:, 0] = 0.75
        r.compute_observations()
        torch.testing.assert_close(r.privileged_obs_buf[:, 282:285], torch.tensor([1.0, -0.4, 0.2]).expand(16, -1))
        torch.testing.assert_close(r.obs_buf[:, -39], torch.full((16,), 1.5))
        r.reset(torch.tensor([2, 4]))
        r.compute_observations()
        for env_id in (2, 4):
            frames = r.obs_buf[env_id].reshape(6, 47)
            torch.testing.assert_close(frames, frames[-1].expand_as(frames))

    def test_policy_step_delay_and_recovery_clipping(self):
        r = self.runtime
        r.control_delay[::2] = 1
        r.process_actions(torch.full((16, 12), 2.0))
        torch.testing.assert_close(r.actions[::2], torch.zeros(8, 12))
        torch.testing.assert_close(r.actions[1::2], torch.full((8, 12), 2.0))
        r.process_actions(torch.full((16, 12), 3.0))
        torch.testing.assert_close(r.actions[::2], torch.full((8, 12), 2.0))
        r.ppo_iteration = 1800
        r.capture_recovery_active[1] = True
        r.process_actions(torch.full((16, 12), 8.0))
        torch.testing.assert_close(r.actions[1], torch.full((12,), 1.5))
        self.assertEqual(r.actions[3, 0], 8.0)

    def test_observation_clip_preserves_internal_history(self):
        r = self.runtime
        r.add_noise = False
        r.dof_vel[:] = 10000
        r.compute_observations()
        self.assertEqual(r.obs_buf.max(), 100)
        self.assertEqual(r.privileged_obs_buf.max(), 100)
        self.assertGreater(r.clean_observation_history.max(), 100)

    def test_stop_requires_adjustment_and_supported_completion(self):
        r = self.runtime
        r.command_steps_remaining[:] = 10000
        r.commands[:, 0] = 0.5
        r.command_targets[:, 0] = 0.5
        r.contact_forces[:, :4, 2] = 35
        r._post_physics_step_callback()
        r.commands[:] = 0
        r.command_targets[:] = 0
        r._post_physics_step_callback()
        self.assertTrue(r.stop_adjustment_active.all())
        # Being motionless with all feet down cannot bypass the required lift/landing window.
        for _ in range(15):
            r._post_physics_step_callback()
        self.assertTrue(r.stop_adjustment_active.all())
        self.assertTrue((r.pose_recovery_blend == 0).all())
        r.reset(torch.tensor([3]))
        self.assertFalse(r.stop_adjustment_active[3])
        self.assertTrue(r.stop_adjustment_active[2])

    def test_all_rewards_finite_and_ji22_terminal_separate(self):
        r = self.runtime
        r.ppo_iteration = 1800
        r.capture_recovery_active[:4] = True
        r.stop_adjustment_active[4:8] = True
        r.commands[8:12, 1] = 0.8
        r.commands[12:, 2] = 0.6
        r.command_targets[:] = r.commands[:, :3]
        r.base_lin_vel[:] = torch.randn(16, 3) * 0.1
        r.dof_vel[:] = torch.randn(16, 12) * 0.1
        r.rigid_body_states[:, :4, 2] = 0.04
        r.contact_forces[:, :4, 2] = 35
        self.assertEqual(len(r.reward_names), 78)
        positive, negative = r.zeros(), r.zeros()
        # Clone because feet_air_time updates its event buffers during reward evaluation.
        expected = copy.deepcopy(r)
        for name, fn in zip(expected.reward_names, expected.reward_functions):
            value = fn() * expected.reward_scales[name]
            self.assertTrue(torch.isfinite(value).all(), name)
            if expected.reward_scales[name] > 0:
                positive += value
            else:
                negative += value
        r.reset_buf[0] = True
        r.time_out_buf[1] = True
        r.reset_buf[1] = True
        r.compute_reward()
        reference = positive * torch.exp((negative / 0.02).clamp(-20, 0))
        reference[0] += -0.04
        torch.testing.assert_close(r.rew_buf, reference)

    def test_push_curriculum_and_capture_activation(self):
        r = self.runtime
        r.episode_length_buf[:] = 500
        calls = []
        r._velocity_writer = lambda ids: calls.append(ids.clone())
        r.ppo_iteration = 1000
        r._push_robots()
        self.assertTrue(calls)
        self.assertLessEqual(r.root_states[:, 7:9].norm(dim=-1).max(), 0.800001)
        self.assertFalse(r.capture_recovery_active.any())
        r.ppo_iteration = 1800
        r._start_capture_recovery(torch.arange(16))
        self.assertTrue(r.capture_recovery_active.all())
        r.reset(torch.tensor([0, 1]))
        self.assertFalse(r.capture_recovery_active[:2].any())
        self.assertTrue(r.capture_recovery_active[2:].all())


class Dog03ModelAndStairsTest(unittest.TestCase):
    def test_him_wrapper_keeps_full_history_and_critic_layout(self):
        from types import ModuleType, SimpleNamespace
        from unittest.mock import patch

        runtime = CyberstarRuntime(8, "cpu")
        runtime.compute_observations()
        bridge = ModuleType("isaaclab_rl.rsl_rl")
        bridge.RslRlVecEnvWrapper = lambda env, **kwargs: env
        spec = importlib.util.spec_from_file_location(
            "dog03_wrapper_cpu", ROOT / "source/robot_lab/robot_lab/him/him_vec_env_wrapper.py"
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {bridge.__name__: bridge}):
            spec.loader.exec_module(module)
        env = SimpleNamespace(
            num_envs=8,
            num_actions=12,
            device="cpu",
            max_episode_length=1000,
            unwrapped=SimpleNamespace(cyberstar=runtime),
            get_observations=lambda: {"policy": runtime.obs_buf, "critic": runtime.privileged_obs_buf},
        )
        wrapper = module.HIMVecEnvWrapper(
            env,
            clip_actions=100.0,
            num_one_step_obs=47,
            history_order="oldest_to_latest",
            policy_frame_term_order=[
                "phase",
                "base_ang_vel",
                "projected_gravity",
                "velocity_commands",
                "joint_pos",
                "joint_vel",
                "actions",
            ],
            estimator_target_term_names=[
                "base_ang_vel",
                "projected_gravity",
                "joint_pos",
                "joint_vel",
                "actions",
                "base_lin_vel",
            ],
        )
        self.assertEqual(wrapper.num_obs, 282)
        self.assertEqual(wrapper.num_privileged_obs, 491)
        self.assertEqual(wrapper.default_estimator_vel_slice, (282, 285))
        self.assertEqual(tuple(wrapper.default_estimator_target_slice), tuple(range(237, 243)) + tuple(range(246, 285)))
        torch.testing.assert_close(wrapper.get_observations(), runtime.obs_buf)
        metadata = wrapper.get_him_deployment_metadata()
        self.assertEqual(metadata["history_order"], "oldest_to_latest")
        self.assertEqual(metadata["cyberstar_contract"]["command_scale"], [2.0, 2.0, 0.25])

    def test_model_defaults_and_limits_match_03dog_urdf(self):
        import ast
        import xml.etree.ElementTree as ET

        runtime = CyberstarRuntime(8, "cpu")
        model_file = ROOT / "source/robot_lab/robot_lab/assets/dog03.py"
        defaults = next(
            ast.literal_eval(n.value)
            for n in ast.parse(model_file.read_text()).body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "DOG03_JOINT_POS" for t in n.targets)
        )
        torch.testing.assert_close(runtime.default_dof_pos[0], torch.tensor([defaults[n] for n in runtime.dof_names]))
        urdf = ET.parse(ROOT / "source/robot_lab/data/Robots/03dog/urdf/03dog.urdf").getroot()
        bounds = {
            j.get("name"): [float(j.find("limit").get(k)) for k in ("lower", "upper")]
            for j in urdf.findall("joint")
            if j.get("type") != "fixed"
        }
        raw = torch.tensor([bounds[n] for n in runtime.dof_names])
        mid = raw.mean(-1)
        half = (raw[:, 1] - raw[:, 0]) * runtime.cfg.rewards.soft_dof_pos_limit / 2
        torch.testing.assert_close(runtime.dof_pos_limits, torch.stack((mid - half, mid + half), -1))
        self.assertEqual(runtime.cfg.init_state.pos, [0.0, 0.0, 0.4])
        self.assertEqual(runtime.cfg.rewards.stand_base_height_target, 0.32)
        self.assertEqual(runtime.cfg.asset.collapse_fixed_joints, False)
        self.assertEqual(runtime.rigid_body_states.shape, (8, 21, 13))
        self.assertEqual(len(runtime.penalised_contact_indices), 8)
        self.assertTrue((runtime.default_dof_pos >= runtime.dof_pos_limits[:, 0]).all())
        self.assertTrue((runtime.default_dof_pos <= runtime.dof_pos_limits[:, 1]).all())

    def test_local_configs_and_rewards_do_not_change_csdog1(self):
        import json

        from dog03_cyberstar_tests.runtime import source_config

        original = (
            ROOT
            / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/csdog1/cyberstar/config.json"
        )
        before = original.read_bytes()
        config = source_config()["env"]
        config["rewards"]["scales"]["tracking_lin_vel_active"] = 99.0
        runtime = CyberstarRuntime(4, "cpu", config=config)
        self.assertAlmostEqual(runtime.reward_scales["tracking_lin_vel_active"], 99.0 * 0.02)
        self.assertEqual(source_config()["env"]["rewards"]["scales"]["tracking_lin_vel_active"], 1.5)
        self.assertEqual(json.loads(original.read_text())["env"]["rewards"]["scales"]["tracking_lin_vel_active"], 1.5)
        self.assertEqual(original.read_bytes(), before)
        self.assertTrue(
            all(fn.__func__.__module__.startswith("dog03_cyberstar_tests.") for fn in runtime.reward_functions)
        )

    def test_stair_stages_are_local_and_control_geometry_and_budget(self):
        import numpy as np
        import os
        from unittest.mock import patch

        from dog03_cyberstar_tests.stairs_geometry import stair_patch_specs
        from dog03_cyberstar_tests.stairs_runtime import stairs_config, stairs_iterations

        for stage in range(5):
            with self.subTest(stage=stage), patch.dict(
                os.environ, {"DOG03_STAIR_STAGE": str(stage), "CSDOG1_STAIR_STAGE": "4"}
            ):
                config = stairs_config()
                selected = config["stages"][str(stage)]
                terrain = config["env"]["terrain"]
                self.assertEqual(config["stage"], stage)
                self.assertEqual(stairs_iterations(), selected["iterations"])
                self.assertEqual(terrain["stair_height_bands"], selected["height_bands"])
                specs = stair_patch_specs(
                    256,
                    terrain["flat_terrain_ratio"],
                    terrain["stair_up_ratio"],
                    terrain["stair_height_bands"],
                    np.random.default_rng(7),
                )
                heights = [abs(height) for height, _ in specs if height]
                self.assertGreaterEqual(min(heights), selected["min_height"])
                self.assertLessEqual(max(heights), selected["max_height"])
                self.assertEqual(sum(height == 0 for height, _ in specs), 77)
        with patch.dict(os.environ, {"DOG03_STAIR_STAGE": "invalid"}):
            with self.assertRaisesRegex(ValueError, "DOG03_STAIR_STAGE"):
                stairs_config()

    def test_stair_history_and_critic_match_robust_flat(self):
        from dog03_cyberstar_tests.stairs_runtime import CyberstarStairsRuntime

        flat, stairs = CyberstarRuntime(16, "cpu"), CyberstarStairsRuntime(16, "cpu")
        for runtime in (flat, stairs):
            runtime.add_noise = False
            runtime.commands[:] = torch.tensor([0.5, -0.1, 0.2, 0])
            runtime.base_lin_vel[:] = torch.tensor([0.2, -0.1, 0.05])
            runtime.compute_observations()
            self.assertEqual(runtime.obs_buf.shape, (16, 282))
            self.assertEqual(runtime.privileged_obs_buf.shape, (16, 491))
        torch.testing.assert_close(stairs.obs_buf, flat.obs_buf)
        torch.testing.assert_close(stairs.privileged_obs_buf, flat.privileged_obs_buf)

    def test_stair_rewards_touchdown_and_partial_reset(self):
        from dog03_cyberstar_tests.stairs_runtime import CyberstarStairsRuntime

        runtime = CyberstarStairsRuntime(32, "cpu")
        runtime.flat_env_mask[8:] = False
        runtime.stair_step_heights[8:20] = -0.05
        runtime.stair_step_heights[20:] = 0.05
        runtime.command_targets[:, 0] = 0.5
        runtime.commands[:, 0] = 0.5
        runtime.base_lin_vel[:, 0] = 0.4
        runtime._post_physics_step_callback()
        runtime.contact_forces[:, :4, 2] = 120
        runtime.previous_foot_velocities[:, :, 2] = -1.0
        runtime._post_physics_step_callback()
        self.assertTrue(runtime.stair_touchdown_event.all())
        self.assertEqual(len(runtime.reward_names), 85)
        for name, function in zip(runtime.reward_names, runtime.reward_functions):
            self.assertTrue(torch.isfinite(function()).all(), name)
        impact = runtime._reward_stair_touchdown_impact()
        self.assertEqual(impact[:8].count_nonzero(), 0)
        torch.testing.assert_close(impact[20:], impact[8:20] * 1.5)
        runtime.episode_length_buf[:] = 10
        runtime.stair_command_progress[8:10] = 2.0
        runtime.reset(torch.tensor([8, 9]))
        self.assertEqual(runtime.stair_success, 1.0)
        self.assertFalse(runtime.stair_touchdown_event[8:10].any())
        self.assertTrue(runtime.stair_touchdown_event[10:].all())

    def test_full_size_him_ppo_update_on_03dog_stairs(self):
        import math
        from types import ModuleType
        from unittest.mock import patch

        from dog03_cyberstar_tests.stairs_runtime import CyberstarStairsRuntime

        him = ModuleType("dog03_him_cpu")
        him.__path__ = [str(ROOT / "source/robot_lab/robot_lab/him")]
        tensorboard = ModuleType("torch.utils.tensorboard")
        tensorboard.SummaryWriter = object
        with patch.dict(sys.modules, {him.__name__: him, "torch.utils.tensorboard": tensorboard}):
            from dog03_him_cpu.him_actor_critic import HIMActorCritic
            from dog03_him_cpu.him_ppo import HIMPPO
        model = HIMActorCritic(
            282,
            491,
            47,
            12,
            estimator_vel_slice=(282, 285),
            estimator_target_slice=tuple(range(237, 243)) + tuple(range(246, 285)),
            actor_current_frame="last",
        )
        self.assertEqual([m.out_features for m in model.critic if isinstance(m, torch.nn.Linear)], [512, 256, 128, 1])
        runtime = CyberstarStairsRuntime(8, "cpu")
        algorithm = HIMPPO(model, num_learning_epochs=1, num_mini_batches=1)
        algorithm.init_storage(8, 4, [282], [491], [12])
        runtime.compute_observations()
        before = model.critic[0].weight.detach().clone()
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
        self.assertFalse(torch.equal(before, model.critic[0].weight))


if __name__ == "__main__":
    unittest.main()
