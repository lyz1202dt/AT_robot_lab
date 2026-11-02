"""CPU regression tests for JSON-driven motors and the ATDog randomization inventory."""

import ast
import importlib.util
import json
import sys
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[1]
VELOCITY = ROOT / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity"
PACKAGE = VELOCITY / "config/quadruped/dog03/cyberstar"
NAME = "dog03_motor_tests"
spec = importlib.util.spec_from_file_location(NAME, PACKAGE / "__init__.py", submodule_search_locations=[str(PACKAGE)])
package = importlib.util.module_from_spec(spec)
sys.modules[NAME] = package
spec.loader.exec_module(package)
from dog03_motor_tests.runtime import CyberstarRuntime, flat_config, joint_values, source_config
from dog03_motor_tests.stairs_runtime import CyberstarStairsRuntime, stairs_config


def adapter_function(name, scope=None):
    tree = ast.parse((PACKAGE / "lab.py").read_text())
    nodes = tree.body
    if scope:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == scope).body
    function = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(PACKAGE / "lab.py"), "exec"), namespace)
    return namespace[name], namespace


class Dog03MotorDynamicsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_json_defaults_and_both_282_stages(self):
        for filename in ("config.json", "stairs_config.json"):
            config = json.loads((PACKAGE / filename).read_text())["env"]
            runtime = (
                CyberstarStairsRuntime(8, "cpu", config=stairs_config()["env"])
                if filename == "stairs_config.json"
                else CyberstarRuntime(8, "cpu", config=config)
            )
            runtime.randomize_joint_friction(torch.arange(8))
            for name in ("coulomb", "viscous"):
                values = config["control"]["motor_friction"][name]
                self.assertEqual(set(values), set(runtime.dof_names))
                self.assertTrue(all(value == 0 for value in values.values()))
                self.assertEqual(getattr(runtime, f"joint_{name}_friction").abs().sum().item(), 0)
                self.assertEqual(config["domain_rand"][f"joint_{name}_friction_offset_range"], [0, 0])
            self.assertEqual(config["init_state"]["reset_dof_velocity_range"], [0, 0])
            runtime.compute_observations()
            self.assertEqual(runtime.obs_buf.shape, (8, 282))
            self.assertEqual(runtime.privileged_obs_buf.shape, (8, 491))

    def test_pd_gains_are_loaded_with_per_joint_overrides(self):
        config = source_config()["env"]
        config["control"]["stiffness"] = {"joint": 8.0, ".*_thigh_joint": 12.0, "FL_hip_joint": 16.0}
        config["control"]["damping"] = {"joint": 0.4, "FR_calf_joint": 0.8}
        runtime = CyberstarRuntime(4, "cpu", config=config)
        targets, positions, velocities = torch.ones(4, 12), torch.zeros(4, 12), torch.full((4, 12), 2.0)
        runtime.kp_scales[:] = 1.2
        runtime.kd_scales[:] = 0.5
        runtime.motor_strengths[:] = 0.8
        runtime.default_joint_angle_offsets[:] = 0.1
        expected_gains = torch.tensor([[16, 12, 8, 8, 12, 8, 8, 12, 8, 8, 12, 8]], dtype=torch.float)
        expected_damping = torch.full((1, 12), 0.4)
        expected_damping[0, 5] = 0.8
        expected = (expected_gains * 1.2 * 1.1 - expected_damping * 0.5 * velocities) * 0.8
        limits = torch.full((4, 12), 10.0)
        actual = runtime.compute_motor_torques(targets, positions, velocities, limits)
        torch.testing.assert_close(actual, expected.clamp(-10, 10))
        torch.testing.assert_close(runtime.p_gains, expected_gains)
        torch.testing.assert_close(runtime.d_gains, expected_damping)
        config["control"]["stiffness"] = {"joint": 0.0}
        config["control"]["damping"] = {"joint": 0.0}
        runtime = CyberstarRuntime(4, "cpu", config=config)
        self.assertEqual(runtime.compute_motor_torques(targets, positions, velocities, limits).abs().sum().item(), 0)

    def test_action_adapter_uses_json_gains_and_motor_not_solver_limits(self):
        runtime = CyberstarRuntime(4, "cpu")
        runtime.torque_limits = torch.full((4, 12), 2.0)
        asset = SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=torch.zeros(4, 12),
                joint_vel=torch.zeros(4, 12),
                joint_effort_limits=torch.full((4, 12), 1.0e9),
            ),
            set_joint_effort_target=Mock(),
        )
        action = SimpleNamespace(
            _env=SimpleNamespace(cyberstar=runtime),
            _asset=asset,
            processed_actions=torch.ones(4, 12),
            _joint_ids=list(range(12)),
        )
        apply_actions, _ = adapter_function("apply_actions", "CyberstarJointAction")
        apply_actions(action)
        torch.testing.assert_close(runtime.torques, torch.full((4, 12), 2.0))
        torch.testing.assert_close(asset.set_joint_effort_target.call_args.args[0], runtime.torques)

    def test_joint_parameter_validation(self):
        names = ["FL_hip_joint", "FR_hip_joint"]
        for values in (
            {"typo_joint": 1.0},
            {"FL_hip_joint": 1.0},
            {"joint": -1.0},
            {"joint": float("nan")},
            {".*": 1.0, ".*_hip_joint": 2.0},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                joint_values(values, names, "cpu", "test")

    def test_friction_sampling_is_partial_and_disabled_modes_keep_nominal(self):
        config = source_config()["env"]
        config["control"]["motor_friction"] = {"coulomb": {"joint": 0.4}, "viscous": {"joint": 0.1}}
        config["domain_rand"]["joint_coulomb_friction_offset_range"] = [-0.1, 0.2]
        config["domain_rand"]["joint_viscous_friction_offset_range"] = [0.05, 0.15]
        runtime = CyberstarRuntime(16, "cpu", config=config)
        ids = torch.tensor([1, 3, 8])
        runtime.randomize_joint_friction(ids)
        self.assertTrue((runtime.joint_coulomb_friction[ids] >= 0.3).all())
        self.assertTrue((runtime.joint_coulomb_friction[ids] <= 0.6).all())
        self.assertTrue((runtime.joint_viscous_friction[ids] >= 0.15).all())
        self.assertTrue((runtime.joint_viscous_friction[ids] <= 0.25).all())
        torch.testing.assert_close(runtime.joint_coulomb_friction[0], torch.full((12,), 0.4))
        before = runtime.joint_coulomb_friction[ids].clone()
        runtime.randomize_joint_friction(ids)
        self.assertFalse(torch.equal(before, runtime.joint_coulomb_friction[ids]))
        runtime.cfg.env.test = True
        runtime.randomize_joint_friction(ids)
        torch.testing.assert_close(runtime.joint_coulomb_friction, torch.full((16, 12), 0.4))
        torch.testing.assert_close(runtime.joint_viscous_friction, torch.full((16, 12), 0.1))
        runtime.cfg.env.test = False
        runtime.cfg.domain_rand.randomize_joint_friction = False
        runtime.randomize_joint_friction(ids)
        torch.testing.assert_close(runtime.joint_coulomb_friction[ids], torch.full((3, 12), 0.4))
        runtime.cfg.domain_rand.randomize_joint_friction = True
        runtime.cfg.domain_rand.joint_coulomb_friction_offset_range = [-2.0, -1.0]
        runtime.randomize_joint_friction(ids)
        self.assertEqual(runtime.joint_coulomb_friction[ids].abs().sum().item(), 0)

    def test_passive_friction_is_after_saturation_and_opposes_motion(self):
        class IdealPDActuator:
            def __init__(self):
                self.applied_effort = torch.zeros(2, 3)

            def compute(self, action, joint_pos, joint_vel):
                self.applied_effort = action.joint_efforts.clamp(-2, 2)
                action.joint_efforts = self.applied_effort
                return action

        tree = ast.parse((PACKAGE / "actuator.py").read_text())
        definition = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        namespace = {"torch": torch, "IdealPDActuator": IdealPDActuator}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(PACKAGE / "actuator.py"), "exec"), namespace)
        actuator = namespace["CyberstarMotorActuator"]()
        actuator.coulomb_friction[:] = 0.5
        actuator.joint_viscous_friction[:] = 0.2
        velocities = torch.tensor([[3.0, -3.0, 0.0], [3.0, -3.0, 0.0]])
        friction = 0.5 * velocities.sign() + 0.2 * velocities
        self.assertTrue((friction * velocities >= 0).all())
        drive = torch.tensor([[10.0, -10.0, 1.0], [-10.0, 10.0, -1.0]])
        action = actuator.compute(SimpleNamespace(joint_efforts=drive), torch.zeros_like(drive), velocities)
        torch.testing.assert_close(action.joint_efforts, drive.clamp(-2, 2) - friction)
        self.assertLess(action.joint_efforts[1, 0].item(), -2)
        actuator.coulomb_friction[:] = 0
        actuator.joint_viscous_friction[:] = 0
        action = actuator.compute(SimpleNamespace(joint_efforts=drive), torch.zeros_like(drive), velocities)
        torch.testing.assert_close(action.joint_efforts, drive.clamp(-2, 2))

    def test_reset_applies_friction_to_correct_actuator_joints_and_joint_velocity(self):
        runtime = CyberstarRuntime(8, "cpu")
        runtime.cfg.control.motor_friction.coulomb.FL_hip_joint = 0.3
        runtime.nominal_joint_coulomb_friction[0, 0] = 0.3
        runtime.cfg.init_state.reset_dof_velocity_range = [0.2, 0.2]
        ids = torch.tensor([1, 3])
        actuator = SimpleNamespace(
            joint_names=list(reversed(runtime.dof_names)),
            coulomb_friction=torch.zeros(8, 12),
            joint_viscous_friction=torch.zeros(8, 12),
        )
        robot = SimpleNamespace(
            data=SimpleNamespace(
                default_root_state=torch.zeros(8, 13),
                default_joint_pos=torch.zeros(8, 12),
                soft_joint_pos_limits=runtime.dof_pos_limits.unsqueeze(0).repeat(8, 1, 1),
                soft_joint_vel_limits=torch.full((8, 12), 30.0),
            ),
            actuators={"legs": actuator},
            write_root_pose_to_sim=Mock(),
            write_root_velocity_to_sim=Mock(),
            write_joint_state_to_sim=Mock(),
        )
        env = SimpleNamespace(
            cyberstar=runtime,
            num_envs=8,
            device="cpu",
            _joint_ids=list(range(12)),
            _sync_cyberstar=Mock(),
            scene=SimpleNamespace(env_origins=torch.zeros(8, 3)),
        )

        class Scene(dict):
            env_origins = torch.zeros(8, 3)

        env.scene = Scene(robot=robot)
        reset, namespace = adapter_function("reset_robot")
        namespace["randomize_reset_pose_and_wrench"] = Mock()
        reset(env, ids)
        torch.testing.assert_close(actuator.coulomb_friction[ids, -1], torch.full((2,), 0.3))
        self.assertEqual(actuator.coulomb_friction[0].sum().item(), 0)
        torch.testing.assert_close(robot.write_joint_state_to_sim.call_args.args[1], torch.full((2, 12), 0.2))
        for direction, limit_index in ((-1, 0), (1, 1)):
            runtime.cfg.domain_rand.default_joint_angle_range = [direction * 4.0] * 2
            runtime.cfg.init_state.reset_dof_scale_range = [2.0, 2.0]
            runtime.cfg.init_state.reset_dof_velocity_range = [direction * 100.0] * 2
            reset(env, ids)
            positions, velocities = robot.write_joint_state_to_sim.call_args.args
            torch.testing.assert_close(positions, robot.data.soft_joint_pos_limits[ids, :, limit_index])
            torch.testing.assert_close(velocities, direction * robot.data.soft_joint_vel_limits[ids])
            torch.testing.assert_close(robot.write_joint_state_to_sim.call_args.kwargs["env_ids"], ids)
        runtime.cfg.env.test = True
        reset(env, ids)
        self.assertEqual(robot.write_joint_state_to_sim.call_args.args[1].sum().item(), 0)

    def test_mass_changes_scale_inertia_and_toggle_preserves_old_behavior(self):
        randomize, _ = adapter_function("_randomize_startup", "Dog03CyberstarEnv")
        for recompute in (False, True):
            config = flat_config()
            config["domain_rand"].update(
                randomize_link_mass=True,
                link_mass_scale_range=[2, 2],
                randomize_base_mass=True,
                added_mass_range=[0.5, 0.5],
                recompute_inertia=recompute,
            )
            runtime = CyberstarRuntime(4, "cpu", config=config)
            masses = torch.ones(4, 3)
            inertias = torch.eye(3).flatten().repeat(4, 3, 1)
            view = SimpleNamespace(
                get_masses=lambda: masses,
                get_coms=lambda: torch.zeros(4, 3, 7),
                get_inertias=lambda: inertias,
                get_material_properties=lambda: torch.ones(4, 1, 3),
                set_masses=Mock(),
                set_inertias=Mock(),
                set_coms=Mock(),
                set_material_properties=Mock(),
            )
            robot = SimpleNamespace(root_physx_view=view, find_bodies=lambda name: ([0], [name]))
            randomize(SimpleNamespace(cyberstar=runtime, num_envs=4, device="cpu", scene={"robot": robot}))
            expected = torch.eye(3).flatten().repeat(4, 3, 1)
            if recompute:
                expected *= masses.unsqueeze(-1)
            torch.testing.assert_close(inertias, expected)

    def test_configurable_delay_buffer_supports_more_than_one_step(self):
        config = source_config()["env"]
        config["domain_rand"]["control_delay_steps"] = [0, 4]
        runtime = CyberstarRuntime(4, "cpu", config=config)
        runtime.control_delay[:] = 4
        for step in range(1, 7):
            actual = runtime.process_actions(torch.full((4, 12), float(step)))
            torch.testing.assert_close(actual, torch.full((4, 12), float(max(0, step - 4))))
        runtime.reset(torch.tensor([1]))
        self.assertEqual(runtime.action_history[1].abs().sum().item(), 0)

    def test_atdog_physics_step_motor_delay_is_enabled_in_both_robust_stages(self):
        tree = ast.parse((ROOT / "source/robot_lab/robot_lab/assets/atdog.py").read_text())
        delays = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "DelayedPDActuatorCfg"
            ):
                parameters = {
                    keyword.arg: ast.literal_eval(keyword.value)
                    for keyword in node.keywords
                    if keyword.arg in ("min_delay", "max_delay")
                }
                delays.add((parameters["min_delay"], parameters["max_delay"]))
        self.assertEqual(delays, {(1, 2)})
        for config in (source_config()["env"], stairs_config()["env"]):
            self.assertTrue(config["domain_rand"]["randomize_motor_delay"])
            self.assertEqual(config["domain_rand"]["motor_delay_steps"], [1, 2])

    def test_motor_delay_sampling_and_partial_reset(self):
        runtime = CyberstarRuntime(64, "cpu")
        runtime.reset(torch.arange(64))
        self.assertTrue((runtime.motor_delay >= 1).all())
        self.assertTrue((runtime.motor_delay <= 2).all())
        self.assertTrue((runtime.motor_delay.float().std(dim=1) > 0).any())
        self.assertTrue((runtime.motor_delay.float().std(dim=0) > 0).all())
        previous = runtime.motor_delay.clone()
        runtime.motor_target_history[:] = 7
        runtime.motor_history_steps[:] = 2
        ids = torch.tensor([1, 3])
        runtime.reset(ids)
        keep = torch.ones(64, dtype=torch.bool)
        keep[ids] = False
        torch.testing.assert_close(runtime.motor_delay[keep], previous[keep])
        self.assertEqual(runtime.motor_target_history[ids].sum().item(), 0)
        self.assertEqual(runtime.motor_history_steps[ids].sum().item(), 0)
        self.assertTrue((runtime.motor_target_history[keep] == 7).all())
        self.assertTrue((runtime.motor_history_steps[keep] == 2).all())
        for test_mode, enabled in ((True, True), (False, False)):
            runtime.cfg.env.test = test_mode
            runtime.cfg.domain_rand.randomize_motor_delay = enabled
            runtime.reset(ids)
            self.assertEqual(runtime.motor_delay[ids].sum().item(), 0)
            torch.testing.assert_close(runtime.motor_delay[keep], previous[keep])
        flat = CyberstarRuntime(4, "cpu", config=flat_config())
        flat.reset(torch.arange(4))
        self.assertEqual(flat.motor_delay.sum().item(), 0)

    def test_motor_delay_delays_setpoints_not_pd_feedback_and_warm_starts(self):
        config = flat_config()
        config["domain_rand"].update(randomize_motor_delay=True, motor_delay_steps=[0, 4])
        runtime = CyberstarRuntime(4, "cpu", config=config)
        runtime.motor_delay[:] = torch.arange(12) % 5
        targets = torch.arange(12).float().expand(4, -1) / 100
        limits = torch.full_like(targets, 100)
        for step in range(7):
            positions = torch.full_like(targets, step / 100)
            velocities = torch.full_like(targets, step / 20)
            effective_steps = (step - runtime.motor_delay).clamp_min(0)
            expected = runtime.p_gains * (targets + effective_steps / 10 - positions) - runtime.d_gains * velocities
            actual = runtime.compute_motor_torques(targets + step / 10, positions, velocities, limits)
            torch.testing.assert_close(actual, expected)
        ids = torch.tensor([1, 3])
        runtime.reset(ids)
        previous_history = runtime.motor_target_history[[0, 2]].clone()
        positions = torch.zeros_like(targets)
        actual = runtime.compute_motor_torques(targets + 2, positions, positions, limits)
        torch.testing.assert_close(actual[ids], runtime.p_gains * (targets[ids] + 2))
        torch.testing.assert_close(runtime.motor_target_history[[0, 2], 1:], previous_history[:, :-1])

    def test_motor_delay_advances_on_each_physics_step_not_each_policy_step(self):
        config = flat_config()
        config["domain_rand"].update(randomize_motor_delay=True, motor_delay_steps=[1, 1])
        runtime = CyberstarRuntime(4, "cpu", config=config)
        runtime.reset(torch.arange(4))
        zeros = torch.zeros(4, 12)
        limits = torch.full_like(zeros, 100)
        runtime.compute_motor_torques(zeros, zeros, zeros, limits)
        runtime.process_actions(torch.ones_like(zeros))
        torch.testing.assert_close(runtime.compute_motor_torques(runtime.actions, zeros, zeros, limits), zeros)
        torch.testing.assert_close(
            runtime.compute_motor_torques(runtime.actions, zeros, zeros, limits), runtime.p_gains.expand_as(zeros)
        )
        self.assertTrue((runtime.action_history[:, 1] == 0).all())

    def test_motor_delay_range_validation(self):
        for delay_range in ((-1, 2), (2, 1), (0, 1.5), (True, 2)):
            with self.subTest(delay_range=delay_range):
                config = flat_config()
                config["domain_rand"]["motor_delay_steps"] = delay_range
                with self.assertRaisesRegex(ValueError, "motor_delay_steps"):
                    CyberstarRuntime(4, "cpu", config=config)

    def test_motor_delay_disabled_and_legacy_configs_leave_targets_unchanged(self):
        for test_mode, legacy in ((False, False), (True, False), (False, True)):
            with self.subTest(test_mode=test_mode, legacy=legacy):
                config = flat_config()
                config["env"]["test"] = test_mode
                if test_mode:
                    config["domain_rand"]["randomize_motor_delay"] = True
                if legacy:
                    del config["domain_rand"]["randomize_motor_delay"]
                    del config["domain_rand"]["motor_delay_steps"]
                runtime = CyberstarRuntime(4, "cpu", config=config)
                for _ in range(3):
                    targets = torch.rand(4, 12)
                    torch.testing.assert_close(runtime.delay_motor_targets(targets), targets)
                self.assertEqual(runtime.motor_delay.sum().item(), 0)
                self.assertEqual(runtime.motor_history_steps.sum().item(), 0)

    def test_playback_applies_nominal_materials_without_randomizing_dynamics(self):
        runtime = CyberstarRuntime(8, "cpu")
        runtime.cfg.env.test = True
        masses = torch.full((8, 3), 2.0)
        coms = torch.full((8, 3, 7), 0.01)
        inertias = torch.eye(3).flatten().repeat(8, 3, 1)
        materials = torch.full((8, 5, 3), 0.2)
        view = SimpleNamespace(
            get_masses=lambda: masses,
            get_coms=lambda: coms,
            get_inertias=lambda: inertias,
            get_material_properties=lambda: materials,
            set_masses=Mock(),
            set_coms=Mock(),
            set_inertias=Mock(),
            set_material_properties=Mock(),
        )
        robot = SimpleNamespace(root_physx_view=view, find_bodies=lambda name: ([0], [name]))
        randomize, _ = adapter_function("_randomize_startup", "Dog03CyberstarEnv")
        randomize(SimpleNamespace(cyberstar=runtime, num_envs=8, device="cpu", scene={"robot": robot}))
        torch.testing.assert_close(masses, torch.full((8, 3), 2.0))
        torch.testing.assert_close(coms, torch.full((8, 3, 7), 0.01))
        torch.testing.assert_close(inertias, torch.eye(3).flatten().repeat(8, 3, 1))
        torch.testing.assert_close(materials[:, :, :2], torch.ones(8, 5, 2))
        self.assertEqual(materials[:, :, 2].sum().item(), 0)
        torch.testing.assert_close(runtime.critic_friction, materials[:, :, 0].mean(dim=1, keepdim=True))
        view.set_material_properties.assert_called_once()

    def test_link_randomization_does_not_change_base_mass_or_com(self):
        config = flat_config()
        config["domain_rand"].update(
            randomize_link_mass=True,
            link_mass_scale_range=[2.0, 2.0],
            randomize_link_com=True,
            link_com_range={"x": [0.01, 0.01], "y": [0.02, 0.02], "z": [0.03, 0.03]},
        )
        runtime = CyberstarRuntime(4, "cpu", config=config)
        masses = torch.ones(4, 3)
        masses[:, 1] = 2.0
        coms = torch.zeros(4, 3, 7)
        coms[:, 1, :3] = torch.tensor([0.04, 0.05, 0.06])
        view = SimpleNamespace(
            get_masses=lambda: masses,
            get_coms=lambda: coms,
            get_inertias=lambda: torch.eye(3).flatten().repeat(4, 3, 1),
            get_material_properties=lambda: torch.ones(4, 1, 3),
            set_masses=Mock(),
            set_coms=Mock(),
            set_inertias=Mock(),
            set_material_properties=Mock(),
        )
        robot = SimpleNamespace(root_physx_view=view, find_bodies=lambda name: ([1], [name]))
        randomize, _ = adapter_function("_randomize_startup", "Dog03CyberstarEnv")
        randomize(SimpleNamespace(cyberstar=runtime, num_envs=4, device="cpu", scene={"robot": robot}))
        torch.testing.assert_close(masses, torch.full((4, 3), 2.0))
        torch.testing.assert_close(coms[:, 1, :3], torch.tensor([[0.04, 0.05, 0.06]]).expand(4, -1))
        torch.testing.assert_close(coms[:, [0, 2], :3], torch.tensor([0.01, 0.02, 0.03]).expand(4, 2, -1))
        self.assertEqual(runtime.critic_base_mass_offset.abs().sum().item(), 0)
        self.assertEqual(runtime.critic_base_com_offset.abs().sum().item(), 0)

    def test_default_push_intervals_vary_in_both_randomized_stages(self):
        for runtime_type, config, base_steps in (
            (CyberstarRuntime, source_config()["env"], 500),
            (CyberstarStairsRuntime, stairs_config()["env"], 750),
        ):
            with self.subTest(runtime_type=runtime_type.__name__):
                runtime = runtime_type(64, "cpu", config=config)
                self.assertTrue((runtime.next_push_steps >= base_steps).all())
                self.assertTrue((runtime.next_push_steps <= base_steps + 250).all())
                self.assertGreater(runtime.next_push_steps.unique().numel(), 1)

    def test_fixed_push_intervals_round_up_and_never_become_zero(self):
        for seconds, expected_steps in ((0.01, 1), (0.04, 2)):
            config = flat_config()
            config["domain_rand"]["push_interval_s"] = seconds
            runtime = CyberstarRuntime(4, "cpu", dt=0.03, config=config)
            self.assertEqual(runtime.cfg.domain_rand.push_interval, expected_steps)
            runtime.episode_length_buf[:] = expected_steps
            torch.testing.assert_close(runtime._scheduled_push_env_ids(expected_steps), torch.arange(4))

    def test_runtime_rejects_invalid_control_timesteps(self):
        for timestep in (0.0, -0.02, float("nan"), float("inf")):
            with self.subTest(timestep=timestep), self.assertRaises(ValueError):
                CyberstarRuntime(4, "cpu", dt=timestep)

    def test_startup_material_friction_is_physically_consistent(self):
        config = flat_config()
        config["domain_rand"].update(
            randomize_friction=True,
            friction_range=[0.6, 0.6],
            randomize_dynamic_friction=True,
            dynamic_friction_range=[0.8, 0.8],
        )
        runtime = CyberstarRuntime(4, "cpu", config=config)
        materials = torch.ones(4, 2, 3)
        view = SimpleNamespace(
            get_masses=lambda: torch.ones(4, 3),
            get_coms=lambda: torch.zeros(4, 3, 7),
            get_inertias=lambda: torch.eye(3).flatten().repeat(4, 3, 1),
            get_material_properties=lambda: materials,
            set_masses=Mock(),
            set_coms=Mock(),
            set_inertias=Mock(),
            set_material_properties=Mock(),
        )
        robot = SimpleNamespace(root_physx_view=view, find_bodies=lambda name: ([0], [name]))
        randomize, _ = adapter_function("_randomize_startup", "Dog03CyberstarEnv")
        randomize(SimpleNamespace(cyberstar=runtime, num_envs=4, device="cpu", scene={"robot": robot}))
        torch.testing.assert_close(materials[:, :, 0], torch.full((4, 2), 0.6))
        torch.testing.assert_close(materials[:, :, 1], materials[:, :, 0])
        torch.testing.assert_close(runtime.critic_friction[:, 0], materials[:, 0, 0])

    def test_material_shape_scope_and_shared_bucket_budget(self):
        randomize, _ = adapter_function("_randomize_startup", "Dog03CyberstarEnv")
        for per_shape in (False, True):
            with self.subTest(per_shape=per_shape):
                config = source_config()["env"]
                config["domain_rand"]["randomize_material_per_shape"] = per_shape
                runtime = CyberstarRuntime(128, "cpu", config=config)
                materials = torch.ones(128, 17, 3)
                view = SimpleNamespace(
                    get_masses=lambda: torch.ones(128, 17),
                    get_coms=lambda: torch.zeros(128, 17, 7),
                    get_inertias=lambda: torch.eye(3).flatten().repeat(128, 17, 1),
                    get_material_properties=lambda: materials,
                    set_masses=Mock(),
                    set_coms=Mock(),
                    set_inertias=Mock(),
                    set_material_properties=Mock(),
                )
                robot = SimpleNamespace(root_physx_view=view, find_bodies=lambda name: ([0], [name]))
                randomize(SimpleNamespace(cyberstar=runtime, num_envs=128, device="cpu", scene={"robot": robot}))
                self.assertLessEqual(torch.unique(materials.reshape(-1, 3), dim=0).shape[0], 64)
                if per_shape:
                    self.assertTrue((materials[:, :, 0].std(dim=1) > 0).all())
                else:
                    torch.testing.assert_close(materials, materials[:, :1].expand_as(materials))
                self.assertTrue((materials[:, :, 0] >= 0.6).all())
                self.assertTrue((materials[:, :, 0] <= 1.3).all())
                self.assertTrue((materials[:, :, 1] <= materials[:, :, 0]).all())
                self.assertTrue((materials[:, :, 2] > 0).any())
                torch.testing.assert_close(runtime.critic_friction, materials[:, :, 0].mean(dim=1, keepdim=True))
                view.set_material_properties.assert_called_once()

    def test_per_shape_material_switch_does_not_enable_disabled_coefficients(self):
        config = flat_config()
        config["domain_rand"].update(randomize_material_per_shape=True, randomize_restitution=True)
        runtime = CyberstarRuntime(16, "cpu", config=config)
        materials = torch.ones(16, 17, 3)
        view = SimpleNamespace(
            get_masses=lambda: torch.ones(16, 17),
            get_coms=lambda: torch.zeros(16, 17, 7),
            get_inertias=lambda: torch.eye(3).flatten().repeat(16, 17, 1),
            get_material_properties=lambda: materials,
            set_masses=Mock(),
            set_coms=Mock(),
            set_inertias=Mock(),
            set_material_properties=Mock(),
        )
        robot = SimpleNamespace(root_physx_view=view, find_bodies=lambda name: ([0], [name]))
        randomize, _ = adapter_function("_randomize_startup", "Dog03CyberstarEnv")
        randomize(SimpleNamespace(cyberstar=runtime, num_envs=16, device="cpu", scene={"robot": robot}))
        torch.testing.assert_close(materials[:, :, :2], torch.ones(16, 17, 2))
        self.assertTrue((materials[:, :, 2].std(dim=1) > 0).all())
        self.assertTrue((materials[:, :, 2] >= 0).all())
        self.assertTrue((materials[:, :, 2] <= 0.5).all())
        torch.testing.assert_close(runtime.critic_friction, torch.ones(16, 1))

    def test_push_safety_uses_ground_clearance_not_world_height(self):
        runtime = CyberstarRuntime(4, "cpu")
        runtime.episode_length_buf[:] = runtime.next_push_steps
        runtime.root_states[:, 2] = torch.tensor([2.4, 2.1, 2.6, 2.4])
        runtime.root_states[3, 9] = 1.0
        runtime._height_query = lambda positions: torch.full_like(positions[..., 0], 2.0)
        runtime._velocity_writer = Mock()
        runtime._push_robots()
        runtime._velocity_writer.assert_called_once()
        torch.testing.assert_close(runtime._velocity_writer.call_args.args[0], torch.tensor([0]))
        self.assertEqual(runtime.root_states[1:, 7:9].abs().sum().item(), 0)

    def test_push_height_gate_queries_only_due_environments(self):
        runtime = CyberstarRuntime(64, "cpu")
        ids = torch.tensor([2, 5])
        runtime.episode_length_buf[ids] = runtime.next_push_steps[ids]
        runtime._height_query = Mock(side_effect=lambda positions: torch.zeros_like(positions[..., 0]))
        runtime._velocity_writer = Mock()
        runtime._push_robots()
        runtime._height_query.assert_called_once()
        self.assertEqual(runtime._height_query.call_args.args[0].shape, (2, 5, 3))
        torch.testing.assert_close(runtime._velocity_writer.call_args.args[0], ids)

    def test_json_timing_and_action_scale_reach_the_simulator_config(self):
        config = source_config()["env"]
        config["sim"]["dt"] = 0.01
        config["control"].update(decimation=3, action_scale=0.15)
        config["env"]["episode_length_s"] = 12.0
        config["init_state"].update(rot=[0.0, 0.0, 0.6, 0.8], lin_vel=[0.1, 0.2, 0.3], ang_vel=[0.4, 0.5, 0.6])
        tree = ast.parse((PACKAGE / "env_cfg.py").read_text())
        definition = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Dog03CyberstarDomainRandEnvCfg"
        )
        method = next(
            node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
        )
        namespace = {
            name: Mock()
            for name in (
                "CollisionPropertiesCfg",
                "spawn_dog03",
                "lab",
                "TerrainGeneratorCfg",
                "MeshPlaneTerrainCfg",
                "CyberstarDiscreteTerrainCfg",
                "ObservationsCfg",
                "CommandsCfg",
                "EventsCfg",
                "TerminationsCfg",
                "RewardTermCfg",
            )
        }
        namespace["super"] = lambda: SimpleNamespace(__post_init__=Mock())
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(PACKAGE / "env_cfg.py"), "exec"), namespace)
        env = Mock(cyberstar_config=config)
        env.scene.robot.actuators = {}
        namespace["__post_init__"](env)
        self.assertEqual(env.sim.dt, 0.01)
        self.assertEqual(env.decimation, 3)
        self.assertEqual(env.sim.render_interval, 3)
        self.assertEqual(env.scene.contact_forces.update_period, 0.01)
        self.assertEqual(env.scene.height_scanner_base.update_period, 0.03)
        self.assertEqual(env.episode_length_s, 12.0)
        self.assertEqual(env.actions.joint_pos.scale, 0.15)
        self.assertEqual(env.scene.terrain.physics_material.restitution_combine_mode, "multiply")
        self.assertEqual(env.scene.terrain.physics_material.restitution, 1.0)
        self.assertEqual(env.scene.robot.init_state.rot, (0.8, 0.0, 0.0, 0.6))
        self.assertEqual(env.scene.robot.init_state.lin_vel, (0.1, 0.2, 0.3))
        self.assertEqual(env.scene.robot.init_state.ang_vel, (0.4, 0.5, 0.6))

    def test_playback_flag_disables_noise_and_pushes_without_cli_overrides(self):
        load, namespace = adapter_function("load_managers", "Dog03CyberstarEnv")
        namespace["super"] = lambda: SimpleNamespace(load_managers=Mock())
        namespace["Ji22RewardManager"] = Mock()
        for playback, config_test_mode in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(playback=playback, config_test_mode=config_test_mode):
                config = source_config()["env"]
                config["env"]["test"] = config_test_mode
                joint_names = [
                    f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")
                ]
                body_names = config["asset"]["body_names"]
                feet = [body_names.index(f"{leg}_FOOT") for leg in ("FL", "FR", "RL", "RR")]
                base = body_names.index("base_link")
                penalty = [index for index, name in enumerate(body_names) if name.endswith(("_thigh", "_calf"))]
                sensor = SimpleNamespace(
                    find_bodies=Mock(
                        side_effect=(
                            (list(range(len(body_names))), body_names),
                            (feet, []),
                            ([base], ["base_link"]),
                            (penalty, []),
                        )
                    )
                )
                robot = SimpleNamespace(
                    body_names=body_names,
                    find_joints=Mock(return_value=(list(range(12)), joint_names)),
                    find_bodies=Mock(return_value=(feet, [])),
                    actuators={
                        "legs": SimpleNamespace(joint_names=joint_names, effort_limit=torch.full((4, 12), 30.0))
                    },
                    data=SimpleNamespace(soft_joint_pos_limits=torch.tensor(config["asset"]["joint_limits"])[None]),
                )
                env = SimpleNamespace(
                    num_envs=4,
                    device="cpu",
                    step_dt=0.02,
                    cfg=SimpleNamespace(cyberstar_config=config, cyberstar_play=playback, rewards={}),
                    runtime_type=CyberstarRuntime,
                    scene={"robot": robot, "contact_forces": sensor},
                    _write_cyberstar_velocity=Mock(),
                    _terrain_height_at=Mock(),
                    _configure_cyberstar_terrain=Mock(),
                    _sync_cyberstar=Mock(),
                    _randomize_startup=Mock(),
                )
                load(env)
                self.assertEqual(env.cyberstar.cfg.env.test, playback or config_test_mode)
                self.assertEqual(env.cyberstar.add_noise, not (playback or config_test_mode))
                self.assertEqual(env.cyberstar.cfg.domain_rand.push_robots, not (playback or config_test_mode))
                self.assertTrue(config["noise"]["add_noise"])
                self.assertTrue(config["domain_rand"]["push_robots"])

    def test_randomized_push_schedule_bounds_resampling_and_partial_reset(self):
        config = source_config()["env"]
        config["domain_rand"]["push_interval_offset_range_s"] = [0, 5]
        runtime = CyberstarRuntime(64, "cpu", config=config)
        self.assertTrue((runtime.next_push_steps >= 500).all())
        self.assertTrue((runtime.next_push_steps <= 750).all())
        self.assertGreater(runtime.next_push_steps.unique().numel(), 1)
        runtime.episode_length_buf[:] = 499
        self.assertEqual(runtime._scheduled_push_env_ids(500).numel(), 0)
        runtime.episode_length_buf[3] = runtime.next_push_steps[3]
        previous = runtime.next_push_steps.clone()
        torch.testing.assert_close(runtime._scheduled_push_env_ids(500), torch.tensor([3]))
        self.assertGreaterEqual(runtime.next_push_steps[3] - previous[3], 500)
        self.assertLessEqual(runtime.next_push_steps[3] - previous[3], 750)
        runtime.reset(torch.tensor([3]))
        self.assertEqual(runtime.episode_length_buf[3], 0)
        torch.testing.assert_close(runtime.next_push_steps[0], previous[0])
        runtime.cfg.domain_rand.randomize_push_interval = False
        runtime.episode_length_buf[:] = 500
        torch.testing.assert_close(runtime._scheduled_push_env_ids(500), torch.arange(64))

    def test_root_reset_ranges_are_per_axis_and_preserve_nominal_velocity(self):
        for runtime in (CyberstarRuntime(8, "cpu"), CyberstarStairsRuntime(8, "cpu")):
            runtime.cfg.init_state.reset_root_position_range = SimpleNamespace(x=[0.2, 0.2], y=[-0.3, -0.3])
            runtime.cfg.init_state.reset_root_velocity_range = SimpleNamespace(x=[0.1, 0.1], yaw=[-0.2, -0.2])
            original = torch.zeros(3, 13)
            original[:, :3] = torch.tensor([2.0, 3.0, 0.4])
            original[:, 7:13] = 0.5
            state = original.clone()
            runtime.randomize_reset_root_state(state)
            torch.testing.assert_close(state[:, :3], original[:, :3] + torch.tensor([0.2, -0.3, 0.0]))
            torch.testing.assert_close(state[:, 7:13], original[:, 7:13] + torch.tensor([0.1, 0, 0, 0, 0, -0.2]))
            stair_state = original.clone()
            runtime.randomize_reset_root_state(stair_state, include_xy=False)
            torch.testing.assert_close(stair_state[:, :3], original[:, :3])
            torch.testing.assert_close(stair_state[:, 7:13], state[:, 7:13])
            runtime.cfg.env.test = True
            runtime.randomize_reset_root_state(original)
            torch.testing.assert_close(original[:, :3], torch.tensor([[2.0, 3.0, 0.4]]).repeat(3, 1))
            torch.testing.assert_close(original[:, 7:13], torch.full((3, 6), 0.5))

    def test_legacy_root_reset_ranges_remain_supported(self):
        runtime = CyberstarRuntime(64, "cpu")
        del runtime.cfg.init_state.reset_root_position_range
        runtime.cfg.init_state.reset_root_velocity_range = 0.1
        state = torch.zeros(64, 13)
        runtime.randomize_reset_root_state(state)
        self.assertTrue((state[:, :2].abs() <= 1).all())
        self.assertTrue((state[:, 7:13].abs() <= 0.1).all())
        self.assertGreater(state[:, :2].abs().sum().item(), 0)
        self.assertGreater(state[:, 7:13].abs().sum().item(), 0)
        torch.testing.assert_close(state[:, 2:7], torch.zeros(64, 5))

    def test_invalid_root_reset_ranges_are_rejected(self):
        for limits in (-0.1, float("inf"), True, {"x": [1, -1]}, {"yaw": [0, float("nan")]}, {"typo": [0, 1]}):
            with self.subTest(limits=limits), self.assertRaisesRegex(ValueError, "reset_root_velocity_range"):
                runtime = CyberstarRuntime(4, "cpu")
                runtime.cfg.init_state.reset_root_velocity_range = limits
                runtime.randomize_reset_root_state(torch.zeros(2, 13))

    def test_root_reset_adapter_uses_axis_ranges_and_skips_stair_xy(self):
        reset, namespace = adapter_function("reset_robot")
        namespace["randomize_reset_pose_and_wrench"] = Mock()
        runtime = CyberstarRuntime(8, "cpu", config=flat_config())
        runtime.cfg.init_state.reset_root_position_range = SimpleNamespace(x=[0.2, 0.2], y=[-0.3, -0.3])
        runtime.cfg.init_state.reset_root_velocity_range = SimpleNamespace(x=[0.1, 0.1], yaw=[-0.2, -0.2])
        data = SimpleNamespace(
            default_root_state=torch.zeros(8, 13),
            default_joint_pos=runtime.default_dof_pos.repeat(8, 1),
            soft_joint_pos_limits=runtime.dof_pos_limits.unsqueeze(0).repeat(8, 1, 1),
            soft_joint_vel_limits=torch.full((8, 12), 30.0),
        )
        data.default_root_state[:, 7:13] = 0.5
        robot = SimpleNamespace(
            data=data,
            actuators={},
            write_root_pose_to_sim=Mock(),
            write_root_velocity_to_sim=Mock(),
            write_joint_state_to_sim=Mock(),
        )

        class Scene(dict):
            env_origins = torch.ones(8, 3)

        env = SimpleNamespace(
            cyberstar=runtime,
            num_envs=8,
            device="cpu",
            _joint_ids=list(range(12)),
            scene=Scene(robot=robot),
            _sync_cyberstar=Mock(),
        )
        ids = torch.tensor([1, 4])
        reset(env, ids)
        torch.testing.assert_close(
            robot.write_root_pose_to_sim.call_args.args[0][:, :2], torch.tensor([[1.2, 0.7]]).repeat(2, 1)
        )
        expected_velocity = torch.tensor([[0.6, 0.5, 0.5, 0.5, 0.5, 0.3]]).repeat(2, 1)
        torch.testing.assert_close(robot.write_root_velocity_to_sim.call_args.args[0], expected_velocity)
        env._place_stair_reset = Mock()
        reset(env, ids)
        torch.testing.assert_close(env._place_stair_reset.call_args.args[0][:, :2], torch.ones(2, 2))

    def test_body_state_velocities_match_link_positions_not_randomized_com(self):
        synchronize, namespace = adapter_function("_sync_cyberstar", "Dog03CyberstarEnv")
        namespace["euler_xyz_from_quat"] = lambda quaternion: (torch.zeros(8),) * 3
        runtime = CyberstarRuntime(8, "cpu")
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(8, 1)
        data = SimpleNamespace(
            root_pos_w=torch.zeros(8, 3),
            root_quat_w=quaternion,
            root_com_vel_w=torch.zeros(8, 6),
            root_lin_vel_b=torch.zeros(8, 3),
            root_ang_vel_b=torch.zeros(8, 3),
            projected_gravity_b=runtime.projected_gravity.clone(),
            joint_pos=runtime.dof_pos.clone(),
            joint_vel=runtime.dof_vel.clone(),
            body_link_pos_w=torch.zeros(8, 5, 3),
            body_link_quat_w=quaternion[:, None].repeat(1, 5, 1),
            body_link_lin_vel_w=torch.full((8, 5, 3), 2.0),
            body_link_ang_vel_w=torch.full((8, 5, 3), 3.0),
            body_com_lin_vel_w=torch.full((8, 5, 3), 9.0),
            body_com_ang_vel_w=torch.full((8, 5, 3), 3.0),
        )
        env = SimpleNamespace(
            cyberstar=runtime,
            _joint_ids=list(range(12)),
            _foot_ids=[1, 2, 3, 4],
            _sensor_ids=list(range(5)),
            _foot_sensor_ids=[1, 2, 3, 4],
            episode_length_buf=torch.zeros(8),
            scene={
                "robot": SimpleNamespace(data=data),
                "contact_forces": SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros(8, 5, 3))),
            },
        )
        synchronize(env)
        torch.testing.assert_close(runtime.rigid_body_states[:, :, 7:10], torch.full((8, 9, 3), 2.0))
        torch.testing.assert_close(runtime.rigid_body_states[:, :, 10:13], torch.full((8, 9, 3), 3.0))

    def test_push_writer_refreshes_body_frame_velocity_observations(self):
        write, _ = adapter_function("_write_cyberstar_velocity", "Dog03CyberstarEnv")
        runtime = CyberstarRuntime(8, "cpu")
        ids = torch.tensor([1, 4])
        runtime.root_states[ids, 7:13] = 0.5
        robot = SimpleNamespace(
            write_root_velocity_to_sim=Mock(),
            data=SimpleNamespace(root_lin_vel_b=torch.ones(8, 3), root_ang_vel_b=torch.full((8, 3), 2.0)),
        )
        write(SimpleNamespace(cyberstar=runtime, scene={"robot": robot}), ids)
        torch.testing.assert_close(robot.write_root_velocity_to_sim.call_args.args[0], torch.full((2, 6), 0.5))
        torch.testing.assert_close(runtime.base_lin_vel[ids], torch.ones(2, 3))
        torch.testing.assert_close(runtime.base_ang_vel[ids], torch.full((2, 3), 2.0))
        self.assertEqual(runtime.base_lin_vel[0].count_nonzero(), 0)
        self.assertEqual(runtime.base_ang_vel[0].count_nonzero(), 0)

    def test_him_episode_staggering_reschedules_pushes_from_the_new_lengths(self):
        path = ROOT / "source/robot_lab/robot_lab/him/him_on_policy_runner.py"
        tree = ast.parse(path.read_text())
        definition = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HIMOnPolicyRunner"
        )
        method = next(node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == "learn")
        namespace = {"torch": torch, "deque": deque}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        runtime = CyberstarRuntime(64, "cpu")
        env = SimpleNamespace(
            num_envs=64,
            max_episode_length=1000,
            episode_length_buf=runtime.episode_length_buf,
            cyberstar=runtime,
            get_observations=lambda: torch.zeros(64, 282),
        )
        runner = SimpleNamespace(
            env=env,
            log_dir=None,
            device="cpu",
            current_learning_iteration=0,
            alg=SimpleNamespace(actor_critic=SimpleNamespace(train=Mock())),
            _get_critic_observations=lambda: torch.zeros(64, 491),
        )
        namespace["learn"](runner, 0, init_at_random_ep_len=True)
        self.assertIs(runtime.episode_length_buf, env.episode_length_buf)
        delays = runtime.next_push_steps - env.episode_length_buf
        self.assertTrue((delays >= 500).all())
        self.assertTrue((delays <= 750).all())
        self.assertEqual(runtime._scheduled_push_env_ids(500).numel(), 0)

    def test_atdog_active_event_inventory_is_covered(self):
        tree = ast.parse((VELOCITY / "velocity_env_cfg.py").read_text())
        event_cfg = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "EventCfg")
        active = {
            node.targets[0].id: node.value
            for node in event_cfg.body
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
        }
        coverage = {
            "randomize_rigid_body_material": (
                "randomize_friction",
                "randomize_dynamic_friction",
                "randomize_restitution",
                "randomize_material_per_shape",
            ),
            "randomize_rigid_body_mass_base": ("randomize_base_mass", "recompute_inertia"),
            "randomize_rigid_body_mass_others": ("randomize_link_mass", "recompute_inertia"),
            "randomize_com_positions": ("randomize_com",),
            "randomize_apply_external_force_torque": ("randomize_external_force_torque",),
            "randomize_reset_joints": ("reset_dof_scale_range", "reset_dof_velocity_range"),
            "randomize_actuator_gains": ("randomize_pd_gains", "kp_scale_range", "kd_scale_range"),
            "randomize_reset_base": (
                "randomize_reset_pose",
                "reset_root_position_range",
                "reset_root_velocity_range",
            ),
            "randomize_push_robot": ("push_robots", "randomize_push_interval"),
        }
        self.assertEqual(set(active), set(coverage))
        for filename in ("config.json", "stairs_config.json"):
            config = json.loads((PACKAGE / filename).read_text())["env"]
            keys = config["domain_rand"].keys() | config["init_state"].keys()
            for event, fields in coverage.items():
                self.assertTrue(set(fields) <= keys, (filename, event))
        paths = [
            path
            for family in ("quadruped/atdog", "wheeled/atdog", "wheeled/atdog_arm")
            for path in (VELOCITY / "config" / family).rglob("*env_cfg.py")
        ]
        for path in paths:
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
                    if ast.unparse(node.value) == "self.events":
                        self.assertIn(node.attr, coverage, str(path))


if __name__ == "__main__":
    unittest.main()
