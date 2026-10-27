"""CPU checks for local 03dog rewards and the retained CSDog1 network contract."""

import ast
import copy
import importlib.util
import inspect
import torch
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped"
DOG = TASKS / "dog03"


class SceneEntityCfg(SimpleNamespace):
    def __init__(self, name="robot", **kwargs):
        super().__init__(name=name, joint_ids=slice(None), body_ids=slice(None), **kwargs)


class RewardTermCfg(SimpleNamespace):
    def __init__(self, func=None, weight=0.0, params=None):
        super().__init__(func=func, weight=weight, params={} if params is None else params)


def configclass(cls):
    """Copy config defaults per instance without loading simulator dependencies."""
    if not hasattr(cls, "__post_init__"):
        cls.__post_init__ = lambda self: None

    def init(self, **kwargs):
        for base in reversed(type(self).__mro__):
            for name, value in vars(base).items():
                if not name.startswith("_") and not callable(value):
                    setattr(self, name, copy.deepcopy(value))
        self.__dict__.update(kwargs)
        self.__post_init__()

    cls.__init__ = init
    return cls


def load_rewards():
    modules = {
        name: ModuleType(name)
        for name in (
            "isaaclab",
            "isaaclab.utils",
            "isaaclab.utils.math",
            "isaaclab.assets",
            "isaaclab.managers",
            "isaaclab.sensors",
            "dog03_cpu",
        )
    }
    modules["dog03_cpu"].__path__ = [str(DOG)]
    modules["isaaclab"].utils = modules["isaaclab.utils"]
    modules["isaaclab.utils"].math = modules["isaaclab.utils.math"]
    modules["isaaclab.utils"].configclass = configclass
    for name in ("quat_apply_inverse", "yaw_quat"):
        setattr(modules["isaaclab.utils.math"], name, lambda *args: None)
    for name in ("Articulation", "RigidObject"):
        setattr(modules["isaaclab.assets"], name, object)
    for name in ("ContactSensor", "RayCaster"):
        setattr(modules["isaaclab.sensors"], name, object)
    managers = modules["isaaclab.managers"]
    managers.ManagerTermBase = object
    managers.RewardTermCfg = RewardTermCfg
    managers.SceneEntityCfg = SceneEntityCfg
    with patch.dict("sys.modules", modules):
        loaded = []
        for name in ("rewards", "rewards_cfg"):
            spec = importlib.util.spec_from_file_location(f"dog03_cpu.{name}", DOG / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            modules[f"dog03_cpu.{name}"] = module
            with patch.dict("sys.modules", modules):
                spec.loader.exec_module(module)
            setattr(modules["dog03_cpu"], name, module)
            loaded.append(module)
    return loaded


rewards, configs = load_rewards()


class Dog03IndependenceTest(unittest.TestCase):
    def test_registered_tasks_follow_csdog1_entry_points(self):
        def registrations(robot):
            entries = {}
            gym = ModuleType("gymnasium")
            gym.register = lambda **entry: entries.update({entry["id"]: entry})
            spec = importlib.util.spec_from_file_location(
                "dog03_registry_cpu",
                TASKS / robot / "__init__.py",
                submodule_search_locations=[str(TASKS / robot)],
            )
            module = importlib.util.module_from_spec(spec)
            with patch.dict("sys.modules", {"gymnasium": gym, spec.name: module}):
                spec.loader.exec_module(module)
            return entries

        source = registrations("csdog1")
        local = registrations("dog03")
        self.assertEqual(len(local), 5)
        for task, entry in source.items():
            task = task.replace("CSDog1", "03Dog")
            expected = {
                key: value.replace("CSDog1", "Dog03") if isinstance(value, str) else value
                for key, value in entry.items()
                if key != "kwargs"
            }
            expected["id"] = task
            expected["kwargs"] = {key: value.replace("CSDog1", "Dog03") for key, value in entry["kwargs"].items()}
            if task == "RobotLab-Isaac-Velocity-Flat-03Dog-282-v0":
                package = "dog03_registry_cpu.cyberstar"
                expected["entry_point"] = f"{package}.lab:Dog03CyberstarEnv"
                expected["kwargs"]["env_cfg_entry_point"] = f"{package}.env_cfg:Dog03CyberstarFlatEnvCfg"
            self.assertEqual(local[task], expected)

    def test_copied_command_and_reward_formulas_match_csdog1(self):
        class Normalize(ast.NodeTransformer):
            def visit_Return(self, node):
                if isinstance(node.value, ast.Constant) and node.value.value is None:
                    node.value = None
                return self.generic_visit(node)

            def visit_FunctionDef(self, node):
                self.generic_visit(node)
                if node.body and isinstance(node.body[-1], ast.Return) and node.body[-1].value is None:
                    node.body.pop()
                return node

        for filename in ("core.py", "stairs_core.py"):
            classes = []
            for robot in ("csdog1", "dog03"):
                text = (TASKS / robot / "cyberstar" / filename).read_text().replace("Csdog1", "Dog03")
                tree = Normalize().visit(ast.parse(text))
                classes.append([ast.dump(n) for n in tree.body if isinstance(n, ast.ClassDef)])
            self.assertEqual(classes[0], classes[1], filename)

    def test_no_csdog1_task_dependencies(self):
        for path in DOG.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn("csdog1", node.module or "", str(path))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn("csdog1", alias.name, str(path))

    def test_all_reward_terms_use_local_implementations(self):
        for cfg_type in (configs.Dog03RewardsCfg, configs.Dog03DomainRandRewardsCfg):
            cfg = cfg_type()
            terms = [v for v in vars(cfg).values() if isinstance(v, RewardTermCfg)]
            self.assertGreater(len(terms), 30)
            for term in terms:
                self.assertEqual(term.func.__module__, rewards.__name__)
                target = term.func.__call__ if inspect.isclass(term.func) else term.func
                inspect.signature(target).bind(*([None, None] if inspect.isclass(term.func) else [None]), **term.params)

    def test_stage_weights_and_instances_are_independent(self):
        flat = configs.Dog03RewardsCfg()
        robust = configs.Dog03DomainRandRewardsCfg()
        self.assertEqual(flat.track_lin_vel_xy_exp.weight, 3.0)
        self.assertEqual(robust.track_lin_vel_xy_exp.weight, 7.5)
        self.assertEqual(flat.action_rate_l2.weight, -0.2)
        self.assertEqual(robust.action_rate_l2.weight, -0.015)
        self.assertEqual(flat.joint_mirror.weight, 0.0)
        self.assertEqual(flat.base_height_l2.params["target_height"], 0.32)
        self.assertEqual(flat.feet_height_body.params["target_height"], -0.26)
        self.assertEqual(flat.feet_distance_y_exp.params["stance_width"], 0.38)
        self.assertEqual(
            flat.feet_gait.params["synced_feet_pair_names"], (("FL_FOOT", "RR_FOOT"), ("FR_FOOT", "RL_FOOT"))
        )
        robust.feet_height.params["asset_cfg"].body_names.append("changed")
        self.assertEqual(flat.feet_height.params["asset_cfg"].body_names, [".*_FOOT"])
        self.assertEqual(configs.Dog03RewardsCfg().feet_height.params["asset_cfg"].body_names, [".*_FOOT"])

    def test_tracking_and_standstill_reward_behavior(self):
        command = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        data = SimpleNamespace(
            root_lin_vel_b=torch.zeros(3, 3),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]),
            joint_pos=torch.ones(3, 12),
            default_joint_pos=torch.zeros(3, 12),
        )
        env = SimpleNamespace(
            scene={"robot": SimpleNamespace(data=data)}, command_manager=SimpleNamespace(get_command=lambda _: command)
        )
        torch.testing.assert_close(
            rewards.track_lin_vel_xy_exp(env, 0.5, "base_velocity"),
            torch.tensor([1.0, torch.exp(torch.tensor(-4.0)).item(), 0.0]),
        )
        torch.testing.assert_close(rewards.stand_still(env, "base_velocity"), torch.tensor([12.0, 0.0, 0.0]))

    def test_network_definitions_match_csdog1(self):
        for filename, class_names in {
            "rsl_rl_him_cfg.py": ("HIMActorCriticCfg", "HIMPPOAlgorithmCfg", "Flat282HIMRunnerCfg"),
            "rsl_rl_ppo_cfg.py": (
                "RoughPPORunnerCfg",
                "FlatPPORunnerCfg",
                "Flat282PPORunnerCfg",
                "Flat282DomainRandPPORunnerCfg",
            ),
            "cusrl_ppo_cfg.py": (
                "RoughTrainerCfg",
                "FlatTrainerCfg",
                "Flat282TrainerCfg",
                "Flat282DomainRandTrainerCfg",
            ),
        }.items():
            # Compare executable model definitions, ignoring robot names and docstrings.
            trees = []
            for robot in ("csdog1", "dog03"):
                source = (TASKS / robot / "agents" / filename).read_text()
                source = source.replace("CSDog1", "Dog03").replace("csdog1_", "dog03_")
                tree = ast.parse(source)
                for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
                    cls.body = [
                        node
                        for node in cls.body
                        if not (
                            isinstance(node, ast.Expr)
                            and isinstance(node.value, ast.Constant)
                            and isinstance(node.value.value, str)
                        )
                    ]
                trees.append({node.name: node for node in tree.body if isinstance(node, ast.ClassDef)})
            for suffix in class_names:
                name = "Dog03" + suffix
                self.assertEqual(ast.dump(trees[0][name]), ast.dump(trees[1][name]), (filename, name))


if __name__ == "__main__":
    unittest.main()
