"""CPU regression check for Dog2's flat-ground foot slip reward."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REWARDS_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/atdog/dog2/rewards.py"
)
reward_node = next(
    node for node in ast.parse(REWARDS_PATH.read_text()).body
    if isinstance(node, ast.FunctionDef) and node.name == "feet_slide_world"
)
namespace = {"torch": torch, "SceneEntityCfg": object}
exec(compile(ast.Module(body=[reward_node], type_ignores=[]), str(REWARDS_PATH), "exec"), namespace)
feet_slide_world = namespace["feet_slide_world"]


class Dog2FlatFootSlideTest(unittest.TestCase):
    def test_stationary_contact_foot_is_not_penalized_when_base_moves(self):
        class Scene(dict):
            pass

        foot_velocities = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]])
        asset = SimpleNamespace(
            data=SimpleNamespace(
                root_lin_vel_w=torch.tensor([[0.4, 0.0, 0.0]]),
                body_lin_vel_w=foot_velocities,
                projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
            )
        )
        sensor = SimpleNamespace(
            data=SimpleNamespace(net_forces_w_history=torch.tensor([[[[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]]]]))
        )
        env = SimpleNamespace(scene=Scene(robot=asset, contacts=sensor))
        env.scene.sensors = {"contacts": sensor}
        sensor_cfg = SimpleNamespace(name="contacts", body_ids=[0, 1])
        asset_cfg = SimpleNamespace(name="robot", body_ids=[0, 1])

        torch.testing.assert_close(feet_slide_world(env, sensor_cfg, asset_cfg), torch.tensor([0.1]))


if __name__ == "__main__":
    unittest.main()
