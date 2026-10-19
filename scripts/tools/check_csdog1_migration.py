"""Exercise the migrated task, recovery, source checkpoints, HIM updates, and exports in Isaac Sim."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="RobotLab-Isaac-Velocity-Flat-CSDog1-282-DomainRand-v0",
                    choices=["RobotLab-Isaac-Velocity-Flat-CSDog1-282-DomainRand-v0", "RobotLab-Isaac-Velocity-Stairs-CSDog1-282-v0"])
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--output_dir", default="outputs/csdog1_migration")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import numpy as np
import omni.usd
import torch
from pxr import Gf, Usd, UsdPhysics
from isaaclab_tasks.utils import parse_env_cfg

import robot_lab.tasks
from robot_lab.him import (
    HIMOnPolicyRunner,
    HIMVecEnvWrapper,
    export_him_deployment_metadata,
    export_him_policy_as_jit,
    export_him_policy_as_onnx,
)
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.csdog1.agents.rsl_rl_him_cfg import (
    CSDog1Flat282DomainRandHIMRunnerCfg,
    CSDog1Stairs282HIMRunnerCfg,
)
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.csdog1.cyberstar.asset import point_inertia

output = Path(args.output_dir).resolve()
output.mkdir(parents=True, exist_ok=True)
task = args.task
is_stairs = "Stairs" in task
cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=args.num_envs)
cfg.scene.terrain.terrain_generator.num_rows = 2
cfg.scene.terrain.terrain_generator.num_cols = 8
cfg.scene.terrain.max_init_terrain_level = None
cfg.scene.terrain.terrain_generator.use_cache = False
env = gym.make(task, cfg=cfg)
agent = CSDog1Stairs282HIMRunnerCfg() if is_stairs else CSDog1Flat282DomainRandHIMRunnerCfg()
wrapped = HIMVecEnvWrapper(
    env,
    clip_actions=100.,
    num_one_step_obs=47,
    history_order="oldest_to_latest",
    policy_frame_term_order=agent.policy.policy_frame_term_order,
    estimator_target_term_names=agent.policy.estimator_target_term_names,
)
runner = HIMOnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=cfg.sim.device)
assert wrapped.num_obs == 282 and wrapped.num_privileged_obs == 491
runtime = env.unwrapped.cyberstar
assert len(runtime.reward_names) == (85 if is_stairs else 78)
if is_stairs:
    assert runtime.flat_env_mask.any() and (~runtime.flat_env_mask).any()
    targets = runtime.command_targets[~runtime.flat_env_mask, :2]
    assert not ((targets[:, 0].abs() > 0) & (targets[:, 1].abs() > 0)).any()
robot = env.unwrapped.scene["robot"]
assert robot.num_bodies == 17
assert len(set(env.unwrapped._foot_sensor_ids)) == 4
calf_ids, _ = env.unwrapped.scene["contact_forces"].find_bodies(".*_calf")
assert set(calf_ids).isdisjoint(env.unwrapped._foot_sensor_ids)

# Compare the authored calf + restored FOOT with the unchanged USD source.
def mass_properties(prim):
    api = UsdPhysics.MassAPI(prim)
    mass = api.GetMassAttr().Get()
    center = np.asarray(api.GetCenterOfMassAttr().Get(), dtype=float)
    rotation = np.asarray(Gf.Matrix3d(Gf.Quatd(api.GetPrincipalAxesAttr().Get())), dtype=float).T
    inertia = rotation @ np.diag(api.GetDiagonalInertiaAttr().Get()) @ rotation.T
    return mass, center, inertia


original = Usd.Stage.Open(cfg.scene.robot.spawn.usd_path)
stage = omni.usd.get_context().get_stage()
for leg in ("FL", "FR", "RL", "RR"):
    original_path = f"{original.GetDefaultPrim().GetPath()}/base_link/{leg}_calf"
    robot_path = "/World/envs/env_0/Robot/base_link"
    mass, center, inertia = mass_properties(original.GetPrimAtPath(original_path))
    calf_mass, calf_center, calf_inertia = mass_properties(stage.GetPrimAtPath(f"{robot_path}/{leg}_calf"))
    foot_mass, foot_center, foot_inertia = mass_properties(stage.GetPrimAtPath(f"{robot_path}/{leg}_FOOT"))
    sign = 1 if leg in ("FL", "RL") else -1
    foot_center += np.array([.13949, -sign * .004, -.14978])
    np.testing.assert_allclose(calf_mass + foot_mass, mass, atol=1e-8)
    np.testing.assert_allclose((calf_mass * calf_center + foot_mass * foot_center) / mass, center, atol=1e-8)
    combined = (
        calf_inertia + calf_mass * point_inertia(calf_center - center)
        + foot_inertia + foot_mass * point_inertia(foot_center - center)
    )
    np.testing.assert_allclose(combined, inertia, atol=1e-9)

# A random fixture tests source-format loading, including Critic297 expansion.
# This fixture is deliberately not a trained locomotion checkpoint.
model = runner.alg.actor_critic
state = {key: value.detach().clone() for key, value in model.state_dict().items()}
old_prefixes = {
    "estimator.encoder.": "history_encoder.",
    "estimator.target.": "target_encoder.",
    "estimator.proto.": "prototypes.",
}
source_state = {}
for key, value in state.items():
    for new, old in old_prefixes.items():
        if key.startswith(new):
            key = old + key[len(new):]
            break
    source_state[key] = value[:, :297].clone() if key == "critic.0.weight" else value
fixture = output / "source_format_random_fixture.pt"
torch.save({"model_state_dict": source_state, "iter": 1000}, fixture)
with torch.inference_mode():
    probe = torch.randn(args.num_envs, 282, device=cfg.sim.device)
    expected_actions = model.act_inference(probe).clone()
    for param in model.parameters():
        param.zero_()
    runner.load(str(fixture))
    torch.testing.assert_close(model.act_inference(probe), expected_actions)
    torch.testing.assert_close(model.critic[0].weight[:, :297], state["critic.0.weight"][:, :297])
    assert model.critic[0].weight[:, 297:].count_nonzero() == 0
assert runner.current_learning_iteration == 1000

runtime.ppo_iteration = 1800
max_foot_force = 0.
with torch.inference_mode():
    for i in range(args.steps):
        obs, reward, done, extra = wrapped.step(torch.randn(args.num_envs, 12, device=cfg.sim.device) * .1)
        if i == 110:
            runtime._start_capture_recovery(torch.arange(args.num_envs, device=cfg.sim.device))
        assert torch.isfinite(obs).all(), f"Nonfinite actor at step {i}"
        assert torch.isfinite(wrapped.get_critic_observations()).all(), f"Nonfinite critic at step {i}"
        assert torch.isfinite(reward).all(), f"Nonfinite rewards at step {i}"
        max_foot_force = max(max_foot_force, runtime.contact_forces[:, :4, 2].max().item())
assert max_foot_force > 1., "Independent FOOT sensors must report ground contact."
runner.current_learning_iteration = 1800
runner.learn(2, init_at_random_ep_len=False)
runner.save(str(output / "smoke_model_1802.pt"))
export_him_policy_as_jit(model, str(output))
export_him_policy_as_onnx(model, str(output))
export_him_deployment_metadata(model, wrapped, str(output))
with torch.inference_mode():
    scripted = torch.jit.load(str(output / "policy.pt"))
    torch.testing.assert_close(scripted(probe.cpu()), model.act_inference(probe).cpu())
print(
    f"CSDOG1_MIGRATION_OK: {task} / 282 actor / 491 critic / {len(runtime.reward_names)} rewards / independent FOOT contacts / "
    "source checkpoint parity / physics rollout / PPO update / JIT + ONNX + metadata export",
    flush=True,
)
env.close()
app.close()
