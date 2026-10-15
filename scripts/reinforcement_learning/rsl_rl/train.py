# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # isort: skip
from rl_utils import enable_mjcf_importer

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
enable_mjcf_importer()

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    cmd = [r"python", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime

import omni
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from robot_lab.him import HIMOnPolicyRunner, HIMVecEnvWrapper
import robot_lab.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def apply_him_observation_history(env_cfg, agent_cfg):
    if hasattr(env_cfg, "cyberstar_config"):
        return
    if agent_cfg.class_name != "HIMOnPolicyRunner":
        return
    policy_obs_cfg = getattr(getattr(env_cfg, "observations", None), "policy", None)
    if policy_obs_cfg is None:
        return
    configured_history = int(getattr(policy_obs_cfg, "history_length", 0) or 0)
    if configured_history <= 1:
        history_length = int(getattr(agent_cfg.policy, "history_length", 6))
        policy_obs_cfg.history_length = history_length
        policy_obs_cfg.flatten_history_dim = True

def load_compatible_checkpoint(runner, resume_path, load_optimizer=False):
    loaded_dict = torch.load(resume_path, map_location="cpu")

    model = getattr(runner.alg, "policy", None)
    if model is None:
        model = getattr(runner.alg, "actor_critic", None)
    if model is None:
        raise AttributeError("Cannot find policy or actor_critic on runner.alg")

    if "model_state_dict" not in loaded_dict:
        raise KeyError(f"Checkpoint at '{resume_path}' does not contain 'model_state_dict'. "
                       f"Available keys: {list(loaded_dict.keys())}")

    current_state = model.state_dict()
    loaded_state = loaded_dict["model_state_dict"]

    compatible_state = {}
    skipped = []

    for key, value in loaded_state.items():
        if key in current_state and current_state[key].shape == value.shape:
            compatible_state[key] = value
        else:
            current_shape = current_state[key].shape if key in current_state else None
            skipped.append((key, tuple(value.shape), tuple(current_shape) if current_shape is not None else None))

    current_state.update(compatible_state)
    model.load_state_dict(current_state, strict=False)

    print(f"[INFO] Compatible params loaded: {len(compatible_state)}")
    if skipped:
        print("[INFO] Skipped incompatible params:")
        for key, old_shape, new_shape in skipped:
            print(f"  - {key}: ckpt={old_shape}, current={new_shape}")

    if load_optimizer and "optimizer_state_dict" in loaded_dict:
        try:
            runner.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            print("[INFO] Optimizer state loaded.")
        except Exception as e:
            print(f"[WARN] Optimizer state not loaded: {e}")

    if hasattr(runner, "current_learning_iteration") and "iter" in loaded_dict:
        runner.current_learning_iteration = loaded_dict["iter"]

    if hasattr(runner.alg, "amp_normalizer") and "amp_normalizer" in loaded_dict:
        runner.alg.amp_normalizer = loaded_dict["amp_normalizer"]

    if hasattr(runner.alg, "discriminator") and "discriminator_state_dict" in loaded_dict:
        try:
            runner.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"], strict=False)
        except Exception as e:
            print(f"[WARN] Discriminator state not loaded: {e}")

    return loaded_dict.get("infos", None)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    apply_him_observation_history(env_cfg, agent_cfg)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = PROJECT_ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name
    log_root_path = str(log_root_path.resolve())
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Resolve the checkpoint before allocating the simulation scene.
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        if getattr(agent_cfg, "source_checkpoint", None) and not args_cli.load_run:
            resume_path = Path(args_cli.checkpoint or agent_cfg.source_checkpoint).expanduser()
            if not resume_path.is_absolute():
                resume_path = PROJECT_ROOT / resume_path
            if not resume_path.is_file():
                raise FileNotFoundError(
                    f"CSDog1 robustness fine-tuning requires the source flat-282 checkpoint: {resume_path}. "
                    "Pass --resume --checkpoint /path/to/model_1000.pt, or use agent.resume=false for a fresh run."
                )
            resume_path = str(resume_path)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    if agent_cfg.class_name == "HIMOnPolicyRunner":
        env = HIMVecEnvWrapper(
            env,
            clip_actions=agent_cfg.clip_actions,
            num_one_step_obs=agent_cfg.policy.num_one_step_obs,
            history_order=getattr(agent_cfg.policy, "history_order", "latest_to_oldest"),
            policy_frame_term_order=getattr(agent_cfg.policy, "policy_frame_term_order", None),
            estimator_target_term_names=getattr(agent_cfg.policy, "estimator_target_term_names", None),
        )
    else:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "HIMOnPolicyRunner":
        runner = HIMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        try:
            runner.load(resume_path)
        except RuntimeError as e:
            if hasattr(env.unwrapped, "cyberstar"):
                raise
            print(f"[WARN] Strict checkpoint loading failed: {e}")
            print("[INFO] Falling back to compatible checkpoint loading.")
            load_compatible_checkpoint(runner, resume_path, load_optimizer=False)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
