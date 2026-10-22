# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""03dog locomotion tasks (dog03 is a valid Python package name)."""

import gymnasium as gym

from . import agents


gym.register(
    id="RobotLab-Isaac-Velocity-Flat-03Dog-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Dog03FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Dog03FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:Dog03FlatTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-03Dog-282-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Dog03Flat282EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:Dog03Flat282HIMRunnerCfg",
        "rsl_rl_ppo_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Dog03Flat282PPORunnerCfg",
        "rsl_rl_him_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:Dog03Flat282HIMRunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:Dog03Flat282TrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-03Dog-282-DomainRand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Dog03Flat282DomainRandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:Dog03Flat282DomainRandHIMRunnerCfg",
        "rsl_rl_ppo_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Dog03Flat282DomainRandPPORunnerCfg",
        "rsl_rl_him_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:Dog03Flat282DomainRandHIMRunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:Dog03Flat282DomainRandTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Stairs-03Dog-282-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stairs_env_cfg:Dog03Stairs282EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:Dog03Stairs282HIMRunnerCfg",
        "rsl_rl_ppo_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Dog03Stairs282PPORunnerCfg",
        "rsl_rl_him_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:Dog03Stairs282HIMRunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:Dog03Stairs282TrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-03Dog-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:Dog03RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Dog03RoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:Dog03RoughTrainerCfg",
    },
)
