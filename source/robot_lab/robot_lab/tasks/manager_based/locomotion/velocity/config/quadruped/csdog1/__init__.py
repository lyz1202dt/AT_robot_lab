# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-CSDog1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:CSDog1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CSDog1FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:CSDog1FlatTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-CSDog1-282-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:CSDog1Flat282EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:CSDog1Flat282HIMRunnerCfg",
        "rsl_rl_ppo_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CSDog1Flat282PPORunnerCfg",
        "rsl_rl_him_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:CSDog1Flat282HIMRunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:CSDog1Flat282TrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-CSDog1-282-DomainRand-v0",
    entry_point=f"{__name__}.cyberstar.lab:CSDog1CyberstarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cyberstar.env_cfg:CSDog1CyberstarDomainRandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:CSDog1Flat282DomainRandHIMRunnerCfg",
        "rsl_rl_ppo_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CSDog1Flat282DomainRandPPORunnerCfg",
        "rsl_rl_him_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:CSDog1Flat282DomainRandHIMRunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:CSDog1Flat282DomainRandTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Stairs-CSDog1-282-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stairs_env_cfg:CSDog1Stairs282EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:CSDog1Stairs282HIMRunnerCfg",
        "rsl_rl_ppo_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CSDog1Stairs282PPORunnerCfg",
        "rsl_rl_him_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:CSDog1Stairs282HIMRunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:CSDog1Stairs282TrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-CSDog1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:CSDog1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CSDog1RoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:CSDog1RoughTrainerCfg",
    },
)
