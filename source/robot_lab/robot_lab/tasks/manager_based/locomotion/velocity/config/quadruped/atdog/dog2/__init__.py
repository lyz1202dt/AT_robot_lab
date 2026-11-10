# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-ATDog-Dog2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:ATDogDog2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDog2FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDog2FlatTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-ATDog-Dog2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:ATDogDog2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDog2RoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDog2RoughTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Stairs-ATDog-Dog2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stairs_env_cfg:ATDogDog2StairsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDog2StairsPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDog2StairsTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Sand-ATDog-Dog2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sand_env_cfg:ATDogDog2SandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDog2SandPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDog2SandTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Bar-ATDog-Dog2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bar_env_cfg:ATDogDog2BarEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDog2BarPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDog2BarTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Slope-ATDog-Dog2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.slope_env_cfg:ATDogDog2SlopeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDog2SlopePPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDog2SlopeTrainerCfg",
    },
)
