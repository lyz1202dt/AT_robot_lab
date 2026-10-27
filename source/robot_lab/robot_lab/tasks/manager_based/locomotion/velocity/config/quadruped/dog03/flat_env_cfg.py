# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from .rewards_cfg import Dog03DomainRandRewardsCfg
from .rough_env_cfg import Dog03RoughEnvCfg

DOG03_LOW_OBSTACLE_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=32,
    num_cols=32,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.95),
        "low_boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.05,
            grid_width=0.45,
            grid_height_range=(0.01, 0.025),
            platform_width=2.0,
            holes=False,
        ),
    },
)


@configclass
class Dog03FlatEnvCfg(Dog03RoughEnvCfg):
    """Flat-ground Dog03 task using the 47-D Cyberstar actor frame."""

    def _configure_task(self):
        super()._configure_task()

        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

        if type(self) in (Dog03FlatEnvCfg, Dog03Flat282EnvCfg):
            # The validated Cyberstar flat-47 task keeps only friction and
            # push randomization. Later robustness stages opt back into mass,
            # COM and gain variation.
            self.events.randomize_rigid_body_mass_base = None
            self.events.randomize_com_positions = None
            self.events.randomize_actuator_gains = None


@configclass
class Dog03Flat282EnvCfg(Dog03FlatEnvCfg):
    """Flat-ground Dog03 task with six-frame actor history.

    Six 47-D frames retain the CSDog1 policy input and history layout.
    """

    def _configure_task(self):
        super()._configure_task()

        self.observations.policy.history_length = 6
        self.observations.policy.flatten_history_dim = True

        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.6, 1.3)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.6, 1.3)
        self.events.randomize_reset_joints.params["position_range"] = (0.95, 1.05)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}

        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.5, 1.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)


@configclass
class Dog03Flat282DomainRandEnvCfg(Dog03Flat282EnvCfg):
    """Robust flat-ground Dog03 task with low obstacle replay and dynamics randomization."""

    rewards: Dog03DomainRandRewardsCfg = Dog03DomainRandRewardsCfg()

    def _configure_task(self):
        super()._configure_task()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = DOG03_LOW_OBSTACLE_TERRAIN_CFG

        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-0.10, 0.10)
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.003, 0.003),
            "y": (-0.003, 0.003),
            "z": (-0.002, 0.002),
        }
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.98, 1.02)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.98, 1.02)
        self.events.randomize_push_robot.interval_range_s = (15.0, 20.0)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}

        self.commands.base_velocity.rel_standing_envs = 0.05
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
