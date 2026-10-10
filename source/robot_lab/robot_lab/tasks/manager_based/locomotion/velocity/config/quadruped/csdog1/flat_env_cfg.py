# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from .rough_env_cfg import CSDog1RoughEnvCfg


CSDOG1_LOW_OBSTACLE_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=32,
    num_cols=32,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.85),
        "low_boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.45,
            grid_height_range=(0.01, 0.04),
            platform_width=2.0,
            holes=False,
        ),
    },
)


@configclass
class CSDog1FlatEnvCfg(CSDog1RoughEnvCfg):
    """Flat-ground CSDog1 task using the 47-D Cyberstar actor frame."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

        if self.__class__.__name__ == "CSDog1FlatEnvCfg":
            # The validated Cyberstar flat-47 task keeps only friction and
            # push randomization. Later robustness stages opt back into mass,
            # COM and gain variation.
            self.events.randomize_rigid_body_mass_base = None
            self.events.randomize_com_positions = None
            self.events.randomize_actuator_gains = None
            self.disable_zero_weight_rewards()


@configclass
class CSDog1Flat282EnvCfg(CSDog1FlatEnvCfg):
    """Flat-ground CSDog1 task with six-frame actor history.

    This mirrors the external ``csdog1_flat_282`` stage at the observation
    level while staying inside RobotLab's manager-based velocity task.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot.init_state.pos = (0.0, 0.0, 0.285)
        self.observations.policy.history_length = 6
        self.observations.policy.flatten_history_dim = True

        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.6, 1.3)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.6, 1.3)
        self.events.randomize_reset_joints.params["position_range"] = (0.95, 1.05)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}

        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.5, 1.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)

        if self.__class__.__name__ == "CSDog1Flat282EnvCfg":
            self.events.randomize_rigid_body_mass_base = None
            self.events.randomize_com_positions = None
            self.events.randomize_actuator_gains = None
            self.disable_zero_weight_rewards()


@configclass
class CSDog1Flat282DomainRandEnvCfg(CSDog1Flat282EnvCfg):
    """Robust flat-ground CSDog1 task with low obstacle replay and dynamics randomization."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = CSDOG1_LOW_OBSTACLE_TERRAIN_CFG

        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-0.25, 0.25)
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.005, 0.005),
            "y": (-0.005, 0.005),
            "z": (-0.003, 0.003),
        }
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.95, 1.05)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.95, 1.05)
        self.events.randomize_push_robot.interval_range_s = (10.0, 10.0)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}

        if self.__class__.__name__ == "CSDog1Flat282DomainRandEnvCfg":
            self.disable_zero_weight_rewards()
