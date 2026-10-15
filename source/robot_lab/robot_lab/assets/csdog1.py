# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Cyberstar CSDog1 robots."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##

CSDOG1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/csdog1/usd/csdog1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": 0.785,
            ".*_calf_joint": -0.823,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.99,
    actuators={
        "hips": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=31.5,
            velocity_limit=30.0,
            stiffness=30.0,
            damping=1.0,
            friction=0.0,
            min_delay=0,
            max_delay=2,
        ),
        "thighs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=31.5,
            velocity_limit=30.0,
            stiffness=30.0,
            damping=1.0,
            friction=0.0,
            min_delay=0,
            max_delay=2,
        ),
        "calves": DelayedPDActuatorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=40.5,
            velocity_limit=30.0,
            stiffness=30.0,
            damping=1.0,
            friction=0.0,
            min_delay=0,
            max_delay=2,
        ),
    },
)
"""Configuration of Cyberstar CSDog1 using delayed PD actuators."""
