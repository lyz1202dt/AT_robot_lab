# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""03dog asset, using the original CAD joint frames and 12 driven joints."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR


# Synced from quadruped-robot-gc/policy/03dog/base.yaml.
# FL, FR, RL, RR; each hip/thigh/calf. All hip axes are +X; right thighs are +Y.
# Foot origins are about z=-0.30635 m relative to the base at this pose.
DOG03_JOINT_POS = {
    "FL_hip_joint": 0.0,
    "FL_thigh_joint": -0.7854,
    "FL_calf_joint": 0.0,
    "FR_hip_joint": 0.0,
    "FR_thigh_joint": -0.7854,
    "FR_calf_joint": 0.0,
    "RL_hip_joint": 0.0,
    "RL_thigh_joint": -0.7854,
    "RL_calf_joint": 0.0,
    "RR_hip_joint": 0.0,
    "RR_thigh_joint": -0.7854,
    "RR_calf_joint": 0.0,
}

DOG03_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/03dog/urdf/03dog.urdf",
        fix_base=False,
        # Keep the real foot bodies for foot positions and contact reports.
        merge_fixed_joints=False,
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
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # The lowest foot mesh points are at z ~= -0.3215 m in the base frame.
        # At 0.40 m the initial clearance is about 7.85 cm, matching MuJoCo.
        pos=(0.0, 0.0, 0.40),
        joint_pos=DOG03_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_(hip|thigh|calf)_joint"],
            # Match the URDF and deployment limits: 30 Nm and 30 rad/s.
            effort_limit=30.0,
            velocity_limit=30.0,
            stiffness=30.0,
            damping=1.0,
            friction=0.0,
            min_delay=0,
            max_delay=2,
        ),
    },
)
