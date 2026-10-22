# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""03dog asset, using the original CAD joint frames and 12 driven joints."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR


# FL, FR, RL, RR; each hip/thigh/calf. The CAD axes have different signs and
# zero offsets on different legs. These angles put the four foot origins at
# (hip_x, +/-0.19, -0.37) m in the base frame (verified by forward kinematics).
DOG03_JOINT_POS = {
    "FL_hip_joint": 0.007997,
    "FL_thigh_joint": -0.916405,
    "FL_calf_joint": 0.517192,
    "FR_hip_joint": -0.007253,
    "FR_thigh_joint": 0.915949,
    "FR_calf_joint": 0.516805,
    "RL_hip_joint": -0.008011,
    "RL_thigh_joint": -1.136067,
    "RL_calf_joint": 0.536377,
    "RR_hip_joint": 0.007946,
    "RR_thigh_joint": 1.169253,
    "RR_calf_joint": 0.517137,
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
        # The lowest foot mesh points are at z ~= -0.3855 m in the base frame.
        pos=(0.0, 0.0, 0.40),
        joint_pos=DOG03_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_(hip|thigh|calf)_joint"],
            # All twelve CAD joints specify 40 Nm and 30 rad/s.
            effort_limit=40.0,
            velocity_limit=30.0,
            stiffness=30.0,
            damping=1.0,
            friction=0.0,
            min_delay=0,
            max_delay=2,
        ),
    },
)
