# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""03dog reward terms and weights for all training stages.

For standard manager stages, edit reward formulas in rewards.py. Set common weights/parameters in
Dog03RewardsCfg, and robustness/stair overrides in
Dog03DomainRandRewardsCfg.__post_init__. Zero weights disable terms.
Registered full Cyberstar DomainRand/Stairs tasks use cyberstar/*.json and core.py instead.
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import rewards

DOG03_JOINT_NAMES = [f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR") for part in ("hip", "thigh", "calf")]
DOG03_FOOT_NAMES = ("FL_FOOT", "FR_FOOT", "RL_FOOT", "RR_FOOT")


@configclass
class Dog03RewardsCfg:
    """Common 03dog rewards; zero weights disable individual terms."""

    # General
    is_terminated = RewTerm(func=rewards.is_terminated, weight=-2.0)

    # Root penalties
    lin_vel_z_l2 = RewTerm(func=rewards.lin_vel_z_l2, weight=-5.0)
    ang_vel_xy_l2 = RewTerm(func=rewards.ang_vel_xy_l2, weight=-1.0)
    flat_orientation_l2 = RewTerm(func=rewards.flat_orientation_l2, weight=-1.0)
    base_height_l2 = RewTerm(
        func=rewards.base_height_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "target_height": 0.32,
        },
    )
    body_lin_acc_l2 = RewTerm(
        func=rewards.body_lin_acc_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["base_link"])},
    )

    # Joint penalties
    joint_torques_l2 = RewTerm(
        func=rewards.joint_torques_l2, weight=-3e-05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_vel_l2 = RewTerm(
        func=rewards.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_acc_l2 = RewTerm(
        func=rewards.joint_acc_l2, weight=-2.5e-07, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )

    def create_joint_deviation_l1_rewterm(self, attr_name, weight, joint_names_pattern):
        rew_term = RewTerm(
            func=rewards.joint_deviation_l1,
            weight=weight,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names_pattern)},
        )
        setattr(self, attr_name, rew_term)

    joint_pos_limits = RewTerm(
        func=rewards.joint_pos_limits, weight=-10.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_vel_limits = RewTerm(
        func=rewards.joint_vel_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "soft_ratio": 1.0},
    )
    joint_power = RewTerm(
        func=rewards.joint_power, weight=-2e-05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )

    stand_still = RewTerm(
        func=rewards.stand_still,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=DOG03_JOINT_NAMES),
        },
    )

    joint_pos_penalty = RewTerm(
        func=rewards.joint_pos_penalty,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=DOG03_JOINT_NAMES),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )

    wheel_vel_penalty = RewTerm(
        func=rewards.wheel_vel_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "command_name": "base_velocity",
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )

    # Raw joint-angle symmetry is disabled because CAD joint zeros differ.
    joint_mirror = RewTerm(
        func=rewards.joint_mirror,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
                ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ],
        },
    )

    action_mirror = RewTerm(
        func=rewards.action_mirror,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]]},
    )

    action_sync = RewTerm(
        func=rewards.action_sync,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_groups": [
                ["FR_hip_joint", "FL_hip_joint", "RL_hip_joint", "RR_hip_joint"],
                ["FR_thigh_joint", "FL_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"],
                ["FR_calf_joint", "FL_calf_joint", "RL_calf_joint", "RR_calf_joint"],
            ],
        },
    )

    # Action penalties
    applied_torque_limits = RewTerm(
        func=rewards.applied_torque_limits,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    action_rate_l2 = RewTerm(func=rewards.action_rate_l2, weight=-0.2)
    action_smoothness_2_l2 = RewTerm(func=rewards.action_smoothness_2_l2, weight=-0.05)

    # Contact sensor
    undesired_contacts = RewTerm(
        func=rewards.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["^(?!.*_FOOT).*"]), "threshold": 1.0},
    )
    contact_forces = RewTerm(
        func=rewards.contact_forces,
        weight=-0.005,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"]), "threshold": 100.0},
    )

    # Velocity-tracking rewards
    track_lin_vel_xy_exp = RewTerm(
        func=rewards.track_lin_vel_xy_exp, weight=3.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=rewards.track_ang_vel_z_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # Others
    feet_air_time = RewTerm(
        func=rewards.feet_air_time,
        weight=0.05,
        params={
            "command_name": "base_velocity",
            "threshold": 0.25,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"]),
        },
    )

    feet_air_time_variance = RewTerm(
        func=rewards.feet_air_time_variance_penalty,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"])},
    )

    wheel_vel_variance = RewTerm(
        func=rewards.wheel_vel_variance_penalty, weight=0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    feet_gait = RewTerm(
        func=rewards.GaitReward,
        weight=1.0,
        params={
            "std": math.sqrt(0.5),
            "command_name": "base_velocity",
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "synced_feet_pair_names": (("FL_FOOT", "RR_FOOT"), ("FR_FOOT", "RL_FOOT")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    diagonal_trot_contact_pattern = RewTerm(
        func=rewards.diagonal_trot_contact_pattern,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "foot_names": DOG03_FOOT_NAMES,
            "command_threshold": 0.1,
        },
    )

    feet_contact = RewTerm(
        func=rewards.feet_contact,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"]),
            "command_name": "base_velocity",
            "expect_contact_num": 2,
        },
    )

    feet_contact_without_cmd = RewTerm(
        func=rewards.feet_contact_without_cmd,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"]),
            "command_name": "base_velocity",
        },
    )

    feet_stumble = RewTerm(
        func=rewards.feet_stumble,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"])},
    )

    feet_slide = RewTerm(
        func=rewards.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_FOOT"]),
        },
    )

    feet_height = RewTerm(
        func=rewards.feet_height,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_FOOT"]),
            "tanh_mult": 2.0,
            "target_height": 0.05,
            "command_name": "base_velocity",
        },
    )

    feet_height_body = RewTerm(
        func=rewards.feet_height_body,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_FOOT"]),
            "tanh_mult": 2.0,
            "target_height": -0.26,
            "command_name": "base_velocity",
        },
    )

    feet_distance_y_exp = RewTerm(
        func=rewards.feet_distance_y_exp,
        weight=0.5,
        params={
            "std": math.sqrt(0.0025),
            "asset_cfg": SceneEntityCfg("robot", body_names=list(DOG03_FOOT_NAMES)),
            "stance_width": 0.38,
        },
    )

    # feet_distance_xy_exp = RewTerm(
    #     func=rewards.feet_distance_xy_exp,
    #     weight=0.0,
    #     params={
    #         "std": math.sqrt(0.25),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=""),
    #         "stance_length": float,
    #         "stance_width": float,
    #     },
    # )

    upward = RewTerm(func=rewards.upward, weight=1.0)


@configclass
class Dog03DomainRandRewardsCfg(Dog03RewardsCfg):
    """Legacy manager-only tuning; registered Cyberstar tasks use their JSON scales."""

    def __post_init__(self):
        super().__post_init__()
        self.track_lin_vel_xy_exp.weight = 7.5
        self.track_lin_vel_xy_exp.params["std"] = 0.707
        self.track_ang_vel_z_exp.weight = 2.0
        self.track_ang_vel_z_exp.params["std"] = 0.707
        self.upward.weight = 0.1
        self.stand_still.weight = -0.5
        self.joint_pos_penalty.weight = -0.35
        self.action_rate_l2.weight = -0.015
        self.action_smoothness_2_l2.weight = -0.005
        self.feet_gait.weight = 0.5
        self.diagonal_trot_contact_pattern.weight = -0.2
        self.feet_distance_y_exp.weight = 0.2
