# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import math

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from robot_lab.assets.csdog1 import CSDOG1_CFG  # isort: skip


@configclass
class CSDog1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Velocity-tracking task for Cyberstar CSDog1.

    The actor observation follows the Cyberstar 47-D flat task:
    45-D proprioception/commands/action frame plus a 2-D gait phase clock.
    """

    base_link_name = "base_link"
    foot_link_name = ".*_calf"
    foot_body_names = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
    # fmt: off
    joint_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]
    # fmt: on

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Scene------------------------------
        self.scene.robot = CSDOG1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.observations.policy.phase = ObsTerm(
            func=mdp.phase,
            params={"cycle_time": 0.5},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        self.observations.critic.phase = ObsTerm(
            func=mdp.phase,
            params={"cycle_time": 0.5},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        self.actions.joint_pos.scale = {".*": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.5, 1.25)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.5, 1.25)
        self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 0.0)
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-0.5, 0.5)
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.01, 0.01),
            "y": (-0.01, 0.01),
            "z": (-0.005, 0.005),
        }
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_reset_joints.params["position_range"] = (0.9, 1.1)
        self.events.randomize_reset_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = self.joint_names
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.9, 1.1)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.9, 1.1)
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.randomize_push_robot.interval_range_s = (15.0, 15.0)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}

        # ------------------------------Rewards------------------------------
        self.rewards.is_terminated.weight = -2.0

        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.base_height_l2.params["target_height"] = 0.275
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0.0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        self.rewards.joint_torques_l2.weight = -3.0e-5
        self.rewards.joint_vel_l2.weight = 0.0
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_pos_limits.weight = -10.0
        self.rewards.joint_vel_limits.weight = 0.0
        self.rewards.applied_torque_limits.weight = -0.01
        self.rewards.joint_power.weight = -2.0e-5
        self.rewards.stand_still.weight = -2.0
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_pos_penalty.params["stand_still_scale"] = 5.0
        self.rewards.joint_mirror.weight = -0.2
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
        ]

        self.rewards.action_rate_l2.weight = -0.03
        self.rewards.action_smoothness_2_l2.weight = -0.01

        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -0.005
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.25)
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.params["std"] = math.sqrt(0.25)

        self.rewards.feet_air_time.weight = 0.05
        self.rewards.feet_air_time.params["threshold"] = 0.25
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.weight = -0.1
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0.0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.5
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0.0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = -0.2
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = -1.0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 1.0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_calf", "RR_calf"), ("FR_calf", "RL_calf"))
        self.rewards.diagonal_trot_contact_pattern.weight = -0.5
        self.rewards.diagonal_trot_contact_pattern.params["foot_names"] = self.foot_body_names
        self.rewards.feet_distance_y_exp.weight = 0.5
        self.rewards.feet_distance_y_exp.params["stance_width"] = 0.32
        self.rewards.feet_distance_y_exp.params["std"] = math.sqrt(0.0025)
        self.rewards.feet_distance_y_exp.params["asset_cfg"].body_names = list(self.foot_body_names)
        self.rewards.upward.weight = 1.0

        if self.__class__.__name__ == "CSDog1RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact = None

        # ------------------------------Curriculums------------------------------
        self.curriculum.command_levels = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.resampling_time_range = (5.0, 5.0)
        self.commands.base_velocity.rel_standing_envs = 0.15
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
