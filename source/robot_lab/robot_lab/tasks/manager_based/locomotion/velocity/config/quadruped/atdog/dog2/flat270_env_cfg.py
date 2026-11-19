# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from robot_lab.assets.atdog import AT_DOG2_CFG

from . import rewards as dog2_rewards
from .flat_env_cfg import ATDogDog2FlatEnvCfg


@configclass
class ATDogDog2Flat270EnvCfg(ATDogDog2FlatEnvCfg):
    """Flat Dog2 task with six historical 45-D actor observations."""

    base_link_name = "base"
    foot_link_name = ".*_foot"
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]

    def __post_init__(self):
        super().__post_init__()

        # Scene: keep this variant on flat ground with the Dog2 asset.
        self.scene.robot = AT_DOG2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.curriculum.terrain_levels = None

        # Flat270-specific reset range.
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-1.0, 1.0),
                "pitch": (-1.0, 1.0),
                "yaw": (-0.0, 0.0),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.3),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        }

        # Observations: retain the existing 45-D actor input per frame.
        self.observations.policy.base_lin_vel = None
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.history_length = 6
        self.observations.policy.flatten_history_dim = True
        self.observations.critic.history_length = 6
        self.observations.critic.flatten_history_dim = True

        # Actions.
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # Randomization.
        self.events.randomize_push_robot.params["velocity_range"] = {
            "x": (-0.7, 0.7), "y": (-0.7, 0.7),
        }
        self.events.randomize_rigid_body_material.params["asset_cfg"].body_names = [self.foot_link_name]
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.5, 0.9)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.4, 0.8)
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-20.0, 20.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-8.0, 8.0)
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.8, 1.2)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.8, 1.2)

        # Rewards.
        self.rewards.is_terminated.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = -20.0
        self.rewards.ang_vel_xy_l2.weight = -2.0
        self.rewards.flat_orientation_l2.func = dog2_rewards.gravity_orientation_l2
        self.rewards.flat_orientation_l2.weight = -6.0
        self.rewards.base_height_l2.func = dog2_rewards.base_height_exp_penalty
        self.rewards.base_height_l2.weight = -2.0
        self.rewards.base_height_l2.params["std"] = 0.1
        self.rewards.base_height_l2.params["target_height"] = 0.30
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.rewards.body_lin_acc_l2.weight = -0.01
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_vel_l2.weight = -1.0e-5
        self.rewards.joint_acc_l2.weight = -1.0e-5
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = 0.0
        self.rewards.joint_power.weight = -2e-5
        self.rewards.stand_still.weight = -8.0
        self.rewards.joint_pos_penalty.weight = -2.5
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.func = dog2_rewards.joint_mirror
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]
        self.rewards.action_rate_l2.weight = -0.5

        self.rewards.undesired_contacts.func = dog2_rewards.undesired_contacts_unconditional
        self.rewards.undesired_contacts.weight = -20.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.track_lin_vel_xy_exp.weight = 10.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.4
        self.rewards.track_ang_vel_z_exp.weight = 10.0

        self.rewards.feet_air_time.weight = 2.0
        self.rewards.feet_air_time.params["threshold"] = 0.2
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.weight = -20.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0.0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.5
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0.0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.func = dog2_rewards.feet_slide_world
        self.rewards.feet_slide.weight = -10.0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = -20.0
        self.rewards.feet_height.params["target_height"] = 0.15
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = -5.0
        self.rewards.feet_height_body.params["target_height"] = -0.15
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 1.0
        self.rewards.feet_gait.params["std"] = 0.05
        self.rewards.feet_gait.params["max_err"] = 0.3
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        self.rewards.upward.weight = 8.0
        self.rewards.feet_distance_y_exp.weight = 0.8
        self.rewards.feet_distance_y_exp.params["stance_width"] = 0.34
        self.rewards.feet_distance_y_exp.params["std"] = 0.15
        self.rewards.feet_distance_y_exp.params["asset_cfg"].body_names = [
            "FL_foot", "FR_foot", "RL_foot", "RR_foot"
        ]

        # Terminations, curriculum, and commands.
        self.terminations.illegal_contact = None
        self.curriculum.command_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.disable_zero_weight_rewards()
