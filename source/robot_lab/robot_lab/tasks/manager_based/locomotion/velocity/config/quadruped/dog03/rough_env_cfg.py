# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
)

from .observations import Dog03JointPositionAction, Dog03Phase, previous_action, velocity_commands
from .rewards_cfg import DOG03_JOINT_NAMES, Dog03RewardsCfg

##
# Pre-defined configs
##
from robot_lab.assets.dog03 import DOG03_CFG  # isort: skip


def dog03_phase_term():
    return ObsTerm(
        func=Dog03Phase,
        params={
            "command_name": "base_velocity",
            "cycle_time": 0.5,
            "stop_cycle_time": 0.5,
            "stop_duration": 0.5,
            "translation_threshold": 0.05,
            "yaw_threshold": 0.05,
        },
        clip=(-1.0, 1.0),
        scale=1.0,
    )


@configclass
class Dog03PolicyCfg(ObsGroup):
    """Deployment's 47-D frame, with phase first for every training backend."""

    phase = dog03_phase_term()
    base_ang_vel = ObservationsCfg.PolicyCfg().base_ang_vel
    projected_gravity = ObservationsCfg.PolicyCfg().projected_gravity
    velocity_commands = ObservationsCfg.PolicyCfg().velocity_commands
    joint_pos = ObservationsCfg.PolicyCfg().joint_pos
    joint_vel = ObservationsCfg.PolicyCfg().joint_vel
    actions = ObsTerm(func=previous_action, clip=(-100.0, 100.0), scale=1.0)

    # Preserve sensor noise augmentation during training.
    enable_corruption = True
    concatenate_terms = True


@configclass
class Dog03RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Independent velocity-tracking task for 03dog.

    The actor observation follows the Cyberstar 47-D flat task:
    45-D proprioception/commands/action frame plus a 2-D gait phase clock.
    """

    base_link_name = "base_link"
    foot_link_name = ".*_FOOT"
    foot_body_names = ("FL_FOOT", "FR_FOOT", "RL_FOOT", "RR_FOOT")
    joint_names = DOG03_JOINT_NAMES
    rewards: Dog03RewardsCfg = Dog03RewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Scene------------------------------
        self.scene.robot = DOG03_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/.*"

        # ------------------------------Observations------------------------------
        self.observations.policy = Dog03PolicyCfg()
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.critic.height_scan = None
        self.observations.policy.velocity_commands.scale = (1.0, 1.0, 0.25)
        self.observations.critic.base_ang_vel.scale = 0.25
        self.observations.critic.joint_vel.scale = 0.05
        self.observations.critic.velocity_commands.scale = (1.0, 1.0, 0.25)
        self.observations.critic.actions.func = previous_action
        self.observations.critic.phase = dog03_phase_term()
        # Isaac Lab clips before scaling; deployment clips the scaled frame
        # to +/-100. Convert the pre-scale limits to preserve that contract.
        for group in (self.observations.policy, self.observations.critic):
            group.base_ang_vel.clip = (-400.0, 400.0)
            group.joint_vel.clip = (-2000.0, 2000.0)
            group.velocity_commands.func = velocity_commands
            group.velocity_commands.clip = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        self.actions.joint_pos.class_type = Dog03JointPositionAction
        self.actions.joint_pos.scale = {".*": 0.25}
        self.actions.joint_pos.clip = None
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
                "roll": (-1.0, 1.0),
                "pitch": (-1.0, 1.0),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.3, 0.3),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.randomize_push_robot.interval_range_s = (15.0, 15.0)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact = TerminationTermCfg(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.base_link_name]), "threshold": 1.0},
        )

        # ------------------------------Curriculums------------------------------
        self.curriculum.command_levels = None

        # ------------------------------Commands------------------------------
        # Preserve slow stair commands: the shared threshold generator would
        # zero every translation command below 0.2 m/s before phase detection.
        self.commands.base_velocity.class_type = mdp.UniformVelocityCommand
        self.commands.base_velocity.resampling_time_range = (5.0, 5.0)
        self.commands.base_velocity.rel_standing_envs = 0.15
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        self.events.randomize_reset_base.params["pose_range"].update(z=(0.0, 0.05), roll=(-0.1, 0.1), pitch=(-0.1, 0.1))
        self._configure_task()
        # Keep inactive terms available until every stage has applied its settings.
        self.disable_zero_weight_rewards()

    def _configure_task(self):
        """Apply stage settings before removing inactive reward terms."""
