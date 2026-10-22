# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""03dog model adaptation shared by the manager-based velocity tasks."""

from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.assets.dog03 import DOG03_CFG

from ..csdog1.rough_env_cfg import CSDog1RoughEnvCfg


class Dog03EnvCfgMixin:
    base_link_name = "base_link"
    foot_link_name = ".*_FOOT"
    foot_body_names = ("FL_FOOT", "FR_FOOT", "RL_FOOT", "RR_FOOT")

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = DOG03_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # URDF-imported bodies are direct children of Robot.
        for scanner in (self.scene.height_scanner, self.scene.height_scanner_base):
            if scanner is not None:
                scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/.*"
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (
            ("FL_FOOT", "RR_FOOT"), ("FR_FOOT", "RL_FOOT")
        )
        self.rewards.base_height_l2.params["target_height"] = 0.385
        self.rewards.feet_height_body.params["target_height"] = -0.32
        self.rewards.feet_distance_y_exp.params["stance_width"] = 0.38
        # Raw-angle symmetry is biased by this CAD model's unequal joint zeros.
        self.rewards.joint_mirror.weight = 0.0
        self.events.randomize_reset_base.params["pose_range"].update(
            z=(0.0, 0.05), roll=(-0.1, 0.1), pitch=(-0.1, 0.1)
        )
        self.terminations.illegal_contact = TerminationTermCfg(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.base_link_name]), "threshold": 1.0},
        )
        self._configure_task()
        self.disable_zero_weight_rewards()

    def _configure_task(self):
        """Apply task-specific settings before removing inactive rewards."""


@configclass
class Dog03RoughEnvCfg(Dog03EnvCfgMixin, CSDog1RoughEnvCfg):
    """03dog rough terrain with the shared 47-D phase/proprioception frame."""
