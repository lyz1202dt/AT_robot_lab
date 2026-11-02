"""Apply identified joint-side friction after motor effort saturation."""

import torch

from isaaclab.actuators import IdealPDActuator


class CyberstarMotorActuator(IdealPDActuator):
    """Keep PD drive limits separate from passive Coulomb and viscous friction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coulomb_friction = torch.zeros_like(self.applied_effort)
        self.joint_viscous_friction = torch.zeros_like(self.applied_effort)

    def compute(self, control_action, joint_pos, joint_vel):
        action = super().compute(control_action, joint_pos, joint_vel)
        friction = self.coulomb_friction * torch.sign(joint_vel) + self.joint_viscous_friction * joint_vel
        self.applied_effort = self.applied_effort - friction
        action.joint_efforts = self.applied_effort
        return action
