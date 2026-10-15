"""Full Cyberstar flat-282 robustness task on the RobotLab Isaac Lab scene."""
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.sim import CollisionPropertiesCfg
from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass

from ..flat_env_cfg import CSDog1Flat282EnvCfg
from . import lab
from .asset import spawn_cyberstar
from .runtime import source_config
from .terrain import CyberstarDiscreteTerrainCfg


@configclass
class PolicyCfg(ObservationGroupCfg):
    history = ObservationTermCfg(func=lab.policy_history)
    enable_corruption = False
    concatenate_terms = True


@configclass
class CriticCfg(ObservationGroupCfg):
    privileged = ObservationTermCfg(func=lab.critic_observation)
    enable_corruption = False
    concatenate_terms = True


@configclass
class ObservationsCfg:
    policy = PolicyCfg()
    critic = CriticCfg()


@configclass
class CommandsCfg:
    base_velocity = UniformVelocityCommandCfg(
        class_type=lab.StructuredVelocityCommand,
        asset_name="robot",resampling_time_range=(5.,5.),heading_command=False,
        ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(-1.5,1.5),lin_vel_y=(-1.5,1.5),ang_vel_z=(-1.5,1.5)),
        debug_vis=False,
    )


@configclass
class EventsCfg:
    reset_robot = EventTermCfg(func=lab.reset_robot,mode="reset")


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=lab.source_timeout,time_out=True)
    base_failure = TerminationTermCfg(func=lab.base_failure)


@configclass
class CSDog1CyberstarDomainRandEnvCfg(CSDog1Flat282EnvCfg):
    cyberstar_config: dict = source_config()["env"]
    cyberstar_play: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.sim.physx.bounce_threshold_velocity = self.cyberstar_config["sim"]["physx"]["bounce_threshold_velocity"]
        self.scene.robot.spawn.collision_props = CollisionPropertiesCfg(contact_offset=.01, rest_offset=0.)
        self.scene.robot.spawn.func = spawn_cyberstar
        self.scene.robot.init_state.joint_pos[".*_calf_joint"] = -.823
        self.actions.joint_pos.class_type = lab.CyberstarJointAction
        # PD is computed explicitly in CyberstarJointAction, including strength scaling before clipping.
        for name,old in list(self.scene.robot.actuators.items()):
            self.scene.robot.actuators[name] = IdealPDActuatorCfg(
                joint_names_expr=old.joint_names_expr,effort_limit=old.effort_limit,
                velocity_limit=old.velocity_limit,stiffness=0.,damping=0.,friction=0.,
            )
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(8.,8.),border_width=25.,num_rows=32,num_cols=32,
            horizontal_scale=.1,vertical_scale=.005,slope_threshold=.75,
            use_cache=False,curriculum=False,
            sub_terrains={"flat":MeshPlaneTerrainCfg(proportion=.85),"discrete":CyberstarDiscreteTerrainCfg(proportion=.15)},
        )
        self.scene.terrain.physics_material.static_friction = 1.
        self.scene.terrain.physics_material.dynamic_friction = 1.
        self.scene.terrain.physics_material.restitution = 0.
        # One existing ray caster supplies a shared terrain mesh for exact arbitrary-point scans.
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/base_link/base_link"
        self.scene.height_scanner_base.update_period = .02
        self.observations = ObservationsCfg()
        self.commands = CommandsCfg()
        self.events = EventsCfg()
        self.terminations = TerminationsCfg()
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels = None
        self.rewards = {name:RewardTermCfg(func=lab.reward_term,weight=weight,params={"name":name}) for name,weight in self.cyberstar_config["rewards"]["scales"].items() if weight}
