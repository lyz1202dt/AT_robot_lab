"""Isaac Lab boundary for the Cyberstar task. No Isaac Gym dependency at runtime."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.managers import CommandTerm, RewardManager
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.utils.warp import raycast_mesh

from .runtime import CyberstarRuntime


class StructuredVelocityCommand(CommandTerm):
    @property
    def command(self):
        return self._env.cyberstar.commands[:, :3]

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.metrics = {
            name: torch.zeros(self.num_envs, device=self.device) for name in ("error_vel_xy", "error_vel_yaw")
        }

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        metrics = {key: value[ids].mean().item() for key, value in self.metrics.items()}
        for value in self.metrics.values():
            value[ids] = 0
        self._env.cyberstar.reset(ids)
        if hasattr(self._env.cyberstar, "stair_success"):
            metrics["stair_success"] = self._env.cyberstar.stair_success.item()
        self._env._cyberstar_obs_step = -1
        return metrics

    def compute(self, dt):
        self._update_metrics()

    def _update_metrics(self):
        runtime = self._env.cyberstar
        self.metrics["error_vel_xy"] += (
            torch.linalg.vector_norm(self.command[:, :2] - runtime.base_lin_vel[:, :2], dim=-1)
            / self._env.max_episode_length
        )
        self.metrics["error_vel_yaw"] += (
            self.command[:, 2] - runtime.base_ang_vel[:, 2]
        ).abs() / self._env.max_episode_length

    def _resample_command(self, env_ids):
        self._env.cyberstar._resample_commands(env_ids)

    def _update_command(self):
        pass


class CyberstarJointAction(JointPositionAction):
    """Delay policy actions once per control step; apply explicit PD each physics step."""

    def process_actions(self, actions):
        delayed = self._env.cyberstar.process_actions(actions)
        super().process_actions(delayed)

    def apply_actions(self):
        runtime = self._env.cyberstar
        q = self._asset.data.joint_pos[:, self._joint_ids]
        dq = self._asset.data.joint_vel[:, self._joint_ids]
        torque = (
            30 * runtime.kp_scales * (self.processed_actions - q) - runtime.kd_scales * dq
        ) * runtime.motor_strengths
        limits = self._asset.data.joint_effort_limits[:, self._joint_ids]
        runtime.torques[:] = torch.clamp(torque, -limits, limits)
        self._asset.set_joint_effort_target(runtime.torques, joint_ids=self._joint_ids)


class Ji22RewardManager(RewardManager):
    def compute(self, dt):
        # Keep individual weighted components and episode metrics in the manager.
        super().compute(dt)
        positive = torch.zeros_like(self._reward_buf)
        negative = torch.zeros_like(positive)
        termination = torch.zeros_like(positive)
        for i, (name, cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            value = self._step_reward[:, i] * dt
            if name == "termination":
                termination += value
            elif cfg.weight > 0:
                positive += value
            else:
                negative += value
        self._reward_buf[:] = (
            positive * torch.exp((negative / self._env.cyberstar.cfg.rewards.sigma_rew_neg).clamp(-20, 0)) + termination
        )
        return self._reward_buf


def reward_term(env, name):
    return getattr(env.cyberstar, "_reward_" + name)()


def policy_history(env):
    env._compute_cyberstar_observations()
    return env.cyberstar.obs_buf


def critic_observation(env):
    env._compute_cyberstar_observations()
    return env.cyberstar.privileged_obs_buf


def base_failure(env):
    runtime = env.cyberstar
    contact = (
        torch.linalg.vector_norm(env.scene["contact_forces"].data.net_forces_w[:, env._base_sensor_id], dim=-1) > 1
    )
    return contact | (runtime.rpy[:, 0].abs() > 0.8) | (runtime.rpy[:, 1].abs() > 1)


def source_timeout(env):
    return env.episode_length_buf > env.max_episode_length


def reset_robot(env, env_ids):
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    runtime, robot = env.cyberstar, env.scene["robot"]
    count = len(ids)
    state = robot.data.default_root_state[ids].clone()
    state[:, :3] += env.scene.env_origins[ids]
    state[:, :2] += torch.empty(count, 2, device=env.device).uniform_(-1, 1)
    if hasattr(env, "_place_stair_reset"):
        env._place_stair_reset(state, ids)
    # Source resets use the initial quaternion, with mild 6-D velocity variation.
    velocity_limit = runtime.cfg.init_state.reset_root_velocity_range
    state[:, 7:13] = (
        torch.empty(count, 6, device=env.device).uniform_(-velocity_limit, velocity_limit)
        if not runtime.cfg.env.test
        else 0
    )
    q = robot.data.default_joint_pos[ids].clone()
    if not runtime.cfg.env.test:
        q *= torch.empty_like(q).uniform_(*runtime.cfg.init_state.reset_dof_scale_range)
    robot.write_root_pose_to_sim(state[:, :7], env_ids=ids)
    robot.write_root_velocity_to_sim(state[:, 7:13], env_ids=ids)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
    dr = runtime.cfg.domain_rand
    for name, enabled, limits in (
        ("motor_strengths", dr.randomize_motor_strength, dr.motor_strength_range),
        ("kp_scales", dr.randomize_pd_gains, dr.kp_scale_range),
        ("kd_scales", dr.randomize_pd_gains, dr.kd_scale_range),
    ):
        getattr(runtime, name)[ids] = (
            torch.empty(count, 12, device=env.device).uniform_(*limits) if enabled and not runtime.cfg.env.test else 1
        )
    low, high = dr.control_delay_steps
    runtime.control_delay[ids] = (
        torch.randint(low, high + 1, (count,), device=env.device)
        if dr.randomize_control_delay and not runtime.cfg.env.test
        else 0
    )
    env._sync_cyberstar()


class Dog03CyberstarEnv(ManagerBasedRLEnv):
    """Preserve source ordering: state/callback → termination/reward → reset/push → history."""

    runtime_type = CyberstarRuntime

    def load_managers(self):
        self.cyberstar = self.runtime_type(self.num_envs, self.device, self.step_dt, self.cfg.cyberstar_config)
        self.cyberstar.cfg.env.test = self.cfg.cyberstar_play
        self._cyberstar_obs_step = -1
        robot = self.scene["robot"]
        self._joint_ids, _ = robot.find_joints(self.cyberstar.dof_names, preserve_order=True)
        self._foot_ids, _ = robot.find_bodies([f"{leg}_FOOT" for leg in ("FL", "FR", "RL", "RR")], preserve_order=True)
        sensor = self.scene["contact_forces"]
        self._sensor_ids, _ = sensor.find_bodies(robot.body_names, preserve_order=True)
        self._foot_sensor_ids, _ = sensor.find_bodies(
            [f"{leg}_FOOT" for leg in ("FL", "FR", "RL", "RR")], preserve_order=True
        )
        base_ids, _ = sensor.find_bodies("base_link")
        self._base_sensor_id = base_ids[0]
        penalty_ids, _ = sensor.find_bodies(".*_(thigh|calf)")
        self.cyberstar.penalised_contact_indices = torch.tensor(
            [4 + self._sensor_ids.index(i) for i in penalty_ids], device=self.device
        )
        self.cyberstar._velocity_writer = self._write_cyberstar_velocity
        self.cyberstar._height_query = self._terrain_height_at
        self._configure_cyberstar_terrain()
        self._sync_cyberstar()
        self.cyberstar.dof_pos_limits = robot.data.soft_joint_pos_limits[0, self._joint_ids].clone()
        super().load_managers()
        # This task uses Ji22 instead of the base manager's linear sum.
        self.reward_manager = Ji22RewardManager(self.cfg.rewards, self)
        self._randomize_startup()
        self._cyberstar_obs_step = -1

    def _configure_cyberstar_terrain(self):
        pass

    def _sync_cyberstar(self):
        runtime, data = self.cyberstar, self.scene["robot"].data
        runtime.root_states[:, :3] = data.root_pos_w
        runtime.root_states[:, 3:7] = data.root_quat_w[:, [1, 2, 3, 0]]
        runtime.root_states[:, 7:13] = data.root_com_vel_w
        runtime.base_lin_vel[:] = data.root_lin_vel_b
        runtime.base_ang_vel[:] = data.root_ang_vel_b
        runtime.projected_gravity[:] = data.projected_gravity_b
        runtime.rpy[:] = torch.stack(euler_xyz_from_quat(data.root_quat_w), dim=-1)
        # Isaac Lab returns Euler angles in [0,2pi); source uses signed angles.
        runtime.rpy[:] = torch.remainder(runtime.rpy + torch.pi, 2 * torch.pi) - torch.pi
        runtime.dof_pos[:] = data.joint_pos[:, self._joint_ids]
        runtime.dof_vel[:] = data.joint_vel[:, self._joint_ids]
        body_pos, body_quat = data.body_link_pos_w, data.body_link_quat_w
        body_vel, body_ang = data.body_com_lin_vel_w, data.body_com_ang_vel_w
        quat = body_quat[:, self._foot_ids]
        foot_pos = body_pos[:, self._foot_ids]
        foot_vel = body_vel[:, self._foot_ids]
        runtime.rigid_body_states = torch.cat(
            (
                torch.cat((foot_pos, quat[:, :, [1, 2, 3, 0]], foot_vel, body_ang[:, self._foot_ids]), dim=-1),
                torch.cat((body_pos, body_quat[:, :, [1, 2, 3, 0]], body_vel, body_ang), dim=-1),
            ),
            dim=1,
        )
        forces = self.scene["contact_forces"].data.net_forces_w
        runtime.contact_forces = torch.cat((forces[:, self._foot_sensor_ids], forces[:, self._sensor_ids]), dim=1)
        runtime.episode_length_buf = self.episode_length_buf

    def _write_cyberstar_velocity(self, ids):
        self.scene["robot"].write_root_velocity_to_sim(self.cyberstar.root_states[ids, 7:13], env_ids=ids)

    def _terrain_height_at(self, positions):
        if self.cfg.scene.terrain.terrain_type == "plane":
            return torch.zeros_like(positions[..., 0])
        sensor = self.scene["height_scanner_base"]
        starts = positions.clone()
        starts[..., 2] += 20
        directions = torch.zeros_like(starts)
        directions[..., 2] = -1
        hits = raycast_mesh(starts, directions, sensor.meshes[sensor.cfg.mesh_prim_paths[0]])[0]
        # A missed ray outside the terrain must not introduce infinite observations.
        return torch.where(torch.isfinite(hits[..., 2]), hits[..., 2], torch.zeros_like(hits[..., 2]))

    def _randomize_startup(self):
        runtime, robot = self.cyberstar, self.scene["robot"]
        if runtime.cfg.env.test:
            return
        view = robot.root_physx_view
        masses, coms = view.get_masses(), view.get_coms()
        base = robot.find_bodies("base_link")[0][0]
        dr = runtime.cfg.domain_rand
        mass = (
            torch.empty(self.num_envs).uniform_(*dr.added_mass_range)
            if dr.randomize_base_mass else torch.zeros(self.num_envs)
        )
        com = torch.stack(
            [
                torch.empty(self.num_envs).uniform_(*limits) if dr.randomize_com else torch.zeros(self.num_envs)
                for limits in (
                    runtime.cfg.domain_rand.com_range_x,
                    runtime.cfg.domain_rand.com_range_y,
                    runtime.cfg.domain_rand.com_range_z,
                )
            ],
            dim=-1,
        )
        masses[:, base] += mass
        coms[:, base, :3] += com
        ids = torch.arange(self.num_envs, dtype=torch.int32)
        view.set_masses(masses, ids)
        view.set_coms(coms, ids)
        runtime.critic_base_mass_offset[:] = mass[:, None].to(self.device)
        runtime.critic_base_com_offset[:] = com.to(self.device)
        # One coefficient per environment and all its collision shapes, as in source.
        if dr.randomize_friction:
            buckets = torch.empty(64).uniform_(*dr.friction_range)
            friction = buckets[torch.randint(0, 64, (self.num_envs,))]
        else:
            friction = torch.ones(self.num_envs)
        materials = view.get_material_properties()
        materials[:, :, :2] = friction[:, None, None]
        materials[:, :, 2] = 0
        view.set_material_properties(materials, ids)
        runtime.critic_friction[:] = friction[:, None].to(self.device)

    def _compute_cyberstar_observations(self):
        if self._cyberstar_obs_step != self.common_step_counter:
            self.cyberstar.compute_observations()
            self._cyberstar_obs_step = self.common_step_counter

    def step(self, action):
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()
        rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self._sync_cyberstar()
        if hasattr(self, "cyberstar_keyboard"):
            self.cyberstar.external_commands = True
            command = torch.as_tensor(self.cyberstar_keyboard(), device=self.device)
            self.cyberstar.commands[:, :3] = command
            self.cyberstar.command_targets[:] = command
            self.cyberstar.command_ramp_active[:] = False
        self.cyberstar._post_physics_step_callback()
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.cyberstar.reset_buf = self.reset_buf
        self.cyberstar.time_out_buf = self.reset_time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()
        ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            self.recorder_manager.record_pre_reset(ids)
            self._reset_idx(ids)
            self.recorder_manager.record_post_reset(ids)
        if self.cyberstar.cfg.domain_rand.push_robots:
            self.cyberstar._push_robots()
        self.command_manager.compute(dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        self.cyberstar.finish_step()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
