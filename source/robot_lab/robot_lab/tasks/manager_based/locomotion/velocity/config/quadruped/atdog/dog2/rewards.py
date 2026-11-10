"""Dog2 reward variants independent of shared locomotion rewards."""

import torch
from isaaclab.managers import SceneEntityCfg


def joint_mirror(env, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    for joint_pair in env.joint_mirror_joints_cache:
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if mirror_joints else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def base_height_exp_penalty(
    env,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        adjusted_target_height = target_height
    height_error = asset.data.root_pos_w[:, 2] - adjusted_target_height
    return 1.0 - torch.exp(-torch.square(height_error) / std**2)


def undesired_contacts_unconditional(env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1).float()


def gravity_orientation_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    gravity_b = asset.data.projected_gravity_b
    return torch.sum(torch.square(gravity_b[:, :2]), dim=1) + torch.square(gravity_b[:, 2] + 1.0)
