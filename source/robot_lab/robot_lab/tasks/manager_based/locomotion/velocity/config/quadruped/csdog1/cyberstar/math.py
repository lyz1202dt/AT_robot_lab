"""Torch equivalents of the source Isaac Gym helpers; quaternion order is xyzw."""
import torch


def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(*shape, device=device) + lower


def wrap_to_pi(angles):
    angles = torch.remainder(angles, 2 * torch.pi)
    return torch.where(angles > torch.pi, angles - 2 * torch.pi, angles)


def quat_apply(q, v):
    xyz = q[..., :3]
    t = 2 * torch.cross(xyz, v, dim=-1)
    return v + q[..., 3:] * t + torch.cross(xyz, t, dim=-1)


def quat_rotate_inverse(q, v):
    inverse = torch.cat((-q[..., :3], q[..., 3:]), dim=-1)
    return quat_apply(inverse, v)


def quat_apply_yaw(q, v):
    yaw = q.clone()
    yaw[..., :2] = 0
    yaw = yaw / torch.linalg.vector_norm(yaw, dim=-1, keepdim=True).clamp_min(1e-12)
    return quat_apply(yaw, v)
