# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def get_activation(act_name: str) -> nn.Module:
    if act_name == "elu":
        return nn.ELU()
    if act_name == "selu":
        return nn.SELU()
    if act_name == "relu":
        return nn.ReLU()
    if act_name == "silu":
        return nn.SiLU()
    if act_name == "lrelu":
        return nn.LeakyReLU()
    if act_name == "tanh":
        return nn.Tanh()
    if act_name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation function: {act_name}")


@torch.no_grad()
def sinkhorn(out: torch.Tensor, eps: float = 0.05, iters: int = 3) -> torch.Tensor:
    q = torch.exp(out / eps).T
    k, batch = q.shape
    q /= q.sum()

    for _ in range(iters):
        q /= torch.sum(q, dim=1, keepdim=True)
        q /= k
        q /= torch.sum(q, dim=0, keepdim=True)
        q /= batch
    return (q * batch).T


class HIMEstimator(nn.Module):
    def __init__(
        self,
        temporal_steps: int,
        num_one_step_obs: int,
        vel_slice: tuple[int, int],
        target_slice: tuple[int, int] | list[int] | tuple[int, ...],
        enc_hidden_dims: list[int] | tuple[int, ...] = (128, 64, 16),
        tar_hidden_dims: list[int] | tuple[int, ...] = (128, 64),
        activation: str = "elu",
        learning_rate: float = 1e-3,
        max_grad_norm: float = 10.0,
        num_prototypes: int = 32,
        temperature: float = 3.0,
    ):
        super().__init__()

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_latent = int(enc_hidden_dims[-1])
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature
        self.vel_slice = tuple(vel_slice)
        target_spec = tuple(int(index) for index in target_slice)
        if len(target_spec) == 2 and target_spec[0] < target_spec[1]:
            self.target_slice = target_spec
            self.target_indices = None
            target_input_dim = self.target_slice[1] - self.target_slice[0]
        else:
            self.target_slice = target_spec
            self.target_indices = torch.tensor(target_spec, dtype=torch.long)
            target_input_dim = len(target_spec)

        activation_layer = get_activation(activation)

        enc_input_dim = self.temporal_steps * self.num_one_step_obs
        enc_layers: list[nn.Module] = []
        for hidden_dim in enc_hidden_dims[:-1]:
            enc_layers.extend([nn.Linear(enc_input_dim, hidden_dim), activation_layer])
            enc_input_dim = hidden_dim
        enc_layers.append(nn.Linear(enc_input_dim, self.num_latent + (self.vel_slice[1] - self.vel_slice[0])))
        self.encoder = nn.Sequential(*enc_layers)

        target_layers: list[nn.Module] = []
        for hidden_dim in tar_hidden_dims:
            target_layers.extend([nn.Linear(target_input_dim, hidden_dim), activation_layer])
            target_input_dim = hidden_dim
        target_layers.append(nn.Linear(target_input_dim, self.num_latent))
        self.target = nn.Sequential(*target_layers)

        self.proto = nn.Embedding(num_prototypes, self.num_latent)
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def _target_observations(self, critic_obs: torch.Tensor) -> torch.Tensor:
        if self.target_indices is not None:
            return critic_obs.index_select(-1, self.target_indices.to(device=critic_obs.device))
        return critic_obs[:, self.target_slice[0] : self.target_slice[1]]

    def forward(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        parts = self.encoder(obs_history.detach())
        pred_vel = parts[..., : self.vel_slice[1] - self.vel_slice[0]]
        latent = F.normalize(parts[..., self.vel_slice[1] - self.vel_slice[0] :], dim=-1, p=2)
        return pred_vel.detach(), latent.detach()

    def update(
        self,
        obs_history: torch.Tensor,
        next_critic_obs: torch.Tensor,
        dones: torch.Tensor | None = None,
        lr: float | None = None,
        velocity_loss_coef: float = 1.0,
        swap_loss_coef: float = 1.0,
    ) -> tuple[float, float]:
        if lr is not None and lr != self.learning_rate:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

        valid_mask = torch.ones(obs_history.shape[0], dtype=torch.bool, device=obs_history.device)
        if dones is not None:
            valid_mask &= ~dones.view(-1).bool()
        if not torch.any(valid_mask):
            return 0.0, 0.0

        obs_history = obs_history[valid_mask]
        next_critic_obs = next_critic_obs[valid_mask]

        target_vel = next_critic_obs[:, self.vel_slice[0] : self.vel_slice[1]].detach()
        target_obs = self._target_observations(next_critic_obs).detach()

        parts = self.encoder(obs_history)
        pred_vel = parts[..., : self.vel_slice[1] - self.vel_slice[0]]
        latent_student = F.normalize(parts[..., self.vel_slice[1] - self.vel_slice[0] :], dim=-1, p=2)
        latent_target = F.normalize(self.target(target_obs), dim=-1, p=2)

        with torch.no_grad():
            proto = F.normalize(self.proto.weight.data.clone(), dim=-1, p=2)
            self.proto.weight.copy_(proto)

        score_student = latent_student @ self.proto.weight.T
        score_target = latent_target @ self.proto.weight.T

        with torch.no_grad():
            assign_student = sinkhorn(score_student)
            assign_target = sinkhorn(score_target)

        log_prob_student = F.log_softmax(score_student / self.temperature, dim=-1)
        log_prob_target = F.log_softmax(score_target / self.temperature, dim=-1)

        swap_loss = -0.5 * (
            torch.sum(assign_target * log_prob_student, dim=-1).mean()
            + torch.sum(assign_student * log_prob_target, dim=-1).mean()
        )
        estimation_loss = F.mse_loss(pred_vel, target_vel)
        total_loss = velocity_loss_coef * estimation_loss + swap_loss_coef * swap_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return float(estimation_loss.item()), float(swap_loss.item())
