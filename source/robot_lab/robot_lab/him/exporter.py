# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import os

import torch
import torch.nn.functional as F

_AT_ROBOT_LAB_TERM_NAME_MAP = {
    "base_ang_vel": "ang_vel",
    "projected_gravity": "gravity_vec",
    "velocity_commands": "commands",
    "joint_pos": "dof_pos",
    "joint_vel": "dof_vel",
    "actions": "actions",
}


class _HIMPolicyExporter(torch.nn.Module):
    def __init__(self, actor_critic, normalizer=None):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder)
        self.num_one_step_obs = int(actor_critic.num_one_step_obs)
        self.num_velocity_features = int(actor_critic.num_velocity_features)
        self.actor_current_frame = getattr(actor_critic, "actor_current_frame", "first")
        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else None

    def forward(self, obs_history: torch.Tensor):
        if self.normalizer is not None:
            obs_history = self.normalizer(obs_history)
        parts = self.estimator(obs_history)
        pred_vel = parts[..., : self.num_velocity_features]
        latent = F.normalize(parts[..., self.num_velocity_features :], dim=-1, p=2.0)
        if self.actor_current_frame == "last":
            current_frame = obs_history[:, -self.num_one_step_obs :]
        else:
            current_frame = obs_history[:, : self.num_one_step_obs]
        actor_input = torch.cat((current_frame, pred_vel, latent), dim=-1)
        return self.actor(actor_input)


def export_him_policy_as_jit(actor_critic, path: str, filename: str = "policy.pt", normalizer=None):
    os.makedirs(path, exist_ok=True)
    exporter = _HIMPolicyExporter(actor_critic, normalizer=normalizer).to("cpu")
    scripted = torch.jit.script(exporter)
    scripted.save(os.path.join(path, filename))


def export_him_policy_as_onnx(actor_critic, path: str, filename: str = "policy.onnx", normalizer=None):
    os.makedirs(path, exist_ok=True)
    exporter = _HIMPolicyExporter(actor_critic, normalizer=normalizer).to("cpu")
    exporter.eval()
    dummy = torch.zeros(1, actor_critic.num_actor_obs, dtype=torch.float32)
    torch.onnx.export(
        exporter,
        dummy,
        os.path.join(path, filename),
        input_names=["obs_history"],
        output_names=["actions"],
        opset_version=17,
    )


def _build_him_deployment_metadata(actor_critic, env) -> dict[str, object]:
    metadata = {
        "num_actor_obs": int(actor_critic.num_actor_obs),
        "num_one_step_obs": int(actor_critic.num_one_step_obs),
        "history_length": int(actor_critic.history_size),
        "num_actions": int(actor_critic.num_actions),
        "estimator_vel_slice": list(actor_critic.vel_slice),
        "estimator_target_slice": list(actor_critic.target_slice),
        "actor_current_frame": getattr(actor_critic, "actor_current_frame", "first"),
        "history_layout": "time_major_current_to_oldest",
    }
    if env is None or not hasattr(env, "get_him_deployment_metadata"):
        return metadata

    env_metadata = env.get_him_deployment_metadata()
    metadata.update(env_metadata)

    policy_terms = env_metadata.get("policy_term_names", [])
    mapped_terms = []
    unsupported_terms = []
    for term_name in policy_terms:
        mapped_term = _AT_ROBOT_LAB_TERM_NAME_MAP.get(term_name)
        if mapped_term is None:
            unsupported_terms.append(term_name)
        else:
            mapped_terms.append(mapped_term)
    if policy_terms:
        metadata["at_robot_lab_term_mapping"] = {
            term_name: _AT_ROBOT_LAB_TERM_NAME_MAP.get(term_name) for term_name in policy_terms
        }
    if unsupported_terms:
        metadata["at_robot_lab_unsupported_terms"] = unsupported_terms
    if mapped_terms and not unsupported_terms:
        metadata["recommended_at_robot_lab_history"] = {
            "num_observations": int(env_metadata["one_step_obs_dim"]),
            "policy_input_dim": int(env_metadata["actor_obs_dim"]),
            "observations": mapped_terms,
            "observations_history": list(range(int(env_metadata["history_length"]))),
            "observations_history_priority": "time",
        }
    return metadata


def export_him_deployment_metadata(
    actor_critic,
    env,
    path: str,
    metadata_filename: str = "policy_metadata.json",
    at_robot_lab_filename: str = "at_robot_lab_history.yaml",
) -> dict[str, str | None]:
    os.makedirs(path, exist_ok=True)

    metadata = _build_him_deployment_metadata(actor_critic, env)
    metadata_path = os.path.join(path, metadata_filename)
    with open(metadata_path, "w", encoding="ascii") as file:
        json.dump(metadata, file, indent=2, sort_keys=False)
        file.write("\n")

    at_robot_lab_path = None
    recommended = metadata.get("recommended_at_robot_lab_history")
    if isinstance(recommended, dict):
        at_robot_lab_path = os.path.join(path, at_robot_lab_filename)
        lines = [
            "# Copy these fields into an AT_robot-lab policy config section.",
            f"# policy_input_dim: {recommended['policy_input_dim']}",
            f"num_observations: {recommended['num_observations']}",
            f"observations: {json.dumps(recommended['observations'])}",
            f"observations_history: {json.dumps(recommended['observations_history'])}",
            f'observations_history_priority: "{recommended["observations_history_priority"]}"',
        ]
        with open(at_robot_lab_path, "w", encoding="ascii") as file:
            file.write("\n".join(lines) + "\n")

    return {
        "metadata_path": metadata_path,
        "at_robot_lab_path": at_robot_lab_path,
    }
