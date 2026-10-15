# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class HIMVecEnvWrapper:
    def __init__(
        self,
        env,
        clip_actions: float,
        num_one_step_obs: int | None = None,
        reorder_history: bool = True,
        history_order: str = "latest_to_oldest",
        policy_frame_term_order: list[str] | tuple[str, ...] | None = None,
        estimator_target_term_names: list[str] | tuple[str, ...] | None = None,
    ):
        self._raw_env = env
        self._wrapped_env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)

        self.num_envs = int(self._wrapped_env.num_envs)
        self.num_actions = int(self._wrapped_env.num_actions)
        self.device = self._wrapped_env.device
        self.max_episode_length = self._wrapped_env.max_episode_length
        # Isaac Lab flattens each term's history independently; HIM expects full
        # observation frames with a task-specific time and term order.
        self.reorder_history = bool(reorder_history)
        if history_order not in {"latest_to_oldest", "oldest_to_latest"}:
            raise ValueError(
                f"Unsupported HIM history_order={history_order!r}; "
                "expected 'latest_to_oldest' or 'oldest_to_latest'."
            )
        self.history_order = history_order
        self.policy_frame_term_order = tuple(policy_frame_term_order or ())
        self.estimator_target_term_names = tuple(estimator_target_term_names or ())

        self.policy_term_slices, self.policy_one_step_term_slices = self._build_group_layout("policy")
        self.critic_term_slices, _ = self._build_group_layout("critic")
        self.policy_term_names = tuple(self.policy_one_step_term_slices.keys())
        self.critic_term_names = tuple(self.critic_term_slices.keys())

        inferred_one_step_obs = self._infer_num_one_step_obs()
        requested_one_step_obs = None if num_one_step_obs is None else int(num_one_step_obs)
        if inferred_one_step_obs is None and requested_one_step_obs is None:
            raise ValueError("Unable to infer HIM one-step actor observation dimension from the policy group.")
        if (
            inferred_one_step_obs is not None
            and requested_one_step_obs is not None
            and inferred_one_step_obs != requested_one_step_obs
        ):
            print(
                "[WARN] HIM policy one-step observation dim mismatch: "
                f"configured={requested_one_step_obs}, inferred={inferred_one_step_obs}. "
                "Using inferred value."
            )
        self.num_one_step_obs = (
            inferred_one_step_obs if inferred_one_step_obs is not None else requested_one_step_obs
        )
        if self.num_one_step_obs is None:
            raise ValueError("HIM one-step actor observation dimension is not available.")
        self.num_one_step_obs = int(self.num_one_step_obs)

        self.default_estimator_vel_slice, self.default_estimator_target_slice = self._infer_estimator_slices()

        actor_obs, critic_obs = self._split_observations(self._wrapped_env.get_observations())
        self._cache_observations(actor_obs, critic_obs)
        self.num_obs = int(actor_obs.shape[-1])
        self.num_privileged_obs = None if critic_obs is None else int(critic_obs.shape[-1])

        print(
            "[INFO] HIM observation manager dimensions: "
            f"policy_terms={self._term_dim_report('policy')}, "
            f"critic_terms={self._term_dim_report('critic')}; "
            f"policy_total={self.num_obs}, critic_total={self.num_privileged_obs}, "
            f"policy_one_step={self.num_one_step_obs}"
        )

        # HIM deployment contract: no heading in velocity_commands and action
        # history must be visible to the estimator/policy input.
        command_slice = self.policy_one_step_term_slices.get("velocity_commands")
        if command_slice is None or command_slice.stop - command_slice.start != 3:
            raise ValueError(
                "HIM requires velocity_commands to be exactly 3-D (vx, vy, yaw), "
                "without heading."
            )
        if "actions" not in self.policy_one_step_term_slices:
            raise ValueError("HIM policy observation history must include the actions term.")

        if self.num_obs % self.num_one_step_obs != 0:
            raise ValueError(
                "Policy observation history is incompatible with HIM. "
                f"actor_obs_dim={self.num_obs}, one_step_dim={self.num_one_step_obs}."
            )
        self.history_length = self.num_obs // self.num_one_step_obs

    @property
    def unwrapped(self):
        return self._wrapped_env.unwrapped

    def reset(self):
        obs_dict, extras = self._wrapped_env.reset()
        actor_obs, critic_obs = self._split_observations(obs_dict)
        self._cache_observations(actor_obs, critic_obs)
        extras = dict(extras)
        if critic_obs is not None:
            extras.setdefault("critic_obs", critic_obs)
        return actor_obs, extras

    def step(self, actions: torch.Tensor):
        obs_dict, rewards, dones, extras = self._wrapped_env.step(actions)
        actor_obs, critic_obs = self._split_observations(obs_dict)
        self._cache_observations(actor_obs, critic_obs)
        extras = dict(extras)
        if critic_obs is not None:
            extras.setdefault("critic_obs", critic_obs)
        return actor_obs, rewards, dones, extras

    def get_observations(self) -> torch.Tensor:
        actor_obs, critic_obs = self._split_observations(self._wrapped_env.get_observations())
        self._cache_observations(actor_obs, critic_obs)
        return actor_obs

    def get_privileged_observations(self) -> torch.Tensor | None:
        if self._latest_critic_obs is not None:
            return self._latest_critic_obs
        _, critic_obs = self._split_observations(self._wrapped_env.get_observations())
        self._latest_critic_obs = critic_obs
        return critic_obs

    def get_critic_observations(self) -> torch.Tensor:
        critic_obs = self.get_privileged_observations()
        if critic_obs is None:
            raise RuntimeError(
                "HIM requires a dedicated 'critic' observation group; "
                "refusing to fall back to actor observations."
            )
        return critic_obs

    def _term_dim_report(self, group_name: str) -> list[dict[str, object]]:
        layout = self.policy_one_step_term_slices if group_name == "policy" else self.critic_term_slices
        return [
            {"name": name, "start": sl.start, "stop": sl.stop, "dim": sl.stop - sl.start}
            for name, sl in layout.items()
        ]

    def get_him_deployment_metadata(self) -> dict[str, object]:
        one_step_obs_dim = self.num_one_step_obs
        history_length = self.history_length
        metadata = {
            "actor_obs_dim": int(self.num_obs),
            "critic_obs_dim": None if self.num_privileged_obs is None else int(self.num_privileged_obs),
            "one_step_obs_dim": one_step_obs_dim,
            "history_length": history_length,
            "history_indices_latest_to_oldest": list(range(history_length)),
            "history_order": self.history_order,
            "history_layout": (
                "time_major_current_to_oldest"
                if self.history_order == "latest_to_oldest"
                else "time_major_oldest_to_current"
            ),
            "policy_frame_term_order": list(self._ordered_policy_term_names()),
            "policy_term_names": list(self.policy_term_names),
            "policy_terms": [
                {
                    "name": term_name,
                    "dim": int(term_slice.stop - term_slice.start),
                    "start": int(term_slice.start),
                    "stop": int(term_slice.stop),
                }
                for term_name, term_slice in self.policy_one_step_term_slices.items()
            ],
            "critic_terms": self._term_dim_report("critic"),
            "critic_term_names": list(self.critic_term_names),
            "default_estimator_vel_slice": self.default_estimator_vel_slice,
            "default_estimator_target_slice": self.default_estimator_target_slice,
        }
        runtime = getattr(self.unwrapped, "cyberstar", None)
        if runtime is not None:
            metadata["cyberstar_contract"] = {
                "default_joint_positions": runtime.default_dof_pos[0].tolist(),
                "joint_names": runtime.dof_names,
                "command_scale": runtime.commands_scale.tolist(),
                "angular_velocity_scale": runtime.obs_scales.ang_vel,
                "joint_velocity_scale": runtime.obs_scales.dof_vel,
                "action_scale": runtime.cfg.control.action_scale,
                "action_clip": runtime.cfg.normalization.clip_actions,
                "cycle_time": runtime.cfg.control.gait_cycle_s,
                "stop_phase": "contact_and_state_dependent_adjustment",
                "reset_history": "repeat_first_valid_frame",
                "policy_dt": runtime.dt,
                "nominal_kp": 30.0,
                "nominal_kd": 1.0,
            }
        return metadata

    def _cache_observations(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor | None):
        self._latest_actor_obs = actor_obs
        self._latest_critic_obs = critic_obs

    def _split_observations(self, observations) -> tuple[torch.Tensor, torch.Tensor | None]:
        actor_obs = self._extract_group_observations(observations, "policy")
        if actor_obs is None:
            raise KeyError("HIM requires a 'policy' observation group.")
        if self.reorder_history:
            actor_obs = self._reorder_policy_history(actor_obs)

        critic_obs = self._extract_group_observations(observations, "critic")
        if critic_obs is None:
            critic_obs = self._compute_group("critic")
        if critic_obs is None:
            raise KeyError("HIM requires a dedicated 'critic' observation group.")
        return actor_obs, critic_obs

    def _reorder_policy_history(self, actor_obs: torch.Tensor) -> torch.Tensor:
        if hasattr(self.unwrapped, "cyberstar"):
            # This task already emits time-major source history, including reset fill.
            return actor_obs
        if not self.policy_term_slices or not self.policy_one_step_term_slices:
            return actor_obs

        term_histories = []
        history_lengths = set()
        for term_name in self._ordered_policy_term_names():
            term_slice = self.policy_term_slices[term_name]
            one_step_slice = self.policy_one_step_term_slices[term_name]
            base_dim = one_step_slice.stop - one_step_slice.start
            total_dim = term_slice.stop - term_slice.start
            if base_dim <= 0 or total_dim % base_dim != 0:
                raise ValueError(
                    f"Invalid HIM policy history layout for term '{term_name}': "
                    f"total_dim={total_dim}, base_dim={base_dim}."
                )
            history_length = total_dim // base_dim
            history_lengths.add(history_length)
            term_histories.append(actor_obs[:, term_slice].reshape(actor_obs.shape[0], history_length, base_dim))

        if len(history_lengths) != 1:
            raise ValueError(f"HIM requires a uniform policy history length, got {sorted(history_lengths)}.")
        history_length = history_lengths.pop()
        if history_length <= 1:
            return actor_obs

        frames = []
        if self.history_order == "latest_to_oldest":
            history_indices = range(history_length - 1, -1, -1)
        else:
            history_indices = range(history_length)
        for history_index in history_indices:
            frames.append(torch.cat([term_history[:, history_index] for term_history in term_histories], dim=-1))
        return torch.cat(frames, dim=-1)

    def _ordered_policy_term_names(self) -> tuple[str, ...]:
        available = tuple(self.policy_term_slices.keys())
        if not self.policy_frame_term_order:
            return available

        missing = [term_name for term_name in self.policy_frame_term_order if term_name not in self.policy_term_slices]
        if missing:
            raise ValueError(f"HIM policy_frame_term_order references missing policy terms: {missing}.")
        extra = [term_name for term_name in available if term_name not in self.policy_frame_term_order]
        if extra:
            raise ValueError(f"HIM policy_frame_term_order omits active policy terms: {extra}.")
        return self.policy_frame_term_order

    def _extract_group_observations(self, observations, group_name: str) -> torch.Tensor | None:
        if observations is None:
            return None

        group_obs = None
        try:
            group_obs = observations[group_name]
        except Exception:
            if isinstance(observations, Mapping):
                group_obs = observations.get(group_name)

        if group_obs is None:
            return None
        return self._flatten_group_tensor(group_obs)

    def _compute_group(self, group_name: str) -> torch.Tensor | None:
        observation_manager = getattr(self.unwrapped, "observation_manager", None)
        if observation_manager is None:
            return None
        try:
            group_obs = observation_manager.compute_group(group_name=group_name)
        except Exception:
            return None
        return self._flatten_group_tensor(group_obs)

    @staticmethod
    def _flatten_group_tensor(group_obs) -> torch.Tensor:
        if not isinstance(group_obs, torch.Tensor):
            group_obs = torch.as_tensor(group_obs)
        if group_obs.ndim == 1:
            group_obs = group_obs.unsqueeze(0)
        elif group_obs.ndim > 2:
            group_obs = group_obs.flatten(start_dim=1)
        return group_obs

    def _infer_num_one_step_obs(self) -> int | None:
        if not self.policy_one_step_term_slices:
            return None
        return max(term_slice.stop for term_slice in self.policy_one_step_term_slices.values())

    def _build_group_layout(self, group_name: str) -> tuple[dict[str, slice], dict[str, slice]]:
        if hasattr(self.unwrapped, "cyberstar"):
            frame = {}
            cursor = 0
            for name, dim in (("phase", 2), ("base_ang_vel", 3), ("projected_gravity", 3),
                              ("velocity_commands", 3), ("joint_pos", 12), ("joint_vel", 12), ("actions", 12)):
                frame[name] = slice(cursor, cursor + dim)
                cursor += dim
            if group_name == "policy":
                return {name: slice(sl.start * 6, sl.stop * 6) for name, sl in frame.items()}, frame
            layout = {name: slice(235 + sl.start, 235 + sl.stop) for name, sl in frame.items()}
            layout["clean_history_prefix"] = slice(0, 235)
            layout["base_lin_vel"] = slice(282, 285)
            layout["privileges"] = slice(285, 491)
            return layout, layout
        observation_manager = getattr(self.unwrapped, "observation_manager", None)
        if observation_manager is None:
            return {}, {}

        term_names = getattr(observation_manager, "_group_obs_term_names", {}).get(group_name, [])
        term_cfgs = getattr(observation_manager, "_group_obs_term_cfgs", {}).get(group_name, [])
        term_dims = observation_manager.group_obs_term_dim.get(group_name, [])
        if not term_names or not term_cfgs or not term_dims:
            return {}, {}

        group_layout: dict[str, slice] = {}
        one_step_layout: dict[str, slice] = {}
        group_cursor = 0
        one_step_cursor = 0

        for term_name, term_cfg, term_dim in zip(term_names, term_cfgs, term_dims, strict=False):
            total_dim = self._shape_numel(term_dim)
            base_dim = self._infer_base_term_dim(term_cfg, total_dim)
            group_layout[term_name] = slice(group_cursor, group_cursor + total_dim)
            one_step_layout[term_name] = slice(one_step_cursor, one_step_cursor + base_dim)
            group_cursor += total_dim
            one_step_cursor += base_dim

        return group_layout, one_step_layout

    def _infer_base_term_dim(self, term_cfg, total_dim: int) -> int:
        try:
            sample = term_cfg.func(self.unwrapped, **term_cfg.params)
            sample = self._flatten_group_tensor(sample)
            return int(sample.shape[-1])
        except Exception:
            history_length = int(getattr(term_cfg, "history_length", 0) or 0)
            if history_length <= 1 or total_dim % history_length != 0:
                return total_dim
            return total_dim // history_length

    def _infer_estimator_slices(self) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        vel_slice = self._term_slice(self.critic_term_slices, "base_lin_vel")
        if self.estimator_target_term_names:
            target_slice = self._target_indices_from_terms(self.estimator_target_term_names)
        else:
            target_slice = self._shared_policy_target_slice()
        return vel_slice, target_slice

    def _target_indices_from_terms(self, term_names: tuple[str, ...]) -> tuple[int, ...] | None:
        indices: list[int] = []
        for term_name in term_names:
            term_slice = self.critic_term_slices.get(term_name)
            if term_slice is None:
                return None
            indices.extend(range(term_slice.start, term_slice.stop))
        return tuple(indices)

    @staticmethod
    def _term_slice(layout: dict[str, slice], term_name: str) -> tuple[int, int] | None:
        term_slice = layout.get(term_name)
        if term_slice is None:
            return None
        return (term_slice.start, term_slice.stop)

    def _shared_policy_target_slice(self) -> tuple[int, int] | None:
        if not self.policy_term_slices or not self.critic_term_slices:
            return None

        # HIM target is exactly policy terms shared with critic, excluding the
        # privileged estimator target (base_lin_vel). Do not include future
        # shared terms implicitly unless they are part of the policy input.
        shared_names = [
            name for name in self.policy_term_slices
            if name != "base_lin_vel" and name in self.critic_term_slices
        ]
        if not shared_names:
            return None

        slices = [self.critic_term_slices[name] for name in shared_names]
        start = slices[0].start
        stop = slices[0].stop
        for term_slice in slices[1:]:
            if term_slice.start != stop:
                return None
            stop = term_slice.stop
        return (start, stop)

    @staticmethod
    def _shape_numel(shape) -> int:
        if isinstance(shape, int):
            return int(shape)
        if len(shape) == 0:
            return 1
        numel = 1
        for dim in shape:
            numel *= int(dim)
        return numel

    def __getattr__(self, name: str):
        return getattr(self._wrapped_env, name)
