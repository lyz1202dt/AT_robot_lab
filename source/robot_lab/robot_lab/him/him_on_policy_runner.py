# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from .him_actor_critic import HIMActorCritic
from .him_ppo import HIMPPO


class HIMOnPolicyRunner:
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        self.cfg = train_cfg
        self.policy_cfg = dict(train_cfg.get("policy", {}))
        self.alg_cfg = dict(train_cfg.get("algorithm", {}))
        self.device = device
        self.env = env
        self.num_steps_per_env = int(train_cfg["num_steps_per_env"])
        self.save_interval = int(train_cfg["save_interval"])
        self.max_iterations = int(train_cfg.get("max_iterations", 0))

        self.policy_cfg.pop("num_one_step_obs", None)
        if self.env.num_privileged_obs is None:
            raise ValueError(
                "HIM requires a dedicated critic observation group; "
                "actor observations cannot be used as a silent fallback."
            )
        num_critic_obs = self.env.num_privileged_obs
        self._resolve_estimator_slices(num_critic_obs)
        actor_critic = HIMActorCritic(
            self.env.num_obs,
            num_critic_obs,
            self.env.num_one_step_obs,
            self.env.num_actions,
            **self.policy_cfg,
        )
        print(
            "[INFO] HIM observation layout: "
            f"actor_obs={self.env.num_obs}, one_step_obs={self.env.num_one_step_obs}, "
            f"history={actor_critic.history_size}, critic_obs={num_critic_obs}, "
            f"vel_slice={actor_critic.vel_slice}, target_slice={actor_critic.target_slice}"
        )
        self.alg = HIMPPO(actor_critic, device=self.device, **self.alg_cfg)
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self._session_time = 0.0
        self._session_iterations = 0
        self.current_learning_iteration = 0
        self.latest_episode_metrics: dict[str, float] = {}
        self.latest_mean_reward: float | None = None
        self.latest_mean_episode_length: float | None = None
        self.episode_metric_history: deque[dict[str, torch.Tensor | float | int]] = deque(maxlen=100)

        _, reset_infos = self.env.reset()
        self._update_episode_metric_cache(self._extract_log_dict(reset_infos))

    def _resolve_estimator_slices(self, num_critic_obs: int):
        vel_slice = self._select_slice(
            self.policy_cfg.get("estimator_vel_slice"),
            getattr(self.env, "default_estimator_vel_slice", None),
            num_critic_obs,
            "estimator_vel_slice",
        )
        target_slice = self._select_target_spec(
            self.policy_cfg.get("estimator_target_slice"),
            getattr(self.env, "default_estimator_target_slice", None),
            num_critic_obs,
            "estimator_target_slice",
        )
        if vel_slice is None:
            raise ValueError("HIM requires a base_lin_vel term in the critic observation group.")
        if target_slice is None:
            raise ValueError("HIM could not infer a contiguous estimator target from policy/critic observations.")
        self.policy_cfg["estimator_vel_slice"] = vel_slice
        self.policy_cfg["estimator_target_slice"] = target_slice

    @staticmethod
    def _is_valid_slice(obs_slice: tuple[int, int] | list[int] | None, upper_bound: int) -> bool:
        if obs_slice is None:
            return False
        if len(obs_slice) != 2:
            return False
        start, stop = int(obs_slice[0]), int(obs_slice[1])
        return 0 <= start < stop <= upper_bound

    def _select_slice(
        self,
        configured_slice: tuple[int, int] | list[int] | None,
        inferred_slice: tuple[int, int] | None,
        upper_bound: int,
        slice_name: str,
    ) -> tuple[int, int] | None:
        if self._is_valid_slice(inferred_slice, upper_bound):
            resolved_slice = (int(inferred_slice[0]), int(inferred_slice[1]))
            configured_valid = self._is_valid_slice(configured_slice, upper_bound)
            configured_value = None
            if configured_valid:
                configured_value = (int(configured_slice[0]), int(configured_slice[1]))
            if configured_slice is not None and configured_value != resolved_slice:
                print(
                    f"[WARN] HIM {slice_name}={tuple(configured_slice)} does not match the environment layout. "
                    f"Using inferred slice {resolved_slice}."
                )
            return resolved_slice
        if self._is_valid_slice(configured_slice, upper_bound):
            return (int(configured_slice[0]), int(configured_slice[1]))
        if configured_slice is not None:
            raise ValueError(f"Invalid HIM {slice_name}={tuple(configured_slice)} for critic dim {upper_bound}.")
        return None

    @staticmethod
    def _is_valid_target_spec(target_spec: tuple[int, ...] | list[int] | None, upper_bound: int) -> bool:
        if target_spec is None:
            return False
        if len(target_spec) == 2 and 0 <= int(target_spec[0]) < int(target_spec[1]) <= upper_bound:
            return True
        return bool(target_spec) and all(0 <= int(index) < upper_bound for index in target_spec)

    def _select_target_spec(
        self,
        configured_spec: tuple[int, ...] | list[int] | None,
        inferred_spec: tuple[int, ...] | list[int] | None,
        upper_bound: int,
        slice_name: str,
    ) -> tuple[int, ...] | None:
        if self._is_valid_target_spec(inferred_spec, upper_bound):
            resolved_spec = tuple(int(index) for index in inferred_spec)
            configured_valid = self._is_valid_target_spec(configured_spec, upper_bound)
            configured_value = None
            if configured_valid:
                configured_value = tuple(int(index) for index in configured_spec)
            if configured_spec is not None and configured_value != resolved_spec:
                print(
                    f"[WARN] HIM {slice_name}={tuple(configured_spec)} does not match the environment layout. "
                    f"Using inferred target spec with {len(resolved_spec)} entries."
                )
            return resolved_spec
        if self._is_valid_target_spec(configured_spec, upper_bound):
            return tuple(int(index) for index in configured_spec)
        if configured_spec is not None:
            raise ValueError(f"Invalid HIM {slice_name}={tuple(configured_spec)} for critic dim {upper_bound}.")
        return None

    def add_git_repo_to_log(self, source_file: str):
        del source_file

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "estimator_optimizer_state_dict": self.alg.actor_critic.estimator.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "tot_time": self.tot_time,
            "tot_timesteps": self.tot_timesteps,
            "infos": {
                "num_one_step_obs": self.alg.actor_critic.num_one_step_obs,
                "vel_slice": self.alg.actor_critic.vel_slice,
                "target_slice": self.alg.actor_critic.target_slice,
            },
        }
        torch.save(payload, path)

    def load(self, path: str, load_optimizer: bool | None = None):
        loaded_dict = torch.load(path, map_location=self.device)
        model_changed = self._load_model_state(loaded_dict["model_state_dict"])
        if load_optimizer is None:
            load_optimizer = self.cfg.get("load_optimizer", True)
        if load_optimizer and model_changed:
            print("[WARN] HIM model tensors changed during loading. Optimizers start fresh.")
            load_optimizer = False
        if load_optimizer and "optimizer_state_dict" in loaded_dict:
            self._load_optimizer_state(self.alg.optimizer, loaded_dict["optimizer_state_dict"], "HIM")
        if load_optimizer and "estimator_optimizer_state_dict" in loaded_dict:
            self._load_optimizer_state(
                self.alg.actor_critic.estimator.optimizer,
                loaded_dict["estimator_optimizer_state_dict"], "HIM estimator",
            )
        elif load_optimizer:
            print("[WARN] HIM checkpoint has no estimator optimizer state. Estimator optimizer starts fresh.")
        self.current_learning_iteration = int(loaded_dict.get("iter", 0))
        # Older checkpoints do not contain cumulative statistics. Keep their
        # unknown history out of the ETA, which uses only this learning session.
        self.tot_time = float(loaded_dict.get("tot_time", 0.0))
        self.tot_timesteps = int(loaded_dict.get("tot_timesteps", 0))
        if "tot_time" not in loaded_dict or "tot_timesteps" not in loaded_dict:
            print("[WARN] HIM checkpoint has no complete timing/step history. Missing counters start at zero.")

    @staticmethod
    def _load_optimizer_state(optimizer, state_dict, name):
        # Adam accepts mismatched moment shapes in load_state_dict and fails only
        # on step(). Validate against live parameters before installing any state.
        previous_state = optimizer.state_dict()
        try:
            groups = state_dict["param_groups"]
            if len(groups) != len(optimizer.param_groups):
                raise ValueError("Parameter group count changed")
            for saved_group, current_group in zip(groups, optimizer.param_groups, strict=True):
                if len(saved_group["params"]) != len(current_group["params"]):
                    raise ValueError("Parameter count changed")
                for saved_id, param in zip(saved_group["params"], current_group["params"], strict=True):
                    for key, value in state_dict["state"].get(saved_id, {}).items():
                        if isinstance(value, torch.Tensor) and key != "step" and value.shape != param.shape:
                            raise ValueError(f"{key}: checkpoint={tuple(value.shape)}, current={tuple(param.shape)}")
            optimizer.load_state_dict(state_dict)
        except (KeyError, RuntimeError, ValueError) as error:
            optimizer.load_state_dict(previous_state)
            print(f"[WARN] {name} optimizer state is incompatible; keeping current state: {error}")

    def _load_model_state(self, loaded_state: dict[str, torch.Tensor]):
        current_state = self.alg.actor_critic.state_dict()
        critic_weight = loaded_state.get("critic.0.weight")
        critic_changed = critic_weight is not None and critic_weight.shape != current_state["critic.0.weight"].shape
        if hasattr(self.env.unwrapped, "cyberstar"):
            from .source_checkpoint import migrate_source_state

            loaded_state = migrate_source_state(loaded_state, current_state)
        compatible_state = {}
        skipped = []
        for key, value in loaded_state.items():
            current_value = current_state.get(key)
            if current_value is not None and current_value.shape == value.shape:
                compatible_state[key] = value
            else:
                current_shape = None if current_value is None else tuple(current_value.shape)
                skipped.append((key, tuple(value.shape), current_shape))
        current_state.update(compatible_state)
        self.alg.actor_critic.load_state_dict(current_state, strict=False)
        if skipped:
            print(f"[WARN] Skipped {len(skipped)} incompatible HIM checkpoint tensors:")
            for key, checkpoint_shape, current_shape in skipped[:20]:
                print(f"  - {key}: checkpoint={checkpoint_shape}, current={current_shape}")
            if len(skipped) > 20:
                print(f"  - ... {len(skipped) - 20} more tensors skipped")
        if not compatible_state:
            raise RuntimeError("No compatible HIM model tensors were found in the checkpoint.")
        return critic_changed or bool(skipped) or len(compatible_state) != len(current_state)

    def get_inference_policy(self, device: str | None = None):
        if device is None:
            device = self.device
        self.alg.actor_critic.eval()

        def policy(obs: torch.Tensor):
            with torch.inference_mode():
                return self.alg.actor_critic.act_inference(obs.to(device))

        return policy

    def _get_critic_observations(self) -> torch.Tensor:
        privileged_obs = self.env.get_critic_observations()
        if privileged_obs is None:
            return self.env.get_observations()
        return privileged_obs

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len and hasattr(self.env, "episode_length_buf") and hasattr(self.env, "max_episode_length"):
            env_unwrapped = getattr(self.env, "unwrapped", self.env)
            env_unwrapped.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        critic_obs = self._get_critic_observations().to(self.device)
        self.alg.actor_critic.train()

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        ep_infos = []
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        self._session_time = 0.0
        self._session_iterations = 0
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for iteration in range(self.current_learning_iteration, tot_iter):
            task_runtime = getattr(self.env.unwrapped, "cyberstar", None)
            if task_runtime is not None:
                task_runtime.ppo_iteration = iteration
            start = time.time()
            episodes_completed = 0
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, rewards, dones, infos = self.env.step(actions)
                    obs = obs.to(self.device)
                    critic_obs = self._get_critic_observations().to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos, critic_obs)

                    cur_reward_sum += rewards.view(-1)
                    cur_episode_length += 1
                    done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                    if done_ids.numel() > 0:
                        episodes_completed += done_ids.numel()
                        rewbuffer.extend(cur_reward_sum[done_ids].detach().cpu().tolist())
                        lenbuffer.extend(cur_episode_length[done_ids].detach().cpu().tolist())
                        self._append_episode_log(ep_infos, infos)
                        cur_reward_sum[done_ids] = 0
                        cur_episode_length[done_ids] = 0

                collection_time = time.time() - start
                learn_start = time.time()
                self.alg.compute_returns(critic_obs)

            (
                mean_value_loss,
                mean_surrogate_loss,
                mean_estimation_loss,
                mean_swap_loss,
                mean_entropy_loss,
            ) = self.alg.update()
            learn_time = time.time() - learn_start

            self._log(
                iteration,
                tot_iter,
                collection_time,
                learn_time,
                mean_value_loss,
                mean_surrogate_loss,
                mean_estimation_loss,
                mean_swap_loss,
                mean_entropy_loss,
                rewbuffer,
                lenbuffer,
                ep_infos,
                episodes_completed,
                float(cur_reward_sum.mean().item()),
                float(cur_episode_length.mean().item()),
            )
            ep_infos.clear()

            self.current_learning_iteration = iteration + 1
            if self.log_dir is not None and self.current_learning_iteration % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _log(
        self,
        iteration: int,
        total_iterations: int,
        collection_time: float,
        learn_time: float,
        mean_value_loss: float,
        mean_surrogate_loss: float,
        mean_estimation_loss: float,
        mean_swap_loss: float,
        mean_entropy_loss: float,
        rewbuffer: deque,
        lenbuffer: deque,
        ep_infos: list[dict[str, torch.Tensor | float | int]],
        episodes_completed: int,
        partial_mean_reward: float,
        partial_mean_episode_length: float,
    ):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += collection_time + learn_time
        iteration_time = collection_time + learn_time
        self._session_time += iteration_time
        self._session_iterations += 1
        remaining_iterations = max(total_iterations - iteration - 1, 0)
        eta = self._session_time / self._session_iterations * remaining_iterations
        fps = int(self.num_steps_per_env * self.env.num_envs / max(collection_time + learn_time, 1e-6))
        mean_reward = statistics.mean(rewbuffer) if rewbuffer else None
        mean_episode_length = statistics.mean(lenbuffer) if lenbuffer else None
        mean_noise_std = self.alg.actor_critic.std.mean().item()
        self.episode_metric_history.extend(info.copy() for info in ep_infos)
        episode_metrics = self._aggregate_episode_infos(list(self.episode_metric_history))
        if mean_reward is not None:
            self.latest_mean_reward = mean_reward
        if mean_episode_length is not None:
            self.latest_mean_episode_length = mean_episode_length
        if episode_metrics:
            self.latest_episode_metrics = dict(episode_metrics)

        display_reward = mean_reward
        display_episode_length = mean_episode_length
        if display_reward is None:
            display_reward = self.latest_mean_reward if self.latest_mean_reward is not None else partial_mean_reward
        if display_episode_length is None:
            if self.latest_mean_episode_length is not None:
                display_episode_length = self.latest_mean_episode_length
            else:
                display_episode_length = partial_mean_episode_length
        display_episode_metrics = episode_metrics if episode_metrics else list(self.latest_episode_metrics.items())
        running_metrics = self._collect_running_episode_metrics(partial_mean_reward, partial_mean_episode_length)

        if self.writer is not None:
            self.writer.add_scalar("Loss/value_function", mean_value_loss, iteration)
            self.writer.add_scalar("Loss/surrogate", mean_surrogate_loss, iteration)
            self.writer.add_scalar("Loss/estimation", mean_estimation_loss, iteration)
            self.writer.add_scalar("Loss/swap", mean_swap_loss, iteration)
            self.writer.add_scalar("Loss/entropy", mean_entropy_loss, iteration)
            self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, iteration)
            self.writer.add_scalar(
                "Loss/estimator_learning_rate",
                self.alg.actor_critic.estimator.learning_rate,
                iteration,
            )
            self.writer.add_scalar("Policy/mean_noise_std", mean_noise_std, iteration)
            self.writer.add_scalar("Perf/fps", fps, iteration)
            self.writer.add_scalar("Perf/collection_time", collection_time, iteration)
            self.writer.add_scalar("Perf/learning_time", learn_time, iteration)
            self.writer.add_scalar("Train/mean_reward", display_reward, iteration)
            self.writer.add_scalar("Train/mean_episode_length", display_episode_length, iteration)
            self.writer.add_scalar("Train/episodes_completed", episodes_completed, iteration)
            self.writer.add_scalar("Train/episode_metric_batches", len(self.episode_metric_history), iteration)
            for key, value in display_episode_metrics:
                self.writer.add_scalar(key, value, iteration)
            for key, value in running_metrics.items():
                self.writer.add_scalar(key, value, iteration)

        width = 80
        pad = 35
        title = f" Learning iteration {iteration + 1}/{total_iterations} "
        log_string = f"{'#' * width}\n{title.center(width, ' ')}\n\n"
        log_string += (
            f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning {learn_time:.3f}s)\n"
            f"{'Mean action noise std:':>{pad}} {mean_noise_std:.2f}\n"
            f"{'Mean value_function loss:':>{pad}} {mean_value_loss:.4f}\n"
            f"{'Mean surrogate loss:':>{pad}} {mean_surrogate_loss:.4f}\n"
            f"{'Mean estimation loss:':>{pad}} {mean_estimation_loss:.4f}\n"
            f"{'Mean swap loss:':>{pad}} {mean_swap_loss:.4f}\n"
            f"{'Mean entropy loss:':>{pad}} {mean_entropy_loss:.4f}\n"
        )
        log_string += (
            f"{'Mean reward:':>{pad}} {display_reward:.2f}\n"
            f"{'Mean episode length:':>{pad}} {display_episode_length:.2f}\n"
            f"{'Episodes completed this iteration:':>{pad}} {episodes_completed}\n"
            f"{'Episode metric history batches:':>{pad}} {len(self.episode_metric_history)}\n"
        )
        for key, value in display_episode_metrics:
            log_string += f"{key:>{pad}}: {value:.4f}\n"
        for key, value in running_metrics.items():
            log_string += f"{key:>{pad}}: {value:.4f}\n"
        log_string += (
            f"{'-' * width}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Time elapsed:':>{pad}} {self._format_duration(self.tot_time)}\n"
            f"{'ETA:':>{pad}} {self._format_duration(eta)}\n"
        )
        print(log_string, flush=True)

    def _collect_running_episode_metrics(self, running_reward: float, running_length: float) -> dict[str, float]:
        """Snapshot metrics for in-progress episodes once per rollout iteration."""
        env = getattr(self.env, "unwrapped", self.env)
        metrics = {
            "Running/episode_reward": running_reward,
            "Running/episode_length": running_length,
        }

        reward_sums = getattr(getattr(env, "reward_manager", None), "_episode_sums", {})
        for term_name, values in reward_sums.items():
            metrics[f"Running/Episode_Reward/{term_name}"] = self._mean_metric_value(values)

        command_manager = getattr(env, "command_manager", None)
        if command_manager is not None:
            command_term = command_manager.get_term("base_velocity")
            for name, values in command_term.metrics.items():
                metrics[f"Running/Metrics/base_velocity/{name}"] = self._mean_metric_value(values)

        terrain = getattr(getattr(env, "scene", None), "terrain", None)
        terrain_levels = getattr(terrain, "terrain_levels", None)
        if terrain_levels is not None:
            metrics["Running/Curriculum/terrain_levels"] = self._mean_metric_value(terrain_levels)
        return metrics

    @staticmethod
    def _mean_metric_value(value) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().mean().item())
        return float(value)

    @staticmethod
    def _append_episode_log(ep_infos: list[dict], infos: dict):
        ep_info = HIMOnPolicyRunner._extract_log_dict(infos)
        if isinstance(ep_info, dict) and ep_info:
            ep_infos.append(ep_info.copy())

    @staticmethod
    def _extract_log_dict(infos: dict | None) -> dict | None:
        if not isinstance(infos, dict):
            return None
        ep_info = infos.get("log")
        return ep_info if isinstance(ep_info, dict) else None

    def _update_episode_metric_cache(self, ep_info: dict | None):
        if not ep_info:
            return
        for key, value in self._aggregate_episode_infos([ep_info]):
            self.latest_episode_metrics[key] = value

    def _aggregate_episode_infos(
        self, ep_infos: list[dict[str, torch.Tensor | float | int]]
    ) -> list[tuple[str, float]]:
        if not ep_infos:
            return []
        metrics: list[tuple[str, float]] = []
        ordered_keys = self._ordered_metric_keys(ep_infos)
        for key in ordered_keys:
            values = []
            for ep_info in ep_infos:
                if key not in ep_info:
                    continue
                value = ep_info[key]
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor([value], device=self.device, dtype=torch.float32)
                else:
                    value = value.to(self.device, dtype=torch.float32)
                if value.ndim == 0:
                    value = value.unsqueeze(0)
                values.append(value.flatten())
            if values:
                metrics.append((key, torch.cat(values).mean().item()))
        return metrics

    @staticmethod
    def _ordered_metric_keys(ep_infos: list[dict[str, torch.Tensor | float | int]]) -> list[str]:
        priority_keys = [
            "Metrics/base_velocity/error_vel_x",
            "Metrics/base_velocity/error_vel_y",
            "Metrics/base_velocity/error_vel_z",
            "Metrics/base_velocity/error_vel_dir",
            "Metrics/base_velocity/error_vel_xy",
            "Metrics/base_velocity/error_vel_yaw",
        ]
        keys = []
        seen = set()
        for key in priority_keys:
            for ep_info in ep_infos:
                if key in ep_info and key not in seen:
                    keys.append(key)
                    seen.add(key)
                    break
        for ep_info in ep_infos:
            for key in ep_info:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        return keys

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = max(int(seconds), 0)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
