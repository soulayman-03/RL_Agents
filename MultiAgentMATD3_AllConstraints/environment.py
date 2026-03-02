from __future__ import annotations

import dataclasses
import random
from collections import Counter
from typing import Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from integrated_system.resource_manager import ResourceManager

try:
    from SingleAgent.utils import DNNLayer, generate_specific_model, set_global_seed
except ModuleNotFoundError:
    from utils import DNNLayer, generate_specific_model, set_global_seed


class MultiAgentIoTEnvAllConstraints(gym.Env):
    """
    Unified multi-agent environment that combines *all constraints*:

    - Resource constraints via ResourceManager:
      sequential_diversity, privacy_clearance, memory, compute, bandwidth,
      privacy_exposure and security_level_exposure (S_l / max_exposure_fraction).
    - Energy hard constraint (per-device battery budget per episode).
    - Trust constraint (hard or soft), based on a minimal trust threshold.

    Reward (success):
      reward = -(t_comp + t_comm + trust_penalty_soft)
      then mixed with group reward: 0.7 individual + 0.3 group average.

    Failures (hard constraints):
      reward = -500.0 and agent terminates, with fail.reason in {"energy","trust",...}.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        *,
        num_agents: int = 3,
        num_devices: int = 5,
        model_types: list[str] | None = None,
        seed: int | None = None,
        shuffle_allocation_order: bool = True,
        max_exposure_fraction: float | None = None,
        queue_per_device: bool = False,
        max_fail_logs_per_episode: int = 0,
        # Privacy (multi-level)
        privacy_max_level: int = 3,
        privacy_profile: str = "linear_front_loaded",
        # Trust
        trust_score_min: float = 0.5,
        trust_score_max: float = 1.0,
        trust_min_for_max_privacy: float = 0.8,
        trust_hard: bool = True,
        trust_lambda: float = 5.0,
        # Energy
        energy_budget_range: tuple[float, float] = (500.0, 1200.0),
        alpha_comp: float = 1.0,
        alpha_comm: float = 1.0,
    ):
        super().__init__()
        self.num_agents = int(num_agents)
        self.num_devices = int(num_devices)
        self.seed = seed
        self.shuffle_allocation_order = bool(shuffle_allocation_order)
        self.max_exposure_fraction = max_exposure_fraction
        self.queue_per_device = bool(queue_per_device)
        self.max_fail_logs_per_episode = max(0, int(max_fail_logs_per_episode))
        self._fail_logs_left = self.max_fail_logs_per_episode

        self.privacy_max_level = max(1, int(privacy_max_level))
        self.privacy_profile = str(privacy_profile or "linear_front_loaded")

        self.trust_score_min = float(trust_score_min)
        self.trust_score_max = float(trust_score_max)
        if not (0.0 <= self.trust_score_min <= 1.0 and 0.0 <= self.trust_score_max <= 1.0):
            raise ValueError("trust_score_min/max must be within [0,1].")
        if self.trust_score_max < self.trust_score_min:
            raise ValueError("trust_score_max must be >= trust_score_min.")
        self.trust_min_for_max_privacy = float(trust_min_for_max_privacy)
        self.trust_hard = bool(trust_hard)
        self.trust_lambda = float(trust_lambda)

        self.energy_budget_range = (float(energy_budget_range[0]), float(energy_budget_range[1]))
        self.alpha_comp = float(alpha_comp)
        self.alpha_comm = float(alpha_comm)

        if model_types is None:
            self.model_types = ["lenet"] * self.num_agents
        else:
            self.model_types = list(model_types)
            if len(self.model_types) < self.num_agents:
                self.model_types.extend(["lenet"] * (self.num_agents - len(self.model_types)))

        set_global_seed(seed)
        self._py_rng = random.Random(seed if seed is not None else None)

        self.resource_manager = ResourceManager(self.num_devices)
        try:
            self.resource_manager.set_max_exposure_fraction(self.max_exposure_fraction)
        except AttributeError:
            pass
        self.resource_manager.reset_devices_with_seed(self.num_devices, seed)
        self._set_multilevel_device_privacy()

        self.device_trust: dict[int, float] = {}
        self._init_device_trust()

        self.device_energy_init: dict[int, float] = {}
        self.device_energy_remaining: dict[int, float] = {}
        self._init_energy_budgets()

        # Spaces
        self.layer_feature_dim = 4
        self.prev_device_feature_dim = self.num_devices + 1
        # Per device: cpu, mem_free, bw, privacy_clearance, step_compute, step_bw, energy_ratio, trust_score
        self.per_device_dim = 8
        self.single_state_dim = 1 + self.layer_feature_dim + self.prev_device_feature_dim + (self.per_device_dim * self.num_devices)
        self.action_spaces = {aid: spaces.Discrete(self.num_devices) for aid in range(self.num_agents)}
        self.observation_spaces = {
            aid: spaces.Box(low=0.0, high=2.0, shape=(self.single_state_dim,), dtype=np.float32) for aid in range(self.num_agents)
        }

        self.tasks: Dict[int, List[DNNLayer]] = {}
        self.agent_progress: Dict[int, int] = {}
        self.agent_prev_device: Dict[int, int] = {}
        self.agent_done: Dict[int, bool] = {}

        self.reset()

    def _set_multilevel_device_privacy(self) -> None:
        devices = list(getattr(self.resource_manager, "devices", {}).values())
        if not devices:
            return
        for d in devices:
            d.privacy_clearance = int(self._py_rng.randint(0, self.privacy_max_level))
        # Ensure at least 2 devices can handle max privacy.
        min_high = min(2, len(devices))
        high = [d for d in devices if int(getattr(d, "privacy_clearance", 0)) >= self.privacy_max_level]
        if len(high) < min_high:
            candidates = [d for d in devices if int(getattr(d, "privacy_clearance", 0)) < self.privacy_max_level]
            for d in self._py_rng.sample(candidates, k=min_high - len(high)):
                d.privacy_clearance = int(self.privacy_max_level)

    def _init_device_trust(self) -> None:
        # Deterministic per-device trust in [trust_score_min .. trust_score_max).
        span = max(0.0, float(self.trust_score_max) - float(self.trust_score_min))
        self.device_trust = {
            int(d_id): float(self.trust_score_min + span * self._py_rng.random()) for d_id in range(self.num_devices)
        }

    def _init_energy_budgets(self) -> None:
        lo, hi = self.energy_budget_range
        self.device_energy_init = {d: float(self._py_rng.uniform(lo, hi)) for d in range(self.num_devices)}
        self.device_energy_remaining = dict(self.device_energy_init)

    def _energy_ratio(self, device_id: int) -> float:
        init_e = float(self.device_energy_init.get(device_id, 1.0))
        if init_e <= 0:
            return 0.0
        return float(self.device_energy_remaining.get(device_id, 0.0)) / init_e

    def _energy_cost(self, layer: DNNLayer, transmission_data_size: float) -> float:
        return (self.alpha_comp * float(getattr(layer, "computation_demand", 0.0))) + (
            self.alpha_comm * float(transmission_data_size)
        )

    def _apply_privacy_profile(self, layers: list[DNNLayer]) -> list[DNNLayer]:
        n = len(layers)
        if n <= 0:
            return layers
        prof = self.privacy_profile.strip().lower()
        max_p = int(self.privacy_max_level)

        if prof in {"first_layer_max", "first"}:
            levels = [max_p] + [0] * (n - 1)
        elif prof in {"linear_front_loaded", "linear"}:
            if n == 1:
                levels = [max_p]
            else:
                levels = [int(round((1.0 - (i / (n - 1))) * max_p)) for i in range(n)]
        elif prof in {"random"}:
            levels = [int(self._py_rng.randint(0, max_p)) for _ in range(n)]
            levels[0] = max(levels[0], min(1, max_p))
        else:
            levels = [max_p] + [0] * (n - 1)

        out: list[DNNLayer] = []
        for layer, lvl in zip(layers, levels, strict=True):
            out.append(dataclasses.replace(layer, privacy_level=int(max(0, min(max_p, int(lvl))))))
        return out

    def _trust_required(self, privacy_level: int) -> float:
        p = float(max(0, min(int(privacy_level), int(self.privacy_max_level))))
        denom = float(max(1, int(self.privacy_max_level)))
        tmin = float(max(0.0, min(1.0, self.trust_min_for_max_privacy)))
        return float(tmin * (p / denom))

    @staticmethod
    def _trust_shortfall(trust_required: float, trust_score: float) -> float:
        return float(max(0.0, float(trust_required) - float(trust_score)))

    def reset(self):
        try:
            self.resource_manager.set_max_exposure_fraction(self.max_exposure_fraction)
        except AttributeError:
            pass
        self.resource_manager.reset(self.num_devices)
        self._fail_logs_left = self.max_fail_logs_per_episode
        self.device_energy_remaining = dict(self.device_energy_init)

        self.tasks = {}
        self.agent_progress = {}
        self.agent_prev_device = {}
        self.agent_done = {}

        for i in range(self.num_agents):
            layers = list(generate_specific_model(self.model_types[i]) or [])
            self.tasks[i] = self._apply_privacy_profile(layers)
            self.agent_progress[i] = 0
            self.agent_prev_device[i] = -1
            self.agent_done[i] = False

        return self._get_observations(), {}

    def _get_observations(self):
        return {i: self._get_agent_observation(i) for i in range(self.num_agents)}

    def _get_agent_observation(self, agent_id: int):
        if self.agent_done[agent_id]:
            return np.zeros(self.single_state_dim, dtype=np.float32)

        progress_val = int(self.agent_progress[agent_id])
        task = self.tasks[agent_id]
        progress_norm = float(progress_val) / float(max(1, len(task)))
        current_obs: list[float] = [progress_norm]

        current_layer = task[progress_val]
        current_obs.extend(
            [
                float(getattr(current_layer, "computation_demand", 0.0)) / 20.0,
                float(getattr(current_layer, "memory_demand", 0.0)) / 200.0,
                float(getattr(current_layer, "output_data_size", 0.0)) / 25.0,
                float(getattr(current_layer, "privacy_level", 0.0)) / float(self.privacy_max_level),
            ]
        )

        prev_dev = int(self.agent_prev_device[agent_id])
        prev_valid = 1.0 if prev_dev != -1 else 0.0
        current_obs.append(prev_valid)
        for d_id in range(self.num_devices):
            current_obs.append(1.0 if prev_dev == d_id else 0.0)

        for d_id in range(self.num_devices):
            d = self.resource_manager.devices[d_id]
            step_load = self.resource_manager.step_resources[d_id]
            current_obs.extend(
                [
                    float(getattr(d, "cpu_speed", 0.0)) / 50.0,
                    float(getattr(d, "memory_capacity", 0.0) - getattr(d, "current_memory_usage", 0.0)) / 600.0,
                    float(getattr(d, "bandwidth", 0.0)) / 300.0,
                    float(getattr(d, "privacy_clearance", 0.0)) / float(self.privacy_max_level),
                    float(step_load.get("compute", 0.0)) / 100.0,
                    float(step_load.get("bw", 0.0)) / 50.0,
                    float(self._energy_ratio(d_id)),
                    float(self.device_trust.get(int(d_id), 0.0)),
                ]
            )

        return np.asarray(current_obs, dtype=np.float32)

    def get_valid_actions(self) -> Dict[int, List[int]]:
        self.resource_manager.reset_step_resources()
        valid_actions_dict: dict[int, list[int]] = {}

        for agent_id in range(self.num_agents):
            if self.agent_done[agent_id]:
                valid_actions_dict[agent_id] = []
                continue

            progress = int(self.agent_progress[agent_id])
            task = self.tasks[agent_id]
            current_layer = task[progress]
            total_layers = len(task)
            is_first = progress == 0
            prev_dev = int(self.agent_prev_device[agent_id])
            prev_out = float(task[progress - 1].output_data_size) if progress > 0 else 5.0

            valid: list[int] = []
            for device_id in range(self.num_devices):
                trans_data = 0.0
                if prev_dev != -1 and prev_dev != device_id:
                    trans_data = float(prev_out)
                elif prev_dev == -1:
                    trans_data = 5.0

                # Energy hard constraint
                cost = self._energy_cost(current_layer, trans_data)
                if float(self.device_energy_remaining.get(device_id, 0.0)) < float(cost):
                    continue

                # Trust hard constraint
                trust_score = float(self.device_trust.get(int(device_id), 0.0))
                trust_required = float(self._trust_required(int(getattr(current_layer, "privacy_level", 0))))
                if self.trust_hard and trust_score < trust_required:
                    continue

                if self.resource_manager.can_allocate(
                    agent_id=agent_id,
                    device_id=device_id,
                    layer=current_layer,
                    total_agent_layers=total_layers,
                    is_first_layer=is_first,
                    prev_device_id=prev_dev,
                    transmission_data_size=trans_data,
                ):
                    valid.append(int(device_id))

            valid_actions_dict[agent_id] = valid

        return valid_actions_dict

    def step(self, actions: Dict[int, int]):
        rewards = {aid: 0.0 for aid in range(self.num_agents)}
        dones: dict[int, bool] = {}
        infos: dict[int, dict] = {}

        self.resource_manager.reset_step_resources()
        device_comp_queue = {d_id: 0.0 for d_id in range(self.num_devices)}
        device_comm_queue = {d_id: 0.0 for d_id in range(self.num_devices)}

        agent_ids = list(actions.keys())
        if self.shuffle_allocation_order:
            random.shuffle(agent_ids)
        else:
            agent_ids.sort()

        for aid in agent_ids:
            if self.agent_done[aid]:
                rewards[aid] = 0.0
                continue

            selected_device_id = int(actions[aid])
            current_layer = self.tasks[aid][self.agent_progress[aid]]
            total_layers = len(self.tasks[aid])
            is_first = self.agent_progress[aid] == 0
            prev_dev = self.agent_prev_device[aid]

            trans_data = 0.0
            if prev_dev != -1 and prev_dev != selected_device_id:
                trans_data = float(self.tasks[aid][self.agent_progress[aid] - 1].output_data_size) if self.agent_progress[aid] > 0 else 5.0
            elif prev_dev == -1:
                trans_data = 5.0

            # Energy hard check
            cost = float(self._energy_cost(current_layer, trans_data))
            if float(self.device_energy_remaining.get(selected_device_id, 0.0)) < cost:
                fail = {
                    "reason": "energy",
                    "device_id": int(selected_device_id),
                    "cost": float(cost),
                    "remaining": float(self.device_energy_remaining.get(selected_device_id, 0.0)),
                }
                rewards[aid] = -500.0
                self.agent_done[aid] = True
                infos[aid] = {"reward_type": "stall_termination", "success": False, "fail": fail}
                continue

            # Trust check
            trust_score = float(self.device_trust.get(int(selected_device_id), 0.0))
            trust_required = float(self._trust_required(int(getattr(current_layer, "privacy_level", 0))))
            trust_shortfall = float(self._trust_shortfall(trust_required, trust_score))
            if self.trust_hard and trust_shortfall > 0.0:
                fail = {
                    "reason": "trust",
                    "device_id": int(selected_device_id),
                    "trust_required": float(trust_required),
                    "trust_score": float(trust_score),
                    "trust_shortfall": float(trust_shortfall),
                }
                rewards[aid] = -500.0
                self.agent_done[aid] = True
                infos[aid] = {"reward_type": "stall_termination", "success": False, "fail": fail}
                continue

            success = self.resource_manager.try_allocate(
                agent_id=aid,
                device_id=selected_device_id,
                layer=current_layer,
                total_agent_layers=total_layers,
                is_first_layer=is_first,
                prev_device_id=prev_dev,
                transmission_data_size=trans_data,
            )

            if not success:
                fail = getattr(self.resource_manager, "last_allocation_fail", {}) or {}
                if self.max_fail_logs_per_episode > 0 and self._fail_logs_left > 0:
                    self._fail_logs_left -= 1
                    reason = fail.get("reason", "unknown") if isinstance(fail, dict) else "unknown"
                    layer_idx = int(self.agent_progress[aid])
                    print(
                        "[ALLOC_FAIL] "
                        f"agent={aid} model={self.model_types[aid]} layer_idx={layer_idx} "
                        f"device={selected_device_id} reason={reason} details={fail}"
                    )
                rewards[aid] = -500.0
                self.agent_done[aid] = True
                infos[aid] = {"reward_type": "stall_termination", "success": False, "fail": fail}
                continue

            # Consume energy
            self.device_energy_remaining[selected_device_id] = float(self.device_energy_remaining.get(selected_device_id, 0.0)) - float(cost)

            # Latency
            dev = self.resource_manager.devices[selected_device_id]
            service_comp = float(getattr(current_layer, "computation_demand", 0.0)) / float(getattr(dev, "cpu_speed", 1.0))
            comp_wait = float(device_comp_queue[selected_device_id]) if self.queue_per_device else 0.0
            comp_latency = float(service_comp + comp_wait)

            trans_latency = 0.0
            comm_wait = 0.0
            if prev_dev != -1 and prev_dev != selected_device_id:
                prev_data = float(self.tasks[aid][self.agent_progress[aid] - 1].output_data_size) if self.agent_progress[aid] > 0 else 5.0
                service_comm = float(prev_data) / float(getattr(dev, "bandwidth", 1.0))
                comm_wait = float(device_comm_queue[selected_device_id]) if self.queue_per_device else 0.0
                trans_latency = float(service_comm + comm_wait)
                if self.queue_per_device:
                    device_comm_queue[selected_device_id] += float(service_comm)
            elif prev_dev == -1:
                service_comm = 5.0 / float(getattr(dev, "bandwidth", 1.0))
                comm_wait = float(device_comm_queue[selected_device_id]) if self.queue_per_device else 0.0
                trans_latency = float(service_comm + comm_wait)
                if self.queue_per_device:
                    device_comm_queue[selected_device_id] += float(service_comm)

            if self.queue_per_device:
                device_comp_queue[selected_device_id] += float(service_comp)

            penalty = 0.0
            if not self.trust_hard:
                penalty = float(self.trust_lambda) * float(trust_shortfall)

            total_latency = float(comp_latency + trans_latency)
            rewards[aid] = -(total_latency + penalty)

            infos[aid] = {
                "t_comp": float(comp_latency),
                "t_comm": float(trans_latency),
                "reward_type": "success",
                "success": True,
                "queue_per_device": bool(self.queue_per_device),
                "t_comp_wait": float(comp_wait),
                "t_comm_wait": float(comm_wait),
                "energy_cost": float(cost),
                "energy_remaining": float(self.device_energy_remaining.get(selected_device_id, 0.0)),
                "trust_required": float(trust_required),
                "trust_score": float(trust_score),
                "trust_shortfall": float(trust_shortfall),
                "trust_penalty": float(penalty),
                "trust_hard": bool(self.trust_hard),
            }

            self.agent_prev_device[aid] = selected_device_id
            self.agent_progress[aid] += 1
            if self.agent_progress[aid] >= len(self.tasks[aid]):
                self.agent_done[aid] = True

        # Mix group reward
        group_reward = sum(rewards.values()) / float(self.num_agents)
        for aid in range(self.num_agents):
            rewards[aid] = (0.7 * float(rewards[aid])) + (0.3 * float(group_reward))

        next_obs = self._get_observations()
        for aid in range(self.num_agents):
            dones[aid] = self.agent_done[aid]

        return next_obs, rewards, dones, False, infos

