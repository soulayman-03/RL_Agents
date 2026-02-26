from __future__ import annotations

import dataclasses
import random
from typing import Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from integrated_system.resource_manager import ResourceManager

try:
    from SingleAgent.utils import DNNLayer, generate_specific_model, set_global_seed
except ModuleNotFoundError:
    from utils import DNNLayer, generate_specific_model, set_global_seed


class MultiAgentIoTEnvPrivacyWeighted(gym.Env):
    """
    Multi-agent scheduling environment with *weighted privacy via minimal trust threshold*.

    Instead of the hard privacy constraint (device.privacy_clearance >= layer.privacy_level),
    allocations are allowed but apply a reward penalty based on a *minimal trust threshold*.

    - Each layer has `privacy_level` in [0..privacy_max_level]
    - Each device has a `trust_score` in [0..1]
    - We map privacy -> required trust:
        trust_required = trust_min_for_max_privacy * (privacy_level / privacy_max_level)
    - shortfall = max(0, trust_required - trust_score)
    - penalty = privacy_lambda * shortfall

    Hard/soft trust:
    - If `trust_hard=True`, allocations with trust_score < trust_required fail immediately (reason="trust").
    - If `trust_hard=False`, allocations are allowed but penalized by `privacy_lambda * shortfall`.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        num_agents: int = 3,
        num_devices: int = 5,
        model_types: list[str] | None = None,
        seed: int | None = None,
        shuffle_allocation_order: bool = True,
        max_exposure_fraction: float | None = None,
        queue_per_device: bool = False,
        max_fail_logs_per_episode: int = 0,
        privacy_max_level: int = 3,
        privacy_profile: str = "linear_front_loaded",
        privacy_lambda: float = 5.0,
        trust_min_for_max_privacy: float = 0.8,
        trust_hard: bool = True,
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
        self.privacy_lambda = float(privacy_lambda)
        self.trust_min_for_max_privacy = float(trust_min_for_max_privacy)
        self.trust_hard = bool(trust_hard)

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
        self._set_weighted_device_privacy()
        self.device_trust: dict[int, float] = {}
        self._init_device_trust()

        self.layer_feature_dim = 4
        self.prev_device_feature_dim = self.num_devices + 1
        self.single_state_dim = 1 + self.layer_feature_dim + self.prev_device_feature_dim + (6 * self.num_devices)

        self.action_spaces = {aid: spaces.Discrete(self.num_devices) for aid in range(self.num_agents)}
        self.observation_spaces = {
            aid: spaces.Box(low=0.0, high=2.0, shape=(self.single_state_dim,), dtype=np.float32)
            for aid in range(self.num_agents)
        }

        self.tasks: Dict[int, List[DNNLayer]] = {}
        self.agent_progress: Dict[int, int] = {}
        self.agent_prev_device: Dict[int, int] = {}
        self.agent_done: Dict[int, bool] = {}

        self.reset()

    def _set_weighted_device_privacy(self) -> None:
        devices = list(getattr(self.resource_manager, "devices", {}).values())
        if not devices:
            return

        for d in devices:
            d.privacy_clearance = int(self._py_rng.randint(0, self.privacy_max_level))

        min_high = min(2, len(devices))
        high = [d for d in devices if int(getattr(d, "privacy_clearance", 0)) >= self.privacy_max_level]
        if len(high) < min_high:
            candidates = [d for d in devices if int(getattr(d, "privacy_clearance", 0)) < self.privacy_max_level]
            for d in self._py_rng.sample(candidates, k=min_high - len(high)):
                d.privacy_clearance = int(self.privacy_max_level)

    def _init_device_trust(self) -> None:
        # Deterministic trust scores per device in [0.5..1.0). Kept constant across episodes.
        self.device_trust = {int(d_id): float(0.5 + 0.5 * self._py_rng.random()) for d_id in range(self.num_devices)}

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

    def reset(self):
        try:
            self.resource_manager.set_max_exposure_fraction(self.max_exposure_fraction)
        except AttributeError:
            pass
        self.resource_manager.reset(self.num_devices)
        self._fail_logs_left = self.max_fail_logs_per_episode

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

        progress_val = self.agent_progress[agent_id]
        task = self.tasks[agent_id]
        progress_norm = float(progress_val) / float(max(1, len(task)))
        current_obs = [progress_norm]

        current_layer = task[progress_val]
        current_obs.extend(
            [
                float(current_layer.computation_demand) / 20.0,
                float(current_layer.memory_demand) / 200.0,
                float(current_layer.output_data_size) / 25.0,
                float(current_layer.privacy_level) / float(self.privacy_max_level),
            ]
        )

        prev_dev = self.agent_prev_device[agent_id]
        prev_valid = 1.0 if prev_dev != -1 else 0.0
        current_obs.append(prev_valid)
        for d_id in range(self.num_devices):
            current_obs.append(1.0 if prev_dev == d_id else 0.0)

        for d_id in range(self.num_devices):
            d = self.resource_manager.devices[d_id]
            current_obs.extend(
                [
                    float(d.cpu_speed) / 50.0,
                    float(d.memory_capacity - d.current_memory_usage) / 600.0,
                    float(d.bandwidth) / 300.0,
                    float(self.device_trust.get(int(d_id), 0.0)),
                ]
            )
            step_load = self.resource_manager.step_resources[d_id]
            current_obs.append(float(step_load["compute"]) / 100.0)
            current_obs.append(float(step_load["bw"]) / 50.0)

        return np.asarray(current_obs, dtype=np.float32)

    @staticmethod
    def _privacy_violation(layer_privacy: int, device_clearance: int) -> int:
        return max(0, int(layer_privacy) - int(device_clearance))

    def _trust_required(self, privacy_level: int) -> float:
        p = float(max(0, min(int(privacy_level), int(self.privacy_max_level))))
        denom = float(max(1, int(self.privacy_max_level)))
        tmin = float(max(0.0, min(1.0, self.trust_min_for_max_privacy)))
        return float(tmin * (p / denom))

    @staticmethod
    def _trust_shortfall(trust_required: float, trust_score: float) -> float:
        return float(max(0.0, float(trust_required) - float(trust_score)))

    @staticmethod
    def _layer_for_resource_manager(layer: DNNLayer, device_clearance: int) -> DNNLayer:
        safe_priv = min(int(getattr(layer, "privacy_level", 0)), int(device_clearance))
        return dataclasses.replace(layer, privacy_level=int(safe_priv))

    def get_valid_actions(self) -> Dict[int, List[int]]:
        self.resource_manager.reset_step_resources()
        valid_actions_dict: dict[int, list[int]] = {}
        for agent_id in range(self.num_agents):
            if self.agent_done[agent_id]:
                valid_actions_dict[agent_id] = []
                continue

            valid: list[int] = []
            progress = self.agent_progress[agent_id]
            task = self.tasks[agent_id]
            current_layer = task[progress]
            total_layers = len(task)
            is_first = progress == 0
            prev_dev = self.agent_prev_device[agent_id]
            prev_out = task[progress - 1].output_data_size if progress > 0 else 5.0

            for device_id in range(self.num_devices):
                trans_data = float(prev_out) if (prev_dev != -1 and prev_dev != device_id) else (5.0 if prev_dev == -1 else 0.0)
                dev = self.resource_manager.devices[int(device_id)]
                if self.trust_hard:
                    trust_score = float(self.device_trust.get(int(device_id), 0.0))
                    trust_required = float(self._trust_required(int(getattr(current_layer, "privacy_level", 0))))
                    if trust_score < trust_required:
                        continue
                layer_rm = self._layer_for_resource_manager(current_layer, int(getattr(dev, "privacy_clearance", 0)))
                if self.resource_manager.can_allocate(
                    agent_id=agent_id,
                    device_id=device_id,
                    layer=layer_rm,
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

            dev = self.resource_manager.devices[selected_device_id]
            dev_priv = int(getattr(dev, "privacy_clearance", 0))
            layer_priv = int(getattr(current_layer, "privacy_level", 0))
            trust_score = float(self.device_trust.get(int(selected_device_id), 0.0))
            trust_required = float(self._trust_required(layer_priv))
            trust_shortfall = float(self._trust_shortfall(trust_required, trust_score))
            trust_penalty = float(self.privacy_lambda) * float(trust_shortfall)

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

            layer_rm = self._layer_for_resource_manager(current_layer, dev_priv)
            success = self.resource_manager.try_allocate(
                agent_id=aid,
                device_id=selected_device_id,
                layer=layer_rm,
                total_agent_layers=total_layers,
                is_first_layer=is_first,
                prev_device_id=prev_dev,
                transmission_data_size=trans_data,
            )

            if not success:
                fail = getattr(self.resource_manager, "last_allocation_fail", {}) or {}
                rewards[aid] = -500.0
                self.agent_done[aid] = True
                infos[aid] = {"reward_type": "stall_termination", "success": False, "fail": fail}
                continue

            service_comp = float(current_layer.computation_demand) / float(dev.cpu_speed)
            comp_wait = float(device_comp_queue[selected_device_id]) if self.queue_per_device else 0.0
            comp_latency = float(service_comp + comp_wait)

            trans_latency = 0.0
            comm_wait = 0.0
            if prev_dev != -1 and prev_dev != selected_device_id:
                prev_data = float(self.tasks[aid][self.agent_progress[aid] - 1].output_data_size) if self.agent_progress[aid] > 0 else 5.0
                service_comm = float(prev_data) / float(dev.bandwidth)
                comm_wait = float(device_comm_queue[selected_device_id]) if self.queue_per_device else 0.0
                trans_latency = float(service_comm + comm_wait)
                if self.queue_per_device:
                    device_comm_queue[selected_device_id] += float(service_comm)
            elif prev_dev == -1:
                service_comm = 5.0 / float(dev.bandwidth)
                comm_wait = float(device_comm_queue[selected_device_id]) if self.queue_per_device else 0.0
                trans_latency = float(service_comm + comm_wait)
                if self.queue_per_device:
                    device_comm_queue[selected_device_id] += float(service_comm)

            if self.queue_per_device:
                device_comp_queue[selected_device_id] += float(service_comp)

            total_latency = float(comp_latency + trans_latency)
            penalty = 0.0 if self.trust_hard else float(trust_penalty)
            rewards[aid] = -(total_latency + penalty)

            infos[aid] = {
                "t_comp": float(comp_latency),
                "t_comm": float(trans_latency),
                "reward_type": "success",
                "success": True,
                "queue_per_device": bool(self.queue_per_device),
                "t_comp_wait": float(comp_wait),
                "t_comm_wait": float(comm_wait),
                "privacy_required": int(layer_priv),
                "privacy_clearance": int(dev_priv),
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

        group_reward = sum(rewards.values()) / float(self.num_agents)
        for aid in range(self.num_agents):
            rewards[aid] = (0.7 * float(rewards[aid])) + (0.3 * float(group_reward))

        next_obs = self._get_observations()
        for aid in range(self.num_agents):
            dones[aid] = self.agent_done[aid]

        return next_obs, rewards, dones, False, infos
