from __future__ import annotations

import random
from typing import Dict, List

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from integrated_system.resource_manager import ResourceManager

try:
    from SingleAgent.utils import generate_specific_model, DNNLayer, set_global_seed
except ModuleNotFoundError:
    from utils import generate_specific_model, DNNLayer, set_global_seed


class MultiAgentIoTEnvEnergyHard(gym.Env):
    """
    Multi-Agent environment that adds a hard energy constraint per device.

    - Each device has an energy budget (battery) for the episode.
    - A layer allocation consumes energy = alpha_comp*compute + alpha_comm*transmitted_data.
    - If remaining energy is insufficient, allocation fails with reason="energy".
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
        # Energy model
        energy_budget_range: tuple[float, float] = (500.0, 1200.0),
        alpha_comp: float = 1.0,
        alpha_comm: float = 1.0,
        # Debug
        max_fail_logs_per_episode: int = 0,
    ):
        super().__init__()
        self.num_agents = int(num_agents)
        self.num_devices = int(num_devices)
        self.seed = seed
        self.shuffle_allocation_order = bool(shuffle_allocation_order)
        self.max_exposure_fraction = max_exposure_fraction
        self.max_fail_logs_per_episode = max(0, int(max_fail_logs_per_episode))
        self._fail_logs_left = self.max_fail_logs_per_episode

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
        self.resource_manager = ResourceManager(self.num_devices)
        try:
            self.resource_manager.set_max_exposure_fraction(self.max_exposure_fraction)
        except AttributeError:
            pass
        self.resource_manager.reset_devices_with_seed(self.num_devices, seed)

        # Spaces
        self.layer_feature_dim = 4
        self.prev_device_feature_dim = self.num_devices + 1
        # +1 energy feature per device (ratio remaining)
        self.single_state_dim = 1 + self.layer_feature_dim + self.prev_device_feature_dim + (7 * self.num_devices)
        self.action_spaces = {aid: spaces.Discrete(self.num_devices) for aid in range(self.num_agents)}
        self.observation_spaces = {
            aid: spaces.Box(low=0.0, high=2.0, shape=(self.single_state_dim,), dtype=np.float32)
            for aid in range(self.num_agents)
        }

        self.tasks: Dict[int, List[DNNLayer]] = {}
        self.agent_progress: Dict[int, int] = {}
        self.agent_prev_device: Dict[int, int] = {}
        self.agent_done: Dict[int, bool] = {}

        self.device_energy_init: dict[int, float] = {}
        self.device_energy_remaining: dict[int, float] = {}
        self._init_energy_budgets()

        self.reset()

    def _init_energy_budgets(self) -> None:
        lo, hi = self.energy_budget_range
        rng = random.Random(self.seed if self.seed is not None else None)
        self.device_energy_init = {d: float(rng.uniform(lo, hi)) for d in range(self.num_devices)}
        self.device_energy_remaining = dict(self.device_energy_init)

    def reset(self):
        self.resource_manager.reset(self.num_devices)
        self._fail_logs_left = self.max_fail_logs_per_episode
        self.device_energy_remaining = dict(self.device_energy_init)
        try:
            self.resource_manager.set_max_exposure_fraction(self.max_exposure_fraction)
        except AttributeError:
            pass

        self.tasks = {}
        self.agent_progress = {}
        self.agent_prev_device = {}
        self.agent_done = {}

        for i in range(self.num_agents):
            m_type = self.model_types[i]
            self.tasks[i] = generate_specific_model(m_type)
            self.agent_progress[i] = 0
            self.agent_prev_device[i] = -1
            self.agent_done[i] = False

        return self._get_observations(), {}

    def _energy_ratio(self, device_id: int) -> float:
        init_e = float(self.device_energy_init.get(device_id, 1.0))
        if init_e <= 0:
            return 0.0
        return float(self.device_energy_remaining.get(device_id, 0.0)) / init_e

    def _get_observations(self):
        return {i: self._get_agent_observation(i) for i in range(self.num_agents)}

    def _get_agent_observation(self, agent_id: int):
        if self.agent_done[agent_id]:
            return np.zeros(self.single_state_dim, dtype=np.float32)

        task = self.tasks[agent_id]
        progress_val = self.agent_progress[agent_id]
        total_layers = len(task)
        progress_norm = float(progress_val) / max(1, total_layers)

        current_obs = [progress_norm]
        current_layer = task[progress_val]
        current_obs.extend(
            [
                current_layer.computation_demand / 20.0,
                current_layer.memory_demand / 200.0,
                current_layer.output_data_size / 25.0,
                float(current_layer.privacy_level),
            ]
        )

        prev_dev = self.agent_prev_device[agent_id]
        prev_valid = 1.0 if prev_dev != -1 else 0.0
        current_obs.append(prev_valid)
        for d_id in range(self.num_devices):
            current_obs.append(1.0 if prev_dev == d_id else 0.0)

        for d_id in range(self.num_devices):
            current_obs.extend(self.resource_manager.get_state_for_device(d_id))  # 4
            step_load = self.resource_manager.step_resources[d_id]
            current_obs.append(step_load["compute"] / 100.0)
            current_obs.append(step_load["bw"] / 50.0)
            current_obs.append(self._energy_ratio(d_id))

        return np.asarray(current_obs, dtype=np.float32)

    def _energy_cost(self, layer: DNNLayer, transmission_data_size: float) -> float:
        return (self.alpha_comp * float(layer.computation_demand)) + (self.alpha_comm * float(transmission_data_size))

    def get_valid_actions(self) -> Dict[int, List[int]]:
        self.resource_manager.reset_step_resources()
        valid_actions_dict: dict[int, list[int]] = {}

        for agent_id in range(self.num_agents):
            if self.agent_done[agent_id]:
                valid_actions_dict[agent_id] = []
                continue

            progress = self.agent_progress[agent_id]
            task = self.tasks[agent_id]
            current_layer = task[progress]
            total_layers = len(task)
            is_first = progress == 0
            prev_dev = self.agent_prev_device[agent_id]
            prev_out = task[progress - 1].output_data_size if progress > 0 else 5.0

            valid: list[int] = []
            for device_id in range(self.num_devices):
                trans_data = 0.0
                if prev_dev != -1 and prev_dev != device_id:
                    trans_data = float(prev_out)
                elif prev_dev == -1:
                    trans_data = 5.0

                cost = self._energy_cost(current_layer, trans_data)
                if float(self.device_energy_remaining.get(device_id, 0.0)) < cost:
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

            cost = self._energy_cost(current_layer, trans_data)
            if float(self.device_energy_remaining.get(selected_device_id, 0.0)) < cost:
                fail = {
                    "reason": "energy",
                    "device_id": selected_device_id,
                    "cost": float(cost),
                    "remaining": float(self.device_energy_remaining.get(selected_device_id, 0.0)),
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
                rewards[aid] = -500.0
                self.agent_done[aid] = True
                infos[aid] = {"reward_type": "stall_termination", "success": False, "fail": fail}
                continue

            self.device_energy_remaining[selected_device_id] = float(self.device_energy_remaining.get(selected_device_id, 0.0)) - float(cost)

            dev = self.resource_manager.devices[selected_device_id]
            comp_latency = float(current_layer.computation_demand) / float(dev.cpu_speed)
            trans_latency = 0.0
            if prev_dev != -1 and prev_dev != selected_device_id:
                prev_data = float(self.tasks[aid][self.agent_progress[aid] - 1].output_data_size) if self.agent_progress[aid] > 0 else 5.0
                trans_latency = prev_data / float(dev.bandwidth)
            elif prev_dev == -1:
                trans_latency = 5.0 / float(dev.bandwidth)

            rewards[aid] = -(comp_latency + trans_latency)
            infos[aid] = {
                "t_comp": float(comp_latency),
                "t_comm": float(trans_latency),
                "reward_type": "success",
                "success": True,
                "energy_cost": float(cost),
                "energy_remaining": float(self.device_energy_remaining.get(selected_device_id, 0.0)),
            }

            self.agent_prev_device[aid] = selected_device_id
            self.agent_progress[aid] += 1
            if self.agent_progress[aid] >= len(self.tasks[aid]):
                self.agent_done[aid] = True

        group_reward = sum(rewards.values()) / float(self.num_agents)
        for aid in range(self.num_agents):
            rewards[aid] = (0.7 * rewards[aid]) + (0.3 * group_reward)

        next_obs = self._get_observations()
        for aid in range(self.num_agents):
            dones[aid] = self.agent_done[aid]
        return next_obs, rewards, dones, False, infos

