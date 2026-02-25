import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Dict
import random

try:
    from SingleAgent.utils import generate_specific_model, DNNLayer, set_global_seed
except ModuleNotFoundError:
    from utils import generate_specific_model, DNNLayer, set_global_seed

from integrated_system.resource_manager import ResourceManager

class MultiAgentIoTEnv(gym.Env):
    """
    Multi-Agent Environment where multiple agents (tasks) schedule their DNN layers
    simultaneously on shared IoT resources.
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        num_agents=3,
        num_devices=5,
        model_types=None,
        seed: int | None = None,
        shuffle_allocation_order: bool = True,
        max_exposure_fraction: float | None = None,
        queue_per_device: bool = False,
        # Set >0 to print resource_manager.last_allocation_fail for debugging.
        max_fail_logs_per_episode: int = 0,
    ):
        super(MultiAgentIoTEnv, self).__init__()

        self.num_agents = num_agents
        self.num_devices = num_devices
        self.seed = seed
        self.shuffle_allocation_order = shuffle_allocation_order
        self.queue_per_device = bool(queue_per_device)
        self.max_fail_logs_per_episode = max(0, int(max_fail_logs_per_episode))
        self._fail_logs_left = self.max_fail_logs_per_episode
        
        # model_types: list of model types per agent
        if model_types is None:
            self.model_types = ["lenet"] * num_agents
        else:
            self.model_types = model_types
            if len(self.model_types) < num_agents:
                self.model_types.extend(["lenet"] * (num_agents - len(self.model_types)))

        set_global_seed(seed)
        self.resource_manager = ResourceManager(num_devices)
        self.max_exposure_fraction = max_exposure_fraction
        try:
            self.resource_manager.set_max_exposure_fraction(max_exposure_fraction)
        except AttributeError:
            pass
        self.resource_manager.reset_devices_with_seed(num_devices, seed)
        
        # Action/Observation space definitions (can be per-agent or global)
        # We'll use dictionaries for multi-agent values
        self.action_spaces = {aid: spaces.Discrete(num_devices) for aid in range(num_agents)}
        
        # State dimension mirroring MonoAgentIoTEnv
        self.layer_feature_dim = 4
        self.prev_device_feature_dim = self.num_devices + 1
        self.single_state_dim = 1 + self.layer_feature_dim + self.prev_device_feature_dim + (6 * num_devices)
        
        self.observation_spaces = {
            aid: spaces.Box(low=0.0, high=2.0, shape=(self.single_state_dim,), dtype=np.float32)
            for aid in range(num_agents)
        }
        
        self.tasks: Dict[int, List[DNNLayer]] = {}
        self.agent_progress: Dict[int, int] = {}
        self.agent_prev_device: Dict[int, int] = {}
        self.agent_done: Dict[int, bool] = {}
        
        self.reset()
        
    def reset(self):
        """Resets the environment and all agents."""
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
            m_type = self.model_types[i]
            self.tasks[i] = generate_specific_model(m_type)
            self.agent_progress[i] = 0
            self.agent_prev_device[i] = -1
            self.agent_done[i] = False
            
        return self._get_observations(), {}

    def _get_observations(self):
        observations = {}
        for i in range(self.num_agents):
            observations[i] = self._get_agent_observation(i)
        return observations

    def _get_agent_observation(self, agent_id):
        if self.agent_done[agent_id]:
            return np.zeros(self.single_state_dim, dtype=np.float32)

        task = self.tasks[agent_id]
        progress_val = self.agent_progress[agent_id]
        total_layers = len(task)
        progress_norm = float(progress_val) / total_layers

        current_obs = [progress_norm]

        # Current layer features
        current_layer = task[progress_val]
        current_obs.extend([
            current_layer.computation_demand / 20.0,
            current_layer.memory_demand / 200.0,
            current_layer.output_data_size / 25.0,
            float(current_layer.privacy_level)
        ])

        # Previous device
        prev_dev = self.agent_prev_device[agent_id]
        prev_valid = 1.0 if prev_dev != -1 else 0.0
        current_obs.append(prev_valid)
        for d_id in range(self.num_devices):
            current_obs.append(1.0 if prev_dev == d_id else 0.0)

        # Shared Device States
        for d_id in range(self.num_devices):
            current_obs.extend(self.resource_manager.get_state_for_device(d_id))
            step_load = self.resource_manager.step_resources[d_id]
            current_obs.append(step_load['compute'] / 100.0)
            current_obs.append(step_load['bw'] / 50.0)

        return np.array(current_obs, dtype=np.float32)

    def get_valid_actions(self) -> Dict[int, List[int]]:
        # Step-level resources are meant to be per-timestep (agents competing in the same step).
        # Ensure we don't leak previous step's allocations into validity checks.
        self.resource_manager.reset_step_resources()
        valid_actions_dict = {}
        for agent_id in range(self.num_agents):
            if self.agent_done[agent_id]:
                valid_actions_dict[agent_id] = []
                continue
            
            valid_actions = []
            progress = self.agent_progress[agent_id]
            task = self.tasks[agent_id]
            current_layer = task[progress]
            total_layers = len(task)
            is_first = (progress == 0)
            prev_dev = self.agent_prev_device[agent_id]

            if progress > 0:
                prev_out = task[progress - 1].output_data_size
            else:
                prev_out = 5.0

            for device_id in range(self.num_devices):
                trans_data = 0.0
                if prev_dev != -1 and prev_dev != device_id:
                    trans_data = prev_out
                elif prev_dev == -1:
                    trans_data = 5.0
                
                if self.resource_manager.can_allocate(
                    agent_id=agent_id,
                    device_id=device_id,
                    layer=current_layer,
                    total_agent_layers=total_layers,
                    is_first_layer=is_first,
                    prev_device_id=prev_dev,
                    transmission_data_size=trans_data
                ):
                    valid_actions.append(device_id)
            valid_actions_dict[agent_id] = valid_actions
        return valid_actions_dict

    def step(self, actions: Dict[int, int]):
        """
        Executes one step for all agents.
        Args:
            actions: Dictionary mapping agent_id to device_id
        """
        rewards = {aid: 0.0 for aid in range(self.num_agents)}
        dones = {}
        infos = {}

        # Important: Reset step-level resources at the start of ORCHESTRATED interaction
        self.resource_manager.reset_step_resources()
        # Optional: model queuing/serialization on each device within a step.
        # When enabled, multiple agents selecting the same device incur waiting time (FIFO)
        # based on the allocation processing order for this step.
        device_comp_queue = {d_id: 0.0 for d_id in range(self.num_devices)}
        device_comm_queue = {d_id: 0.0 for d_id in range(self.num_devices)}

        # 1. Order for processing allocations (Social Fairness)
        # Randomization prevents Agent 0 from always having priority.
        agent_ids = list(actions.keys())
        if self.shuffle_allocation_order:
            random.shuffle(agent_ids)
        else:
            agent_ids.sort()

        for aid in agent_ids:
            if self.agent_done[aid]:
                rewards[aid] = 0.0
                continue

            action = actions[aid]
            selected_device_id = int(action)
            current_layer = self.tasks[aid][self.agent_progress[aid]]
            total_layers = len(self.tasks[aid])
            is_first = (self.agent_progress[aid] == 0)
            prev_dev = self.agent_prev_device[aid]

            # Transmission data calc
            trans_data = 0.0
            if prev_dev != -1 and prev_dev != selected_device_id:
                if self.agent_progress[aid] > 0:
                    trans_data = self.tasks[aid][self.agent_progress[aid] - 1].output_data_size
                else:
                    trans_data = 5.0
            elif prev_dev == -1:
                trans_data = 5.0

            # 2. Environment calls ResourceManager
            success = self.resource_manager.try_allocate(
                agent_id=aid,
                device_id=selected_device_id,
                layer=current_layer,
                total_agent_layers=total_layers,
                is_first_layer=is_first,
                prev_device_id=prev_dev,
                transmission_data_size=trans_data
            )

            if not success:
                fail = getattr(self.resource_manager, "last_allocation_fail", {}) or {}
                if self.max_fail_logs_per_episode > 0 and self._fail_logs_left > 0:
                    self._fail_logs_left -= 1
                    reason = fail.get("reason", "unknown") if isinstance(fail, dict) else "unknown"
                    layer_idx = int(self.agent_progress[aid])
                    layer = self.tasks[aid][layer_idx] if layer_idx < len(self.tasks[aid]) else None
                    layer_desc = (
                        f"{getattr(layer, 'name', f'layer_{layer_idx}')}"
                        f"(c={getattr(layer, 'computation_demand', '?')},m={getattr(layer, 'memory_demand', '?')},p={getattr(layer, 'privacy_level', '?')})"
                        if layer is not None
                        else f"layer_{layer_idx}"
                    )
                    print(
                        "[ALLOC_FAIL] "
                        f"agent={aid} model={self.model_types[aid]} layer={layer_desc} "
                        f"device={selected_device_id} reason={reason} details={fail}"
                    )
                rewards[aid] = -500.0
                self.agent_done[aid] = True
                infos[aid] = {"reward_type": "stall_termination", "success": False, "fail": fail}
                continue

            # 3. Latency Calculation
            dev = self.resource_manager.devices[selected_device_id]
            service_comp = current_layer.computation_demand / dev.cpu_speed
            comp_wait = device_comp_queue[selected_device_id] if self.queue_per_device else 0.0
            comp_latency = service_comp + comp_wait
            trans_latency = 0.0
            comm_wait = 0.0
            if prev_dev != -1 and prev_dev != selected_device_id:
                prev_data = self.tasks[aid][self.agent_progress[aid] - 1].output_data_size if self.agent_progress[aid] > 0 else 5.0
                service_comm = prev_data / dev.bandwidth
                comm_wait = device_comm_queue[selected_device_id] if self.queue_per_device else 0.0
                trans_latency = service_comm + comm_wait
                if self.queue_per_device:
                    device_comm_queue[selected_device_id] += float(service_comm)
            elif prev_dev == -1:
                service_comm = 5.0 / dev.bandwidth
                comm_wait = device_comm_queue[selected_device_id] if self.queue_per_device else 0.0
                trans_latency = service_comm + comm_wait
                if self.queue_per_device:
                    device_comm_queue[selected_device_id] += float(service_comm)

            if self.queue_per_device:
                device_comp_queue[selected_device_id] += float(service_comp)

            total_latency = comp_latency + trans_latency
            rewards[aid] = -total_latency
             
            infos[aid] = {
                't_comp': comp_latency,
                't_comm': trans_latency,
                'reward_type': "success",
                "success": True,
                "queue_per_device": bool(self.queue_per_device),
                "t_comp_wait": float(comp_wait),
                "t_comm_wait": float(comm_wait),
            }

            # Advance agent
            self.agent_prev_device[aid] = selected_device_id
            self.agent_progress[aid] += 1
            if self.agent_progress[aid] >= len(self.tasks[aid]):
                self.agent_done[aid] = True

        # 4. Social Reward Adjustment (Cooperation)
        # Mix individual reward with the average reward of the group
        # This encourages agents to not block each other
        group_reward = sum(rewards.values()) / self.num_agents
        for aid in range(self.num_agents):
            # 70% Individual, 30% Group
            rewards[aid] = (0.7 * rewards[aid]) + (0.3 * group_reward)

        next_obs = self._get_observations()
        for aid in range(self.num_agents):
            dones[aid] = self.agent_done[aid]
        
        return next_obs, rewards, dones, False, infos
