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

    def __init__(self, num_agents=3, num_devices=5, model_types=None, seed: int | None = None):
        super(MultiAgentIoTEnv, self).__init__()

        self.num_agents = num_agents
        self.num_devices = num_devices
        self.seed = seed
        
        # model_types: list of model types per agent
        if model_types is None:
            self.model_types = ["lenet"] * num_agents
        else:
            self.model_types = model_types
            if len(self.model_types) < num_agents:
                self.model_types.extend(["lenet"] * (num_agents - len(self.model_types)))

        set_global_seed(seed)
        self.resource_manager = ResourceManager(num_devices)
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
        self.resource_manager.reset(self.num_devices)
        
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
        rewards = {}
        dones = {}
        infos = {}

        # Important: Reset step-level resources at the start of ORCHESTRATED interaction
        self.resource_manager.reset_step_resources()

        # 1. Randomize order for processing allocations (Social Fairness)
        # This prevents Agent 0 from always having priority over Agent 1 and 2
        agent_ids = list(actions.keys())
        random.shuffle(agent_ids)

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
                rewards[aid] = -500.0
                self.agent_done[aid] = True
                infos[aid] = {"reward_type": "stall_termination", "success": False}
                continue

            # 3. Latency Calculation
            dev = self.resource_manager.devices[selected_device_id]
            comp_latency = current_layer.computation_demand / dev.cpu_speed
            trans_latency = 0.0
            if prev_dev != -1 and prev_dev != selected_device_id:
                prev_data = self.tasks[aid][self.agent_progress[aid] - 1].output_data_size if self.agent_progress[aid] > 0 else 5.0
                trans_latency = prev_data / dev.bandwidth
            elif prev_dev == -1:
                trans_latency = 5.0 / dev.bandwidth

            total_latency = comp_latency + trans_latency
            rewards[aid] = -total_latency
            
            infos[aid] = {
                't_comp': comp_latency,
                't_comm': trans_latency,
                'reward_type': "success",
                "success": True
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
