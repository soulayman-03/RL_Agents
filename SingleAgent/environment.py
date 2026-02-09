import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List
from utils import generate_dummy_dnn_model, DNNLayer, set_global_seed
from integrated_system.resource_manager import ResourceManager

class MonoAgentIoTEnv(gym.Env):
    """
    Multi-Agent Environment where K agents (tasks) compete for resources.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agents=3, num_devices=5, model_types=None, seed: int | None = None):
        super(MonoAgentIoTEnv, self).__init__()
        
        self.num_agents = num_agents
        self.num_devices = num_devices
        self.seed = seed
        # model_types: List of strings e.g. ["lenet", "resnet18", "mobilenet"]
        self.model_types = model_types if model_types else ["lenet"] * num_agents

        set_global_seed(seed)
        self.resource_manager = ResourceManager(num_devices)
        self.resource_manager.reset_devices_with_seed(num_devices, seed)
        
        # Action Space: K agents, each chooses a device (0..D-1)
        self.action_space = spaces.Discrete(num_devices) # Per agent
        
        # State Space Per Agent:
        # [OwnProgress, OtherAgentsProgress..., LayerComp, LayerMem, LayerOut, LayerPriv,
        #  PrevDeviceValid, PrevDeviceOneHot..., Dev1_CPU, Dev1_Mem, Dev1_BW, Dev1_Priv, Dev1_StepLoad, ...]
        self.layer_feature_dim = 4
        self.prev_device_feature_dim = self.num_devices + 1  # valid-flag + one-hot
        self.single_state_dim = self.num_agents + self.layer_feature_dim + self.prev_device_feature_dim + (6 * num_devices)
        self.observation_space = spaces.Box(low=0.0, high=2.0,
                                            shape=(self.single_state_dim,), 
                                            dtype=np.float32)
        
        self.agents_tasks: List[List[DNNLayer]] = []
        self.agents_progress: List[int] = [] # Current layer index for each agent
        self.agents_prev_device: List[int] = [] 
        self.agents_done: List[bool] = []
        
        self.reset()
        
    def reset(self):
        """Resets the environment and the shared resource manager."""
        self.resource_manager.reset(self.num_devices)
        
        self.agents_tasks = []
        self.agents_progress = []
        self.agents_prev_device = []
        self.agents_done = []
        
        from utils import generate_specific_model # Imported here to avoid circular
        
        for i in range(self.num_agents):
            m_type = self.model_types[i] if i < len(self.model_types) else "lenet"
            self.agents_tasks.append(generate_specific_model(m_type))
            self.agents_progress.append(0)
            self.agents_prev_device.append(-1)
            self.agents_done.append(False)
            
        return self._get_all_observations(), {}

    def _get_all_observations(self):
        obs_list = []
        # Progresses for all agents
        progresses = []
        for i in range(self.num_agents):
            if self.agents_done[i]:
                progresses.append(1.0)
            else:
                total_layers = len(self.agents_tasks[i])
                progresses.append(float(self.agents_progress[i]) / total_layers)

        for i in range(self.num_agents):
            if self.agents_done[i]:
                obs_list.append(np.zeros(self.single_state_dim))
                continue
                
            # Construct state for Agent i
            # 1. Progres (Own first, then others)
            current_obs = [progresses[i]]
            for j in range(self.num_agents):
                if i != j:
                    current_obs.append(progresses[j])

            # 2. Current layer features (normalized)
            current_layer = self.agents_tasks[i][self.agents_progress[i]]
            current_obs.extend([
                current_layer.computation_demand / 20.0,
                current_layer.memory_demand / 200.0,
                current_layer.output_data_size / 25.0,
                float(current_layer.privacy_level)
            ])

            # 3. Previous device (valid flag + one-hot)
            prev_dev = self.agents_prev_device[i]
            prev_valid = 1.0 if prev_dev != -1 else 0.0
            current_obs.append(prev_valid)
            for d_id in range(self.num_devices):
                current_obs.append(1.0 if prev_dev == d_id else 0.0)
            
            # 4. Shared Device States (from Resource Manager)
            for d_id in range(self.num_devices):
                # Static state (CPU, Mem, BW, Priv)
                current_obs.extend(self.resource_manager.get_state_for_device(d_id))
                # Dynamic state (Current Step Load)
                step_load = self.resource_manager.step_resources[d_id]
                current_obs.append(step_load['compute'] / 100.0)
                current_obs.append(step_load['bw'] / 50.0)
            
            obs_list.append(np.array(current_obs, dtype=np.float32))
        return obs_list

    def get_valid_actions(self, agent_id: int) -> List[int]:
        """
        Returns a list of valid actions for the given agent in the current state.
        This mirrors the constraints used in step(), including sequential diversity.
        """
        if self.agents_done[agent_id]:
            return []

        # Mimic start-of-step resource conditions
        self.resource_manager.reset_step_resources()

        progress = self.agents_progress[agent_id]
        current_layer = self.agents_tasks[agent_id][progress]
        total_layers = len(self.agents_tasks[agent_id])
        is_first = (progress == 0)
        prev_dev = self.agents_prev_device[agent_id]

        # Pre-compute previous layer output (if any)
        if progress > 0:
            prev_out = self.agents_tasks[agent_id][progress - 1].output_data_size
        else:
            prev_out = 5.0

        valid_actions = []
        for device_id in range(self.num_devices):
            if prev_dev != -1 and prev_dev != device_id:
                trans_data = prev_out
            elif prev_dev == -1:
                trans_data = 5.0
            else:
                trans_data = 0.0

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

        return valid_actions

    def step(self, actions: List[int]):
        """
        Executes one step for ALL agents.
        Args:
            actions: List of ints, one for each agent.
        """
        rewards = [0.0] * self.num_agents
        dones = [False] * self.num_agents
        infos = [{} for _ in range(self.num_agents)]
        
        # Reset step-level resources (CPU/BW bottlenecks)
        self.resource_manager.reset_step_resources()
        
        # Conflict resolution: First come first serve based on index
        agent_indices = list(range(self.num_agents))
        # np.random.shuffle(agent_indices) # Optional: shuffle for fairness
        
        for idx in agent_indices:
            if self.agents_done[idx]:
                dones[idx] = True
                continue
                
            action = actions[idx]
            selected_device_id = int(action)
            current_layer = self.agents_tasks[idx][self.agents_progress[idx]]
            total_layers = len(self.agents_tasks[idx])
            is_first = (self.agents_progress[idx] == 0)
            
            # Transmission data calculation
            prev_dev = self.agents_prev_device[idx]
            trans_data = 0.0
            if prev_dev != -1 and prev_dev != selected_device_id:
                if self.agents_progress[idx] > 0:
                    trans_data = self.agents_tasks[idx][self.agents_progress[idx]-1].output_data_size
                else:
                    trans_data = 5.0 # Initial input
            elif prev_dev == -1:
                trans_data = 5.0 # Input
            
            # 1. Try Allocate (Enforces Hard Constraints 4.a, 4.b, 4.c, 4.f, 4.h)
            success = self.resource_manager.try_allocate(
                agent_id=idx, 
                device_id=selected_device_id, 
                layer=current_layer,
                total_agent_layers=total_layers,
                is_first_layer=is_first,
                prev_device_id=prev_dev,
                transmission_data_size=trans_data
            )
            
            if not success:
                # Penalty for failure (Constraint violation or Resource Full)
                # The agent terminates immediately to avoid infinite stalls
                rewards[idx] = -500.0 # Large stalling penalty
                self.agents_done[idx] = True # Terminate on stall
                infos[idx]['reward_type'] = "stall_termination"
                continue 

            else:
                # 2. Calculate Latency (Cost)
                dev = self.resource_manager.devices[selected_device_id]
                
                # Compute
                comp_latency = current_layer.computation_demand / dev.cpu_speed
                
                # Transmit
                trans_latency = 0
                prev_dev = self.agents_prev_device[idx]
                
                if prev_dev != -1 and prev_dev != selected_device_id:
                     # Get output size of PREVIOUS layer
                     if self.agents_progress[idx] > 0:
                         prev_data = self.agents_tasks[idx][self.agents_progress[idx]-1].output_data_size
                     else:
                         prev_data = 5.0 # Input
                     trans_latency = prev_data / dev.bandwidth
                elif prev_dev == -1:
                    trans_latency = 5.0 / dev.bandwidth
                    
                total_latency = comp_latency + trans_latency
                rewards[idx] = -total_latency
                
                # Add detailed latency info for mapping analysis
                infos[idx]['t_comp'] = comp_latency
                infos[idx]['t_comm'] = trans_latency
                infos[idx]['reward_type'] = "success"
                
                self._advance_agent(idx, selected_device_id)

        next_obs = self._get_all_observations()
        
        # In Gymnasium, step returns (obs, reward, terminated, truncated, info)
        terminated = all(self.agents_done)
        truncated = False
        
        return next_obs, rewards, self.agents_done, truncated, infos

    def _advance_agent(self, idx, device_id):
        self.agents_prev_device[idx] = device_id
        self.agents_progress[idx] += 1
        if self.agents_progress[idx] >= len(self.agents_tasks[idx]):
            self.agents_done[idx] = True
