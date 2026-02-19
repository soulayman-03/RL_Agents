import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List
try:
    from SingleAgent.utils import generate_dummy_dnn_model, DNNLayer, set_global_seed
except ModuleNotFoundError:
    from utils import generate_dummy_dnn_model, DNNLayer, set_global_seed
from integrated_system.resource_manager import ResourceManager

class MonoAgentIoTEnv(gym.Env):
    """
    Single-Agent Environment where one task schedules its DNN layers.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agents=1, num_devices=5, model_types=None, seed: int | None = None):
        super(MonoAgentIoTEnv, self).__init__()

        if num_agents != 1:
            raise ValueError("MonoAgentIoTEnv supports only num_agents=1")

        self.num_agents = 1
        self.num_devices = num_devices
        self.seed = seed
        # model_types: List of strings e.g. ["lenet", "resnet18", "mobilenet"]
        if model_types is None:
            self.model_types = ["lenet"]
        elif isinstance(model_types, list):
            self.model_types = model_types
        else:
            self.model_types = [model_types]

        set_global_seed(seed)
        self.resource_manager = ResourceManager(num_devices)
        self.resource_manager.reset_devices_with_seed(num_devices, seed)
        
        # Action Space: K agents, each chooses a device (0..D-1)
        self.action_space = spaces.Discrete(num_devices)
        
        # State Space Per Agent:
        # [OwnProgress, LayerComp, LayerMem, LayerOut, LayerPriv,
        #  PrevDeviceValid, PrevDeviceOneHot..., Dev1_CPU, Dev1_Mem, Dev1_BW, Dev1_Priv, Dev1_StepLoad, ...]
        self.layer_feature_dim = 4
        self.prev_device_feature_dim = self.num_devices + 1  # valid-flag + one-hot
        self.single_state_dim = 1 + self.layer_feature_dim + self.prev_device_feature_dim + (6 * num_devices)
        self.observation_space = spaces.Box(low=0.0, high=2.0,
                                            shape=(self.single_state_dim,), 
                                            dtype=np.float32)
        
        self.task: List[DNNLayer] = []
        self.progress = 0
        self.prev_device = -1
        self.done = False
        
        self.reset()
        
    def reset(self):
        """Resets the environment and the shared resource manager."""
        self.resource_manager.reset(self.num_devices)
        
        try:
            from SingleAgent.utils import generate_specific_model  # Imported here to avoid circular
        except ModuleNotFoundError:
            from utils import generate_specific_model  # Imported here to avoid circular
        
        m_type = self.model_types[0] if self.model_types else "lenet"
        self.task = generate_specific_model(m_type)
        self.progress = 0
        self.prev_device = -1
        self.done = False
            
        return self._get_observation(), {}

    def _get_observation(self):
        if self.done:
            return np.zeros(self.single_state_dim, dtype=np.float32)

        total_layers = len(self.task)
        progress = float(self.progress) / total_layers

        current_obs = [progress]

        # Current layer features (normalized)
        current_layer = self.task[self.progress]
        current_obs.extend([
            current_layer.computation_demand / 20.0,
            current_layer.memory_demand / 200.0,
            current_layer.output_data_size / 25.0,
            float(current_layer.privacy_level)
        ])

        # Previous device (valid flag + one-hot)
        prev_valid = 1.0 if self.prev_device != -1 else 0.0
        current_obs.append(prev_valid)
        for d_id in range(self.num_devices):
            current_obs.append(1.0 if self.prev_device == d_id else 0.0)

        # Shared Device States (from Resource Manager)
        for d_id in range(self.num_devices):
            # Static state (CPU, Mem, BW, Priv)
            current_obs.extend(self.resource_manager.get_state_for_device(d_id))
            # Dynamic state (Current Step Load)
            step_load = self.resource_manager.step_resources[d_id]
            current_obs.append(step_load['compute'] / 100.0)
            current_obs.append(step_load['bw'] / 50.0)

        return np.array(current_obs, dtype=np.float32)

    def get_valid_actions(self) -> List[int]:
        """
        Returns a list of valid actions in the current state.
        Mirrors the constraints used in step().
        """
        if self.done:
            return []

        # Mimic start-of-step resource conditions
        self.resource_manager.reset_step_resources()

        progress = self.progress
        current_layer = self.task[progress]
        total_layers = len(self.task)
        is_first = (progress == 0)
        prev_dev = self.prev_device

        # Pre-compute previous layer output (if any)
        if progress > 0:
            prev_out = self.task[progress - 1].output_data_size
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
                agent_id=0,
                device_id=device_id,
                layer=current_layer,
                total_agent_layers=total_layers,
                is_first_layer=is_first,
                prev_device_id=prev_dev,
                transmission_data_size=trans_data
            ):
                valid_actions.append(device_id)

        return valid_actions

    def step(self, action: int):
        """
        Executes one step for the single agent.
        Args:
            action: Device id (int)
        """
        if self.done:
            obs = self._get_observation()
            return obs, 0.0, True, False, {"reward_type": "already_done"}

        selected_device_id = int(action)
        current_layer = self.task[self.progress]
        total_layers = len(self.task)
        is_first = (self.progress == 0)

        info = {
            "layer_idx": int(self.progress),
            "device": int(selected_device_id),
            "prev_device": int(self.prev_device),
        }

        # Transmission data calculation
        prev_dev = self.prev_device
        trans_data = 0.0
        if prev_dev != -1 and prev_dev != selected_device_id:
            if self.progress > 0:
                trans_data = self.task[self.progress - 1].output_data_size
            else:
                trans_data = 5.0  # Initial input
        elif prev_dev == -1:
            trans_data = 5.0  # Input

        # Reset step-level resources (CPU/BW bottlenecks)
        self.resource_manager.reset_step_resources()

        # 1. Try Allocate (Enforces Hard Constraints 4.a, 4.b, 4.c, 4.f, 4.h)
        success = self.resource_manager.try_allocate(
            agent_id=0,
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
            reward = -500.0  # Large stalling penalty
            self.done = True
            info["reward_type"] = "stall_termination"
            info["allocation_fail"] = dict(self.resource_manager.last_allocation_fail) if self.resource_manager.last_allocation_fail else {}
            info["t_comp"] = 0.0
            info["t_comm"] = 0.0
            info["trans_data"] = float(trans_data)
            next_obs = self._get_observation()
            return next_obs, reward, True, False, info

        # 2. Calculate Latency (Cost)
        dev = self.resource_manager.devices[selected_device_id]

        # Compute
        comp_latency = current_layer.computation_demand / dev.cpu_speed

        # Transmit
        trans_latency = 0.0
        prev_dev = self.prev_device
        if prev_dev != -1 and prev_dev != selected_device_id:
            # Get output size of PREVIOUS layer
            if self.progress > 0:
                prev_data = self.task[self.progress - 1].output_data_size
            else:
                prev_data = 5.0  # Input
            trans_latency = prev_data / dev.bandwidth
        elif prev_dev == -1:
            trans_latency = 5.0 / dev.bandwidth

        total_latency = comp_latency + trans_latency
        reward = -total_latency

        # Add detailed latency info for mapping analysis
        info["t_comp"] = comp_latency
        info["t_comm"] = trans_latency
        info["reward_type"] = "success"
        info["trans_data"] = float(trans_data)

        self._advance_agent(selected_device_id)

        next_obs = self._get_observation()

        # In Gymnasium, step returns (obs, reward, terminated, truncated, info)
        terminated = self.done
        truncated = False
        
        return next_obs, reward, terminated, truncated, info

    def _advance_agent(self, device_id):
        self.prev_device = device_id
        self.progress += 1
        if self.progress >= len(self.task):
            self.done = True
