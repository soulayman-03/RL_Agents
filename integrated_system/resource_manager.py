from typing import List, Dict
import math
from SingleAgent.utils import IoTDevice, DNNLayer, generate_iot_network, set_global_seed
class ResourceManager:
    """
    Singleton-like class to manage shared resources of IoT devices.
    Thread-safe if we ever move to threaded simulation, though currently sequential-multi-agent.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ResourceManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, num_devices=5):
        # Avoid re-initialization if already created
        if not hasattr(self, 'initialized'):
            self.devices: Dict[int, IoTDevice] = {}
            for d in generate_iot_network(num_devices):
                self.devices[d.device_id] = d
            
            # History: {agent_id: {device_id: count_of_layers}}
            self.assignment_history: Dict[int, Dict[int, int]] = {}
            # Step resources: {device_id: {'compute': float, 'bandwidth': float}}
            self.step_resources: Dict[int, Dict[str, float]] = {d_id: {'compute': 0.0, 'bw': 0.0} for d_id in range(num_devices)}
            # Last allocation failure info for debugging
            self.last_allocation_fail: Dict[str, object] = {}

            # Security-level constraint: max fraction of an agent's model layers that can be exposed
            # to a single device (per episode). Set to 1.0 to disable.
            self.max_exposure_fraction: float = 1.0
             
            self.initialized = True

    def set_max_exposure_fraction(self, value: float | None) -> None:
        """
        Sets the max fraction (S_l) of an agent's model that may be assigned to a single device.
        - value=None or value>=1.0 disables the constraint
        - value in (0,1) enables it (e.g., 0.5 means at most floor(L*0.5) layers per device)
        """
        if value is None:
            self.max_exposure_fraction = 1.0
            return
        v = float(value)
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError(f"max_exposure_fraction must be > 0, got {value}")
        self.max_exposure_fraction = v

    def reset(self, num_devices=5):
        """
        Resets the state of all devices WITHOUT regenerating them.
        Preserves device characteristics (CPU, memory capacity, bandwidth, privacy).
        Only resets: memory usage, assignment history, and step resources.
        """
        # Only reset memory usage for existing devices
        for device in self.devices.values():
            device.current_memory_usage = 0.0
        
        # Reset allocation tracking
        self.assignment_history = {}
        self.step_resources = {d_id: {'compute': 0.0, 'bw': 0.0} for d_id in self.devices.keys()}
        self.reset_step_resources()

    def reset_step_resources(self):
        """Resets compute and bandwidth usage at the start of each env step."""
        for d_id in self.step_resources:
            self.step_resources[d_id] = {'compute': 0.0, 'bw': 0.0}

    def set_devices(self, devices: List[IoTDevice]):
        """Replaces the current device set (for reproducible evaluation)."""
        self.devices = {d.device_id: d for d in devices}
        for device in self.devices.values():
            device.current_memory_usage = 0.0
        self.assignment_history = {}
        self.step_resources = {d_id: {'compute': 0.0, 'bw': 0.0} for d_id in self.devices.keys()}
        self.reset_step_resources()

    def reset_devices_with_seed(self, num_devices: int, seed: int | None):
        """Regenerates devices deterministically for a given seed."""
        set_global_seed(seed)
        devices = generate_iot_network(num_devices, seed=seed)
        self.set_devices(devices)
    
    def get_state_for_device(self, device_id: int) -> List[float]:
        """Returns the normalized OBSERVATION variables for a specific device."""
        d = self.devices[device_id]
        # CPU, Mem Free, Bandwidth, Privacy
        return [
            d.cpu_speed / 50.0,
            (d.memory_capacity - d.current_memory_usage) / 600.0,
            d.bandwidth / 300.0,
            float(d.privacy_clearance)
        ]

    def _validate_device_characteristics(self, device_id: int) -> None:
        """
        Sanity-check immutable device characteristics.
        This should be called before and after each allocation.
        """
        if device_id not in self.devices:
            raise ValueError(f"Unknown device_id: {device_id}")
        d = self.devices[device_id]
        if d.cpu_speed <= 0 or d.memory_capacity <= 0 or d.bandwidth <= 0:
            raise ValueError(
                f"Invalid device capacities for device_id={device_id}: "
                f"cpu_speed={d.cpu_speed}, memory_capacity={d.memory_capacity}, bandwidth={d.bandwidth}"
            )
        if d.privacy_clearance < 0:
            raise ValueError(
                f"Invalid privacy_clearance for device_id={device_id}: {d.privacy_clearance}"
            )

    def can_allocate(self, agent_id: int, device_id: int, layer: DNNLayer, total_agent_layers: int, 
                     is_first_layer: bool, prev_device_id: int = -1, transmission_data_size: float = 0.0) -> bool:
        """
        Checks if a layer can be allocated to a device WITHOUT mutating state.
        Mirrors try_allocate constraints.
        """
        self.last_allocation_fail = {}
        self._validate_device_characteristics(device_id)
        device = self.devices[device_id]
        
        # 0. Check Sequential Diversity (4.e)
        if prev_device_id != -1 and device_id == prev_device_id:
            self.last_allocation_fail = {
                "reason": "sequential_diversity",
                "device_id": device_id,
                "prev_device_id": prev_device_id
            }
            return False
            
        # 1. Check Privacy (4.f)
        if device.privacy_clearance < layer.privacy_level:
            self.last_allocation_fail = {
                "reason": "privacy",
                "device_id": device_id,
                "device_privacy": device.privacy_clearance,
                "layer_privacy": layer.privacy_level
            }
            return False
            
        # 2. Check Memory (4.a)
        if device.current_memory_usage + layer.memory_demand > device.memory_capacity:
            self.last_allocation_fail = {
                "reason": "memory",
                "device_id": device_id,
                "current_memory": device.current_memory_usage,
                "layer_memory": layer.memory_demand,
                "capacity": device.memory_capacity
            }
            return False
            
        # 3. Check Privacy Exposure (4.h)
        if agent_id in self.assignment_history:
            prev_layers_count = sum(self.assignment_history[agent_id].values())
            if prev_layers_count > 0 and self.assignment_history[agent_id].get(device_id, 0) == prev_layers_count:
                if prev_layers_count + 1 == total_agent_layers:
                    self.last_allocation_fail = {
                        "reason": "privacy_exposure",
                        "device_id": device_id,
                        "prev_layers_count": prev_layers_count,
                        "total_layers": total_agent_layers
                    }
                    return False

        # 3b. Security level constraint (S_l): cap how many layers of an agent can be exposed to one device.
        # Enforced per agent, per device, across the episode.
        max_frac = float(getattr(self, "max_exposure_fraction", 1.0))
        if max_frac < 1.0:
            total_layers = max(1, int(total_agent_layers))
            max_layers_on_device = int(math.floor(total_layers * max_frac))
            max_layers_on_device = max(1, max_layers_on_device)
            already_on_device = int(self.assignment_history.get(agent_id, {}).get(device_id, 0))
            if already_on_device + 1 > max_layers_on_device:
                self.last_allocation_fail = {
                    "reason": "security_level_exposure",
                    "device_id": device_id,
                    "already_on_device": already_on_device,
                    "max_layers_on_device": max_layers_on_device,
                    "total_layers": total_layers,
                    "max_exposure_fraction": max_frac,
                }
                return False

        # 4. Check Compute Constraint (4.b)
        current_step_compute = self.step_resources[device_id]['compute']
        if current_step_compute + layer.computation_demand > device.cpu_speed * 10.0: 
            self.last_allocation_fail = {
                "reason": "compute",
                "device_id": device_id,
                "current_step_compute": current_step_compute,
                "layer_compute": layer.computation_demand,
                "limit": device.cpu_speed * 10.0
            }
            return False

        # 5. Check Bandwidth Constraint (4.c)
        if prev_device_id != -1 and prev_device_id != device_id:
            current_step_bw = self.step_resources[device_id]['bw']
            if current_step_bw + transmission_data_size > device.bandwidth * 5.0:
                self.last_allocation_fail = {
                    "reason": "bandwidth",
                    "device_id": device_id,
                    "current_step_bw": current_step_bw,
                    "transmission": transmission_data_size,
                    "limit": device.bandwidth * 5.0
                }
                return False

        self.last_allocation_fail = {}
        return True

    def try_allocate(self, agent_id: int, device_id: int, layer: DNNLayer, total_agent_layers: int, 
                     is_first_layer: bool, prev_device_id: int = -1, transmission_data_size: float = 0.0) -> bool:
        """
        Attempts to allocate a layer to a device.
        Returns True if successful (and updates state), False otherwise.
        Strictly enforces (4.a, 4.b, 4.c, 4.f, 4.h).
        """
        self._validate_device_characteristics(device_id)
        device = self.devices[device_id]
        # Snapshot immutable characteristics to ensure they don't change due to allocation.
        static_snapshot = (device.cpu_speed, device.memory_capacity, device.bandwidth, device.privacy_clearance)

        if not self.can_allocate(
            agent_id=agent_id,
            device_id=device_id,
            layer=layer,
            total_agent_layers=total_agent_layers,
            is_first_layer=is_first_layer,
            prev_device_id=prev_device_id,
            transmission_data_size=transmission_data_size
        ):
            return False

        # Allocate
        device.current_memory_usage += layer.memory_demand
        
        # Update step resources
        self.step_resources[device_id]['compute'] += layer.computation_demand
        if prev_device_id != -1 and prev_device_id != device_id:
            self.step_resources[device_id]['bw'] += transmission_data_size
        
        # Record history
        if agent_id not in self.assignment_history:
            self.assignment_history[agent_id] = {}
        self.assignment_history[agent_id][device_id] = self.assignment_history[agent_id].get(device_id, 0) + 1

        # Validate after allocation
        self._validate_device_characteristics(device_id)
        if static_snapshot != (device.cpu_speed, device.memory_capacity, device.bandwidth, device.privacy_clearance):
            raise RuntimeError(f"Device characteristics changed during allocation for device_id={device_id}")
        
        return True

    def release(self, device_id: int, memory_amount: float):
        """Releases memory from a device (e.g., after task completion)."""
        if device_id in self.devices:
            self.devices[device_id].current_memory_usage = max(0.0, self.devices[device_id].current_memory_usage - memory_amount)
