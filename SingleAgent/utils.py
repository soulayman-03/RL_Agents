import dataclasses
import torch
import os
from typing import List, Dict
import random
import numpy as np

def set_global_seed(seed: int | None) -> None:
    """Sets global RNG seeds for reproducibility."""
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@dataclasses.dataclass
class DNNLayer:
    """Represents a single layer of the Neural Network task."""
    layer_id: int
    name: str # e.g. "conv1", "fc1"
    computation_demand: float # Estimated FLOPs or time units
    memory_demand: float # MB required to store weights/output
    output_data_size: float # MB of data to transmit to next layer
    privacy_level: int # Privacy requirement (0=Public, 1=Private)

@dataclasses.dataclass
class IoTDevice:
    """Represents an IoT device in the network."""
    device_id: int
    cpu_speed: float # Relative CPU speed factor (e.g. 1.0 = baseline)
    memory_capacity: float # Total MB
    current_memory_usage: float # Used MB
    bandwidth: float # Mbps link speed
    privacy_clearance: int # Max privacy level this device is allowed to see (0 or 1)
    
    def can_host(self, layer: DNNLayer) -> bool:
        """Check if device has resources and clearance for the layer."""
        if self.current_memory_usage + layer.memory_demand > self.memory_capacity:
            return False # Out of memory
        if self.privacy_clearance < layer.privacy_level:
            return False # Privacy violation
        return True

def generate_specific_model(model_type="lenet") -> List[DNNLayer]:
    """Generates a sequence of layers for specific model profiles."""
    layers = []
    
    if model_type == "lenet": # 6 layers
        profiles = [
            (2.0, 5.0, 3.0),   # Conv1
            (1.0, 5.0, 1.0),   # Conv2
            (1.0, 1.0, 0.5),   # Flatten/Overhead
            (1.5, 120.0, 0.1), # FC1
            (0.5, 10.0, 0.01), # FC2
            (0.1, 0.1, 0.01)   # Output
        ]
        
    elif model_type == "simplecnn": # 5 layers
        profiles = [
             (5.0, 20.0, 15.0), # Conv1 (32)
             (8.0, 40.0, 8.0),  # Conv2 (64)
             (0.5, 0.5, 0.5),   # Flatten
             (5.0, 150.0, 0.2), # Dense (128)
             (0.1, 1.0, 0.01)   # Output
        ]

    elif model_type == "deepcnn": # Deeper, more compute
        profiles = [
             (10.0, 40.0, 20.0), # Block 1 (2x Conv32 + Pool)
             (15.0, 80.0, 10.0), # Block 2 (2x Conv64 + Pool)
             (0.5, 0.5, 0.5),    # Flatten
             (8.0, 200.0, 0.5),  # FC1 (256)
             (0.2, 2.0, 0.01)    # Output
        ]

    elif model_type == "miniresnet": # Skip connections, heavy compute
        profiles = [
             (12.0, 50.0, 25.0), # Block 1 (Conv32)
             (18.0, 60.0, 25.0), # Block 2 (ResBlock)
             (1.0, 1.0, 1.0),    # Pool + Flatten
             (6.0, 150.0, 0.2),  # FC1 (128)
             (0.1, 1.0, 0.01)    # Output
        ]
    else:
        return generate_specific_model("simplecnn") # Fallback

    for i, (comp, mem, out_size) in enumerate(profiles):
        layers.append(DNNLayer(
            layer_id=i,
            name=f"{model_type}_L{i}",
            computation_demand=comp,
            memory_demand=mem,
            output_data_size=out_size,
            privacy_level=1 if i == 0 else 0
        ))
    return layers

def generate_dummy_dnn_model(num_layers=6):
    """Legacy wrapper."""
    return generate_specific_model("lenet")

def generate_iot_network(num_devices=5, seed: int | None = None) -> List[IoTDevice]:
    """Generates a set of heterogeneous IoT devices."""
    if seed is not None:
        set_global_seed(seed)
    devices = []
    for i in range(num_devices):
        # Deterministic random for reproducibility if needed
        devices.append(IoTDevice(
            device_id=i,
            cpu_speed=random.uniform(0.5, 2.0),
            memory_capacity=random.uniform(200, 600), # Increased for multi-agent
            current_memory_usage=0.0,
            bandwidth=random.uniform(20, 100), # Increased
            privacy_clearance=random.choice([0, 1]) 
        ))
    return devices

def load_and_remap_weights(model, path, model_type):
    """
    Centralized utility to load weights and remap keys if necessary.
    Handles the difference between old (conv1, fc1) and new (layers.0.0, layers.3.0) structures.
    """
    if not os.path.exists(path):
        return False
        
    try:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        
        # Mappings for various architectures
        mappings = {
            "simplecnn": {
                "conv1.weight": "layers.0.0.weight", "conv1.bias": "layers.0.0.bias",
                "conv2.weight": "layers.1.0.weight", "conv2.bias": "layers.1.0.bias",
                "fc1.weight": "layers.3.0.weight", "fc1.bias": "layers.3.0.bias",
                "fc2.weight": "layers.4.weight", "fc2.bias": "layers.4.bias"
            },
            "deepcnn": {
                "conv1.weight": "layers.0.0.weight", "conv1.bias": "layers.0.0.bias",
                "conv2.weight": "layers.0.2.weight", "conv2.bias": "layers.0.2.bias",
                "conv3.weight": "layers.1.0.weight", "conv3.bias": "layers.1.0.bias",
                "conv4.weight": "layers.1.2.weight", "conv4.bias": "layers.1.2.bias",
                "fc1.weight": "layers.3.0.weight", "fc1.bias": "layers.3.0.bias",
                "fc2.weight": "layers.4.weight", "fc2.bias": "layers.4.bias"
            },
            "miniresnet": {
                "conv1.weight": "layers.0.0.weight", "conv1.bias": "layers.0.0.bias",
                "conv2.weight": "layers.0.2.weight", "conv2.bias": "layers.0.2.bias",
                "conv3.weight": "layers.1.conv.weight", "conv3.bias": "layers.1.conv.bias",
                "fc1.weight": "layers.3.0.weight", "fc1.bias": "layers.3.0.bias",
                "fc2.weight": "layers.4.weight", "fc2.bias": "layers.4.bias"
            }
        }
        
        if model_type in mappings:
            mapping = mappings[model_type]
            new_state_dict = {}
            for old_key, val in state_dict.items():
                if old_key in mapping:
                    new_state_dict[mapping[old_key]] = val
                else:
                    new_state_dict[old_key] = val
            model.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
            
        return True
    except Exception as e:
        print(f"Error remapping weights for {model_type}: {e}")
        return False
