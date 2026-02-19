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
    if not isinstance(model_type, str):
        model_type = str(model_type)
    model_type = model_type.lower()
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
    elif model_type == "bigcnn": # Larger CNN (more compute/mem than DeepCNN)
        profiles = [
             (20.0, 80.0, 30.0),  # Block 1 (2x Conv64 + Pool)
             (30.0, 140.0, 18.0), # Block 2 (2x Conv128 + Pool)
             (18.0, 90.0, 12.0),  # Conv256
             (1.0, 1.0, 1.0),     # Flatten/Overhead
             (15.0, 220.0, 1.0),  # FC1 (512)
             (8.0, 120.0, 0.2),   # FC2 (256)
             (0.3, 2.0, 0.01)     # Output
        ]
    elif model_type == "biggercnn": # Larger++ CNN
        profiles = [
             (25.0, 95.0, 35.0),   # Block 1
             (40.0, 180.0, 22.0),  # Block 2
             (30.0, 160.0, 14.0),  # Block 3 (2x Conv256)
             (1.0, 1.0, 1.0),      # Flatten/Overhead
             (24.0, 320.0, 1.5),   # FC1 (1024)
             (16.0, 200.0, 0.8),   # FC2 (512)
             (10.0, 150.0, 0.3),   # FC3 (256)
             (0.5, 2.5, 0.01),     # Output
        ]
    elif model_type == "hugecnn": # Largest CNN (very heavy)
        profiles = [
             (35.0, 130.0, 45.0),  # Block 1 (96)
             (55.0, 240.0, 28.0),  # Block 2 (192)
             (45.0, 260.0, 18.0),  # Block 3 (2x Conv384)
             (30.0, 220.0, 18.0),  # Extra Conv384
             (1.0, 1.0, 1.0),      # Flatten/Overhead
             (40.0, 520.0, 2.0),   # FC1 (2048)
             (26.0, 360.0, 1.0),   # FC2 (1024)
             (18.0, 260.0, 0.6),   # FC3 (512)
             (12.0, 180.0, 0.3),   # FC4 (256)
             (0.8, 3.0, 0.01),     # Output
        ]
    elif model_type == "cnn7": # 7 layers
        profiles = [(2.0, 5.0, 2.0)] * 7
    elif model_type == "cnn10": # 10 layers
        profiles = [(1.5, 4.0, 1.5)] * 10
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
        devices.append(IoTDevice(
            device_id=i,
            cpu_speed=random.uniform(35.0, 45.0), # Bottleneck for 3 agents (~15c * 3 = 45c)
            memory_capacity=random.uniform(400, 1000), # Bottleneck for 3 agents (~200m * 2 = 400m)
            current_memory_usage=0.0,
            bandwidth=random.uniform(100, 300), 
            privacy_clearance=random.choice([0, 1]) 
        ))

    # Ensure at least 2 devices can host privacy_level=1 layers (e.g., first layer).
    min_private = min(2, num_devices)
    private_count = sum(1 for d in devices if d.privacy_clearance == 1)
    if private_count < min_private:
        candidates = [d for d in devices if d.privacy_clearance == 0]
        for d in random.sample(candidates, k=min_private - private_count):
            d.privacy_clearance = 1
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
