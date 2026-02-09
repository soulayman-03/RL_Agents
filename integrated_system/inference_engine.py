import torch
import time
import sys

class DistributedRunner:
    """
    Executes a PyTorch model in a split fashion based on an allocation map.
    Simulates the network transfer delays between devices.
    """
    def __init__(self, model):
        self.model = model
        self.device_logs = []
        
    def run(self, input_data, allocation_map):
        """
        Args:
            input_data: The input tensor (image).
            allocation_map: List of device IDs, one for each layer in model.layers.
        """
        x = input_data
        current_device = -1 # Source
        
        print("\n--- Starting Distributed Inference ---")
        
        for i, layer in enumerate(self.model.layers):
            target_device = allocation_map[i]
            
            # Simulate Network Transfer if crossing boundaries
            if target_device != current_device:
                data_size_mb = x.element_size() * x.nelement() / (1024 * 1024)
                print(f"[Network] Transferring {data_size_mb:.4f} MB from Dev {current_device} -> Dev {target_device}")
                # Time scaling for demo
                # time.sleep(0.01) 
                
            # Simulate Computation on Device
            print(f"[Device {target_device}] Executing Layer {i}")
            with torch.no_grad():
                x = layer(x)
            
            current_device = target_device
            
        print("--- Inference Complete ---\n")
        return x

class MultiTaskRunner:
    """
    Executes MULTIPLE PyTorch models simultaneously (simulated interleaved execution).
    """
    def __init__(self, model):
        self.model = model
        
    def run(self, input_data, allocation_map):
        """Wrapper for single-task execution."""
        return self.run_all([input_data], [allocation_map])[0]
        
    def run_all(self, inputs: list, allocation_maps: list):
        """
        Args:
           inputs: List of input tensors.
           allocation_maps: List of allocation maps (one per input/task).
        """
        num_tasks = len(inputs)
        print(f"\n=== Starting Multi-Task Execution (N={num_tasks}) ===")
        
        # We will interleave layer execution to simulate concurrency
        # State tracking for each task
        task_states = []
        for k in range(num_tasks):
            task_states.append({
                "id": k,
                "data": inputs[k],
                "current_device": -1,
                "layer_idx": 0,
                "done": False,
                "map": allocation_maps[k]
            })
            
        active_tasks = True
        step = 0
        
        while active_tasks:
            active_tasks = False
            print(f"\n[Time Step {step}]")
            
            for k in range(num_tasks):
                state = task_states[k]
                if state["done"]:
                    continue
                    
                active_tasks = True
                
                # Execute ONE layer for this task
                layer_idx = state["layer_idx"]
                allocation_map = state["map"]
                
                # Safety check
                if layer_idx >= len(self.model.layers):
                    state["done"] = True
                    continue

                target_device = allocation_map[layer_idx]
                current_device = state["current_device"]
                x = state["data"]
                
                # Transfer Log
                if target_device != current_device:
                    print(f"  [Task {k}] Transfer Dev {current_device} -> Dev {target_device}")
                    
                # Compute Log
                print(f"  [Task {k}] Compute Layer {layer_idx} on Dev {target_device}")
                
                # Execute
                layer = self.model.layers[layer_idx]
                with torch.no_grad():
                    state["data"] = layer(x)
                
                # Update State
                state["current_device"] = target_device
                state["layer_idx"] += 1
                if state["layer_idx"] >= len(self.model.layers):
                    state["done"] = True
                    print(f"  [Task {k}] FINISHED")
            
            step += 1
            # time.sleep(0.1) # Demo delay
            
        print("\n=== All Tasks Complete ===")
        return [t["data"] for t in task_states]
