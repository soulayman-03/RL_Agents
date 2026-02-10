import torch
import torch.nn as nn
from split_inference.cnn_model import SimpleCNN, DeepCNN, MiniResNet
from MultiAgent.manager import MultiAgentManager
from MultiAgent.environment import MultiAgentIoTEnv
from SingleAgent.utils import load_and_remap_weights
import os

def main():
    # 1. Configuration
    NUM_AGENTS = 3
    NUM_DEVICES = 5
    MODEL_TYPES = ["simplecnn", "deepcnn", "miniresnet"]
    
    # Paths to trained weights
    WEIGHT_PATHS = {
        0: "split_inference/mnist_simplecnn.pth",
        1: "split_inference/mnist_deepcnn.pth",
        2: "split_inference/mnist_miniresnet.pth"
    }

    # 2. Initialize System
    env = MultiAgentIoTEnv(num_agents=NUM_AGENTS, num_devices=NUM_DEVICES, model_types=MODEL_TYPES)
    manager = MultiAgentManager(
        agent_ids=list(range(NUM_AGENTS)), 
        state_dim=env.single_state_dim, 
        action_dim=env.num_devices
    )

    # 3. Instantiate Real CNN Models
    real_models = {
        0: SimpleCNN(),
        1: DeepCNN(),
        2: MiniResNet()
    }

    # 4. Load Weights and Assign to Agents
    print("Loading real CNN weights into agents...")
    for aid, model in real_models.items():
        weight_path = WEIGHT_PATHS[aid]
        if os.path.exists(weight_path):
            # Use the remapping utility to handle any key differences
            success = load_and_remap_weights(model, weight_path, MODEL_TYPES[aid])
            if success:
                print(f"  Agent {aid} ({MODEL_TYPES[aid]}): Weights loaded successfully.")
            else:
                print(f"  Agent {aid} ({MODEL_TYPES[aid]}): Failed to load weights.")
        else:
            print(f"  Agent {aid} ({MODEL_TYPES[aid]}): Weight file not found at {weight_path}")

    # Assign models to agents via MultiAgentManager
    manager.assign_models(real_models)

    print("\nSystem ready with real CNN models assigned to RL agents.")
    print("Each agent can now use self.inference_model for actual image classification.")

if __name__ == "__main__":
    main()
