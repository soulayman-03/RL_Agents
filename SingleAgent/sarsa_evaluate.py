import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import torch

# Define paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Add parent directory to path
sys.path.append(PROJECT_ROOT)

try:
    from .agent import DeepSARSAAgent
    from .environment import MonoAgentIoTEnv
except ImportError:
    from agent import DeepSARSAAgent
    from environment import MonoAgentIoTEnv
from split_inference.cnn_model import SimpleCNN
try:
    from .utils import load_and_remap_weights, set_global_seed
except ImportError:
    from utils import load_and_remap_weights, set_global_seed

def plot_layer_latency(all_episode_traces, task, results_dir, filename):
    num_layers = len(task)
    comp = np.zeros(num_layers)
    comm = np.zeros(num_layers)
    counts = np.zeros(num_layers)

    for trace in all_episode_traces:
        for t in trace:
            if not isinstance(t, dict):
                continue
            if "layer" not in t:
                continue

            layer = int(t["layer"])
            comp[layer] += float(t.get("t_comp", 0))
            comm[layer] += float(t.get("t_comm", 0))
            counts[layer] += 1

    counts[counts == 0] = 1
    comp /= counts
    comm /= counts

    layer_names = [layer.name for layer in task]
    x = np.arange(num_layers)
    width = 0.4

    plt.figure(figsize=(14, 5))
    plt.bar(x - width / 2, comp, width, label="Computation")
    plt.bar(x + width / 2, comm, width, label="Communication")

    plt.xticks(x, layer_names, rotation=60)
    plt.ylabel("Latency (s)")
    plt.xlabel("Layers")
    plt.title("Average Layer-wise Latency Breakdown")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(results_dir, filename))
    plt.close()

def sarsa_evaluate():
    NUM_EPISODES = 10
    NUM_DEVICES = 5
    MODEL_TYPE = "simplecnn"
    SEED = 42
    LAYER_LATENCY_SEED = 46
    LAYER_LATENCY_EPISODES = 50
    
    RESULTS_ROOT = os.path.join(SCRIPT_DIR, "results", "resultSARSA")
    RESULTS_DIR = os.path.join(RESULTS_ROOT, "eval")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    MODEL_PATHS = [
        os.path.join(SCRIPT_DIR, "models", "sarsa_agent.pth"),    # preferred (global models folder)
        os.path.join(RESULTS_ROOT, "models", "sarsa_agent.pth"),  # backward-compat
    ]
    MODEL_PATH = None
    for p in MODEL_PATHS:
        if os.path.exists(p):
            MODEL_PATH = p
            break
    if MODEL_PATH is None:
        print(f"Error: Trained SARSA model not found. Looked in: {MODEL_PATHS}.")
        return

    print(f"=== Starting Deep SARSA Evaluation ({MODEL_TYPE}) ===")
    
    # Initialize Environment
    set_global_seed(SEED)
    env = MonoAgentIoTEnv(num_agents=1, num_devices=NUM_DEVICES, model_types=[MODEL_TYPE], seed=SEED)
    
    # Initialize Agent
    agent = DeepSARSAAgent(state_dim=env.single_state_dim, action_dim=NUM_DEVICES)
    agent.load(MODEL_PATH)
    print(f"Loaded trained agent from {MODEL_PATH}")
    
    cv_model = SimpleCNN()
    agent.assign_inference_model(cv_model)
    agent.epsilon = 0.0 # No exploration during evaluation

    # Layer-wise latency plot (requested: seed=46)
    set_global_seed(LAYER_LATENCY_SEED)
    latency_env = MonoAgentIoTEnv(
        num_agents=1,
        num_devices=NUM_DEVICES,
        model_types=[MODEL_TYPE],
        seed=LAYER_LATENCY_SEED
    )
    all_episode_traces = []
    for _ in range(LAYER_LATENCY_EPISODES):
        state, _ = latency_env.reset()
        done = False
        trace = []

        valid_actions = latency_env.get_valid_actions()
        action = agent.act(state, valid_actions)

        step = 0
        while not done and step < 100:
            prev_progress = latency_env.progress
            next_state, reward, terminated, truncated, info = latency_env.step(action)
            done = terminated or truncated

            if latency_env.progress > prev_progress:
                trace.append({
                    "layer": prev_progress,
                    "device": int(action),
                    "t_comp": info.get("t_comp", 0),
                    "t_comm": info.get("t_comm", 0),
                })

            next_valid_actions = latency_env.get_valid_actions()
            if not done:
                next_action = agent.act(next_state, next_valid_actions)
            else:
                next_action = 0

            state = next_state
            action = next_action
            step += 1

        all_episode_traces.append(trace)

    plot_layer_latency(
        all_episode_traces,
        latency_env.task,
        RESULTS_DIR,
        f"layer_latency_avg_seed_{LAYER_LATENCY_SEED}.png",
    )
    print(f" - Layer latency plot saved to {RESULTS_DIR}/layer_latency_avg_seed_{LAYER_LATENCY_SEED}.png")
    
    for e in range(NUM_EPISODES):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        ep_stalls = 0
        trace = []
        
        # Initial action for SARSA
        valid_actions = env.get_valid_actions()
        action = agent.act(state, valid_actions)
        
        step = 0
        while not done and step < 100:
            device = env.resource_manager.devices[int(action)]
            prev_progress = env.progress
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if reward == -500.0:
                ep_stalls += 1
            
            if env.progress > prev_progress:
                layer_idx = prev_progress
                layer = env.task[layer_idx]
                trace.append({
                    'layer': layer_idx,
                    'device': int(action),
                    'l_comp': float(layer.computation_demand),
                    'l_mem': float(layer.memory_demand),
                    'l_out': float(layer.output_data_size),
                    'd_cpu': float(device.cpu_speed),
                    'd_mem': float(device.memory_capacity),
                    'd_bw': float(device.bandwidth),
                    'priv': int(layer.privacy_level)
                })
            
            # Choose next action (SARSA style, but epsilon=0)
            next_valid_actions = env.get_valid_actions()
            if not done:
                next_action = agent.act(next_state, next_valid_actions)
            else:
                next_action = 0
                
            ep_reward += reward
            state = next_state
            action = next_action
            step += 1
            
        respected = "YES" if ep_stalls == 0 else "NO"
        print(f"Episode {e+1}/{NUM_EPISODES} | Reward: {ep_reward:8.2f} | Stalls: {ep_stalls:2} | Constraints Respected: {respected}")
        print("  Trace: layer -> device | L_comp L_mem L_out | D_cpu D_mem D_bw | priv")
        for t in trace:
             print(f"   L{t['layer']:02d} -> D{t['device']} | {t['l_comp']:6.1f} {t['l_mem']:6.1f} {t['l_out']:6.1f} | {t['d_cpu']:5.2f} {t['d_mem']:5.1f} {t['d_bw']:5.1f} | {t['priv']}")
        print("-" * 80)

if __name__ == "__main__":
    sarsa_evaluate()
