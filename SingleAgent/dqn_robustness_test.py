import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json

# Define paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(PROJECT_ROOT)

try:
    from .agent import DQNAgent
    from .environment import MonoAgentIoTEnv
    from .utils import load_and_remap_weights, set_global_seed
except ImportError:
    from agent import DQNAgent
    from environment import MonoAgentIoTEnv
    from utils import load_and_remap_weights, set_global_seed
from split_inference.cnn_model import SimpleCNN

def _slugify(name):
    return "".join(c.lower() if c.isalnum() else "_" for c in str(name)).strip("_")

def plot_layer_latency(all_episode_traces, task, results_dir, filename):
    """
    Plot average layer-wise latency breakdown (computation vs communication).
    Expects `all_episode_traces` to be a list of episode traces, where each trace is a list of dicts:
      { "layer": int, "t_comp": float, "t_comm": float, ... }
    """
    num_layers = len(task)
    comp = np.zeros(num_layers, dtype=float)
    comm = np.zeros(num_layers, dtype=float)
    counts = np.zeros(num_layers, dtype=float)

    for trace in all_episode_traces:
        for t in trace:
            if not isinstance(t, dict) or "layer" not in t:
                continue
            layer = int(t["layer"])
            if layer < 0 or layer >= num_layers:
                continue
            comp[layer] += float(t.get("t_comp", 0.0))
            comm[layer] += float(t.get("t_comm", 0.0))
            counts[layer] += 1.0

    counts[counts == 0] = 1.0
    comp /= counts
    comm /= counts

    layer_names = []
    for i, layer in enumerate(task):
        layer_names.append(getattr(layer, "name", f"Layer_{i}"))

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

def run_test(agent, env, scenario_name, episodes=20, collect_traces=False):
    rewards = []
    stalls = []
    cpu_utils = []
    mem_utils = []
    all_episode_traces = []
    all_devices = []
    
    agent.epsilon = 0.0 # No exploration for testing
    
    for e in range(episodes):
        state, _ = env.reset()
        dep_reward = 0
        ep_stalls = 0
        done = False
        
        ep_cpu = []
        ep_mem = []
        trace = []
        
        step = 0
        while not done and step < 100:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
                
            action = agent.act(state, valid_actions)
            prev_progress = env.progress
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if env.progress > prev_progress:
                trace.append({
                    "layer": int(prev_progress),
                    "device": int(action),
                    "t_comp": info.get("t_comp", 0.0),
                    "t_comm": info.get("t_comm", 0.0),
                })

            # Pull stats from environment's resource manager
            device_idx = int(action)
            device = env.resource_manager.devices[device_idx]
            
            layer_idx = env.progress - 1
            if layer_idx >= 0:
                layer = env.task[layer_idx]
                cpu_util = (layer.computation_demand / device.cpu_speed) if device.cpu_speed > 0 else 0
                mem_util = (layer.memory_demand / device.memory_capacity) if device.memory_capacity > 0 else 0
                ep_cpu.append(cpu_util)
                ep_mem.append(mem_util)

            if reward == -500.0:
                ep_stalls += 1
            
            dep_reward += reward
            state = next_state
            all_devices.append(int(action))
            step += 1
            
        rewards.append(dep_reward)
        stalls.append(ep_stalls)
        cpu_utils.append(np.mean(ep_cpu) if ep_cpu else 0)
        mem_utils.append(np.mean(ep_mem) if ep_mem else 0)
        all_episode_traces.append(trace)
        
    summary = {
        "scenario": scenario_name,
        "avg_reward": np.mean(rewards),
        "success_rate": np.mean([1 if s == 0 else 0 for s in stalls]) * 100,
        "avg_cpu": np.mean(cpu_utils),
        "avg_mem": np.mean(mem_utils),
        "device_usage": np.bincount(np.asarray(all_devices, dtype=int), minlength=env.num_devices).tolist() if all_devices else [0] * env.num_devices,
    }

    if collect_traces:
        return summary, all_episode_traces
    return summary

def robustness_comparison_dqn():
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "resultDQN", "robustness")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "single_agent_simplecnn.pth")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained DQN model not found at {MODEL_PATH}.")
        return

    # Initialize Agent
    set_global_seed(42)
    temp_env = MonoAgentIoTEnv(num_agents=1, num_devices=5, model_types=["simplecnn"], seed=42)
    agent = DQNAgent(state_dim=temp_env.single_state_dim, action_dim=5)
    agent.load(MODEL_PATH)
    
    cv_model = SimpleCNN()
    agent.assign_inference_model(cv_model)
    
    results = []
    
    # Scenario 1: Base Environment
    print("Testing Base Scenario...")
    env_base = MonoAgentIoTEnv(num_agents=1, num_devices=5, model_types=["simplecnn"], seed=42)
    summary, traces = run_test(agent, env_base, "Base", collect_traces=True)
    results.append(summary)
    
    # Scenario 2: Degraded Device (Slow CPU)
    print("Testing Degraded Device (Slow CPU)...")
    env_slow = MonoAgentIoTEnv(num_agents=1, num_devices=5, model_types=["simplecnn"], seed=42)
    env_slow.resource_manager.devices[0].cpu_speed *= 0.2 
    summary, traces = run_test(agent, env_slow, "Slow CPU (D0)", collect_traces=True)
    results.append(summary)
    
    # Scenario 3: Network Drop (Low Bandwidth)
    print("Testing Network Drop (Low Bandwidth)...")
    env_net = MonoAgentIoTEnv(num_agents=1, num_devices=5, model_types=["simplecnn"], seed=42)
    for d in env_net.resource_manager.devices.values():
        d.bandwidth *= 0.1
    summary, traces = run_test(agent, env_net, "Low BW", collect_traces=True)
    results.append(summary)
    
    # Scenario 4: High Load
    print("Testing High Load...")
    env_load = MonoAgentIoTEnv(num_agents=1, num_devices=5, model_types=["simplecnn"], seed=42)
    for t in env_load.task:
        t.computation_demand *= 2.0
    summary, traces = run_test(agent, env_load, "High Load", collect_traces=True)
    results.append(summary)

    # Plotting results
    scenarios = [r['scenario'] for r in results]
    rewards = [r['avg_reward'] for r in results]
    success = [r['success_rate'] for r in results]
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.bar(scenarios, rewards, color='skyblue')
    plt.title("DQN Reward across Robustness Scenarios")
    plt.ylabel("Avg Reward")
    
    plt.subplot(2, 1, 2)
    plt.bar(scenarios, success, color='salmon')
    plt.title("DQN Success Rate (%) across Robustness Scenarios")
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "robustness_comparison.png"))
    
    print("\nDQN Robustness tests complete. Results:")
    for r in results:
        print(f"- {r['scenario']}: Reward={r['avg_reward']:.2f}, Success={r['success_rate']:.1f}%, CPU={r['avg_cpu']:.2f}")

    with open(os.path.join(RESULTS_DIR, "robustness_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    robustness_comparison_dqn()
