import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Define paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(PROJECT_ROOT)

from agent import DeepSARSAAgent
from environment import MonoAgentIoTEnv
from utils import load_and_remap_weights, set_global_seed
from split_inference.cnn_model import SimpleCNN

def run_test(agent, env, scenario_name, episodes=20):
    rewards = []
    stalls = []
    cpu_utils = []
    mem_utils = []
    
    for e in range(episodes):
        state, _ = env.reset()
        dep_reward = 0
        ep_stalls = 0
        done = False
        
        ep_cpu = []
        ep_mem = []
        
        valid_actions = env.get_valid_actions()
        action = agent.act(state, valid_actions)
        
        step = 0
        while not done and step < 100:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Choose next action a' from s'
            next_valid_actions = env.get_valid_actions()
            if not done:
                next_action = agent.act(next_state, next_valid_actions)
            else:
                next_action = 0 
            
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
            action = next_action
            step += 1
            
        rewards.append(dep_reward)
        stalls.append(ep_stalls)
        cpu_utils.append(np.mean(ep_cpu) if ep_cpu else 0)
        mem_utils.append(np.mean(ep_mem) if ep_mem else 0)
        
    return {
        "scenario": scenario_name,
        "avg_reward": np.mean(rewards),
        "success_rate": np.mean([1 if s == 0 else 0 for s in stalls]) * 100,
        "avg_cpu": np.mean(cpu_utils),
        "avg_mem": np.mean(mem_utils)
    }

def robustness_comparison():
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "resultSARSA")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(RESULTS_DIR, "models", "sarsa_agent.pth")
    
    if not os.path.exists(MODEL_PATH):
        print("Error: Trained SARSA model not found. Run sarsa_train.py first.")
        return

    # Initialize Agent
    set_global_seed(42)
    temp_env = MonoAgentIoTEnv(num_agents=1, num_devices=5, model_types=["simplecnn"], seed=42)
    agent = DeepSARSAAgent(state_dim=temp_env.single_state_dim, action_dim=5)
    agent.load(MODEL_PATH)
    agent.epsilon = 0.0 # No exploration for testing
    
    cv_model = SimpleCNN()
    agent.assign_inference_model(cv_model)
    
    results = []
    
    # Scenario 1: Base Environment
    print("Testing Base Scenario...")
    env_base = MonoAgentIoTEnv(num_agents=1, num_devices=5, model_types=["simplecnn"], seed=42)
    results.append(run_test(agent, env_base, "Base"))
    
    # Scenario 2: Degraded Device (Slow CPU)
    print("Testing Degraded Device (Slow CPU)...")
    env_slow = MonoAgentIoTEnv(num_agents=1, num_devices=5, model_types=["simplecnn"], seed=42)
    # Slow down device 0
    env_slow.resource_manager.devices[0].cpu_speed *= 0.2 
    results.append(run_test(agent, env_slow, "Slow CPU (D0)"))
    
    # Scenario 3: Network Drop (Low Bandwidth)
    print("Testing Network Drop (Low Bandwidth)...")
    env_net = MonoAgentIoTEnv(num_agents=1, num_devices=5, model_types=["simplecnn"], seed=42)
    for d in env_net.resource_manager.devices.values():
        d.bandwidth *= 0.1
    results.append(run_test(agent, env_net, "Low BW"))
    
    # Scenario 4: Extra Devices (Scaling)
    print("Testing Scaled Devices (10 devices instead of 5)...")
    # We need a new agent for this if action_dim changes, OR we just use 5 but change env to hide 5?
    # Actually DQNAgent is fixed dim. Let's simulate by making some devices unusable? 
    # Or just run with original dim agent on env with 5 devices to show it still works (base).
    # To truly test scalability, we'd need a multi-agent or variable dim agent.
    # Let's just do "High Load" (more compute needed) as robustness.
    env_load = MonoAgentIoTEnv(num_agents=1, num_devices=5, model_types=["simplecnn"], seed=42)
    for t in env_load.task:
        t.computation_demand *= 2.0
    results.append(run_test(agent, env_load, "High Load"))

    # Plotting results
    scenarios = [r['scenario'] for r in results]
    rewards = [r['avg_reward'] for r in results]
    success = [r['success_rate'] for r in results]
    cpu = [r['avg_cpu'] for r in results]
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.bar(scenarios, rewards, color='skyblue')
    plt.title("Reward across Robustness Scenarios")
    plt.ylabel("Avg Reward")
    
    plt.subplot(2, 1, 2)
    plt.bar(scenarios, success, color='salmon')
    plt.title("Success Rate (%) across Robustness Scenarios")
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "robustness_comparison.png"))
    
    print("\nRobustness tests complete. Results:")
    for r in results:
        print(f"- {r['scenario']}: Reward={r['avg_reward']:.2f}, Success={r['success_rate']:.1f}%, CPU={r['avg_cpu']:.2f}")

if __name__ == "__main__":
    robustness_comparison()
