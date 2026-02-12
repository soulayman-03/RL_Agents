import sys
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

# Define paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Add parent directory to path
sys.path.append(PROJECT_ROOT)

from agent import DeepSARSAAgent
from environment import MonoAgentIoTEnv
from utils import load_and_remap_weights, set_global_seed
from split_inference.cnn_model import SimpleCNN

def plot_execution_strategy(trace, results_dir):
    """
    Plots the device assignment strategy for a single successful episode.
    """
    if not trace:
        return
        
    devices = [t['device'] for t in trace]
    comps = [t['comp'] for t in trace]
    comms = [t['comm'] for t in trace]
    
    plt.figure(figsize=(10, 6))
    # Create step data
    x = np.arange(len(devices) + 1)
    y = devices + [devices[-1]]
    
    plt.step(x, y, where='post', color='green', linewidth=1.5)
    plt.scatter(range(len(devices)), devices, color='green', zorder=5)
    
    # Annotate with C and T
    for i, (d, c, t) in enumerate(zip(devices, comps, comms)):
        plt.text(i, d + 0.1, f"C:{c:.1f}\nT:{t:.1f}", ha='center', va='bottom', fontsize=8)

    plt.yticks(range(5), [f"Device {i}" for i in range(5)])
    plt.xticks(range(len(devices) + 1))
    plt.xlabel("Layer Index")
    plt.ylabel("Device ID")
    plt.title(f"Execution Strategy - SARSA (Trace Length: {len(devices)}/{len(devices)})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "execution_strategy.png"))
    plt.close()

def plot_training_reward_history(rewards, results_dir):
    """
    Plots the training rewards with a moving average.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, color='blue', alpha=0.3, label='Raw Reward')
    
    # Moving average
    window = 50
    if len(rewards) >= window:
        mv_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), mv_avg, color='red', linewidth=2, label='Moving Avg')
        
    plt.title("Training Rewards History - SARSA")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, "sarsa_training_history.png"))
    plt.close()

def train_sarsa_agent():
    """
    Trains a Deep SARSA agent.
    """
    NUM_AGENTS = 1
    NUM_DEVICES = 5
    MODEL_TYPES = ["simplecnn"]
    EPISODES = 1000
    SEED = 42
    
    print(f"=== Starting Deep SARSA Agent Training (Model: {MODEL_TYPES[0]}) ===")
    
    # Initialize Environment
    set_global_seed(SEED)
    env = MonoAgentIoTEnv(num_agents=NUM_AGENTS, num_devices=NUM_DEVICES, model_types=MODEL_TYPES, seed=SEED)
    
    # Initialize Agent
    agent = DeepSARSAAgent(state_dim=env.single_state_dim, action_dim=NUM_DEVICES)
    
    # Prepare the CNN model and load pre-trained weights
    cv_model = SimpleCNN()
    weight_path = os.path.join(PROJECT_ROOT, "split_inference", "mnist_simplecnn.pth")
    
    success = load_and_remap_weights(cv_model, weight_path, "simplecnn")
    if success:
        print(f"Successfully loaded weights from {weight_path}")
    
    agent.assign_inference_model(cv_model)
    
    # Define results directory
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "resultSARSA")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    MODELS_DIR = os.path.join(RESULTS_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Training Loop
    start_time = time.time()
    history_rewards = []
    history_stalls = []
    
    # Resource metrics history
    history_cpu_util = []
    history_mem_util = []
    history_network_traffic = []
    
    final_successful_trace = []
    
    for e in range(EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        stalls = 0
        done = False
        
        # Resource metrics for this episode
        ep_cpu = []
        ep_mem = []
        ep_net = []
        current_trace = []
        
        # In SARSA, we need the initial action
        valid_actions = env.get_valid_actions()
        action = agent.act(state, valid_actions)
        
        step = 0
        MAX_STEPS = 100
        
        while not done and step < MAX_STEPS:
            # Take action a, observe r, s'
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Choose next action a' from s'
            next_valid_actions = env.get_valid_actions()
            if not done:
                next_action = agent.act(next_state, next_valid_actions)
            else:
                next_action = 0 
            
            # Metrics Logging
            if "device_stats" in info:
                # Assuming info contains current step stats
                # If not explicitly in info, we can pull from env.resource_manager
                pass
            
            # Pull stats from environment's resource manager
            device_idx = int(action)
            device = env.resource_manager.devices[device_idx]
            
            # Record resources
            # CPU utilization: layer computation / device cpu speed (simplified)
            layer_idx = env.progress - 1
            if layer_idx >= 0:
                layer = env.task[layer_idx]
                cpu_v = (layer.computation_demand / device.cpu_speed) if device.cpu_speed > 0 else 0
                mem_v = (layer.memory_demand / device.memory_capacity) if device.memory_capacity > 0 else 0
                net_v = layer.output_data_size
                
                ep_cpu.append(cpu_v)
                ep_mem.append(mem_v)
                ep_net.append(net_v)
                
                # Capture trace for final successful strategy plot if no stall yet
                if reward != -500.0:
                    current_trace.append({
                        'device': action,
                        'comp': info.get('t_comp', 0),
                        'comm': info.get('t_comm', 0)
                    })

            if reward == -500.0:
                stalls += 1
            
            # Store experience (s, a, r, s', a', done)
            agent.remember(state, action, reward, next_state, next_action, terminated)
            episode_reward += reward
            
            # Move to next state and action
            state = next_state
            action = next_action
            step += 1
            
        # Resources for trace
        if not stalls and done:
            final_successful_trace = current_trace
            
        # Replay at end of episode
        for _ in range(5):
            agent.replay()
            
        history_rewards.append(episode_reward)
        history_stalls.append(stalls)
        
        if ep_cpu:
            history_cpu_util.append(np.mean(ep_cpu))
            history_mem_util.append(np.mean(ep_mem))
            history_network_traffic.append(np.sum(ep_net))
        else:
            history_cpu_util.append(0)
            history_mem_util.append(0)
            history_network_traffic.append(0)
        
        if e % 100 == 0:
            print(f"Episode {e}/{EPISODES} | Reward: {episode_reward:7.2f} | Stalls: {stalls} | CPU: {history_cpu_util[-1]:.2f} | Mem: {history_mem_util[-1]:.2f}")

    end_time = time.time()
    
    # Save results to sarsaResult
    agent.save(os.path.join(MODELS_DIR, "sarsa_agent.pth"))
    np.save(os.path.join(RESULTS_DIR, "sarsa_rewards.npy"), np.array(history_rewards))
    np.save(os.path.join(RESULTS_DIR, "sarsa_stalls.npy"), np.array(history_stalls))
    np.save(os.path.join(RESULTS_DIR, "sarsa_cpu.npy"), np.array(history_cpu_util))
    np.save(os.path.join(RESULTS_DIR, "sarsa_mem.npy"), np.array(history_mem_util))
    np.save(os.path.join(RESULTS_DIR, "sarsa_net.npy"), np.array(history_network_traffic))
    
    # Generate Training Plots
    import matplotlib.pyplot as plt
    plot_training_reward_history(history_rewards, RESULTS_DIR)
    plot_execution_strategy(final_successful_trace, RESULTS_DIR)
    
    # Keep standard overview plot as well
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(history_rewards)
    plt.title("Training Rewards")
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(history_cpu_util)
    plt.title("CPU Utilization")
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(history_mem_util)
    plt.title("Memory Utilization")
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(history_network_traffic)
    plt.title("Network Traffic (Output Size)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_metrics.png"))
    
    print(f"\nTraining Complete. All files saved to {RESULTS_DIR}")

if __name__ == "__main__":
    train_sarsa_agent()
