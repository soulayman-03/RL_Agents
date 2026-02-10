import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Define paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Add parent directory to path
sys.path.append(PROJECT_ROOT)

from environment import MonoAgentIoTEnv
from agent import DQNAgent
from split_inference.cnn_model import SimpleCNN
from utils import load_and_remap_weights, set_global_seed
import torch

def plot_execution_flow(trace, max_progress, num_devices, seed, results_dir, filename):
    """Helper to plot the execution strategy trace for a single episode."""
    plt.figure(figsize=(10, 6))
    layers = [t['layer'] for t in trace]
    devices = [t['device'] for t in trace]
    
    # Add final point for visualization
    layers.append(max_progress)
    devices.append(devices[-1])
    
    plt.step(layers, devices, where='post', marker='o', color='green', label='Allocation')
    
    # Add latency annotations
    for t in trace:
        plt.text(t['layer'], t['device'] + 0.1, f"C:{t['t_comp']:.1f}\n T:{t['t_comm']:.1f}", 
                 fontsize=8, verticalalignment='bottom', horizontalalignment='center',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    plt.yticks(range(num_devices), [f"Device {d}" for d in range(num_devices)])
    plt.xticks(range(max_progress + 1))
    plt.xlim(-0.5, max_progress + 0.5)
    plt.xlabel("Layer Index")
    plt.ylabel("Device ID")
    plt.title(f"Execution Strategy - Seed {seed} (Trace Length: {len(trace)}/{max_progress})")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()

def evaluate_single_agent():
    NUM_EPISODES = 50
    NUM_DEVICES = 5
    MODEL_TYPE = "simplecnn"
    MODEL_TYPE = "simplecnn"
    SEEDS = [42, 43, 44, 45, 46]
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
    MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "single_agent_simplecnn.pth")
    HISTORY_PATH = os.path.join(SCRIPT_DIR, "models", "single_train_history.npy")
    STALL_HISTORY_PATH = os.path.join(SCRIPT_DIR, "models", "single_stall_history.npy")
    
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    print(f"=== Starting Single Agent Evaluation ({MODEL_TYPE}) ===")
    
    # Setup Agent (weights reused across seeds)
    dummy_env = MonoAgentIoTEnv(num_agents=1, num_devices=NUM_DEVICES, model_types=[MODEL_TYPE], seed=SEEDS[0])
    agent = DQNAgent(state_dim=dummy_env.single_state_dim, action_dim=NUM_DEVICES)

    if os.path.exists(MODEL_PATH):
        agent.load(MODEL_PATH)
        print(f"Loaded trained agent from {MODEL_PATH}")
    else:
        print(f"WARNING: No trained model found at {MODEL_PATH}. Using random weights.")

    agent.epsilon = 0.0 # No exploration during evaluation
    
    # 1. Plot Training History if available
    if os.path.exists(HISTORY_PATH):
        history = np.load(HISTORY_PATH)
        stalls_history = np.load(STALL_HISTORY_PATH) if os.path.exists(STALL_HISTORY_PATH) else None
        
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Rewards
        plt.subplot(1, 2, 1)
        plt.plot(history, alpha=0.3, color='blue')
        # Moving average
        window = 20
        if len(history) > window:
            smoothed = np.convolve(history, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(history)), smoothed, color='red', linewidth=2, label='Moving Avg')
        plt.title("Training Rewards History")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        
        # Subplot 2: Stalls
        if stalls_history is not None:
            plt.subplot(1, 2, 2)
            plt.plot(stalls_history, color='orange')
            plt.title("Stall Count per Episode")
            plt.xlabel("Episode")
            plt.ylabel("Stall Count")
        
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/training_trends.png")
        plt.close()
        print(f" - Training trends plot saved to {RESULTS_DIR}/training_trends.png")

    # 2. Run Test Episodes across seeds
    all_seed_rewards = []
    all_seed_stalls = []
    all_seed_t_comp = []
    all_seed_t_comm = []
    all_seed_traces = []
    sample_trace = None
    sample_trace_full = None
    
    for seed in SEEDS:
        set_global_seed(seed)
        env = MonoAgentIoTEnv(num_agents=1, num_devices=NUM_DEVICES, model_types=[MODEL_TYPE], seed=seed)
        max_progress = len(env.task)
        test_rewards = []
        test_stalls = []
        test_t_comp = []
        test_t_comm = []
        
        print(f"Running {NUM_EPISODES} test episodes (seed={seed})...")
        for e in range(NUM_EPISODES):
            state, _ = env.reset()
            done = False
            ep_reward = 0
            ep_stalls = 0
            ep_t_comp = 0
            ep_t_comm = 0
            trace = []
            
            step = 0
            while not done and step < 100:
                valid_actions = env.get_valid_actions()
                action = agent.act(state, valid_actions)
                prev_progress = env.progress
                
                next_state, reward, terminated, truncated, info = env.step(action)
                
                if reward == -500.0:
                    ep_stalls += 1
                else:
                    ep_t_comp += info.get('t_comp', 0)
                    ep_t_comm += info.get('t_comm', 0)
                
                if env.progress > prev_progress:
                    layer = env.task[prev_progress]
                    device = env.resource_manager.devices[int(action)]
                    trace_entry = {
                        "layer": prev_progress,
                        "device": int(action),
                        "comp": float(layer.computation_demand),
                        "mem": float(layer.memory_demand),
                        "out": float(layer.output_data_size),
                        "priv": int(layer.privacy_level),
                        "d_cpu": float(device.cpu_speed),
                        "d_mem": float(device.memory_capacity),
                        "d_bw": float(device.bandwidth),
                        "t_comp": info.get('t_comp', 0),
                        "t_comm": info.get('t_comm', 0)
                    }
                    trace.append(trace_entry)
                
                ep_reward += reward
                state = next_state
                done = terminated
                step += 1
                
            test_rewards.append(ep_reward)
            test_stalls.append(ep_stalls)
            test_t_comp.append(ep_t_comp)
            test_t_comm.append(ep_t_comm)
            
            if e % 10 == 0:
                if ep_stalls > 0:
                    status = "FAILED (stall)"
                elif done:
                    status = "COMPLETED"
                else:
                    status = f"STUCK at Layer {env.progress}"
                print(f" Episode {e:2} | Reward: {ep_reward:8.2f} | T_comp: {ep_t_comp:6.2f} | T_comm: {ep_t_comm:6.2f} | Stalls: {ep_stalls:3} | {status}")
                
                # Print full trace for the very first episode across all seeds for visibility
                if e == 0:
                    print("   Full Trace (Lxx -> Dx | L_comp L_mem L_out | D_cpu D_mem D_bw | priv)")
                    for t in trace:
                         print(f"    L{t['layer']:02d} -> D{t['device']} | {t['comp']:6.1f} {t['mem']:6.1f} {t['out']:6.1f} | {t['d_cpu']:5.2f} {t['d_mem']:5.1f} {t['d_bw']:5.1f} | {t['priv']}")
                    
                    # Generate per-seed plot
                    plot_execution_flow(trace, max_progress, NUM_DEVICES, seed, RESULTS_DIR, f"execution_flow_seed_{seed}.png")

            if sample_trace_full is None or (done and len(trace) > len(sample_trace_full)):
                sample_trace_full = trace
                sample_trace = [(t['layer'], t['device']) for t in trace]

        all_seed_rewards.extend(test_rewards)
        all_seed_stalls.extend(test_stalls)
        all_seed_t_comp.extend(test_t_comp)
        all_seed_t_comm.extend(test_t_comm)
        all_seed_traces.append(trace) # Store one sample trace per seed

    # 3. Plot Distribution (Rewards)
    plt.figure(figsize=(10, 5))
    plt.boxplot(all_seed_rewards)
    plt.title(f"Test Reward Distribution ({NUM_EPISODES * len(SEEDS)} episodes)")
    plt.ylabel("Reward")
    plt.savefig(f"{RESULTS_DIR}/reward_distribution.png")
    plt.close()
    
    # 3.b Plot Latency Composition (Sample of episodes)
    plt.figure(figsize=(10, 5))
    ep_indices = range(min(20, len(all_seed_t_comp)))
    plt.bar(ep_indices, all_seed_t_comp[:20], label='Compute Latency', color='skyblue')
    plt.bar(ep_indices, all_seed_t_comm[:20], bottom=all_seed_t_comp[:20], label='Comm Latency', color='salmon')
    plt.title("Latency Composition (First 20 Evaluation Episodes)")
    plt.xlabel("Episode Index")
    plt.ylabel("Latency (Time Units)")
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/latency_composition.png")
    plt.close()

    # 3.c Plot Device Usage Balance
    all_devices_used = []
    if sample_trace_full: # Use the full traces collected
        for ep_trace in all_seed_traces:
            for step in ep_trace:
                all_devices_used.append(step['device'])
    
    if all_devices_used:
        plt.figure(figsize=(10, 5))
        counts = [all_devices_used.count(d) for d in range(NUM_DEVICES)]
        plt.bar(range(NUM_DEVICES), counts, color='lightgreen')
        plt.title("Device Load Distribution (Total Layer Assignments)")
        plt.xlabel("Device ID")
        plt.ylabel("Number of Layers Assigned")
        plt.xticks(range(NUM_DEVICES))
        plt.savefig(f"{RESULTS_DIR}/device_usage_stats.png")
        plt.close()

    # 4. Plot Execution Flow (Summary Sample)
    if sample_trace_full:
        plot_execution_flow(sample_trace_full, max_progress, NUM_DEVICES, "Sample", RESULTS_DIR, "sample_execution_flow.png")

    avg_reward = float(np.mean(all_seed_rewards)) if all_seed_rewards else 0.0
    std_reward = float(np.std(all_seed_rewards)) if all_seed_rewards else 0.0
    avg_stalls = float(np.mean(all_seed_stalls)) if all_seed_stalls else 0.0
    print(f"\nSummary: avg_reward={avg_reward:.2f} | std_reward={std_reward:.2f} | avg_stalls={avg_stalls:.2f}")
    print(f"Evaluation Complete. Results in {RESULTS_DIR}/")

if __name__ == "__main__":
    evaluate_single_agent()
