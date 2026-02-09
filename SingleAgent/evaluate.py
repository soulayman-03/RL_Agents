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
    sample_trace = None
    
    for seed in SEEDS:
        set_global_seed(seed)
        env = MonoAgentIoTEnv(num_agents=1, num_devices=NUM_DEVICES, model_types=[MODEL_TYPE], seed=seed)
        max_progress = len(env.agents_tasks[0])
        test_rewards = []
        test_stalls = []
        test_t_comp = []
        test_t_comm = []
        
        print(f"Running {NUM_EPISODES} test episodes (seed={seed})...")
        for e in range(NUM_EPISODES):
            states, _ = env.reset()
            done = False
            ep_reward = 0
            ep_stalls = 0
            ep_t_comp = 0
            ep_t_comm = 0
            trace = []
            
            step = 0
            while not done and step < 100:
                valid_actions = env.get_valid_actions(0)
                action = agent.act(states[0], valid_actions)
                prev_progress = env.agents_progress[0]
                
                next_states, rewards, now_dones, truncated, infos = env.step([action])
                
                if rewards[0] == -500.0:
                    ep_stalls += 1
                else:
                    ep_t_comp += infos[0].get('t_comp', 0)
                    ep_t_comm += infos[0].get('t_comm', 0)
                
                if env.agents_progress[0] > prev_progress:
                     trace.append((prev_progress, int(action)))
                
                ep_reward += rewards[0]
                states = next_states
                done = now_dones[0]
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
                    status = f"STUCK at Layer {env.agents_progress[0]}"
                print(f" Episode {e:2} | Reward: {ep_reward:8.2f} | T_comp: {ep_t_comp:6.2f} | T_comm: {ep_t_comm:6.2f} | Stalls: {ep_stalls:3} | {status}")
                
            if sample_trace is None or (done and len(trace) > len(sample_trace)):
                sample_trace = trace

        all_seed_rewards.extend(test_rewards)
        all_seed_stalls.extend(test_stalls)
        all_seed_t_comp.extend(test_t_comp)
        all_seed_t_comm.extend(test_t_comm)

    # 3. Plot Distribution
    plt.figure(figsize=(10, 5))
    plt.boxplot(all_seed_rewards)
    plt.title(f"Test Reward Distribution ({NUM_EPISODES * len(SEEDS)} episodes)")
    plt.ylabel("Reward")
    plt.savefig(f"{RESULTS_DIR}/reward_distribution.png")
    plt.close()
    
    # 4. Plot Execution Flow (Sample)
    if sample_trace:
        plt.figure(figsize=(10, 6))
        layers = [t[0] for t in sample_trace]
        devices = [t[1] for t in sample_trace]
        
        # Add the final state if completed
        if len(sample_trace) == max_progress:
             # Just for visualization, add a point for the "end"
             layers.append(max_progress)
             devices.append(devices[-1])
             
        plt.step(layers, devices, where='post', marker='o', color='green', label='Allocation')
        plt.yticks(range(NUM_DEVICES), [f"Device {d}" for d in range(NUM_DEVICES)])
        plt.xticks(range(max_progress + 1))
        plt.xlim(-0.5, max_progress + 0.5)
        plt.xlabel("Layer Index")
        plt.ylabel("Device ID")
        plt.title(f"Execution Strategy Trace (Length: {len(sample_trace)}/{max_progress})")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{RESULTS_DIR}/sample_execution_flow.png")
        plt.close()

    avg_reward = float(np.mean(all_seed_rewards)) if all_seed_rewards else 0.0
    std_reward = float(np.std(all_seed_rewards)) if all_seed_rewards else 0.0
    avg_stalls = float(np.mean(all_seed_stalls)) if all_seed_stalls else 0.0
    print(f"Summary: avg_reward={avg_reward:.2f} | std_reward={std_reward:.2f} | avg_stalls={avg_stalls:.2f}")
    print(f"Evaluation Complete. Results in {RESULTS_DIR}/")

if __name__ == "__main__":
    evaluate_single_agent()
