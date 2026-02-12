import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt

# Define paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Add parent directory to path to resolve MultiAgent.xxx imports
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from MultiAgent.environment import MultiAgentIoTEnv
from MultiAgent.manager import MultiAgentManager

def print_device_info(resource_manager):
    print("\n" + "="*50)
    print("DEVICE SPECIFICATIONS")
    print("="*50)
    for d_id, d in resource_manager.devices.items():
        print(f"Device {d_id}: CPU={d.cpu_speed:.2f}, RAM={d.memory_capacity:.1f}MB, BW={d.bandwidth:.1f}Mbps, Privacy={d.privacy_clearance}")
    print("="*50 + "\n")

def train():
    # Parameters
    NUM_AGENTS = 3
    NUM_DEVICES = 5
    EPISODES = 1000 # Lower for testing
    MODEL_TYPES = ["simplecnn", "cnn7", "cnn10"]
    SAVE_PATH = os.path.join(SCRIPT_DIR, "models", "marl_test")
    
    if not os.path.exists(os.path.join(SCRIPT_DIR, "models")):
        os.makedirs(os.path.join(SCRIPT_DIR, "models"))

    # Initialize Environment
    TRAIN_SEED = 42
    env = MultiAgentIoTEnv(num_agents=NUM_AGENTS, num_devices=NUM_DEVICES, model_types=MODEL_TYPES, seed=TRAIN_SEED)
    
    # Initialize Manager
    state_dim = env.single_state_dim
    action_dim = env.num_devices
    manager = MultiAgentManager(agent_ids=list(range(NUM_AGENTS)), state_dim=state_dim, action_dim=action_dim)

    # Print initial device info
    print_device_info(env.resource_manager)

    # History for plotting
    episode_rewards = []
    agent_reward_history = {i: [] for i in range(NUM_AGENTS)}
    agent_success_history = {i: [] for i in range(NUM_AGENTS)}
    
    results_path = os.path.join(SCRIPT_DIR, "results")
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    print(f"Starting Multi-Agent Test Training with {NUM_AGENTS} agents (Heterogeneous)...")

    for ep in range(EPISODES):
        # Structured seed rotation every 100 episodes
        if ep % 100 == 0:
            current_seed = TRAIN_SEED + (ep // 100)
            env.resource_manager.reset_devices_with_seed(NUM_DEVICES, current_seed)
        
        obs, _ = env.reset()
        total_ep_reward = 0
        agent_ep_rewards = {i: 0 for i in range(NUM_AGENTS)}
        agent_ep_success = {i: False for i in range(NUM_AGENTS)}
        agent_ep_mappings = {i: [] for i in range(NUM_AGENTS)}
        done = {i: False for i in range(NUM_AGENTS)}
        
        while not all(done.values()):
            valid_actions = env.get_valid_actions()
            actions = manager.get_actions(obs, valid_actions)
            
            # Record choices
            for aid, device_id in actions.items():
                if not done[aid]:
                    agent_ep_mappings[aid].append(device_id)

            next_obs, rewards, next_done, truncated, infos = env.step(actions)
            manager.remember(obs, actions, rewards, next_obs, next_done)
            
            obs = next_obs
            done = next_done
            
            for i in range(NUM_AGENTS):
                if i in rewards:
                    agent_ep_rewards[i] += rewards[i]
                if i in infos and infos[i].get("success") and env.agent_progress[i] == len(env.tasks[i]):
                     agent_ep_success[i] = True
            total_ep_reward += sum(rewards.values())

        manager.train()
        
        episode_rewards.append(total_ep_reward)
        for i in range(NUM_AGENTS):
            agent_reward_history[i].append(agent_ep_rewards[i])
            agent_success_history[i].append(1 if agent_ep_success[i] else 0)
        
        if (ep + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            agent_stats = [f"A{i}: {np.mean(agent_success_history[i][-50:])*100:.0f}%" for i in range(NUM_AGENTS)]
            print(f"Ep {ep+1:4d}/{EPISODES} - Avg: {avg_reward:.1f} - [{ ' | '.join(agent_stats) }]")
            
            # Detailed Mapping for the last episode in this block
            print(f"  --- Detailed Mapping (Ep {ep+1}) ---")
            for i in range(NUM_AGENTS):
                mapping_str = " -> ".join([str(d) for d in agent_ep_mappings[i]])
                success_tag = "[SUCCESS]" if agent_ep_success[i] else "[FAILED]"
                task_layers = env.tasks[i]
                print(f"  Agent {i} ({env.model_types[i]}): {mapping_str} {success_tag} | Reward: {agent_ep_rewards[i]:.2f}")
                layer_details = [f"L{l.layer_id}({l.computation_demand}c,{l.memory_demand}m)" for l in task_layers]
                print(f"    Layer sizes: {' | '.join(layer_details)}")
            print("-" * 30)

    # Save models
    manager.save_agents(SAVE_PATH)
    print(f"Test Training finished. Models saved to {SAVE_PATH}")

    # Plot results
    plt.figure(figsize=(10, 6))
    for i in range(NUM_AGENTS):
        successes = agent_success_history[i]
        window = 20
        if len(successes) >= window:
            smoothed = np.convolve(successes, np.ones(window)/window, mode='valid') * 100
            plt.plot(smoothed, label=f'Agent {i} ({MODEL_TYPES[i]})')
    plt.title("Success Rate Convergence (Heterogeneous Test)")
    plt.ylabel("Success Rate (%)")
    plt.xlabel("Episode")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_path}/test_trends.png")
    plt.close()

if __name__ == "__main__":
    train()
