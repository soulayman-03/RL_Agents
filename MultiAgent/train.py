import numpy as np
import torch
from MultiAgent.environment import MultiAgentIoTEnv
from MultiAgent.manager import MultiAgentManager
import matplotlib.pyplot as plt
import os

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
    EPISODES = 5000
    MODEL_TYPES = ["simplecnn", "deepcnn", "miniresnet"]
    SAVE_PATH = "MultiAgent/models/marl"
    
    if not os.path.exists("MultiAgent/models"):
        os.makedirs("MultiAgent/models")

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
    
    results_path = "MultiAgent/results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    print(f"Starting Multi-Agent Training with {NUM_AGENTS} agents...")

    for ep in range(EPISODES):
        # Structured seed rotation every 100 episodes for generalization
        if ep % 100 == 0:
            current_seed = TRAIN_SEED + (ep // 100)
            env.resource_manager.reset_devices_with_seed(NUM_DEVICES, current_seed)
            if ep > 0:
                print(f"\n--- Rotated IoT Network Seed to {current_seed} for Generalized Learning ---")
        
        obs, _ = env.reset()
        total_ep_reward = 0
        agent_ep_rewards = {i: 0 for i in range(NUM_AGENTS)}
        agent_ep_success = {i: False for i in range(NUM_AGENTS)}
        agent_ep_mappings = {i: [] for i in range(NUM_AGENTS)}
        done = {i: False for i in range(NUM_AGENTS)}
        
        while not all(done.values()):
            # 1. MultiAgentManager collects actions
            valid_actions = env.get_valid_actions()
            actions = manager.get_actions(obs, valid_actions)
            
            # Record choices
            for aid, device_id in actions.items():
                if not done[aid]:
                    agent_ep_mappings[aid].append(device_id)

            # 2. Environment steps
            next_obs, rewards, next_done, truncated, infos = env.step(actions)
            
            # 3. Store experience
            manager.remember(obs, actions, rewards, next_obs, next_done)
            
            obs = next_obs
            done = next_done
            
            for i in range(NUM_AGENTS):
                if i in rewards:
                    agent_ep_rewards[i] += rewards[i]
                if i in infos and infos[i].get("success") and env.agent_progress[i] == len(env.tasks[i]):
                     agent_ep_success[i] = True

            total_ep_reward += sum(rewards.values())

        # 4. Training (replay)
        manager.train()
        
        episode_rewards.append(total_ep_reward)
        for i in range(NUM_AGENTS):
            agent_reward_history[i].append(agent_ep_rewards[i])
            agent_success_history[i].append(1 if agent_ep_success[i] else 0)
        
        # Periodic Detailed Logging
        if (ep + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            agent_stats = []
            for i in range(NUM_AGENTS):
                a_reward = np.mean(agent_reward_history[i][-50:])
                a_success = np.mean(agent_success_history[i][-50:]) * 100
                agent_stats.append(f"A{i}: {a_reward:.1f} ({a_success:.0f}%)")
            
            stats_str = " | ".join(agent_stats)
            print(f"Ep {ep+1:4d}/{EPISODES} - Avg: {avg_reward:.1f} - [{stats_str}] - Eps: {manager.agents[0].epsilon:.3f}")
            
            # Detailed Mapping for the last episode in this block
            print(f"  --- Detailed Mapping (Ep {ep+1}) ---")
            for i in range(NUM_AGENTS):
                mapping_str = " -> ".join([str(d) for d in agent_ep_mappings[i]])
                success_tag = "[SUCCESS]" if agent_ep_success[i] else "[FAILED]"
                task_layers = env.tasks[i]
                print(f"  Agent {i} ({env.model_types[i]}): {mapping_str} {success_tag}")
                # Print layer sizes for each agent
                layer_details = [f"L{l.layer_id}({l.computation_demand}c,{l.memory_demand}m)" for l in task_layers]
                print(f"    Layer sizes: {' | '.join(layer_details)}")
            print("-" * 30)

    # Save models
    manager.save_agents(SAVE_PATH)
    print("Training finished. Models saved.")

    # Save Histories
    np.save(f"{results_path}/agent_reward_history.npy", agent_reward_history)
    np.save(f"{results_path}/agent_success_history.npy", agent_success_history)

    # Enhanced Plotting: Training Summary
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    window = 50
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green
    
    # 1. Individual Reward Convergence (Moving Average)
    for i in range(NUM_AGENTS):
        rewards = agent_reward_history[i]
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(rewards)), smoothed, label=f'Agent {i} ({MODEL_TYPES[i]})', color=colors[i % len(colors)], linewidth=2)
    ax1.set_title("Reward Convergence (Moving Average)", fontsize=14)
    ax1.set_ylabel("Reward", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)

    # 2. Success Rate Convergence (Moving Average)
    for i in range(NUM_AGENTS):
        successes = agent_success_history[i]
        if len(successes) >= window:
            smoothed = np.convolve(successes, np.ones(window)/window, mode='valid') * 100
            ax2.plot(range(window-1, len(successes)), smoothed, label=f'Agent {i}', color=colors[i % len(colors)], linewidth=2)
    ax2.set_title("Success Rate (%) Convergence", fontsize=14)
    ax2.set_ylabel("Success Rate %", fontsize=12)
    ax2.set_ylim(-5, 105)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)

    # 3. Epsilon Decay
    # Assuming epsilon decay is the same for all agents since they share parameters/logic
    first_agent = manager.agents[0]
    epsilons = [first_agent.epsilon_min + (1.0 - first_agent.epsilon_min) * np.exp(-first_agent.epsilon_decay * step) for step in range(EPISODES)]
    ax3.plot(epsilons, color='purple', linewidth=2, label='Epsilon')
    ax3.set_title("Exploration Rate (Epsilon) Decay", fontsize=14)
    ax3.set_xlabel("Episode", fontsize=12)
    ax3.set_ylabel("Epsilon", fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f"{results_path}/training_trends.png", dpi=150)
    plt.close()
    print(f"Comprehensive training trends saved to {results_path}/training_trends.png")

if __name__ == "__main__":
    train()
