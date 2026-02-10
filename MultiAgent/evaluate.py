import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import torch

# Define paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Add parent directory to path to resolve MultiAgent.xxx imports
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from MultiAgent.environment import MultiAgentIoTEnv
from MultiAgent.manager import MultiAgentManager
from SingleAgent.utils import set_global_seed

def plot_execution_flow(all_traces, num_agents, num_devices, results_dir, filename):
    """Plots the execution flow for all agents in a single episode."""
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    for aid in range(num_agents):
        trace = all_traces.get(aid, [])
        if not trace:
            continue
            
        layers = [t['layer'] for t in trace]
        devices = [t['device'] for t in trace]
        
        # Add final point for visualization
        max_layers = max(layers) if layers else 0
        layers.append(max_layers + 0.5)
        devices.append(devices[-1])
        
        plt.step(layers, devices, where='post', marker='o', color=colors[aid % len(colors)], 
                 label=f'Agent {aid} ({trace[0]["model"]})', alpha=0.8)
        
        # Annotate total latency for the agent
        total_lat = sum([t['lat'] for t in trace])
        plt.text(layers[-1], devices[-1], f"Agent {aid}: {total_lat:.2f}s", 
                 color=colors[aid % len(colors)], fontweight='bold')

    plt.yticks(range(num_devices), [f"Device {d}" for d in range(num_devices)])
    plt.xlabel("Layer Index")
    plt.ylabel("Device ID")
    plt.title("Multi-Agent Execution Flow (Resource Allocation Strategy)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()

def evaluate():
    NUM_AGENTS = 3
    NUM_DEVICES = 5
    MODEL_TYPES = ["simplecnn", "deepcnn", "miniresnet"]
    MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "marl")
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
    NUM_EVAL_EPISODES = 10
    
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    print(f"=== Starting Multi-Agent Evaluation (Agents: {NUM_AGENTS}) ===")
    
    # Initialize Environment with same seed as we will set in training
    EVAL_SEED = 42
    env = MultiAgentIoTEnv(num_agents=NUM_AGENTS, num_devices=NUM_DEVICES, model_types=MODEL_TYPES, seed=EVAL_SEED)
    
    # Initialize and Load Agents
    manager = MultiAgentManager(agent_ids=list(range(NUM_AGENTS)), 
                                state_dim=env.single_state_dim, 
                                action_dim=env.num_devices)
    
    expected_models = [os.path.exists(f"{MODEL_PATH}_agent_{i}.pt") for i in range(NUM_AGENTS)]
    if all(expected_models):
        manager.load_agents(MODEL_PATH)
        print(f"Successfully loaded all agents from {MODEL_PATH}")
    else:
        print(f"WARNING: Some models missing at {MODEL_PATH}. Proceeding anyway.")
    
    # Set epsilon to 0 for all agents
    for agent in manager.agents.values():
        agent.epsilon = 0.0

    eval_rewards = {i: [] for i in range(NUM_AGENTS)}
    eval_successes = {i: [] for i in range(NUM_AGENTS)}
    
    # Seeds to test for generalization
    EVAL_SEEDS = [42, 55, 66, 77, 88]
    
    # To store for execution flow plot (from last seed, last episode)
    last_episode_traces = {i: [] for i in range(NUM_AGENTS)}

    for seed_idx, current_seed in enumerate(EVAL_SEEDS):
        print(f"\n--- Testing Generalization - Seed: {current_seed} ---")
        env.resource_manager.reset_devices_with_seed(NUM_DEVICES, current_seed)
        
        for ep in range(NUM_EVAL_EPISODES):
            obs, _ = env.reset()
            done = {i: False for i in range(NUM_AGENTS)}
            # To track agent mappings for this episode
            agent_ep_mappings = {i: [] for i in range(NUM_AGENTS)}
            ep_rewards = {i: 0 for i in range(NUM_AGENTS)}
            ep_success = {i: False for i in range(NUM_AGENTS)}
            
            while not all(done.values()):
                valid_actions = env.get_valid_actions()
                actions = manager.get_actions(obs, valid_actions)
                
                # Record choices
                for aid, device_id in actions.items():
                    if not done[aid]:
                        agent_ep_mappings[aid].append(device_id)

                # Record trace for plotting (only for the very last episode of the last seed)
                if seed_idx == len(EVAL_SEEDS) - 1 and ep == NUM_EVAL_EPISODES - 1:
                    for aid, dev_id in actions.items():
                        if not done[aid]:
                            last_episode_traces[aid].append({
                                'layer': env.agent_progress[aid],
                                'device': dev_id,
                                'model': env.model_types[aid],
                                'lat': 0 # Updated after step
                            })

                next_obs, rewards, next_done, truncated, infos = env.step(actions)
                
                # Update latencies in trace
                if seed_idx == len(EVAL_SEEDS) - 1 and ep == NUM_EVAL_EPISODES - 1:
                    for aid in range(NUM_AGENTS):
                        if aid in rewards and rewards[aid] > -500:
                            last_episode_traces[aid][-1]['lat'] = -rewards[aid]

                obs = next_obs
                done = next_done
                
                for i in range(NUM_AGENTS):
                    if i in rewards:
                        ep_rewards[i] += rewards[i]
                    if i in infos and infos[i].get("success") and env.agent_progress[i] == len(env.tasks[i]):
                         ep_success[i] = True

            for i in range(NUM_AGENTS):
                eval_rewards[i].append(ep_rewards[i])
                eval_successes[i].append(1 if ep_success[i] else 0)
            
            # Print only for the first few episodes per seed to avoid clutter
            if ep < 2:
                avg_ep_reward = sum(ep_rewards.values())/NUM_AGENTS
                print(f"Seed {current_seed} | Ep {ep+1} | Avg Reward: {avg_ep_reward:.2f}")

    # --- RESULTS & PLOTTING ---
    
    # 1. Execution Flow Plot
    plot_execution_flow(last_episode_traces, NUM_AGENTS, NUM_DEVICES, RESULTS_DIR, "execution_flow.png")
    print(f"\n - Execution flow plot (last seed) saved to {RESULTS_DIR}/execution_flow.png")

    # 2. Performance Summary Plot
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Average Reward per Agent
    plt.subplot(1, 2, 1)
    agent_labels = [f"Agent {i}\n({MODEL_TYPES[i]})" for i in range(NUM_AGENTS)]
    avg_rewards = [np.mean(eval_rewards[i]) for i in range(NUM_AGENTS)]
    plt.bar(agent_labels, avg_rewards, color=['skyblue', 'lightgreen', 'salmon'])
    plt.title("Average Reward per Agent")
    plt.ylabel("Reward")

    # Subplot 2: Success Rate per Agent
    plt.subplot(1, 2, 2)
    success_rates = [np.mean(eval_successes[i]) * 100 for i in range(NUM_AGENTS)]
    plt.bar(agent_labels, success_rates, color=['skyblue', 'lightgreen', 'salmon'])
    plt.title("Success Rate per Agent")
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, 110)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "evaluation_summary.png"))
    plt.close()
    print(f" - Evaluation summary plot saved to {RESULTS_DIR}/evaluation_summary.png")

    # 3. Training Trends
    train_curve_old = "MultiAgent/training_curve.png"
    train_curve_new = os.path.join(RESULTS_DIR, "training_curve.png")
    
    if os.path.exists(train_curve_old):
        print(f" - Moving {train_curve_old} to {train_curve_new}")
        os.rename(train_curve_old, train_curve_new)
    
    if os.path.exists(train_curve_new):
        print(f" - Training trends curve is available at {train_curve_new}")
    
    print("\n=== Evaluation Results ===")
    for i in range(NUM_AGENTS):
        print(f"Agent {i} ({MODEL_TYPES[i]}): Avg Reward = {np.mean(eval_rewards[i]):.2f}, Success Rate = {np.mean(eval_successes[i])*100:.1f}%")

if __name__ == "__main__":
    evaluate()
