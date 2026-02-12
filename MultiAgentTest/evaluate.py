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

def plot_execution_flow(all_traces, num_agents, num_devices, results_dir, filename):
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    for aid in range(num_agents):
        trace = all_traces.get(aid, [])
        if not trace: continue
            
        layers = [t['layer'] for t in trace]
        devices = [t['device'] for t in trace]
        max_layers = max(layers) if layers else 0
        layers.append(max_layers + 0.5)
        devices.append(devices[-1])
        
        plt.step(layers, devices, where='post', marker='o', color=colors[aid % len(colors)], 
                 label=f'Agent {aid} ({trace[0]["model"]})', alpha=0.8)
        
        total_lat = sum([t['lat'] for t in trace])
        plt.text(layers[-1], devices[-1], f"A{aid}: {total_lat:.2f}s", color=colors[aid % len(colors)], fontweight='bold')

    plt.yticks(range(num_devices), [f"Device {d}" for d in range(num_devices)])
    plt.xlabel("Layer Index")
    plt.ylabel("Device ID")
    plt.title("Heterogeneous Multi-Agent Execution Flow (5, 7, 10 Layers)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()

def evaluate():
    NUM_AGENTS = 3
    NUM_DEVICES = 5
    MODEL_TYPES = ["simplecnn", "cnn7", "cnn10"]
    MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "marl_test")
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
    NUM_EVAL_EPISODES = 5
    
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    print(f"=== Starting Multi-Agent Test Evaluation ===")
    env = MultiAgentIoTEnv(num_agents=NUM_AGENTS, num_devices=NUM_DEVICES, model_types=MODEL_TYPES, seed=42)
    manager = MultiAgentManager(agent_ids=list(range(NUM_AGENTS)), state_dim=env.single_state_dim, action_dim=env.num_devices)
    
    if os.path.exists(f"{MODEL_PATH}_agent_0.pt"):
        manager.load_agents(MODEL_PATH)
        print(f"Loaded models from {MODEL_PATH}")
    else:
        print(f"WARNING: Models not found. Using untrained agents.")

    for agent in manager.agents.values(): agent.epsilon = 0.0

    eval_rewards = {i: [] for i in range(NUM_AGENTS)}
    eval_successes = {i: [] for i in range(NUM_AGENTS)}
    EVAL_SEEDS = [42, 99] # Just two seeds for quick test
    last_episode_traces = {i: [] for i in range(NUM_AGENTS)}

    for seed_idx, current_seed in enumerate(EVAL_SEEDS):
        print(f"\n--- Testing Seed: {current_seed} ---")
        env.resource_manager.reset_devices_with_seed(NUM_DEVICES, current_seed)
        
        for ep in range(NUM_EVAL_EPISODES):
            obs, _ = env.reset()
            done = {i: False for i in range(NUM_AGENTS)}
            agent_ep_mappings = {i: [] for i in range(NUM_AGENTS)}
            ep_rewards = {i: 0 for i in range(NUM_AGENTS)}
            ep_success = {i: False for i in range(NUM_AGENTS)}
            
            while not all(done.values()):
                valid_actions = env.get_valid_actions()
                actions = manager.get_actions(obs, valid_actions)
                
                for aid, device_id in actions.items():
                    if not done[aid]: agent_ep_mappings[aid].append(device_id)

                if seed_idx == len(EVAL_SEEDS) - 1 and ep == NUM_EVAL_EPISODES - 1:
                    for aid, dev_id in actions.items():
                        if not done[aid]:
                            last_episode_traces[aid].append({
                                'layer': env.agent_progress[aid], 'device': dev_id,
                                'model': env.model_types[aid], 'lat': 0
                            })

                next_obs, rewards, next_done, truncated, infos = env.step(actions)
                
                if seed_idx == len(EVAL_SEEDS) - 1 and ep == NUM_EVAL_EPISODES - 1:
                    for aid in range(NUM_AGENTS):
                        if aid in rewards and rewards[aid] > -500:
                            last_episode_traces[aid][-1]['lat'] = -rewards[aid]

                obs, done = next_obs, next_done
                for i in range(NUM_AGENTS):
                    if i in rewards: ep_rewards[i] += rewards[i]
                    if i in infos and infos[i].get("success") and env.agent_progress[i] == len(env.tasks[i]):
                         ep_success[i] = True

            for i in range(NUM_AGENTS):
                eval_rewards[i].append(ep_rewards[i])
                eval_successes[i].append(1 if ep_success[i] else 0)
            
            if ep == 0:
                print(f"  Ep {ep+1} | Avg Reward: {sum(ep_rewards.values())/NUM_AGENTS:.2f}")
                for i in range(NUM_AGENTS):
                    mapping = " -> ".join([str(d) for d in agent_ep_mappings[i]])
                    print(f"    Agent {i}: {mapping} {'[SUCCESS]' if ep_success[i] else '[FAILED]'}")

    plot_execution_flow(last_episode_traces, NUM_AGENTS, NUM_DEVICES, RESULTS_DIR, "test_execution_flow.png")
    print(f"\nEvaluation complete. Results saved in {RESULTS_DIR}")

if __name__ == "__main__":
    evaluate()
