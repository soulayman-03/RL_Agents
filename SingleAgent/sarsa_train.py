import sys
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import json

# Define paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Add parent directory to path
sys.path.append(PROJECT_ROOT)

try:
    from .agent import DeepSARSAAgent
    from .environment import MonoAgentIoTEnv
    from .utils import load_and_remap_weights, set_global_seed
except ImportError:
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
    comps = [t['t_comp'] for t in trace]
    comms = [t['t_comm'] for t in trace]
    
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
    def _fmt_seconds(sec: float) -> str:
        sec = max(0.0, float(sec))
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    NUM_AGENTS = 1
    NUM_DEVICES = 5
    MODEL_TYPES = ["simplecnn"]
    EPISODES = 5000
    SEED = 42
    LOG_EVERY = 50
    
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
    
    RESULTS_ROOT = os.path.join(SCRIPT_DIR, "results", "resultSARSA")
    TRAIN_DIR = os.path.join(RESULTS_ROOT, "train")
    os.makedirs(TRAIN_DIR, exist_ok=True)

    # Keep existing model path to avoid breaking other scripts
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    # Save all .npy files under SingleAgent/models
    GLOBAL_MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
    os.makedirs(GLOBAL_MODELS_DIR, exist_ok=True)

    log_path = os.path.join(TRAIN_DIR, "train_log.jsonl")
    try:
        log_f = open(log_path, "w", encoding="utf-8")
    except Exception:
        log_f = None
    
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
                current_trace.append({
                    'layer': layer_idx,
                    'device': action,
                    'l_comp': float(layer.computation_demand),
                    'l_mem': float(layer.memory_demand),
                    'l_out': float(layer.output_data_size),
                    'd_cpu': float(device.cpu_speed),
                    'd_mem': float(device.memory_capacity),
                    'd_bw': float(device.bandwidth),
                    'priv': int(layer.privacy_level),
                    't_comp': info.get('t_comp', 0),
                    't_comm': info.get('t_comm', 0)
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
        
        if log_f is not None:
            try:
                log_f.write(
                    json.dumps(
                        {
                            "episode": int(e),
                            "reward": float(episode_reward),
                            "stalls": int(stalls),
                            "epsilon": float(agent.epsilon),
                            "cpu_util_mean": float(history_cpu_util[-1]),
                            "mem_util_mean": float(history_mem_util[-1]),
                            "net_traffic_sum": float(history_network_traffic[-1]),
                            "devices": [int(t["device"]) for t in current_trace],
                            "trace": current_trace,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                log_f.flush()
            except Exception:
                pass

        if e % LOG_EVERY == 0 or e == EPISODES - 1:
            start = max(0, e - LOG_EVERY + 1)
            last_rewards = history_rewards[start : e + 1]
            last_stalls = history_stalls[start : e + 1]
            avg_r = float(np.mean(last_rewards)) if last_rewards else 0.0
            avg_s = float(np.mean(last_stalls)) if last_stalls else 0.0

            respected = "YES" if stalls == 0 else f"NO ({stalls} violations)"
            devices = [int(t["device"]) for t in current_trace]

            print(
                f"[{start}-{e}] | AvgReward: {avg_r:8.2f} | AvgStalls: {avg_s:5.2f} | "
                f"LastReward: {episode_reward:8.2f} | LastStalls: {stalls:2} | Respected: {respected} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )
            if devices:
                print(f"  Devices: {devices}")
                print("  Trace: layer -> device | t_comp t_comm")
                for t in current_trace:
                    print(f"   L{int(t['layer']):02d} -> D{int(t['device'])} | {float(t.get('t_comp',0)):6.3f} {float(t.get('t_comm',0)):6.3f}")

    total_time = time.time() - start_time
    # Save results to sarsaResult
    agent.save(os.path.join(GLOBAL_MODELS_DIR, "sarsa_agent.pth"))
    np.save(os.path.join(GLOBAL_MODELS_DIR, "sarsa_train_history.npy"), np.array(history_rewards))
    np.save(os.path.join(GLOBAL_MODELS_DIR, "sarsa_stall_history.npy"), np.array(history_stalls))
    np.save(os.path.join(GLOBAL_MODELS_DIR, "sarsa_cpu.npy"), np.array(history_cpu_util))
    np.save(os.path.join(GLOBAL_MODELS_DIR, "sarsa_mem.npy"), np.array(history_mem_util))
    np.save(os.path.join(GLOBAL_MODELS_DIR, "sarsa_net.npy"), np.array(history_network_traffic))
    
    # Generate Training Plots
    import matplotlib.pyplot as plt
    plot_training_reward_history(history_rewards, TRAIN_DIR)
    plot_execution_strategy(final_successful_trace, TRAIN_DIR)
    
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
    plt.savefig(os.path.join(TRAIN_DIR, "training_metrics.png"))

    if log_f is not None:
        try:
            log_f.close()
        except Exception:
            pass
    
    print(f"\nTraining Complete. Total training time: {_fmt_seconds(total_time)}")
    print(f"All files saved to {TRAIN_DIR}")

if __name__ == "__main__":
    train_sarsa_agent()
