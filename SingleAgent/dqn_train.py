import sys
import os
import time
import json

# Define paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Add parent directory to path to allow imports from rl_pdnn and split_inference
sys.path.append(PROJECT_ROOT)

import torch
import numpy as np
try:
    from .agent import DQNAgent
    from .environment import MonoAgentIoTEnv
    from .utils import load_and_remap_weights, set_global_seed
except ImportError:
    from agent import DQNAgent
    from environment import MonoAgentIoTEnv
    from utils import load_and_remap_weights, set_global_seed
from split_inference.cnn_model import SimpleCNN


def plot_training_scalar(values, title, ylabel, out_path, window=50):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(values, color="blue", alpha=0.25, label="Raw")
    if len(values) >= window:
        mv_avg = np.convolve(values, np.ones(window) / window, mode="valid")
        plt.plot(range(window - 1, len(values)), mv_avg, color="red", linewidth=2, label=f"MA{window}")
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def train_single_agent():
    """
    Trains a single agent to schedule its DNN layers across 5 devices.
    Uses the MultiAgentIoTEnv configured with num_agents=1.
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
    
    print(f"=== Starting Single Agent Training (Model: {MODEL_TYPES[0]}) ===")
    
    # Initialize Environment
    set_global_seed(SEED)
    env = MonoAgentIoTEnv(num_agents=NUM_AGENTS, num_devices=NUM_DEVICES, model_types=MODEL_TYPES, seed=SEED)
    
    # Initialize Agent
    agent = DQNAgent(state_dim=env.single_state_dim, action_dim=NUM_DEVICES)
    
    # Prepare the CNN model and load pre-trained weights
    cv_model = SimpleCNN()
    weight_path = os.path.join(PROJECT_ROOT, "split_inference", "mnist_simplecnn.pth")
    
    success = load_and_remap_weights(cv_model, weight_path, "simplecnn")
    if success:
        print(f"Successfully loaded and remapped weights from {weight_path}")
    else:
        print(f"Warning: Could not load weights from {weight_path}. Using random initialization.")
    
    agent.assign_inference_model(cv_model)
    
    # Training Loop
    start_time = time.time()
    history_rewards = []
    history_stalls = []
    history_epsilons = []

    RESULTS_ROOT = os.path.join(SCRIPT_DIR, "results", "resultDQN")
    TRAIN_DIR = os.path.join(RESULTS_ROOT, "train")
    os.makedirs(TRAIN_DIR, exist_ok=True)

    log_path = os.path.join(TRAIN_DIR, "train_log.jsonl")
    try:
        log_f = open(log_path, "w", encoding="utf-8")
    except Exception:
        log_f = None
    
    for e in range(EPISODES):
        # Gymnasium reset returns (obs, info)
        state, _ = env.reset()
        episode_reward = 0
        stalls = 0
        done = False
        episode_trace = []
        
        step = 0
        MAX_STEPS = 100
        
        while not done and step < MAX_STEPS:
            valid_actions = env.get_valid_actions()
            action = agent.act(state, valid_actions)
            
            # Gymnasium step returns (obs, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, info = env.step(action)

            # Log layer/device choice with layer characteristics
            layer_idx = env.progress - 1
            if layer_idx >= 0:
                layer = env.task[layer_idx]
                device = env.resource_manager.devices[int(action)]
                episode_trace.append({
                    "layer": layer_idx,
                    "device": int(action),
                    "comp": float(layer.computation_demand),
                    "mem": float(layer.memory_demand),
                    "out": float(layer.output_data_size),
                    "priv": int(layer.privacy_level),
                    "d_cpu": float(device.cpu_speed),
                    "d_mem": float(device.memory_capacity),
                    "d_bw": float(device.bandwidth),
                    "t_comp": float(info.get("t_comp", 0.0)),
                    "t_comm": float(info.get("t_comm", 0.0)),
                })
            
            # Record stalls (constraint violations)
            if reward == -500.0:
                stalls += 1
            
            # Store experience
            agent.remember(state, action, reward, next_state, terminated)
            episode_reward += reward
            
            state = next_state
            done = terminated
            step += 1
            
        # Replay at end of episode for faster training
        for _ in range(5):
            agent.replay()
            
        history_rewards.append(episode_reward)
        history_stalls.append(stalls)
        history_epsilons.append(float(agent.epsilon))

        if log_f is not None:
            try:
                log_f.write(
                    json.dumps(
                        {
                            "episode": int(e),
                            "reward": float(episode_reward),
                            "stalls": int(stalls),
                            "epsilon": float(agent.epsilon),
                            "devices": [int(t["device"]) for t in episode_trace],
                            "trace": episode_trace,
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
            devices = [int(t["device"]) for t in episode_trace]

            print(
                f"[{start}-{e}] | AvgReward: {avg_r:8.2f} | AvgStalls: {avg_s:5.2f} | "
                f"LastReward: {episode_reward:8.2f} | LastStalls: {stalls:2} | Respected: {respected} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )
            if devices:
                print(f"  Devices: {devices}")
                print("  Trace: layer -> device | t_comp t_comm")
                for t in episode_trace:
                    print(f"   L{int(t['layer']):02d} -> D{int(t['device'])} | {float(t.get('t_comp',0)):6.3f} {float(t.get('t_comm',0)):6.3f}")

    total_time = time.time() - start_time

    # Save trained RL agent
    MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)
    
    agent.save(os.path.join(MODELS_DIR, "single_agent_simplecnn.pth"))
    np.save(os.path.join(MODELS_DIR, "single_train_history.npy"), np.array(history_rewards))
    np.save(os.path.join(MODELS_DIR, "single_stall_history.npy"), np.array(history_stalls))

    plot_training_scalar(
        history_rewards,
        "Training Rewards History - DQN",
        "Reward",
        os.path.join(TRAIN_DIR, "dqn_training_history.png"),
    )
    plot_training_scalar(
        history_stalls,
        "Training Stalls History - DQN",
        "Stalls",
        os.path.join(TRAIN_DIR, "dqn_stalls_history.png"),
    )
    plot_training_scalar(
        history_epsilons,
        "Epsilon Schedule - DQN",
        "Epsilon",
        os.path.join(TRAIN_DIR, "dqn_epsilon.png"),
        window=200,
    )

    if log_f is not None:
        try:
            log_f.close()
        except Exception:
            pass
    
    print("\nTraining Complete.")
    print(f"Total training time: {_fmt_seconds(total_time)}")
    print(f"Model saved to {os.path.join(MODELS_DIR, 'single_agent_simplecnn.pth')}")
    print(f"Training results in {TRAIN_DIR}/")

if __name__ == "__main__":
    train_single_agent()
