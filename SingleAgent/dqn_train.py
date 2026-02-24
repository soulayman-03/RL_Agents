import sys
import os
import time
import json
import argparse

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


def plot_execution_strategy(trace, results_dir, num_devices=5, out_name="execution_strategy.png"):
    """
    Visualizes the selected device per layer as a step plot, with compute/comm latencies annotated.
    Expects trace items like: {"layer": int, "device": int, "t_comp": float, "t_comm": float}
    """
    if not trace:
        return

    import matplotlib.pyplot as plt

    trace_sorted = sorted(trace, key=lambda t: int(t.get("layer", 0)))
    devices = [int(t.get("device", 0)) for t in trace_sorted]
    comps = [float(t.get("t_comp", 0.0)) for t in trace_sorted]
    comms = [float(t.get("t_comm", 0.0)) for t in trace_sorted]

    if not devices:
        return

    plt.figure(figsize=(10, 6))
    x = np.arange(len(devices) + 1)
    y = devices + [devices[-1]]
    plt.step(x, y, where="post", color="green", linewidth=1.5)
    plt.scatter(range(len(devices)), devices, color="green", zorder=5)

    for i, (d, c, t) in enumerate(zip(devices, comps, comms)):
        plt.text(i, d + 0.1, f"C:{c:.2f}\nT:{t:.2f}", ha="center", va="bottom", fontsize=8)

    plt.yticks(range(num_devices), [f"Device {i}" for i in range(num_devices)])
    plt.xticks(range(len(devices) + 1))
    plt.xlabel("Layer Index")
    plt.ylabel("Device ID")
    plt.title(f"Execution Strategy - DQN (Trace Length: {len(devices)}/{len(devices)})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, out_name))
    plt.close()


def _sl_tag(max_exposure_fraction: float) -> str:
    return f"sl_{float(max_exposure_fraction):.2f}".replace(".", "p")


def _make_run_dirs(*, algorithm: str, model: str, max_exposure_fraction: float, seed: int) -> tuple[str, str]:
    safe_algo = (algorithm or "algo").replace(os.sep, "_")
    safe_model = (model or "model").replace(os.sep, "_")
    run_root = os.path.join(
        SCRIPT_DIR,
        "results",
        "resultDQN",
        safe_algo,
        safe_model,
        _sl_tag(max_exposure_fraction),
        f"seed_{int(seed)}",
    )
    train_dir = os.path.join(run_root, "train")
    os.makedirs(train_dir, exist_ok=True)
    return run_root, train_dir


def train_single_agent(
    *,
    max_exposure_fraction: float = 1.0,
    episodes: int = 5000,
    seed: int = 42,
    model_type: str = "simplecnn",
):
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
    MODEL_TYPES = [str(model_type)]
    EPISODES = int(episodes)
    SEED = int(seed)
    LOG_EVERY = 50
     
    print(f"=== Starting Single Agent Training (Model: {MODEL_TYPES[0]}) ===")
    print(f"  S_l (max exposure fraction): {float(max_exposure_fraction):.3f}")
     
    # Initialize Environment
    set_global_seed(SEED)
    env = MonoAgentIoTEnv(
        num_agents=NUM_AGENTS,
        num_devices=NUM_DEVICES,
        model_types=MODEL_TYPES,
        seed=SEED,
        max_exposure_fraction=max_exposure_fraction,
    )
    
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
    final_successful_trace = []

    RUN_ROOT, TRAIN_DIR = _make_run_dirs(
        algorithm="DQN",
        model=str(MODEL_TYPES[0]),
        max_exposure_fraction=float(max_exposure_fraction),
        seed=SEED,
    )
    print(f"  RunDir: {RUN_ROOT}")

    config_path = os.path.join(RUN_ROOT, "run_config.json")
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "algorithm": "DQN",
                    "model_type": str(MODEL_TYPES[0]),
                    "episodes": int(EPISODES),
                    "seed": int(SEED),
                    "num_devices": int(NUM_DEVICES),
                    "max_exposure_fraction": float(max_exposure_fraction),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception:
        pass

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
            
        if stalls == 0 and done and episode_trace:
            final_successful_trace = episode_trace

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
                            "max_exposure_fraction": float(max_exposure_fraction),
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
    plot_execution_strategy(final_successful_trace, TRAIN_DIR, num_devices=NUM_DEVICES, out_name="execution_strategy.png")

    if log_f is not None:
        try:
            log_f.close()
        except Exception:
            pass

    try:
        last_k = 100
        avg_reward_last = float(np.mean(history_rewards[-last_k:])) if history_rewards else 0.0
        avg_stalls_last = float(np.mean(history_stalls[-last_k:])) if history_stalls else 0.0
        summary_path = os.path.join(RUN_ROOT, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "episodes": int(EPISODES),
                    "seed": int(SEED),
                    "model_type": str(MODEL_TYPES[0]),
                    "max_exposure_fraction": float(max_exposure_fraction),
                    "avg_reward_last_100": avg_reward_last,
                    "avg_stalls_last_100": avg_stalls_last,
                    "best_reward": float(np.max(history_rewards)) if history_rewards else None,
                    "worst_reward": float(np.min(history_rewards)) if history_rewards else None,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception:
        pass
     
    print("\nTraining Complete.")
    print(f"Total training time: {_fmt_seconds(total_time)}")
    print(f"Model saved to {os.path.join(MODELS_DIR, 'single_agent_simplecnn.pth')}")
    print(f"Training results in {TRAIN_DIR}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Single-Agent DQN for DNN layer scheduling.")
    parser.add_argument("--max-exposure-fraction", type=float, default=1.0, help="S_l in (0,1]. 1.0 disables.")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-type", type=str, default="simplecnn")
    args = parser.parse_args()

    train_single_agent(
        max_exposure_fraction=float(args.max_exposure_fraction),
        episodes=int(args.episodes),
        seed=int(args.seed),
        model_type=str(args.model_type),
    )
