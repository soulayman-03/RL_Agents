import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
import json

try:
    from .agent import ActorCriticAgent
    from .environment import MonoAgentIoTEnv
    from .utils import load_and_remap_weights, set_global_seed
except ImportError:
    from agent import ActorCriticAgent
    from environment import MonoAgentIoTEnv
    from utils import load_and_remap_weights, set_global_seed

from split_inference.cnn_model import SimpleCNN


def plot_training_reward_history(rewards, results_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, color="blue", alpha=0.3, label="Raw Reward")
    window = 50
    if len(rewards) >= window:
        mv_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(range(window - 1, len(rewards)), mv_avg, color="red", linewidth=2, label="Moving Avg")
    plt.title("Training Rewards History - Actor-Critic")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "ac_training_history.png"))
    plt.close()

def plot_training_scalar(values, title, ylabel, out_path, window=50):
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


def plot_execution_strategy(trace, results_dir, num_devices=5):
    if not trace:
        return

    devices = [t["device"] for t in trace]
    comps = [t["t_comp"] for t in trace]
    comms = [t["t_comm"] for t in trace]

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
    plt.title(f"Execution Strategy - Actor-Critic (Trace Length: {len(devices)}/{len(devices)})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "execution_strategy.png"))
    plt.close()


def train_actor_critic():
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

    print(f"=== Starting Actor-Critic Training (Model: {MODEL_TYPES[0]}) ===")

    set_global_seed(SEED)
    env = MonoAgentIoTEnv(num_agents=NUM_AGENTS, num_devices=NUM_DEVICES, model_types=MODEL_TYPES, seed=SEED)
    agent = ActorCriticAgent(state_dim=env.single_state_dim, action_dim=NUM_DEVICES)
    agent.set_entropy_schedule(start=0.02, end=0.001, decay_steps=EPISODES)
    agent.set_gae_lambda(0.95)

    cv_model = SimpleCNN()
    weight_path = os.path.join(PROJECT_ROOT, "split_inference", "mnist_simplecnn.pth")
    success = load_and_remap_weights(cv_model, weight_path, "simplecnn")
    if success:
        print(f"Successfully loaded and remapped weights from {weight_path}")
    else:
        print(f"Warning: Could not load weights from {weight_path}. Using random initialization.")
    agent.assign_inference_model(cv_model)

    MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    RESULTS_ROOT = os.path.join(SCRIPT_DIR, "results", "resultAC")
    TRAIN_DIR = os.path.join(RESULTS_ROOT, "train")
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_ROOT, "eval"), exist_ok=True)

    start_time = time.time()
    history_rewards = []
    history_stalls = []
    history_policy_loss = []
    history_value_loss = []
    history_entropy = []
    final_successful_trace = []

    MAX_STEPS = 100

    log_path = os.path.join(TRAIN_DIR, "train_log.jsonl")
    try:
        log_f = open(log_path, "w", encoding="utf-8")
    except Exception:
        log_f = None

    for e in range(EPISODES):
        state, _ = env.reset()
        episode_reward = 0.0
        stalls = 0
        done = False
        step = 0

        log_probs = []
        values = []
        rewards = []
        entropies = []
        dones = []
        current_trace = []

        while not done and step < MAX_STEPS:
            valid_actions = env.get_valid_actions()
            prev_progress = env.progress

            action, log_prob, value, entropy = agent.select_action(state, valid_actions)
            next_state, reward, terminated, truncated, info = env.step(action)

            if reward == -500.0:
                stalls += 1
            else:
                if env.progress > prev_progress:
                    current_trace.append(
                        {
                            "layer": prev_progress,
                            "device": int(action),
                            "t_comp": float(info.get("t_comp", 0.0)),
                            "t_comm": float(info.get("t_comm", 0.0)),
                        }
                    )

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(float(reward))
            entropies.append(entropy)
            dones.append(bool(terminated or truncated))

            episode_reward += float(reward)
            state = next_state
            done = terminated or truncated
            step += 1

        bootstrap_value = 0.0 if done else agent.predict_value(state)
        p_loss, v_loss, ent = agent.update(log_probs, values, rewards, entropies, bootstrap_value=bootstrap_value, dones=dones)

        history_rewards.append(episode_reward)
        history_stalls.append(stalls)
        history_policy_loss.append(p_loss)
        history_value_loss.append(v_loss)
        history_entropy.append(ent)

        if stalls == 0 and done and current_trace:
            final_successful_trace = current_trace

        if log_f is not None:
            try:
                log_f.write(
                    json.dumps(
                        {
                            "episode": int(e),
                            "reward": float(episode_reward),
                            "stalls": int(stalls),
                            "policy_loss": float(p_loss),
                            "value_loss": float(v_loss),
                            "entropy": float(ent),
                            "entropy_coef": float(getattr(agent, "entropy_coef", 0.0)),
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
                f"P_Loss: {p_loss:7.4f} | V_Loss: {v_loss:7.4f} | Ent: {ent:6.3f}"
            )
            if devices:
                print(f"  Devices: {devices}")
                print("  Trace: layer -> device | t_comp t_comm")
                for t in current_trace:
                    print(f"   L{int(t['layer']):02d} -> D{int(t['device'])} | {float(t['t_comp']):6.3f} {float(t['t_comm']):6.3f}")

    total_time = time.time() - start_time

    model_path = os.path.join(MODELS_DIR, "single_agent_simplecnn_ac.pth")
    agent.save(model_path)
    np.save(os.path.join(MODELS_DIR, "single_ac_train_history.npy"), np.asarray(history_rewards, dtype=float))
    np.save(os.path.join(MODELS_DIR, "single_ac_stall_history.npy"), np.asarray(history_stalls, dtype=float))
    np.save(os.path.join(MODELS_DIR, "single_ac_policy_loss.npy"), np.asarray(history_policy_loss, dtype=float))
    np.save(os.path.join(MODELS_DIR, "single_ac_value_loss.npy"), np.asarray(history_value_loss, dtype=float))
    np.save(os.path.join(MODELS_DIR, "single_ac_entropy.npy"), np.asarray(history_entropy, dtype=float))

    plot_training_reward_history(history_rewards, TRAIN_DIR)
    plot_training_scalar(history_stalls, "Training Stalls History - Actor-Critic", "Stalls", os.path.join(TRAIN_DIR, "ac_stalls_history.png"))
    plot_training_scalar(history_policy_loss, "Policy Loss - Actor-Critic", "Loss", os.path.join(TRAIN_DIR, "ac_policy_loss.png"))
    plot_training_scalar(history_value_loss, "Value Loss - Actor-Critic", "Loss", os.path.join(TRAIN_DIR, "ac_value_loss.png"))
    plot_training_scalar(history_entropy, "Entropy - Actor-Critic", "Entropy", os.path.join(TRAIN_DIR, "ac_entropy.png"))
    plot_execution_strategy(final_successful_trace, TRAIN_DIR, num_devices=NUM_DEVICES)

    if log_f is not None:
        try:
            log_f.close()
        except Exception:
            pass

    print("\nTraining Complete.")
    print(f"Total training time: {_fmt_seconds(total_time)}")
    print(f"Model saved to {model_path}")
    print(f"Training results in {TRAIN_DIR}/")


if __name__ == "__main__":
    train_actor_critic()
