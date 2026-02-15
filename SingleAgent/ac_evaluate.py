import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt

try:
    from .agent import ActorCriticAgent
    from .environment import MonoAgentIoTEnv
    from .utils import set_global_seed
except ImportError:
    from agent import ActorCriticAgent
    from environment import MonoAgentIoTEnv
    from utils import set_global_seed


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
    plt.savefig(os.path.join(results_dir, "execution_strategy_eval.png"))
    plt.close()


def evaluate_single_agent():
    MODEL_TYPE = "simplecnn"
    NUM_DEVICES = 5
    NUM_EPISODES = 50
    SEEDS = [0, 1, 2, 3, 4]

    MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "single_agent_simplecnn_ac.pth")
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "resultAC", "eval")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"=== Starting Single Agent Evaluation - Actor-Critic ({MODEL_TYPE}) ===")

    dummy_env = MonoAgentIoTEnv(num_agents=1, num_devices=NUM_DEVICES, model_types=[MODEL_TYPE], seed=SEEDS[0])
    agent = ActorCriticAgent(state_dim=dummy_env.single_state_dim, action_dim=NUM_DEVICES)

    if os.path.exists(MODEL_PATH):
        agent.load(MODEL_PATH)
        print(f"Loaded trained agent from {MODEL_PATH}")
    else:
        print(f"WARNING: No trained model found at {MODEL_PATH}. Using random weights.")

    all_seed_rewards = []
    all_seed_stalls = []
    all_seed_t_comp = []
    all_seed_t_comm = []
    sample_trace_full = None

    for seed in SEEDS:
        set_global_seed(seed)
        env = MonoAgentIoTEnv(num_agents=1, num_devices=NUM_DEVICES, model_types=[MODEL_TYPE], seed=seed)

        test_rewards = []
        test_stalls = []
        test_t_comp = []
        test_t_comm = []

        print(f"Running {NUM_EPISODES} test episodes (seed={seed})...")

        for e in range(NUM_EPISODES):
            state, _ = env.reset()
            done = False
            step = 0

            ep_reward = 0.0
            ep_stalls = 0
            ep_t_comp = 0.0
            ep_t_comm = 0.0
            trace = []

            while not done and step < 100:
                valid_actions = env.get_valid_actions()
                prev_progress = env.progress
                action = agent.act(state, valid_actions, deterministic=True)

                next_state, reward, terminated, truncated, info = env.step(action)

                if reward == -500.0:
                    ep_stalls += 1
                else:
                    ep_t_comp += float(info.get("t_comp", 0.0))
                    ep_t_comm += float(info.get("t_comm", 0.0))

                if env.progress > prev_progress:
                    trace.append(
                        {
                            "layer": prev_progress,
                            "device": int(action),
                            "t_comp": float(info.get("t_comp", 0.0)),
                            "t_comm": float(info.get("t_comm", 0.0)),
                        }
                    )

                ep_reward += float(reward)
                state = next_state
                done = terminated or truncated
                step += 1

            test_rewards.append(ep_reward)
            test_stalls.append(ep_stalls)
            test_t_comp.append(ep_t_comp)
            test_t_comm.append(ep_t_comm)

            if sample_trace_full is None and done and ep_stalls == 0 and trace:
                sample_trace_full = trace

            if e % 10 == 0:
                status = "FAILED (stall)" if ep_stalls > 0 else ("COMPLETED" if done else f"STUCK at Layer {env.progress}")
                print(
                    f" Episode {e:2} | Reward: {ep_reward:8.2f} | "
                    f"T_comp: {ep_t_comp:6.2f} | T_comm: {ep_t_comm:6.2f} | "
                    f"Stalls: {ep_stalls:3} | {status}"
                )

        all_seed_rewards.extend(test_rewards)
        all_seed_stalls.extend(test_stalls)
        all_seed_t_comp.extend(test_t_comp)
        all_seed_t_comm.extend(test_t_comm)

    avg_reward = float(np.mean(all_seed_rewards)) if all_seed_rewards else 0.0
    std_reward = float(np.std(all_seed_rewards)) if all_seed_rewards else 0.0
    avg_stalls = float(np.mean(all_seed_stalls)) if all_seed_stalls else 0.0
    avg_t_comp = float(np.mean(all_seed_t_comp)) if all_seed_t_comp else 0.0
    avg_t_comm = float(np.mean(all_seed_t_comm)) if all_seed_t_comm else 0.0

    print(f"\nSummary: avg_reward={avg_reward:.2f} | std_reward={std_reward:.2f} | avg_stalls={avg_stalls:.2f}")

    plt.figure(figsize=(10, 5))
    plt.hist(all_seed_rewards, bins=20, color="steelblue", alpha=0.8)
    plt.title("Episode Reward Distribution - Actor-Critic")
    plt.xlabel("Episode Reward")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "reward_distribution.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(["T_comp", "T_comm"], [avg_t_comp, avg_t_comm], color=["orange", "purple"])
    plt.title("Average Latency Breakdown - Actor-Critic")
    plt.ylabel("Latency (s)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "avg_latency_breakdown.png"))
    plt.close()

    if sample_trace_full:
        plot_execution_strategy(sample_trace_full, RESULTS_DIR, num_devices=NUM_DEVICES)

    print(f"Evaluation Complete. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    evaluate_single_agent()
