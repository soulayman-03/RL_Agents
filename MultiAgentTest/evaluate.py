import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Define paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from MultiAgent.environment import MultiAgentIoTEnv
from MultiAgent.manager import MultiAgentManager
from SingleAgent.utils import set_global_seed


def plot_execution_flow(all_traces, num_agents, num_devices, results_dir, filename):
    """Plots the execution flow for all agents in a single episode."""
    plt.figure(figsize=(12, 8))
    colors = ["blue", "green", "red", "orange", "purple"]

    for aid in range(num_agents):
        trace = all_traces.get(aid, [])
        if not trace:
            continue

        layers = [t["layer"] for t in trace]
        devices = [t["device"] for t in trace]

        max_layers = max(layers) if layers else 0
        layers.append(max_layers + 0.5)
        devices.append(devices[-1])

        plt.step(
            layers,
            devices,
            where="post",
            marker="o",
            color=colors[aid % len(colors)],
            label=f'Agent {aid} ({trace[0]["model"]})',
            alpha=0.8,
        )

        total_lat = sum([t["lat"] for t in trace])
        plt.text(
            layers[-1],
            devices[-1],
            f"Agent {aid}: {total_lat:.2f}s",
            color=colors[aid % len(colors)],
            fontweight="bold",
        )

    plt.yticks(range(num_devices), [f"Device {d}" for d in range(num_devices)])
    plt.xlabel("Layer Index")
    plt.ylabel("Device ID")
    plt.title("Multi-Agent Execution Flow (Resource Allocation Strategy)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()


def plot_layer_latency(all_episode_traces_by_agent, tasks_by_agent, results_dir, filename_prefix):
    """
    Plot average layer-wise latency breakdown (computation vs communication) per agent.
    `all_episode_traces_by_agent[aid]` is a list of traces; each trace is list of dicts with:
      {layer, t_comp, t_comm, ...}
    """
    for aid, traces in all_episode_traces_by_agent.items():
        task = tasks_by_agent.get(aid)
        if not task:
            continue

        num_layers = len(task)
        comp = np.zeros(num_layers, dtype=float)
        comm = np.zeros(num_layers, dtype=float)
        counts = np.zeros(num_layers, dtype=float)

        for trace in traces:
            for t in trace:
                if not isinstance(t, dict) or "layer" not in t:
                    continue
                layer = int(t["layer"])
                if layer < 0 or layer >= num_layers:
                    continue
                comp[layer] += float(t.get("t_comp", 0.0))
                comm[layer] += float(t.get("t_comm", 0.0))
                counts[layer] += 1.0

        if float(np.sum(counts)) == 0.0:
            plt.figure(figsize=(14, 5))
            plt.text(
                0.5,
                0.5,
                f"No successful allocations recorded for Agent {aid}\n"
                f"(try more episodes or enable shuffle_allocation_order).",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"{filename_prefix}_agent_{aid}.png"))
            plt.close()
            continue

        counts[counts == 0] = 1.0
        comp /= counts
        comm /= counts

        layer_names = [getattr(l, "name", f"Layer_{i}") for i, l in enumerate(task)]
        x = np.arange(num_layers)
        width = 0.4

        plt.figure(figsize=(14, 5))
        plt.bar(x - width / 2, comp, width, label="Computation")
        plt.bar(x + width / 2, comm, width, label="Communication")
        plt.xticks(x, layer_names, rotation=60)
        plt.ylabel("Latency (s)")
        plt.xlabel("Layers")
        plt.title(f"Average Layer-wise Latency Breakdown - Agent {aid}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{filename_prefix}_agent_{aid}.png"))
        plt.close()


def evaluate(
    eval_seeds=None,
    num_eval_episodes: int = 10,
    results_dir: str | None = None,
    model_path: str | None = None,
    model_types=None,
):
    NUM_AGENTS = 3
    NUM_DEVICES = 5
    MODEL_TYPES = model_types or ["simplecnn", "cnn7", "cnn10"]
    MODEL_PATH = model_path or os.path.join(SCRIPT_DIR, "models", "marl_test")
    RESULTS_DIR = results_dir or os.path.join(SCRIPT_DIR, "results")
    NUM_EVAL_EPISODES = int(num_eval_episodes)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"=== Starting Multi-Agent Test Evaluation (Agents: {NUM_AGENTS}) ===")

    set_global_seed(42)
    env = MultiAgentIoTEnv(
        num_agents=NUM_AGENTS,
        num_devices=NUM_DEVICES,
        model_types=MODEL_TYPES,
        seed=42,
        shuffle_allocation_order=True,
    )

    manager = MultiAgentManager(
        agent_ids=list(range(NUM_AGENTS)),
        state_dim=env.single_state_dim,
        action_dim=env.num_devices,
        shared_policy=True,
    )

    if os.path.exists(f"{MODEL_PATH}_shared.pt") or os.path.exists(f"{MODEL_PATH}_agent_0.pt"):
        manager.load_agents(MODEL_PATH)
        print(f"Loaded trained agents from {MODEL_PATH}")
    else:
        print(f"WARNING: Models not found at {MODEL_PATH}. Using untrained agents.")

    for agent in {id(a): a for a in manager.agents.values()}.values():
        agent.epsilon = 0.0

    eval_rewards = {i: [] for i in range(NUM_AGENTS)}
    eval_successes = {i: [] for i in range(NUM_AGENTS)}

    EVAL_SEEDS = list(eval_seeds) if eval_seeds is not None else [42]

    last_episode_traces = {i: [] for i in range(NUM_AGENTS)}
    all_episode_traces_by_agent = {i: [] for i in range(NUM_AGENTS)}
    tasks_by_agent = {}

    for seed_idx, current_seed in enumerate(EVAL_SEEDS):
        print(f"\n--- Testing Seed: {current_seed} ---")
        env.resource_manager.reset_devices_with_seed(NUM_DEVICES, int(current_seed))

        for ep in range(NUM_EVAL_EPISODES):
            obs, _ = env.reset()
            if not tasks_by_agent:
                tasks_by_agent = {i: env.tasks[i] for i in range(NUM_AGENTS)}

            done = {i: False for i in range(NUM_AGENTS)}
            agent_ep_mappings = {i: [] for i in range(NUM_AGENTS)}
            ep_rewards = {i: 0.0 for i in range(NUM_AGENTS)}
            ep_success = {i: False for i in range(NUM_AGENTS)}
            ep_traces = {i: [] for i in range(NUM_AGENTS)}

            while not all(done.values()):
                valid_actions = env.get_valid_actions()
                actions = manager.get_actions(obs, valid_actions)

                step_layers = {aid: env.agent_progress[aid] for aid in range(NUM_AGENTS) if not done[aid]}

                for aid, device_id in actions.items():
                    if not done[aid]:
                        agent_ep_mappings[aid].append(device_id)

                if seed_idx == len(EVAL_SEEDS) - 1 and ep == NUM_EVAL_EPISODES - 1:
                    for aid, dev_id in actions.items():
                        if not done[aid]:
                            last_episode_traces[aid].append(
                                {"layer": env.agent_progress[aid], "device": dev_id, "model": env.model_types[aid], "lat": 0}
                            )

                next_obs, rewards, next_done, truncated, infos = env.step(actions)

                if seed_idx == len(EVAL_SEEDS) - 1 and ep == NUM_EVAL_EPISODES - 1:
                    for aid in range(NUM_AGENTS):
                        if aid in rewards and rewards[aid] > -500 and last_episode_traces[aid]:
                            last_episode_traces[aid][-1]["lat"] = -float(rewards[aid])

                for aid in range(NUM_AGENTS):
                    if aid in infos and infos[aid].get("success") and aid in step_layers:
                        ep_traces[aid].append(
                            {
                                "layer": int(step_layers[aid]),
                                "device": int(actions[aid]),
                                "t_comp": float(infos[aid].get("t_comp", 0.0)),
                                "t_comm": float(infos[aid].get("t_comm", 0.0)),
                                "model": env.model_types[aid],
                            }
                        )

                obs, done = next_obs, next_done

                for i in range(NUM_AGENTS):
                    if i in rewards:
                        ep_rewards[i] += float(rewards[i])
                    if i in infos and infos[i].get("success") and env.agent_progress[i] == len(env.tasks[i]):
                        ep_success[i] = True

            for i in range(NUM_AGENTS):
                all_episode_traces_by_agent[i].append(ep_traces[i])
                eval_rewards[i].append(float(ep_rewards[i]))
                eval_successes[i].append(1 if ep_success[i] else 0)

            if ep == 0:
                avg_ep_reward = sum(ep_rewards.values()) / NUM_AGENTS
                print(f"Seed {current_seed} | Ep {ep+1} | Avg Reward: {avg_ep_reward:.2f}")
                for i in range(NUM_AGENTS):
                    mapping_str = " -> ".join([str(d) for d in agent_ep_mappings[i]])
                    print(f"  Agent {i} ({env.model_types[i]}): {mapping_str} {'[SUCCESS]' if ep_success[i] else '[FAILED]'}")

        seed_avg_reward = np.mean([np.mean(eval_rewards[i][-NUM_EVAL_EPISODES:]) for i in range(NUM_AGENTS)])
        seed_success_rate = np.mean([np.mean(eval_successes[i][-NUM_EVAL_EPISODES:]) for i in range(NUM_AGENTS)]) * 100
        print(f"--- Seed {current_seed} Summary: Avg Reward: {seed_avg_reward:.2f}, Success Rate: {seed_success_rate:.1f}% ---")

    # --- RESULTS & PLOTTING ---
    plot_execution_flow(last_episode_traces, NUM_AGENTS, NUM_DEVICES, RESULTS_DIR, "execution_flow.png")
    print(f"\n - Execution flow plot saved to {RESULTS_DIR}/execution_flow.png")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    agent_labels = [f"Agent {i}\n({MODEL_TYPES[i]})" for i in range(NUM_AGENTS)]
    avg_rewards = [np.mean(eval_rewards[i]) if eval_rewards[i] else 0.0 for i in range(NUM_AGENTS)]
    plt.bar(agent_labels, avg_rewards, color=["skyblue", "lightgreen", "salmon"])
    plt.title("Average Reward per Agent")
    plt.ylabel("Reward")

    plt.subplot(1, 2, 2)
    success_rates = [np.mean(eval_successes[i]) * 100 if eval_successes[i] else 0.0 for i in range(NUM_AGENTS)]
    plt.bar(agent_labels, success_rates, color=["skyblue", "lightgreen", "salmon"])
    plt.title("Success Rate per Agent")
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, 110)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "evaluation_summary.png"))
    plt.close()
    print(f" - Evaluation summary plot saved to {RESULTS_DIR}/evaluation_summary.png")

    plot_layer_latency(all_episode_traces_by_agent, tasks_by_agent, RESULTS_DIR, "layer_latency_avg")
    print(f" - Layer-wise latency breakdown saved to {RESULTS_DIR}/layer_latency_avg_agent_*.png")

    # Reward/success trends (per agent)
    plt.figure(figsize=(14, 6))
    for i in range(NUM_AGENTS):
        plt.plot(eval_rewards[i], label=f"Agent {i} Reward")
    plt.title("Reward per Evaluation Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "reward_trends.png"))
    plt.close()

    plt.figure(figsize=(14, 6))
    for i in range(NUM_AGENTS):
        plt.plot(eval_successes[i], label=f"Agent {i} Success")
    plt.title("Success per Evaluation Episode")
    plt.xlabel("Episode")
    plt.ylabel("Success (0/1)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "success_trends.png"))
    plt.close()

    np.savez_compressed(
        os.path.join(RESULTS_DIR, "evaluation_rewards.npz"),
        **{f"agent_{i}": np.array(eval_rewards[i], dtype=float) for i in range(NUM_AGENTS)},
    )
    np.savez_compressed(
        os.path.join(RESULTS_DIR, "evaluation_successes.npz"),
        **{f"agent_{i}": np.array(eval_successes[i], dtype=float) for i in range(NUM_AGENTS)},
    )

    print("\n=== Evaluation Results ===")
    for i in range(NUM_AGENTS):
        avg_r = float(np.mean(eval_rewards[i])) if eval_rewards[i] else 0.0
        sr = float(np.mean(eval_successes[i]) * 100) if eval_successes[i] else 0.0
        print(f"Agent {i} ({MODEL_TYPES[i]}): Avg Reward = {avg_r:.2f}, Success Rate = {sr:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="MultiAgentTest evaluation (same plots as MultiAgent).")
    parser.add_argument("--seed", type=int, default=42, help="Single seed to evaluate (default: 42).")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds (overrides --seed).")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per seed.")
    parser.add_argument("--results-dir", type=str, default=None, help="Output directory for plots/results.")
    parser.add_argument("--model-path", type=str, default=None, help="Base path for saved models.")
    parser.add_argument("--model-types", type=str, default=None, help="Comma-separated model types.")
    args = parser.parse_args()

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [int(args.seed)]

    model_types = [s.strip() for s in args.model_types.split(",") if s.strip()] if args.model_types else None
    evaluate(
        eval_seeds=seeds,
        num_eval_episodes=int(args.episodes),
        results_dir=args.results_dir,
        model_path=args.model_path,
        model_types=model_types,
    )


if __name__ == "__main__":
    main()

