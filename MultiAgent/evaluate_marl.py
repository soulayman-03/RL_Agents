import argparse
import json
import os
import sys
from collections import Counter, defaultdict

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


def _safe_mean(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained MARL agents in MultiAgentIoTEnv.")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per seed.")
    parser.add_argument("--seeds", type=str, default="42,55,66,77,88", help="Comma-separated seeds.")
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-devices", type=int, default=5)
    parser.add_argument("--model-types", type=str, default="simplecnn,deepcnn,miniresnet", help="Comma-separated model types.")
    parser.add_argument("--model-path", type=str, default=os.path.join(SCRIPT_DIR, "models", "marl"), help="Base path for saved models.")
    parser.add_argument("--results-dir", type=str, default=os.path.join(SCRIPT_DIR, "results"))
    parser.add_argument("--shared-policy", action="store_true", default=True, help="Use shared policy (matches training default).")
    parser.add_argument("--no-shared-policy", dest="shared_policy", action="store_false")
    parser.add_argument("--shuffle-order", action="store_true", default=True, help="Shuffle allocation order to avoid starvation.")
    parser.add_argument("--no-shuffle-order", dest="shuffle_order", action="store_false")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    model_types = [s.strip() for s in args.model_types.split(",") if s.strip()]
    os.makedirs(args.results_dir, exist_ok=True)

    set_global_seed(seeds[0] if seeds else 42)
    env = MultiAgentIoTEnv(
        num_agents=args.num_agents,
        num_devices=args.num_devices,
        model_types=model_types,
        seed=seeds[0] if seeds else 42,
        shuffle_allocation_order=bool(args.shuffle_order),
    )
    manager = MultiAgentManager(
        agent_ids=list(range(args.num_agents)),
        state_dim=env.single_state_dim,
        action_dim=args.num_devices,
        shared_policy=bool(args.shared_policy),
    )

    if os.path.exists(f"{args.model_path}_shared.pt") or os.path.exists(f"{args.model_path}_agent_0.pt"):
        manager.load_agents(args.model_path)
        print(f"Loaded MARL models from {args.model_path}")
    else:
        print("WARNING: Models not found. Evaluating untrained policy structure.")

    # Deterministic evaluation
    for agent in {id(a): a for a in manager.agents.values()}.values():
        agent.epsilon = 0.0

    per_agent = {
        aid: {
            "success_eps": 0,
            "episodes": 0,
            "latencies": [],
            "t_comp": [],
            "t_comm": [],
            "device_counts": Counter(),
            "fail_reasons": Counter(),
        }
        for aid in range(args.num_agents)
    }

    for seed in seeds:
        env.resource_manager.reset_devices_with_seed(args.num_devices, seed)
        for _ in range(int(args.episodes)):
            obs, _ = env.reset()
            done = {aid: False for aid in range(args.num_agents)}

            for aid in range(args.num_agents):
                per_agent[aid]["episodes"] += 1

            while not all(done.values()):
                valid_actions = env.get_valid_actions()
                actions = manager.get_actions(obs, valid_actions)
                next_obs, rewards, next_done, truncated, infos = env.step(actions)

                for aid in range(args.num_agents):
                    info = infos.get(aid, {}) if isinstance(infos, dict) else {}
                    if info.get("success"):
                        tc = float(info.get("t_comp", 0.0))
                        tm = float(info.get("t_comm", 0.0))
                        per_agent[aid]["t_comp"].append(tc)
                        per_agent[aid]["t_comm"].append(tm)
                        per_agent[aid]["latencies"].append(tc + tm)
                        if aid in actions:
                            per_agent[aid]["device_counts"][int(actions[aid])] += 1
                    else:
                        if info.get("reward_type") == "stall_termination":
                            fail = info.get("fail", {})
                            reason = fail.get("reason", "unknown") if isinstance(fail, dict) else "unknown"
                            per_agent[aid]["fail_reasons"][str(reason)] += 1

                obs, done = next_obs, next_done

            for aid in range(args.num_agents):
                if env.agent_progress[aid] == len(env.tasks[aid]):
                    per_agent[aid]["success_eps"] += 1

    report = {"seeds": seeds, "episodes_per_seed": int(args.episodes), "per_agent": {}}
    for aid in range(args.num_agents):
        a = per_agent[aid]
        episodes = int(a["episodes"]) if a["episodes"] else 1
        report["per_agent"][str(aid)] = {
            "model_type": model_types[aid] if aid < len(model_types) else "unknown",
            "episodes": episodes,
            "success_rate": float(a["success_eps"]) / episodes * 100.0,
            "avg_latency": _safe_mean(a["latencies"]),
            "avg_t_comp": _safe_mean(a["t_comp"]),
            "avg_t_comm": _safe_mean(a["t_comm"]),
            "device_counts": dict(a["device_counts"]),
            "fail_reasons": dict(a["fail_reasons"]),
        }

    out_json = os.path.join(args.results_dir, "marl_eval_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # --- Plots ---
    agent_ids = list(range(args.num_agents))
    success_rates = [report["per_agent"][str(aid)]["success_rate"] for aid in agent_ids]
    avg_lat = [report["per_agent"][str(aid)]["avg_latency"] for aid in agent_ids]
    labels = [f"A{aid} ({report['per_agent'][str(aid)]['model_type']})" for aid in agent_ids]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.bar(np.arange(len(agent_ids)), success_rates, color="seagreen")
    plt.xticks(np.arange(len(agent_ids)), labels, rotation=15, ha="right")
    plt.ylim(0, 100)
    plt.title("Success Rate (%)")
    plt.grid(True, axis="y", alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.bar(np.arange(len(agent_ids)), avg_lat, color="steelblue")
    plt.xticks(np.arange(len(agent_ids)), labels, rotation=15, ha="right")
    plt.title("Avg Latency (s)")
    plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_summary = os.path.join(args.results_dir, "marl_eval_summary.png")
    plt.savefig(out_summary, dpi=150)
    plt.close()

    # Device usage heatmap (agent x device)
    usage = np.zeros((args.num_agents, args.num_devices), dtype=float)
    for aid in agent_ids:
        counts = report["per_agent"][str(aid)]["device_counts"]
        for d in range(args.num_devices):
            usage[aid, d] = float(counts.get(str(d), counts.get(d, 0)))  # tolerate json casting

    plt.figure(figsize=(10, 4))
    plt.imshow(usage, aspect="auto", cmap="viridis")
    plt.colorbar(label="Selections (count)")
    plt.yticks(np.arange(args.num_agents), labels)
    plt.xticks(np.arange(args.num_devices), [f"D{d}" for d in range(args.num_devices)])
    plt.title("Device Selection Heatmap")
    plt.tight_layout()
    out_heat = os.path.join(args.results_dir, "marl_device_heatmap.png")
    plt.savefig(out_heat, dpi=150)
    plt.close()

    print(f"Saved: {out_json}")
    print(f"Saved: {out_summary}")
    print(f"Saved: {out_heat}")


if __name__ == "__main__":
    main()

