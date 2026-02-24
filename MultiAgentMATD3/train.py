import os
import sys
import time
import argparse
import json
from collections import Counter

import numpy as np

# Allow running both:
# - as a module:  python -m MultiAgentMATD3.train
# - as a script:  python MultiAgentMATD3/train.py
if __package__:
    from .environment import MultiAgentIoTEnv
    from .manager import MATD3Manager
    from .plots import plot_avg_cumulative_rewards, plot_training_trends, plot_per_agent_training_rewards
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from MultiAgentMATD3.environment import MultiAgentIoTEnv
    from MultiAgentMATD3.manager import MATD3Manager
    from MultiAgentMATD3.plots import plot_avg_cumulative_rewards, plot_training_trends, plot_per_agent_training_rewards


def print_device_info(resource_manager):
    print("\n" + "=" * 50)
    print("DEVICE SPECIFICATIONS")
    print("=" * 50)
    for d_id, d in resource_manager.devices.items():
        print(
            f"Device {d_id}: CPU={d.cpu_speed:.2f}, RAM={d.memory_capacity:.1f}MB, "
            f"BW={d.bandwidth:.1f}Mbps, Privacy={d.privacy_clearance}"
        )
    print("=" * 50 + "\n")


def _model_stats(tasks: dict[int, list[object]], num_agents: int) -> dict[str, dict[str, float]]:
    """
    Summarize model (task) characteristics per agent based on generated layers.
    Returns JSON-serializable floats/ints only.
    """
    stats: dict[str, dict[str, float]] = {}
    for aid in range(int(num_agents)):
        layers = list(tasks.get(aid, []) or [])
        comp = [float(getattr(l, "computation_demand", 0.0)) for l in layers]
        mem = [float(getattr(l, "memory_demand", 0.0)) for l in layers]
        out = [float(getattr(l, "output_data_size", 0.0)) for l in layers]
        priv = [int(getattr(l, "privacy_level", 0)) for l in layers]
        stats[str(aid)] = {
            "n_layers": float(len(layers)),
            "total_compute": float(sum(comp)),
            "total_memory": float(sum(mem)),
            "total_output": float(sum(out)),
            "max_layer_compute": float(max(comp)) if comp else 0.0,
            "max_layer_memory": float(max(mem)) if mem else 0.0,
            "n_private_layers": float(sum(1 for p in priv if int(p) > 0)),
        }
    return stats


def _sl_tag(v: float) -> str:
    return f"sl_{float(v):.2f}".replace(".", "p")


def _scenario_tag(model_types: list[str]) -> str:
    cleaned = [str(m).strip().lower() for m in (model_types or []) if str(m).strip()]
    if not cleaned:
        return "default"
    return "models_" + "_".join(cleaned)


def _normalize_model_types(model_types: list[str] | None, num_agents: int) -> list[str]:
    if not model_types:
        return ["simplecnn", "deepcnn", "miniresnet"][:num_agents] + ["lenet"] * max(0, num_agents - 3)
    mt = [str(m).strip() for m in model_types if str(m).strip()]
    if len(mt) == 1:
        return mt * num_agents
    if len(mt) != num_agents:
        raise ValueError(f"--models must be length 1 or {num_agents}, got {len(mt)}")
    return mt


def train(
    *,
    sl: float = 1.0,
    episodes: int = 5000,
    seed: int = 42,
    model_types: list[str] | None = None,
    max_fail_logs_per_episode: int = 0,
    log_every: int = 50,
    log_trace: bool = False,
    trace_max_steps: int = 200,
):
    NUM_AGENTS = 3
    NUM_DEVICES = 5
    EPISODES = int(episodes)
    MODEL_TYPES = _normalize_model_types(model_types, NUM_AGENTS)
    TERMINATE_ON_FAIL = True

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    scenario = _scenario_tag(MODEL_TYPES)
    SAVE_DIR = os.path.join(SCRIPT_DIR, "models", scenario, _sl_tag(float(sl)))
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", scenario, _sl_tag(float(sl)))
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    env = MultiAgentIoTEnv(
        num_agents=NUM_AGENTS,
        num_devices=NUM_DEVICES,
        model_types=MODEL_TYPES,
        seed=int(seed),
        shuffle_allocation_order=True,
        max_exposure_fraction=float(sl),
        max_fail_logs_per_episode=int(max_fail_logs_per_episode),
    )

    manager = MATD3Manager(
        agent_ids=list(range(NUM_AGENTS)),
        obs_dim=env.single_state_dim,
        action_dim=env.num_devices,
        state_dim=NUM_AGENTS * env.single_state_dim,
        batch_size=256,
        shared_policy=False,
    )

    episode_team_rewards: list[float] = []
    episode_team_rewards_sum: list[float] = []
    losses: list[float] = []
    eps_history: list[float] = []
    episode_steps: list[int] = []
    episode_success: list[int] = []
    agent_reward_history = {i: [] for i in range(NUM_AGENTS)}
    agent_success_history = {i: [] for i in range(NUM_AGENTS)}
    total_env_steps = 0
    total_fail_episodes = 0
    fail_reasons_history: list[Counter[str]] = []

    log_path = os.path.join(RESULTS_DIR, "train_log.jsonl")
    try:
        log_f = open(log_path, "w", encoding="utf-8")
    except Exception:
        log_f = None

    def _fmt_seconds(sec: float) -> str:
        sec = max(0.0, float(sec))
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    t0 = time.time()
    print(
        "MATD3 Training (CTDE)\n"
        f"  Agents: {NUM_AGENTS} | Devices: {NUM_DEVICES} | Episodes: {EPISODES}\n"
        f"  Models: {MODEL_TYPES}\n"
        f"  Scenario: {scenario}\n"
        f"  Seed: {int(seed)}\n"
        f"  S_l (max exposure fraction): {float(sl):.3f}\n"
        f"  SaveDir: {SAVE_DIR}\n"
        f"  Results: {RESULTS_DIR}\n"
        f"  ShuffleAllocationOrder: {True}\n"
        f"  TerminateOnFail: {TERMINATE_ON_FAIL}\n"
    )
    print_device_info(env.resource_manager)
    model_stats = _model_stats(getattr(env, "tasks", {}) or {}, NUM_AGENTS)
    print(f"MODEL STATS (per agent): {model_stats}\n")

    for ep in range(EPISODES):
        obs, _ = env.reset()
        done = {i: False for i in range(NUM_AGENTS)}  # done before step

        team_return_mean = 0.0
        team_return_sum = 0.0
        steps = 0
        ep_failed = False
        agent_ep_reward = {i: 0.0 for i in range(NUM_AGENTS)}
        agent_failed = {i: False for i in range(NUM_AGENTS)}
        ep_fail_reasons: Counter[str] = Counter()
        ep_trace: list[dict] = []
        ep_layer_device_map: dict[str, dict[str, int]] = {str(i): {} for i in range(NUM_AGENTS)}

        while not all(done.values()):
            valid_actions = env.get_valid_actions()
            actions = manager.get_actions(obs, valid_actions)
            layer_idx_before = {aid: int(env.agent_progress.get(aid, 0)) for aid in range(NUM_AGENTS) if not done.get(aid, False)}

            next_obs, rewards, next_done, truncated, infos = env.step(actions)
            steps += 1

            for aid in range(NUM_AGENTS):
                if not done.get(aid, False):
                    agent_ep_reward[aid] += float(rewards.get(aid, 0.0))

            any_fail = False
            for aid, info in infos.items():
                if isinstance(info, dict) and info.get("success") is False:
                    any_fail = True
                    agent_failed[int(aid)] = True
                    fail = info.get("fail", {}) if isinstance(info, dict) else {}
                    if isinstance(fail, dict):
                        ep_fail_reasons[str(fail.get("reason", "unknown"))] += 1
                    else:
                        ep_fail_reasons["unknown"] += 1

            for aid in range(NUM_AGENTS):
                if done.get(aid, False):
                    continue
                info = infos.get(aid, {}) if isinstance(infos, dict) else {}
                if not isinstance(info, dict):
                    continue
                if bool(info.get("success", True)) is True and aid in layer_idx_before:
                    layer_idx = int(layer_idx_before[aid])
                    ep_layer_device_map[str(aid)][str(layer_idx)] = int(actions.get(aid, 0))

            if log_trace and len(ep_trace) < int(trace_max_steps):
                for aid in range(NUM_AGENTS):
                    if done.get(aid, False):
                        continue
                    info = infos.get(aid, {}) if isinstance(infos, dict) else {}
                    fail = info.get("fail", {}) if isinstance(info, dict) else {}
                    fail_reason = str(fail.get("reason", "")) if isinstance(fail, dict) else ""
                    ep_trace.append(
                        {
                            "step": int(steps - 1),
                            "agent": int(aid),
                            "layer": int(layer_idx_before.get(aid, 0)),
                            "device": int(actions.get(aid, 0)),
                            "success": bool(info.get("success", True)) if isinstance(info, dict) else True,
                            "reward": float(rewards.get(aid, 0.0)),
                            "t_comp": float(info.get("t_comp", 0.0)) if isinstance(info, dict) else 0.0,
                            "t_comm": float(info.get("t_comm", 0.0)) if isinstance(info, dict) else 0.0,
                            "fail_reason": fail_reason,
                        }
                    )

            if any_fail and TERMINATE_ON_FAIL:
                ep_failed = True
                next_done = {i: True for i in range(NUM_AGENTS)}

            next_valid_actions = env.get_valid_actions()
            manager.remember(
                obs_dict=obs,
                actions_dict=actions,
                rewards_dict=rewards,
                next_obs_dict=next_obs,
                dones_dict=next_done,
                active_before_dict=done,
                valid_actions=valid_actions,
                next_valid_actions=next_valid_actions,
            )

            loss = manager.train()
            if loss is not None:
                losses.append(float(loss))

            team_r = float(sum(rewards.values())) / float(NUM_AGENTS)
            team_return_mean += team_r
            team_return_sum += float(sum(rewards.values()))

            done = next_done
            obs = next_obs

        finished_steps = max(1, steps)
        episode_team_rewards.append(team_return_mean / float(finished_steps))
        episode_team_rewards_sum.append(team_return_sum)
        episode_steps.append(steps)
        episode_success.append(0 if ep_failed else 1)
        fail_reasons_history.append(ep_fail_reasons)

        for aid in range(NUM_AGENTS):
            agent_reward_history[aid].append(agent_ep_reward[aid])
            finished = bool(env.agent_progress.get(aid, 0) >= len(env.tasks.get(aid, [])))
            agent_success_history[aid].append(1 if (not agent_failed[aid] and finished) else 0)

        total_env_steps += steps
        if ep_failed:
            total_fail_episodes += 1

        eps = manager._unique_agents()[0].eps.epsilon
        eps_history.append(float(eps))

        if log_f is not None:
            device_counts: dict[str, dict[str, int]] = {}
            if ep_trace:
                per_agent = {str(a): Counter() for a in range(NUM_AGENTS)}
                for t in ep_trace:
                    per_agent[str(int(t.get("agent", 0)))][str(int(t.get("device", 0)))] += 1
                device_counts = {a: dict(c) for a, c in per_agent.items()}
            try:
                log_f.write(
                    json.dumps(
                        {
                            "episode": int(ep),
                            "seed": int(seed),
                            "sl": float(sl),
                            "scenario": str(scenario),
                            "models": list(MODEL_TYPES),
                            "model_stats": model_stats,
                            "team_reward_sum": float(team_return_sum),
                            "team_reward_mean_step": float(team_return_mean / float(finished_steps)),
                            "steps": int(steps),
                            "ep_failed": bool(ep_failed),
                            "fail_reasons": dict(ep_fail_reasons),
                            "per_agent_reward_sum": {str(a): float(agent_ep_reward[a]) for a in range(NUM_AGENTS)},
                            "per_agent_failed": {str(a): bool(agent_failed[a]) for a in range(NUM_AGENTS)},
                            "device_counts": device_counts,
                            "layer_device_map": ep_layer_device_map,
                            "trace": ep_trace if log_trace else None,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                log_f.flush()
            except Exception:
                pass

        if (ep + 1) % max(1, int(log_every)) == 0:
            avg = float(np.mean(episode_team_rewards_sum[-50:]))
            avg_steps = float(np.mean(episode_steps[-50:]))
            avg_loss = float(np.mean(losses[-200:])) if len(losses) >= 1 else float("nan")
            elapsed = time.time() - t0
            eps_per_sec = (ep + 1) / max(elapsed, 1e-9)
            eta = (EPISODES - (ep + 1)) / max(eps_per_sec, 1e-9)
            replay_len = len(manager.buffer)

            last_k = 50
            recent_fail = Counter()
            for c in fail_reasons_history[-last_k:]:
                recent_fail.update(c)
            top_fail = ", ".join([f"{k}={v}" for k, v in recent_fail.most_common(3)]) if recent_fail else "none"

            agent_stats = []
            for aid in range(NUM_AGENTS):
                a_reward = float(np.mean(agent_reward_history[aid][-50:]))
                a_success = float(np.mean(agent_success_history[aid][-50:]) * 100.0)
                agent_stats.append(f"A{aid}: {a_reward:.1f} ({a_success:.0f}%)")
            stats_str = " | ".join(agent_stats)
            print(
                f"Ep {ep+1:4d}/{EPISODES} - Avg: {avg:.1f} - [{stats_str}] - Eps: {eps:.3f} "
                f"- Loss: {avg_loss:.4f} - Steps(50): {avg_steps:.1f} - Replay: {replay_len} "
                f"- Elapsed: {_fmt_seconds(elapsed)} - ETA: {_fmt_seconds(eta)} - FailReasons(50): {top_fail}"
            )

    base = os.path.join(SAVE_DIR, "matd3")
    manager.save(base)
    np.save(os.path.join(RESULTS_DIR, "team_reward_history.npy"), np.asarray(episode_team_rewards, dtype=np.float32))
    np.save(os.path.join(RESULTS_DIR, "team_reward_sum_history.npy"), np.asarray(episode_team_rewards_sum, dtype=np.float32))
    np.save(os.path.join(RESULTS_DIR, "loss_history.npy"), np.asarray(losses, dtype=np.float32))
    np.save(os.path.join(RESULTS_DIR, "epsilon_history.npy"), np.asarray(eps_history, dtype=np.float32))
    np.save(os.path.join(RESULTS_DIR, "episode_steps.npy"), np.asarray(episode_steps, dtype=np.int32))
    np.save(os.path.join(RESULTS_DIR, "episode_success.npy"), np.asarray(episode_success, dtype=np.int32))
    np.savez_compressed(
        os.path.join(RESULTS_DIR, "agent_success_history.npz"),
        **{f"agent_{i}": np.asarray(agent_success_history[i], dtype=np.int32) for i in range(NUM_AGENTS)},
    )

    plot_training_trends(
        out_path=os.path.join(RESULTS_DIR, "training_trends.png"),
        team_rewards=episode_team_rewards,
        losses=losses,
        eps_history=None,
        window=50,
    )
    plot_avg_cumulative_rewards(
        out_path=os.path.join(RESULTS_DIR, "avg_cumulative_rewards.png"),
        episode_team_reward_sums=episode_team_rewards_sum,
        num_agents=NUM_AGENTS,
        window=50,
    )
    plot_avg_cumulative_rewards(
        out_path=os.path.join(RESULTS_DIR, "avg_cumulative_rewards_fixed.png"),
        episode_team_reward_sums=episode_team_rewards_sum,
        num_agents=NUM_AGENTS,
        window=50,
        ylim=(-500, 0),
    )
    np.savez_compressed(
        os.path.join(RESULTS_DIR, "agent_reward_history.npz"),
        **{f"agent_{i}": np.asarray(agent_reward_history[i], dtype=np.float32) for i in range(NUM_AGENTS)},
    )
    plot_per_agent_training_rewards(
        out_path=os.path.join(RESULTS_DIR, "training_agent_rewards.png"),
        agent_reward_history=agent_reward_history,
        model_types={i: MODEL_TYPES[i] for i in range(NUM_AGENTS)},
        window=50,
    )
    plot_per_agent_training_rewards(
        out_path=os.path.join(RESULTS_DIR, "training_agent_rewards_fixed.png"),
        agent_reward_history=agent_reward_history,
        model_types={i: MODEL_TYPES[i] for i in range(NUM_AGENTS)},
        window=50,
        ylim=(-500, 0),
    )

    total_time = time.time() - t0
    overall_succ = float(np.mean(episode_success) * 100.0) if len(episode_success) else 0.0
    print(
        "Training finished\n"
        f"  Time: {_fmt_seconds(total_time)}\n"
        f"  EnvSteps: {total_env_steps}\n"
        f"  Success: {overall_succ:.1f}% | Fail episodes: {total_fail_episodes}/{EPISODES}\n"
        f"  Models: {base}_actor_agent_*.pt and {base}_critic*_agent_*.pt\n"
        f"  Plot: {os.path.join(RESULTS_DIR, 'training_trends.png')}\n"
        f"  Plot: {os.path.join(RESULTS_DIR, 'avg_cumulative_rewards.png')}\n"
        f"  Plot: {os.path.join(RESULTS_DIR, 'training_agent_rewards.png')}\n"
        f"  Log: {log_path}\n"
    )

    if log_f is not None:
        try:
            log_f.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MATD3 Training (CTDE)")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sl", type=float, nargs="*", default=[1.0], help="List of S_l values to run.")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["simplecnn", "deepcnn", "miniresnet"],
        help="Model types per agent (length 3) or a single model to replicate for all agents.",
    )
    parser.add_argument("--max-fail-logs", type=int, default=0, help="Print up to N allocation failures per episode.")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--log-trace", action="store_true", help="Include per-step (layer->device) trace in JSONL logs.")
    parser.add_argument("--trace-max-steps", type=int, default=200, help="Max trace entries per episode when --log-trace is set.")
    args = parser.parse_args()

    for sl in (args.sl or [1.0]):
        train(
            sl=float(sl),
            episodes=int(args.episodes),
            seed=int(args.seed),
            model_types=list(args.models),
            max_fail_logs_per_episode=int(args.max_fail_logs),
            log_every=int(args.log_every),
            log_trace=bool(args.log_trace),
            trace_max_steps=int(args.trace_max_steps),
        )
