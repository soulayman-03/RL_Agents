from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

# Allow running both:
# - as a module:  python -m MultiAgentMADDPG_Energy.train
# - as a script:  python MultiAgentMADDPG_Energy/train.py
if __package__:
    from .environment import MultiAgentIoTEnvEnergyHard
    from .manager import MADDPGManager
    from .plots import (
        EvalEpisodeFlow,
        plot_avg_cumulative_rewards,
        plot_execution_flow,
        plot_marl_eval_summary,
        plot_per_agent_layer_latency,
        plot_per_agent_training_rewards,
        plot_training_execution_strategy,
        plot_training_trends,
    )
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from MultiAgentMADDPG_Energy.environment import MultiAgentIoTEnvEnergyHard
    from MultiAgentMADDPG_Energy.manager import MADDPGManager
    from MultiAgentMADDPG_Energy.plots import (
        EvalEpisodeFlow,
        plot_avg_cumulative_rewards,
        plot_execution_flow,
        plot_marl_eval_summary,
        plot_per_agent_layer_latency,
        plot_per_agent_training_rewards,
        plot_training_execution_strategy,
        plot_training_trends,
    )


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


def _model_stats(tasks: dict[int, list[object]], num_agents: int) -> dict[str, dict[str, float]]:
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


def _layer_specs(tasks: dict[int, list[object]], num_agents: int) -> dict[str, list[dict[str, float | int | str]]]:
    """
    Per-agent, per-layer requirements (compute/memory/output/privacy).
    Kept JSON-serializable and stable across episodes for a given seed/model_types.
    """
    out: dict[str, list[dict[str, float | int | str]]] = {}
    for aid in range(int(num_agents)):
        layers = list(tasks.get(aid, []) or [])
        specs: list[dict[str, float | int | str]] = []
        for idx, l in enumerate(layers):
            specs.append(
                {
                    "idx": int(idx),
                    "name": str(getattr(l, "name", f"layer_{idx}")),
                    "compute": float(getattr(l, "computation_demand", 0.0)),
                    "memory": float(getattr(l, "memory_demand", 0.0)),
                    "output": float(getattr(l, "output_data_size", 0.0)),
                    "privacy": int(getattr(l, "privacy_level", 0)),
                }
            )
        out[str(aid)] = specs
    return out


def _fmt_seconds(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def train(
    *,
    sl: float = 1.0,
    episodes: int = 5000,
    seed: int = 42,
    model_types: list[str] | None = None,
    log_every: int = 50,
    log_trace: bool = False,
    trace_max_steps: int = 200,
    max_fail_logs_per_episode: int = 0,
    energy_min: float = 500.0,
    energy_max: float = 1200.0,
    alpha_comp: float = 1.0,
    alpha_comm: float = 1.0,
    log_layer_specs: bool = False,
):
    NUM_AGENTS = 3
    NUM_DEVICES = 5
    EPISODES = int(episodes)
    MODEL_TYPES = _normalize_model_types(model_types, NUM_AGENTS)
    TERMINATE_ON_FAIL = True

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    scenario = _scenario_tag(MODEL_TYPES)
    energy_tag = f"energy_{float(energy_min):.0f}_{float(energy_max):.0f}_ac{float(alpha_comp):g}_am{float(alpha_comm):g}".replace(
        ".", "p"
    )
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", scenario, _sl_tag(sl), f"seed_{int(seed)}", energy_tag)
    SAVE_DIR = os.path.join(SCRIPT_DIR, "models", scenario, _sl_tag(sl), f"seed_{int(seed)}", energy_tag)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    env = MultiAgentIoTEnvEnergyHard(
        num_agents=NUM_AGENTS,
        num_devices=NUM_DEVICES,
        model_types=MODEL_TYPES,
        seed=int(seed),
        shuffle_allocation_order=True,
        max_exposure_fraction=float(sl),
        energy_budget_range=(float(energy_min), float(energy_max)),
        alpha_comp=float(alpha_comp),
        alpha_comm=float(alpha_comm),
        max_fail_logs_per_episode=int(max_fail_logs_per_episode),
    )

    manager = MADDPGManager(
        agent_ids=list(range(NUM_AGENTS)),
        obs_dim=env.single_state_dim,
        action_dim=env.num_devices,
        state_dim=NUM_AGENTS * env.single_state_dim,
        batch_size=256,
        shared_policy=False,
    )

    run_cfg = {
        "algo": "MADDPG",
        "variant": "EnergyHard",
        "seed": int(seed),
        "episodes": int(EPISODES),
        "sl": float(sl),
        "models": list(MODEL_TYPES),
        "num_agents": int(NUM_AGENTS),
        "num_devices": int(NUM_DEVICES),
        "energy_budget_range": [float(energy_min), float(energy_max)],
        "alpha_comp": float(alpha_comp),
        "alpha_comm": float(alpha_comm),
        "terminate_on_fail": bool(TERMINATE_ON_FAIL),
    }
    try:
        with open(os.path.join(RESULTS_DIR, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(run_cfg, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    episode_team_rewards: list[float] = []
    episode_team_rewards_sum: list[float] = []
    losses: list[float] = []
    eps_history: list[float] = []
    episode_steps: list[int] = []
    episode_success: list[int] = []
    agent_reward_history: dict[int, list[float]] = {i: [] for i in range(NUM_AGENTS)}
    agent_success_history: dict[int, list[int]] = {i: [] for i in range(NUM_AGENTS)}
    total_env_steps = 0
    total_fail_episodes = 0
    fail_reasons_history: list[Counter[str]] = []

    per_agent_comp_sum_hist: dict[int, list[float]] = {i: [] for i in range(NUM_AGENTS)}
    per_agent_comm_sum_hist: dict[int, list[float]] = {i: [] for i in range(NUM_AGENTS)}
    per_agent_step_count_hist: dict[int, list[int]] = {i: [] for i in range(NUM_AGENTS)}
    per_agent_device_counts_hist: dict[int, list[Counter[int]]] = {i: [] for i in range(NUM_AGENTS)}
    latency_traces_by_agent: dict[int, list[list[dict]]] = {i: [] for i in range(NUM_AGENTS)}

    plots_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    log_path = os.path.join(RESULTS_DIR, "train_log.jsonl")
    try:
        log_f = open(log_path, "w", encoding="utf-8")
    except Exception:
        log_f = None

    model_stats = _model_stats(getattr(env, "tasks", {}) or {}, NUM_AGENTS)
    layer_specs = _layer_specs(getattr(env, "tasks", {}) or {}, NUM_AGENTS)
    print(
        "MADDPG Training (CTDE) - EnergyHard\n"
        f"  Agents: {NUM_AGENTS} | Devices: {NUM_DEVICES} | Episodes: {EPISODES}\n"
        f"  Models: {MODEL_TYPES}\n"
        f"  Scenario: {scenario}\n"
        f"  Seed: {int(seed)}\n"
        f"  S_l (max exposure fraction): {float(sl):.3f}\n"
        f"  EnergyBudget: [{float(energy_min):.1f}, {float(energy_max):.1f}] | alpha_comp={float(alpha_comp):g} | alpha_comm={float(alpha_comm):g}\n"
        f"  SaveDir: {SAVE_DIR}\n"
        f"  Results: {RESULTS_DIR}\n"
        f"  TerminateOnFail: {TERMINATE_ON_FAIL}\n"
        f"  DeviceEnergyInit: {getattr(env, 'device_energy_init', {})}\n"
        f"  ModelStats: {model_stats}\n"
    )

    last_success_strategy: dict[str, dict[str, int]] | None = None
    last_success_ep: int | None = None
    last_success_flow: EvalEpisodeFlow | None = None

    t0 = time.time()
    for ep in range(EPISODES):
        obs, _ = env.reset()
        done = {i: False for i in range(NUM_AGENTS)}

        team_return_mean = 0.0
        team_return_sum = 0.0
        steps = 0
        ep_failed = False
        agent_ep_reward = {i: 0.0 for i in range(NUM_AGENTS)}
        agent_failed = {i: False for i in range(NUM_AGENTS)}
        ep_fail_reasons: Counter[str] = Counter()
        ep_trace: list[dict] = []
        ep_layer_device_map: dict[str, dict[str, int]] = {str(i): {} for i in range(NUM_AGENTS)}
        ep_device_counts: dict[int, Counter[int]] = {i: Counter() for i in range(NUM_AGENTS)}
        ep_t_comp_sum: dict[int, float] = {i: 0.0 for i in range(NUM_AGENTS)}
        ep_t_comm_sum: dict[int, float] = {i: 0.0 for i in range(NUM_AGENTS)}
        ep_step_count: dict[int, int] = {i: 0 for i in range(NUM_AGENTS)}
        ep_flow_device_choices: dict[int, list[int]] = {i: [] for i in range(NUM_AGENTS)}
        ep_latency_trace: dict[int, list[dict]] = {i: [] for i in range(NUM_AGENTS)}
        ep_fail_step: int | None = None
        ep_fail_agent: int | None = None

        while not all(done.values()):
            valid_actions = env.get_valid_actions()
            actions = manager.get_actions(obs, valid_actions)
            layer_idx_before = {
                aid: int(env.agent_progress.get(aid, 0)) for aid in range(NUM_AGENTS) if not done.get(aid, False)
            }

            next_obs, rewards, next_done, _truncated, infos = env.step(actions)
            steps += 1
            total_env_steps += 1

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
                    if ep_fail_step is None:
                        ep_fail_step = int(steps - 1)
                        ep_fail_agent = int(aid)

            for aid in range(NUM_AGENTS):
                if done.get(aid, False):
                    continue
                info = infos.get(aid, {}) if isinstance(infos, dict) else {}
                if not isinstance(info, dict):
                    continue
                dev = int(actions.get(aid, 0))
                ep_device_counts[aid][dev] += 1
                ep_flow_device_choices[aid].append(dev)
                ep_step_count[aid] += 1
                if bool(info.get("success", True)) is True:
                    ep_t_comp_sum[aid] += float(info.get("t_comp", 0.0))
                    ep_t_comm_sum[aid] += float(info.get("t_comm", 0.0))
                if bool(info.get("success", True)) is True and aid in layer_idx_before:
                    layer_idx = int(layer_idx_before[aid])
                    ep_layer_device_map[str(aid)][str(layer_idx)] = dev
                    ep_latency_trace[aid].append(
                        {"layer": int(layer_idx), "t_comp": float(info.get("t_comp", 0.0)), "t_comm": float(info.get("t_comm", 0.0))}
                    )

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
                            "energy_cost": float(info.get("energy_cost", 0.0)) if isinstance(info, dict) else 0.0,
                            "energy_remaining": float(info.get("energy_remaining", 0.0)) if isinstance(info, dict) else 0.0,
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
        if ep_failed:
            total_fail_episodes += 1

        for aid in range(NUM_AGENTS):
            agent_reward_history[aid].append(float(agent_ep_reward[aid]))
            agent_success_history[aid].append(0 if agent_failed[aid] else 1)
            per_agent_comp_sum_hist[aid].append(float(ep_t_comp_sum[aid]))
            per_agent_comm_sum_hist[aid].append(float(ep_t_comm_sum[aid]))
            per_agent_step_count_hist[aid].append(int(ep_step_count[aid]))
            per_agent_device_counts_hist[aid].append(ep_device_counts[aid])
            latency_traces_by_agent[aid].append(ep_latency_trace[aid])

        eps_now = float(manager.agents[0].eps.epsilon) if 0 in manager.agents else float("nan")
        eps_history.append(eps_now)

        if not ep_failed:
            last_success_strategy = ep_layer_device_map
            last_success_ep = int(ep + 1)
            last_success_flow = EvalEpisodeFlow(
                agent_ids=list(range(NUM_AGENTS)),
                device_choices={i: list(ep_flow_device_choices[i]) for i in range(NUM_AGENTS)},
                fail_step=ep_fail_step,
                fail_agent=ep_fail_agent,
                model_types={i: MODEL_TYPES[i] for i in range(NUM_AGENTS)},
            )

        if log_f is not None:
            try:
                device_energy_init = {str(k): float(v) for k, v in getattr(env, "device_energy_init", {}).items()}
                device_energy_remaining = {
                    str(k): float(v) for k, v in getattr(env, "device_energy_remaining", {}).items()
                }
                energy_spent = {
                    k: float(device_energy_init.get(k, 0.0)) - float(device_energy_remaining.get(k, 0.0))
                    for k in device_energy_init.keys()
                }
                energy_spent_total = float(sum(energy_spent.values()))
                remaining_ratios = []
                for k, init_v in device_energy_init.items():
                    init_v = float(init_v)
                    rem_v = float(device_energy_remaining.get(k, 0.0))
                    remaining_ratios.append(rem_v / init_v if init_v > 0 else 0.0)
                min_remaining_ratio = float(min(remaining_ratios)) if remaining_ratios else 0.0

                device_counts = {str(a): {str(k): int(v) for k, v in ep_device_counts[a].items()} for a in range(NUM_AGENTS)}
                log_f.write(
                    json.dumps(
                        {
                            "episode": int(ep + 1),
                            "seed": int(seed),
                            "sl": float(sl),
                            "energy_budget_range": [float(energy_min), float(energy_max)],
                            "alpha_comp": float(alpha_comp),
                            "alpha_comm": float(alpha_comm),
                            "models": list(MODEL_TYPES),
                            "model_stats": model_stats,
                            # Log layer requirements once (episode 1) by default to avoid large JSONL files.
                            "layer_specs": layer_specs if (bool(log_layer_specs) and int(ep + 1) == 1) else None,
                            "team_reward_sum": float(team_return_sum),
                            "team_reward_mean_step": float(team_return_mean / float(finished_steps)),
                            "steps": int(steps),
                            "ep_failed": bool(ep_failed),
                            "fail_reasons": dict(ep_fail_reasons),
                            "per_agent_reward_sum": {str(a): float(agent_ep_reward[a]) for a in range(NUM_AGENTS)},
                            "per_agent_failed": {str(a): bool(agent_failed[a]) for a in range(NUM_AGENTS)},
                            "device_counts": device_counts,
                            "layer_device_map": ep_layer_device_map,
                            "device_energy_init": device_energy_init,
                            "device_energy_remaining": device_energy_remaining,
                            "energy_spent": energy_spent,
                            "energy_spent_total": float(energy_spent_total),
                            "min_energy_remaining_ratio": float(min_remaining_ratio),
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
                f"Ep {ep+1:4d}/{EPISODES} - Avg: {avg:.1f} - [{stats_str}] - Eps: {eps_now:.3f} "
                f"- Loss: {avg_loss:.4f} - Steps(50): {avg_steps:.1f} - Replay: {replay_len} "
                f"- Elapsed: {_fmt_seconds(elapsed)} - ETA: {_fmt_seconds(eta)} - FailReasons(50): {top_fail}"
            )

    base = os.path.join(SAVE_DIR, "maddpg")
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
    np.savez_compressed(
        os.path.join(RESULTS_DIR, "agent_reward_history.npz"),
        **{f"agent_{i}": np.asarray(agent_reward_history[i], dtype=np.float32) for i in range(NUM_AGENTS)},
    )

    plot_training_trends(
        out_path=os.path.join(plots_dir, "training_trends.png"),
        team_rewards=episode_team_rewards,
        losses=losses,
        eps_history=None,
        window=50,
    )
    plot_avg_cumulative_rewards(
        out_path=os.path.join(plots_dir, "avg_cumulative_rewards.png"),
        episode_team_reward_sums=episode_team_rewards_sum,
        num_agents=NUM_AGENTS,
        window=50,
    )

    plot_per_agent_training_rewards(
        out_path=os.path.join(plots_dir, "training_agent_rewards.png"),
        agent_reward_history=agent_reward_history,
        model_types={i: MODEL_TYPES[i] for i in range(NUM_AGENTS)},
        window=50,
    )
    plot_per_agent_training_rewards(
        out_path=os.path.join(plots_dir, "training_agent_rewards_fixed.png"),
        agent_reward_history=agent_reward_history,
        model_types={i: MODEL_TYPES[i] for i in range(NUM_AGENTS)},
        window=50,
        ylim=(-500, 0),
    )
    if last_success_strategy:
        plot_training_execution_strategy(
            out_path=os.path.join(plots_dir, "execution_strategy.png"),
            layer_device_map=last_success_strategy,
            model_types={i: MODEL_TYPES[i] for i in range(NUM_AGENTS)},
            num_devices=NUM_DEVICES,
            title=f"Execution Strategy (episode {last_success_ep})",
        )
    if last_success_flow is not None:
        plot_execution_flow(out_path=os.path.join(plots_dir, "execution_flow.png"), flow=last_success_flow)

    last_k = 100
    per_agent_summary: dict[int, dict] = {}
    for aid in range(NUM_AGENTS):
        succ = float(np.mean(agent_success_history[aid][-last_k:]) * 100.0) if agent_success_history[aid] else 0.0
        comp_sum = float(np.sum(per_agent_comp_sum_hist[aid][-last_k:])) if per_agent_comp_sum_hist[aid] else 0.0
        comm_sum = float(np.sum(per_agent_comm_sum_hist[aid][-last_k:])) if per_agent_comm_sum_hist[aid] else 0.0
        steps_sum = int(np.sum(per_agent_step_count_hist[aid][-last_k:])) if per_agent_step_count_hist[aid] else 0
        avg_comp = comp_sum / max(1, steps_sum)
        avg_comm = comm_sum / max(1, steps_sum)
        dc = Counter()
        for c in per_agent_device_counts_hist[aid][-last_k:]:
            dc.update(c)
        per_agent_summary[aid] = {
            "success_rate": succ,
            "avg_t_comp": avg_comp,
            "avg_t_comm": avg_comm,
            "device_counts": {str(k): int(v) for k, v in dc.items()},
            "model_type": MODEL_TYPES[aid],
        }
    plot_marl_eval_summary(out_path=os.path.join(plots_dir, "training_marl_summary.png"), per_agent=per_agent_summary)
    plot_per_agent_layer_latency(
        out_path=os.path.join(plots_dir, "layer_latency_breakdown.png"),
        per_agent_traces=latency_traces_by_agent,
        per_agent_tasks={i: env.tasks.get(i, []) for i in range(NUM_AGENTS)},
        model_types={i: MODEL_TYPES[i] for i in range(NUM_AGENTS)},
        title="Average Layer-wise Latency Breakdown (training)",
    )

    total_time = time.time() - t0
    overall_succ = float(np.mean(episode_success) * 100.0) if len(episode_success) else 0.0
    print(
        "Training finished\n"
        f"  Time: {_fmt_seconds(total_time)}\n"
        f"  EnvSteps: {total_env_steps}\n"
        f"  Success: {overall_succ:.1f}% | Fail episodes: {total_fail_episodes}/{EPISODES}\n"
        f"  Models: {base}_actor_agent_*.pt and {base}_critic_agent_*.pt\n"
        f"  PlotDir: {plots_dir}\n"
        f"  Log: {log_path}\n"
    )

    if log_f is not None:
        try:
            log_f.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MADDPG Training (CTDE) - EnergyHard variant")
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
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--log-trace", action="store_true", help="Include per-step trace in JSONL logs.")
    parser.add_argument("--trace-max-steps", type=int, default=200)
    parser.add_argument("--max-fail-logs", type=int, default=0)
    parser.add_argument("--energy-min", type=float, default=500.0)
    parser.add_argument("--energy-max", type=float, default=1200.0)
    parser.add_argument("--alpha-comp", type=float, default=1.0)
    parser.add_argument("--alpha-comm", type=float, default=1.0)
    parser.add_argument(
        "--log-layer-specs",
        action="store_true",
        help="Include per-layer requirements (compute/memory/output/privacy) in episode 1 log line.",
    )
    args = parser.parse_args()

    for sl in (args.sl or [1.0]):
        train(
            sl=float(sl),
            episodes=int(args.episodes),
            seed=int(args.seed),
            model_types=list(args.models),
            log_every=int(args.log_every),
            log_trace=bool(args.log_trace),
            trace_max_steps=int(args.trace_max_steps),
            max_fail_logs_per_episode=int(args.max_fail_logs),
            energy_min=float(args.energy_min),
            energy_max=float(args.energy_max),
            alpha_comp=float(args.alpha_comp),
            alpha_comm=float(args.alpha_comm),
            log_layer_specs=bool(args.log_layer_specs),
        )
