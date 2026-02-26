from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

from MultiAgentMADDPG.manager import MADDPGManager
from MultiAgentMADDPG.plots import (
    EvalEpisodeFlow,
    plot_actor_critic_losses,
    plot_avg_cumulative_rewards,
    plot_execution_flow,
    plot_per_agent_success_rate,
    plot_per_agent_training_rewards,
    plot_training_execution_strategy,
    plot_training_trends,
)

# Allow running both:
# - as a module:  python -m MultiAgentMADDPG_PrivacyWeighted.train
# - as a script:  python MultiAgentMADDPG_PrivacyWeighted/train.py
if __package__:
    from .environment import MultiAgentIoTEnvPrivacyWeighted
    from .plots import plot_privacy_violations
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from MultiAgentMADDPG_PrivacyWeighted.environment import MultiAgentIoTEnvPrivacyWeighted
    from MultiAgentMADDPG_PrivacyWeighted.plots import plot_privacy_violations


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


def _fmt_seconds(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


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


def _layer_specs(tasks: dict[int, list[object]], num_agents: int) -> dict[str, list[dict[str, float | int | str]]]:
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


def _device_summary(resource_manager) -> dict[str, dict[str, float | int]]:
    devices = {}
    for d_id, d in getattr(resource_manager, "devices", {}).items():
        devices[str(int(d_id))] = {
            "cpu_speed": float(getattr(d, "cpu_speed", 0.0)),
            "memory_capacity": float(getattr(d, "memory_capacity", 0.0)),
            "bandwidth": float(getattr(d, "bandwidth", 0.0)),
            "privacy_clearance": int(getattr(d, "privacy_clearance", 0)),
        }
    return devices


def train(
    *,
    sl: float = 1.0,
    episodes: int = 5000,
    seed: int = 42,
    model_types: list[str] | None = None,
    log_every: int = 50,
    log_trace: bool = False,
    trace_max_steps: int = 200,
    queue_per_device: bool = False,
    privacy_max_level: int = 3,
    privacy_profile: str = "linear_front_loaded",
    privacy_lambda: float = 5.0,
    trust_min_for_max_privacy: float = 0.8,
    trust_hard: bool = True,
):
    NUM_AGENTS = 3
    NUM_DEVICES = 5
    EPISODES = int(episodes)
    MODEL_TYPES = _normalize_model_types(model_types, NUM_AGENTS)

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    scenario = (
        _scenario_tag(MODEL_TYPES)
        + f"_pmax{int(privacy_max_level)}"
        + f"_tmin{float(trust_min_for_max_privacy):g}"
        + f"_lam{float(privacy_lambda):g}"
        + f"_{str(privacy_profile)}"
        + ("_hard" if bool(trust_hard) else "_soft")
    )
    SAVE_DIR = os.path.join(SCRIPT_DIR, "models", scenario, _sl_tag(float(sl)))
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", scenario, _sl_tag(float(sl)))
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    env = MultiAgentIoTEnvPrivacyWeighted(
        num_agents=NUM_AGENTS,
        num_devices=NUM_DEVICES,
        model_types=MODEL_TYPES,
        seed=int(seed),
        shuffle_allocation_order=True,
        max_exposure_fraction=float(sl),
        queue_per_device=bool(queue_per_device),
        privacy_max_level=int(privacy_max_level),
        privacy_profile=str(privacy_profile),
        privacy_lambda=float(privacy_lambda),
        trust_min_for_max_privacy=float(trust_min_for_max_privacy),
        trust_hard=bool(trust_hard),
    )

    manager = MADDPGManager(
        agent_ids=list(range(NUM_AGENTS)),
        obs_dim=env.single_state_dim,
        action_dim=env.num_devices,
        state_dim=NUM_AGENTS * env.single_state_dim,
        batch_size=256,
        shared_policy=False,
    )

    plots_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    try:
        with open(os.path.join(RESULTS_DIR, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "algorithm": "MADDPG",
                    "variant": "PrivacyWeighted",
                    "episodes": int(EPISODES),
                    "seed": int(seed),
                    "sl": float(sl),
                    "num_agents": int(NUM_AGENTS),
                    "num_devices": int(NUM_DEVICES),
                    "models": list(MODEL_TYPES),
                    "scenario": str(scenario),
                    "queue_per_device": bool(queue_per_device),
                    "privacy_max_level": int(privacy_max_level),
                    "privacy_profile": str(privacy_profile),
                    "privacy_lambda": float(privacy_lambda),
                    "trust_min_for_max_privacy": float(trust_min_for_max_privacy),
                    "trust_hard": bool(trust_hard),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    except Exception:
        pass

    try:
        with open(os.path.join(RESULTS_DIR, "model_summary.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"models": list(MODEL_TYPES), "layer_specs": _layer_specs(getattr(env, "tasks", {}) or {}, NUM_AGENTS)},
                f,
                indent=2,
                ensure_ascii=False,
            )
    except Exception:
        pass

    try:
        with open(os.path.join(RESULTS_DIR, "device_summary.json"), "w", encoding="utf-8") as f:
            device_trust = {str(k): float(v) for k, v in getattr(env, "device_trust", {}).items()}
            devices = _device_summary(env.resource_manager)
            for d_id, spec in devices.items():
                try:
                    spec["trust_score"] = float(device_trust.get(str(d_id), 0.0))
                except Exception:
                    spec["trust_score"] = 0.0
            json.dump({"devices": devices}, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    print(
        "MADDPG Training (CTDE) - PrivacyWeighted\n"
        f"  Agents: {NUM_AGENTS} | Devices: {NUM_DEVICES} | Episodes: {EPISODES}\n"
        f"  Models: {MODEL_TYPES}\n"
        f"  Scenario: {scenario}\n"
        f"  Seed: {int(seed)}\n"
        f"  S_l (max exposure fraction): {float(sl):.3f}\n"
        f"  Trust: min_for_max_privacy={float(trust_min_for_max_privacy):g} | hard={bool(trust_hard)} | privacy_max={int(privacy_max_level)} | profile={privacy_profile} | lambda={float(privacy_lambda):g}\n"
        f"  SaveDir: {SAVE_DIR}\n"
        f"  Results: {RESULTS_DIR}\n"
        f"  QueuePerDevice: {bool(queue_per_device)}\n"
    )
    print_device_info(env.resource_manager)

    episode_team_rewards: list[float] = []
    episode_team_rewards_sum: list[float] = []
    losses_total: list[float] = []
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    eps_history: list[float] = []
    episode_success: list[int] = []

    agent_reward_history = {i: [] for i in range(NUM_AGENTS)}
    agent_success_history = {i: [] for i in range(NUM_AGENTS)}
    per_agent_trust_shortfall_sum = {i: [] for i in range(NUM_AGENTS)}

    fail_reasons_history: list[Counter[str]] = []

    last_success_strategy: dict[str, dict[str, int]] | None = None
    last_success_ep: int | None = None
    last_success_flow: EvalEpisodeFlow | None = None

    log_path = os.path.join(RESULTS_DIR, "train_log.jsonl")
    try:
        log_f = open(log_path, "w", encoding="utf-8")
    except Exception:
        log_f = None
    log_write_error_printed = False

    t0 = time.time()
    total_env_steps = 0
    total_fail_episodes = 0

    for ep in range(EPISODES):
        obs, _ = env.reset()
        done = {i: False for i in range(NUM_AGENTS)}

        team_return_mean = 0.0
        team_return_sum = 0.0
        steps = 0
        ep_failed = False
        agent_ep_reward = {i: 0.0 for i in range(NUM_AGENTS)}
        agent_step_count = {i: 0 for i in range(NUM_AGENTS)}
        agent_last_step_reward = {i: 0.0 for i in range(NUM_AGENTS)}
        agent_failed = {i: False for i in range(NUM_AGENTS)}
        ep_fail_reasons: Counter[str] = Counter()
        ep_trust_shortfall = {i: 0.0 for i in range(NUM_AGENTS)}
        ep_trace: list[dict] = []
        ep_layer_device_map: dict[str, dict[str, int]] = {str(i): {} for i in range(NUM_AGENTS)}
        ep_flow_device_choices: dict[int, list[int]] = {i: [] for i in range(NUM_AGENTS)}
        ep_actor_updates: list[float] = []
        ep_critic_updates: list[float] = []

        while not all(done.values()):
            valid_actions = env.get_valid_actions()
            actions = manager.get_actions(obs, valid_actions)
            layer_idx_before = {
                aid: int(env.agent_progress.get(aid, 0)) for aid in range(NUM_AGENTS) if not done.get(aid, False)
            }
            next_obs, rewards, next_done, _truncated, infos = env.step(actions)
            steps += 1

            for aid in range(NUM_AGENTS):
                if not done.get(aid, False):
                    agent_ep_reward[aid] += float(rewards.get(aid, 0.0))
                    agent_step_count[aid] += 1
                    agent_last_step_reward[aid] = float(rewards.get(aid, 0.0))

            for aid, info in (infos or {}).items():
                if isinstance(info, dict) and info.get("success") is False:
                    agent_failed[int(aid)] = True
                    fail = info.get("fail", {}) if isinstance(info, dict) else {}
                    if isinstance(fail, dict):
                        ep_fail_reasons[str(fail.get("reason", "unknown"))] += 1
                    else:
                        ep_fail_reasons["unknown"] += 1

            for aid, info in (infos or {}).items():
                if not isinstance(info, dict) or info.get("success") is not True:
                    continue
                ep_trust_shortfall[int(aid)] += float(info.get("trust_shortfall", 0.0))
                a = int(aid)
                d = int(actions.get(a, 0))
                ep_flow_device_choices[a].append(d)
                lidx = int(layer_idx_before.get(a, int(env.agent_progress.get(a, 0)) - 1))
                ep_layer_device_map[str(a)][str(lidx)] = int(d)
                if log_trace and len(ep_trace) < int(trace_max_steps):
                    ep_trace.append(
                        {
                            "agent": int(aid),
                            "device": int(actions.get(aid, 0)),
                            "layer": int(lidx),
                            "t_comp": float(info.get("t_comp", 0.0)),
                            "t_comm": float(info.get("t_comm", 0.0)),
                            "privacy_required": int(info.get("privacy_required", 0)),
                            "privacy_clearance": int(info.get("privacy_clearance", 0)),
                            "trust_required": float(info.get("trust_required", 0.0)),
                            "trust_score": float(info.get("trust_score", 0.0)),
                            "trust_shortfall": float(info.get("trust_shortfall", 0.0)),
                            "trust_penalty": float(info.get("trust_penalty", 0.0)),
                        }
                    )

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
                losses_total.append(float(loss.get("total", 0.0)))
                actor_losses.append(float(loss.get("actor", 0.0)))
                critic_losses.append(float(loss.get("critic", 0.0)))
                ep_actor_updates.append(float(loss.get("actor", 0.0)))
                ep_critic_updates.append(float(loss.get("critic", 0.0)))

            team_r = float(sum(rewards.values())) / float(NUM_AGENTS)
            team_return_mean += team_r
            team_return_sum += float(sum(rewards.values()))

            done = next_done
            obs = next_obs

        total_env_steps += steps
        if ep_failed:
            total_fail_episodes += 1

        episode_team_rewards.append(team_return_mean / float(max(1, steps)))
        episode_team_rewards_sum.append(team_return_sum)
        episode_success.append(0 if ep_failed else 1)
        fail_reasons_history.append(ep_fail_reasons)

        for aid in range(NUM_AGENTS):
            agent_reward_history[aid].append(float(agent_ep_reward[aid]))
            finished = bool(env.agent_progress.get(aid, 0) >= len(env.tasks.get(aid, [])))
            agent_success_history[aid].append(1 if (not agent_failed[aid] and finished) else 0)
            per_agent_trust_shortfall_sum[aid].append(float(ep_trust_shortfall[aid]))

        if all(bool(agent_success_history[aid][-1]) for aid in range(NUM_AGENTS)):
            last_success_strategy = {a: dict(m) for a, m in ep_layer_device_map.items()}
            last_success_ep = int(ep)
            last_success_flow = EvalEpisodeFlow(
                agent_ids=list(range(NUM_AGENTS)),
                device_choices={i: list(ep_flow_device_choices[i]) for i in range(NUM_AGENTS)},
                fail_step=None,
                fail_agent=None,
                model_types={i: MODEL_TYPES[i] for i in range(NUM_AGENTS)},
            )

        eps_now = float(manager._unique_agents()[0].eps.epsilon)
        eps_history.append(eps_now)
        ep_actor_mean = float(np.mean(ep_actor_updates)) if ep_actor_updates else None
        ep_critic_mean = float(np.mean(ep_critic_updates)) if ep_critic_updates else None
        stats_window = 50
        agent_stats = []
        for aid in range(NUM_AGENTS):
            a_reward = float(np.mean(agent_reward_history[aid][-stats_window:])) if agent_reward_history[aid] else 0.0
            a_success = float(np.mean(agent_success_history[aid][-stats_window:]) * 100.0) if agent_success_history[aid] else 0.0
            agent_stats.append(f"A{aid}: {a_reward:.1f} ({a_success:.0f}%)")
        agent_stats_str = "[" + " | ".join(agent_stats) + "]"

        if log_f is not None:
            try:
                log_f.write(
                    json.dumps(
                        {
                            "episode": int(ep),
                            "seed": int(seed),
                            "sl": float(sl),
                            "scenario": str(scenario),
                            "models": list(MODEL_TYPES),
                            "queue_per_device": bool(queue_per_device),
                            "privacy_max_level": int(privacy_max_level),
                            "privacy_profile": str(privacy_profile),
                            "privacy_lambda": float(privacy_lambda),
                            "trust_min_for_max_privacy": float(trust_min_for_max_privacy),
                            "trust_hard": bool(trust_hard),
                            "device_trust": {str(k): float(v) for k, v in getattr(env, "device_trust", {}).items()},
                            "agent_stats_window": int(stats_window),
                            "agent_stats": str(agent_stats_str),
                            "team_reward_sum": float(team_return_sum),
                            "team_reward_mean_step": float(team_return_mean / float(max(1, steps))),
                            "steps": int(steps),
                            "fail_reasons": dict(ep_fail_reasons),
                            "per_agent_reward_sum": {str(a): float(agent_ep_reward[a]) for a in range(NUM_AGENTS)},
                            "per_agent_reward_mean_step": {
                                str(a): float(agent_ep_reward[a]) / float(max(1, int(agent_step_count[a])))
                                for a in range(NUM_AGENTS)
                            },
                            "per_agent_last_step_reward": {str(a): float(agent_last_step_reward[a]) for a in range(NUM_AGENTS)},
                            "per_agent_steps": {str(a): int(agent_step_count[a]) for a in range(NUM_AGENTS)},
                            "per_agent_failed": {str(a): bool(agent_failed[a]) for a in range(NUM_AGENTS)},
                            "trust_shortfall_sum": {str(a): float(ep_trust_shortfall[a]) for a in range(NUM_AGENTS)},
                            "actor_loss": ep_actor_mean,
                            "critic_loss": ep_critic_mean,
                            "layer_device_map": ep_layer_device_map,
                            "trace": ep_trace if log_trace else None,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                log_f.flush()
            except Exception as e:
                if not log_write_error_printed:
                    log_write_error_printed = True
                    print(f"[LOG_WRITE_ERROR] Failed writing {log_path}: {e}", file=sys.stderr)
                raise

        if (ep + 1) % max(1, int(log_every)) == 0:
            avg = float(np.mean(episode_team_rewards_sum[-50:]))
            avg_loss = float(np.mean(losses_total[-200:])) if losses_total else float("nan")
            elapsed = time.time() - t0
            eps_per_sec = (ep + 1) / max(elapsed, 1e-9)
            eta = (EPISODES - (ep + 1)) / max(eps_per_sec, 1e-9)
            recent_fail = Counter()
            for c in fail_reasons_history[-50:]:
                recent_fail.update(c)
            top_fail = ", ".join([f"{k}={v}" for k, v in recent_fail.most_common(3)]) if recent_fail else "none"
            print(
                f"Ep {ep+1:4d}/{EPISODES} - AvgSum(50): {avg:.1f} - Eps: {eps_now:.3f} - Loss: {avg_loss:.4f} "
                f"- Elapsed: {_fmt_seconds(elapsed)} - ETA: {_fmt_seconds(eta)} - FailReasons(50): {top_fail}"
            )

    if log_f is not None:
        try:
            log_f.close()
        except Exception:
            pass

    np.save(os.path.join(RESULTS_DIR, "team_reward_history.npy"), np.asarray(episode_team_rewards, dtype=np.float32))
    np.save(os.path.join(RESULTS_DIR, "team_reward_sum_history.npy"), np.asarray(episode_team_rewards_sum, dtype=np.float32))
    np.save(os.path.join(RESULTS_DIR, "loss_history.npy"), np.asarray(losses_total, dtype=np.float32))
    np.save(os.path.join(RESULTS_DIR, "actor_loss_history.npy"), np.asarray(actor_losses, dtype=np.float32))
    np.save(os.path.join(RESULTS_DIR, "critic_loss_history.npy"), np.asarray(critic_losses, dtype=np.float32))
    np.save(os.path.join(RESULTS_DIR, "epsilon_history.npy"), np.asarray(eps_history, dtype=np.float32))
    np.save(os.path.join(RESULTS_DIR, "episode_success.npy"), np.asarray(episode_success, dtype=np.int32))
    np.savez_compressed(
        os.path.join(RESULTS_DIR, "agent_trust_shortfall_sum.npz"),
        **{f"agent_{i}": np.asarray(per_agent_trust_shortfall_sum[i], dtype=np.float32) for i in range(NUM_AGENTS)},
    )

    plot_training_trends(
        out_path=os.path.join(plots_dir, "training_trends.png"),
        team_rewards=episode_team_rewards,
        losses=losses_total,
        eps_history=None,
        window=50,
    )
    plot_actor_critic_losses(
        out_path=os.path.join(plots_dir, "policy_value_losses.png"),
        actor_losses=actor_losses,
        critic_losses=critic_losses,
        window=200,
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
    plot_per_agent_success_rate(
        out_path=os.path.join(plots_dir, "training_agent_success_rate.png"),
        agent_success_history=agent_success_history,
        window=50,
    )
    plot_privacy_violations(
        out_path=os.path.join(plots_dir, "trust_shortfall.png"),
        per_agent_violation_sum=per_agent_trust_shortfall_sum,
        window=50,
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

    overall_succ = float(np.mean(episode_success) * 100.0) if episode_success else 0.0
    per_agent_summary = {}
    for aid in range(NUM_AGENTS):
        per_agent_summary[str(aid)] = {
            "success_rate": float(np.mean(agent_success_history[aid][-100:]) * 100.0) if agent_success_history[aid] else 0.0,
            "avg_trust_shortfall_sum_last_100": float(np.mean(per_agent_trust_shortfall_sum[aid][-100:])) if per_agent_trust_shortfall_sum[aid] else 0.0,
        }
    try:
        with open(os.path.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "episodes": int(EPISODES),
                    "seed": int(seed),
                    "sl": float(sl),
                    "scenario": str(scenario),
                    "models": list(MODEL_TYPES),
                    "variant": "PrivacyWeighted",
                    "privacy_max_level": int(privacy_max_level),
                    "privacy_profile": str(privacy_profile),
                    "privacy_lambda": float(privacy_lambda),
                    "trust_min_for_max_privacy": float(trust_min_for_max_privacy),
                    "trust_hard": bool(trust_hard),
                    "overall_success_rate": float(overall_succ),
                    "env_steps": int(total_env_steps),
                    "fail_episodes": int(total_fail_episodes),
                    "per_agent_summary": per_agent_summary,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MADDPG Training (CTDE) - PrivacyWeighted variant")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sl", type=float, nargs="*", default=[1.0], help="List of S_l values to run.")
    parser.add_argument("--models", type=str, nargs="+", default=["hugcnn", "cnn15", "simplecnn"], help="List of model types for agents. Length 1 to apply to all, or length=num_agents.")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--log-trace", action="store_true")
    parser.add_argument("--trace-max-steps", type=int, default=200)
    parser.add_argument("--queue-per-device", action="store_true")
    parser.add_argument("--privacy-max-level", type=int, default=3)
    parser.add_argument("--privacy-profile", type=str, default="linear_front_loaded")
    parser.add_argument("--privacy-lambda", type=float, default=5.0)
    parser.add_argument(
        "--trust-min-for-max-privacy",
        type=float,
        default=0.8,
        help="Minimal trust required when privacy_level == privacy_max_level (0..1).",
    )
    parser.add_argument(
        "--trust-soft",
        action="store_true",
        help="Use soft trust penalty instead of hard trust constraint.",
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
            queue_per_device=bool(args.queue_per_device),
            privacy_max_level=int(args.privacy_max_level),
            privacy_profile=str(args.privacy_profile),
            privacy_lambda=float(args.privacy_lambda),
            trust_min_for_max_privacy=float(args.trust_min_for_max_privacy),
            trust_hard=not bool(args.trust_soft),
        )
