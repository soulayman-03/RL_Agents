import os
import sys

import numpy as np
import json
from collections import defaultdict, Counter
import time

# Allow running both:
# - as a module:  python -m MultiAgentVDN.evaluate
# - as a script:  python MultiAgentVDN/evaluate.py
if __package__:
    from .environment import MultiAgentIoTEnv
    from .manager import VDNManager
    from .plots import (
        EvalEpisodeFlow,
        plot_evaluation_summary,
        plot_execution_flow,
        plot_marl_eval_summary,
    )
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from MultiAgentVDN.environment import MultiAgentIoTEnv
    from MultiAgentVDN.manager import VDNManager
    from MultiAgentVDN.plots import (
        EvalEpisodeFlow,
        plot_evaluation_summary,
        plot_execution_flow,
        plot_marl_eval_summary,
    )


def evaluate(episodes: int = 50):
    NUM_AGENTS = 3
    NUM_DEVICES = 5
    MODEL_TYPES = ["simplecnn", "deepcnn", "miniresnet"]

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    base = os.path.join(SCRIPT_DIR, "models", "vdn")

    def _fmt_seconds(sec: float) -> str:
        sec = max(0.0, float(sec))
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    manager = VDNManager(
        agent_ids=list(range(NUM_AGENTS)),
        obs_dim=MultiAgentIoTEnv(
            num_agents=NUM_AGENTS,
            num_devices=NUM_DEVICES,
            model_types=MODEL_TYPES,
            seed=0,
            shuffle_allocation_order=True,
        ).single_state_dim,
        action_dim=NUM_DEVICES,
        shared_policy=False,
    )
    manager.load(base)

    # Greedy evaluation
    for a in manager._unique_agents():
        a.eps.epsilon = 0.0

    t0 = time.time()
    print(
        "VDN Evaluation (greedy)\n"
        f"  Episodes: {episodes}\n"
        f"  Models: {base}_agent_*.pt\n"
        f"  ModelsTypes: {MODEL_TYPES}\n"
        f"  ShuffleAllocationOrder: {True}\n"
        f"  SeedBase: 10000\n"
    )

    returns: list[float] = []
    success_rate: list[float] = []
    episode_steps: list[int] = []

    per_agent_success_eps = {i: 0 for i in range(NUM_AGENTS)}
    per_agent_steps = {i: 0 for i in range(NUM_AGENTS)}
    per_agent_t_comp = defaultdict(float)
    per_agent_t_comm = defaultdict(float)
    per_agent_device_counts: dict[int, Counter] = {i: Counter() for i in range(NUM_AGENTS)}
    per_agent_fail_reasons: dict[int, Counter] = {i: Counter() for i in range(NUM_AGENTS)}

    # For execution_flow plot (record the first evaluation episode)
    flow_device_choices: dict[int, list[int]] = {i: [] for i in range(NUM_AGENTS)}
    flow_fail_step: int | None = None
    flow_fail_agent: int | None = None

    for ep in range(int(episodes)):
        # Use a fresh env per episode to vary seeds and avoid identical rollouts.
        env = MultiAgentIoTEnv(
            num_agents=NUM_AGENTS,
            num_devices=NUM_DEVICES,
            model_types=MODEL_TYPES,
            seed=10_000 + ep,
            shuffle_allocation_order=True,
        )
        obs, _ = env.reset()
        done = {i: False for i in range(NUM_AGENTS)}

        team_return = 0.0
        team_success = True
        steps = 0
        agent_failed = {i: False for i in range(NUM_AGENTS)}

        while not all(done.values()):
            valid_actions = env.get_valid_actions()
            actions = manager.get_actions(obs, valid_actions)
            next_obs, rewards, next_done, truncated, infos = env.step(actions)
            team_return += float(sum(rewards.values())) / float(NUM_AGENTS)
            steps += 1

            # Accounting: chosen devices
            for aid, dev in actions.items():
                if not done.get(aid, False):
                    per_agent_device_counts[int(aid)][int(dev)] += 1
                    if ep == 0:
                        flow_device_choices[int(aid)].append(int(dev))

            # Accounting: latency components and failures
            any_fail = False
            for aid, info in infos.items():
                if not isinstance(info, dict):
                    continue
                if info.get("reward_type") == "success":
                    per_agent_steps[int(aid)] += 1
                    per_agent_t_comp[int(aid)] += float(info.get("t_comp", 0.0))
                    per_agent_t_comm[int(aid)] += float(info.get("t_comm", 0.0))
                if info.get("success") is False:
                    any_fail = True
                    agent_failed[int(aid)] = True
                    fail = info.get("fail", {}) or {}
                    if isinstance(fail, dict):
                        reason = fail.get("reason", "unknown")
                    else:
                        reason = "unknown"
                    per_agent_fail_reasons[int(aid)][str(reason)] += 1
                    if ep == 0 and flow_fail_step is None:
                        flow_fail_step = max(0, steps - 1)
                        flow_fail_agent = int(aid)

            if any_fail:
                team_success = False
                done = {i: True for i in range(NUM_AGENTS)}
            else:
                done = next_done

            obs = next_obs

        returns.append(team_return)
        success_rate.append(1.0 if team_success else 0.0)
        episode_steps.append(steps)

        for aid in range(NUM_AGENTS):
            if not agent_failed[aid]:
                per_agent_success_eps[aid] += 1

        if (ep + 1) % 10 == 0 or (ep + 1) == int(episodes):
            elapsed = time.time() - t0
            r_mean = float(np.mean(returns)) if len(returns) else 0.0
            sr = float(np.mean(success_rate) * 100.0) if len(success_rate) else 0.0
            steps_mean = float(np.mean(episode_steps)) if len(episode_steps) else 0.0
            eps_per_sec = (ep + 1) / max(elapsed, 1e-9)
            eta = (int(episodes) - (ep + 1)) / max(eps_per_sec, 1e-9)
            print(
                f"Eval {ep+1:3d}/{int(episodes)} | "
                f"ReturnMean: {r_mean:7.1f} | "
                f"TeamSucc: {sr:5.1f}% | "
                f"StepsMean: {steps_mean:5.1f} | "
                f"Elapsed: {_fmt_seconds(elapsed)} | ETA: {_fmt_seconds(eta)}"
            )

    total_time = time.time() - t0
    print(f"Episodes: {episodes}")
    print(f"Team return: mean={np.mean(returns):.1f} std={np.std(returns):.1f}")
    print(f"All-agents success: {np.mean(success_rate)*100:.1f}%")
    print(f"Eval time: {_fmt_seconds(total_time)}")

    # Build marl_eval_report.json similar to MultiAgent/results/marl_eval_report.json
    per_agent_report: dict[int, dict] = {}
    for aid in range(NUM_AGENTS):
        ep_count = int(episodes)
        succ = float(per_agent_success_eps[aid]) / float(ep_count) * 100.0 if ep_count else 0.0
        steps_count = max(1, int(per_agent_steps[aid]))
        avg_t_comp = float(per_agent_t_comp[aid]) / float(steps_count)
        avg_t_comm = float(per_agent_t_comm[aid]) / float(steps_count)
        per_agent_report[aid] = {
            "model_type": MODEL_TYPES[aid] if aid < len(MODEL_TYPES) else "",
            "episodes": ep_count,
            "success_rate": succ,
            "avg_t_comp": avg_t_comp,
            "avg_t_comm": avg_t_comm,
            "avg_latency": avg_t_comp + avg_t_comm,
            "device_counts": {str(k): int(v) for k, v in per_agent_device_counts[aid].items()},
            "fail_reasons": dict(per_agent_fail_reasons[aid]),
        }

    report = {
        "algo": "VDN",
        "episodes": int(episodes),
        "shuffle_allocation_order": True,
        "seed_base": 10_000,
        "per_agent": {str(k): v for k, v in per_agent_report.items()},
        "team": {
            "success_rate": float(np.mean(success_rate) * 100.0) if len(success_rate) else 0.0,
            "return_mean": float(np.mean(returns)) if len(returns) else 0.0,
            "return_std": float(np.std(returns)) if len(returns) else 0.0,
            "episode_steps_mean": float(np.mean(episode_steps)) if len(episode_steps) else 0.0,
        },
    }

    with open(os.path.join(RESULTS_DIR, "marl_eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Plots
    plot_evaluation_summary(
        out_path=os.path.join(RESULTS_DIR, "evaluation_summary.png"),
        team_returns=returns,
        team_success=success_rate,
        episode_steps=episode_steps,
    )
    plot_marl_eval_summary(
        out_path=os.path.join(RESULTS_DIR, "marl_eval_summary.png"),
        per_agent=per_agent_report,
    )

    flow = EvalEpisodeFlow(
        agent_ids=list(range(NUM_AGENTS)),
        device_choices=flow_device_choices,
        fail_step=flow_fail_step,
        fail_agent=flow_fail_agent,
        model_types={i: MODEL_TYPES[i] for i in range(NUM_AGENTS)},
    )
    plot_execution_flow(out_path=os.path.join(RESULTS_DIR, "execution_flow.png"), flow=flow)
    print(
        "Saved\n"
        f"  Report: {os.path.join(RESULTS_DIR, 'marl_eval_report.json')}\n"
        f"  Plot: {os.path.join(RESULTS_DIR, 'evaluation_summary.png')}\n"
        f"  Plot: {os.path.join(RESULTS_DIR, 'marl_eval_summary.png')}\n"
        f"  Plot: {os.path.join(RESULTS_DIR, 'execution_flow.png')}\n"
    )


if __name__ == "__main__":
    evaluate()
