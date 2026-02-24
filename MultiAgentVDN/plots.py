from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class EvalEpisodeFlow:
    agent_ids: List[int]
    device_choices: Dict[int, List[int]]  # agent_id -> list of device ids (per step when active)
    fail_step: int | None
    fail_agent: int | None
    model_types: Dict[int, str]


def plot_training_trends(
    out_path: str,
    team_rewards: Sequence[float],
    losses: Sequence[float],
    eps_history: Sequence[float] | None = None,
    window: int = 50,
) -> None:
    team_rewards = np.asarray(team_rewards, dtype=np.float32)
    losses = np.asarray(losses, dtype=np.float32)

    # NOTE: We intentionally do not plot exploration (epsilon) here to keep the figure focused.
    # `eps_history` is kept for backward compatibility with older training scripts.
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), constrained_layout=True)

    ax = axes[0]
    ax.plot(team_rewards, color="#1f77b4", alpha=0.35, linewidth=1, label="Team return (mean/step)")
    if len(team_rewards) >= window:
        ma = np.convolve(team_rewards, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window - 1, len(team_rewards)), ma, color="#1f77b4", linewidth=2, label=f"Moving avg ({window})")
    ax.set_title("Training Trends: Team Return")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    ax = axes[1]
    if len(losses) > 0:
        ax.plot(losses, color="#ff7f0e", alpha=0.5, linewidth=1, label="TD loss")
        w = min(500, max(10, len(losses) // 50))
        if len(losses) >= w:
            ma = np.convolve(losses, np.ones(w) / w, mode="valid")
            ax.plot(np.arange(w - 1, len(losses)), ma, color="#ff7f0e", linewidth=2, label=f"Moving avg ({w})")
    ax.set_title("Training Trends: Loss")
    ax.set_xlabel("Train update")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_avg_cumulative_rewards(
    out_path: str,
    episode_team_reward_sums: Sequence[float],
    num_agents: int,
    window: int = 50,
    ylim: tuple[float, float] | None = None,
) -> None:
    """
    Average cumulative rewards vs training episodes.

    - "cumulative" = sum of step rewards over an episode
    - "average" = divided by number of agents
    """
    sums = np.asarray(episode_team_reward_sums, dtype=np.float32)
    denom = float(max(1, int(num_agents)))
    avg_cum = sums / denom

    fig, ax = plt.subplots(1, 1, figsize=(12, 5.5), constrained_layout=True)
    ax.plot(avg_cum, color="#2ca02c", alpha=0.35, linewidth=1, label="Avg cumulative reward/episode (per agent)")
    if avg_cum.size >= window:
        ma = np.convolve(avg_cum, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window - 1, len(avg_cum)), ma, color="#2ca02c", linewidth=2, label=f"Moving avg ({window})")
    ax.set_title("Average Cumulative Rewards vs Training Episodes")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg cumulative reward")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    if ylim is not None:
        y0, y1 = float(ylim[0]), float(ylim[1])
        ax.set_ylim(min(y0, y1), max(y0, y1))

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_per_agent_training_rewards(
    out_path: str,
    agent_reward_history: Dict[int, Sequence[float]],
    model_types: Dict[int, str] | None = None,
    window: int = 50,
    ylim: tuple[float, float] | None = None,
) -> None:
    agent_ids = sorted(int(a) for a in agent_reward_history.keys())
    fig, ax = plt.subplots(1, 1, figsize=(12, 5.5), constrained_layout=True)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for idx, aid in enumerate(agent_ids):
        rewards = np.asarray(list(agent_reward_history.get(aid, [])), dtype=np.float32)
        if rewards.size == 0:
            continue
        label = f"Agent {aid}"
        if model_types and aid in model_types:
            label += f" ({model_types[aid]})"
        c = colors[idx % len(colors)]
        ax.plot(rewards, color=c, alpha=0.25, linewidth=1)
        if rewards.size >= window:
            ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(
                np.arange(window - 1, len(rewards)),
                ma,
                color=c,
                linewidth=2,
                label=f"{label} MA({window})",
            )
        else:
            ax.plot(rewards, color=c, linewidth=2, label=label)

    ax.set_title("Training: Per-Agent Episode Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (sum per episode)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
    if ylim is not None:
        y0, y1 = float(ylim[0]), float(ylim[1])
        ax.set_ylim(min(y0, y1), max(y0, y1))

    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def plot_marl_eval_summary(
    out_path: str,
    per_agent: Dict[int, dict],
) -> None:
    agent_ids = sorted(int(k) for k in per_agent.keys())
    success = [float(per_agent[i]["success_rate"]) for i in agent_ids]
    avg_comp = [float(per_agent[i].get("avg_t_comp", 0.0)) for i in agent_ids]
    avg_comm = [float(per_agent[i].get("avg_t_comm", 0.0)) for i in agent_ids]
    labels = [f"A{i}\n{per_agent[i].get('model_type','')}" for i in agent_ids]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    ax = axes[0]
    ax.bar(labels, success, color="#2ca02c", alpha=0.9)
    ax.set_ylim(0, 100)
    ax.set_title("Per-Agent Success Rate")
    ax.set_ylabel("%")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)

    ax = axes[1]
    x = np.arange(len(agent_ids))
    ax.bar(x, avg_comp, color="#1f77b4", alpha=0.9, label="t_comp")
    ax.bar(x, avg_comm, bottom=avg_comp, color="#ff7f0e", alpha=0.9, label="t_comm")
    ax.set_xticks(x, labels)
    ax.set_title("Latency Components (avg)")
    ax.set_ylabel("Seconds (normalized units)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.legend()

    ax = axes[2]
    # Device counts heat-ish bar: show top-3 devices per agent.
    width = 0.8 / max(1, len(agent_ids))
    for j, aid in enumerate(agent_ids):
        counts = per_agent[aid].get("device_counts", {}) or {}
        devices = sorted((int(d), int(c)) for d, c in counts.items())
        if not devices:
            continue
        xs = np.array([d for d, _ in devices], dtype=np.int32) + (j - (len(agent_ids) - 1) / 2) * width
        ys = np.array([c for _, c in devices], dtype=np.int32)
        ax.bar(xs, ys, width=width, alpha=0.85, label=f"A{aid}")
    ax.set_title("Device Selection Counts")
    ax.set_xlabel("Device id")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.legend()

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_execution_flow(out_path: str, flow: EvalEpisodeFlow) -> None:
    agent_ids = flow.agent_ids
    max_len = max((len(flow.device_choices.get(aid, [])) for aid in agent_ids), default=0)
    if max_len == 0:
        max_len = 1

    data = np.full((len(agent_ids), max_len), np.nan, dtype=np.float32)
    for i, aid in enumerate(agent_ids):
        choices = flow.device_choices.get(aid, [])
        if choices:
            data[i, : len(choices)] = np.asarray(choices, dtype=np.float32)

    fig, ax = plt.subplots(1, 1, figsize=(12, 0.9 * len(agent_ids) + 2), constrained_layout=True)
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="tab20", vmin=0)

    yt = []
    for aid in agent_ids:
        mt = flow.model_types.get(aid, "")
        yt.append(f"A{aid} {mt}")
    ax.set_yticks(np.arange(len(agent_ids)), yt)
    ax.set_xticks(np.arange(max_len))
    ax.set_xlabel("Allocation step (while active)")
    ax.set_title("Execution Flow: Device Choice per Agent")

    if flow.fail_step is not None:
        ax.axvline(flow.fail_step, color="red", linestyle="--", linewidth=2, alpha=0.8)
        if flow.fail_agent is not None:
            ax.text(
                min(flow.fail_step + 0.2, max_len - 1),
                max(0, agent_ids.index(flow.fail_agent)),
                "FAIL",
                color="red",
                fontsize=10,
                fontweight="bold",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Device id")

    fig.savefig(out_path, dpi=160)
    plt.close(fig)
