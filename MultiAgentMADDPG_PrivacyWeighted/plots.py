from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_privacy_violations(
    out_path: str,
    per_agent_violation_sum: Dict[int, Sequence[float]],
    window: int = 50,
) -> None:
    agent_ids = sorted(int(a) for a in per_agent_violation_sum.keys())
    if not agent_ids:
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 5.5), constrained_layout=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, aid in enumerate(agent_ids):
        v = np.asarray(list(per_agent_violation_sum.get(aid, []) or []), dtype=np.float32)
        if v.size == 0:
            continue
        ax.plot(v, color=colors[idx % len(colors)], alpha=0.25, linewidth=1, label=f"Agent {aid} (raw)")
        if v.size >= window:
            ma = np.convolve(v, np.ones(window) / window, mode="valid")
            ax.plot(
                np.arange(window - 1, v.size),
                ma,
                color=colors[idx % len(colors)],
                linewidth=2,
                label=f"Agent {aid} (MA{window})",
            )

    ax.set_title("Trust Shortfall Severity (per episode)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Trust shortfall (sum)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(ncols=2, fontsize=9)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)
