"""
Reuse plotting utilities from MultiAgentVDN.

MADDPG produces the same kind of artifacts (reward/loss/epsilon histories + eval summary),
so the same plotting functions apply.
"""

from MultiAgentVDN.plots import (  # noqa: F401
    EvalEpisodeFlow,
    plot_avg_cumulative_rewards,
    plot_layer_latency,
    plot_per_agent_layer_latency,
    plot_training_trends,
    plot_per_agent_training_rewards,
    plot_execution_flow,
    plot_training_execution_strategy,
    plot_marl_eval_summary,
)
