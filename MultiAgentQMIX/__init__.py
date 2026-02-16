"""QMIX (Monotonic Value Function Factorisation) CTDE learner for this project.

This package reuses the existing `MultiAgentIoTEnv` and swaps the learner to
QMIX (centralized mixing network conditioned on a global state), while keeping
decentralized execution (per-agent argmax with action masking).
"""

