"""VDN (Value Decomposition Networks) CTDE baseline for this project.

This package intentionally reuses the existing MultiAgentIoTEnv, while swapping
the learner from independent DQN to a centralized VDN-style trainer with
decentralized execution (per-agent argmax).
"""

