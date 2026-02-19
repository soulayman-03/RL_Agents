import os
import sys

# Re-export the existing environment to avoid duplication.
# Allow running both:
# - as a module:  python -m MultiAgentMADDPG.train
# - as a script:  python MultiAgentMADDPG/train.py
if __package__:
    from MultiAgent.environment import MultiAgentIoTEnv
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from MultiAgent.environment import MultiAgentIoTEnv

__all__ = ["MultiAgentIoTEnv"]

