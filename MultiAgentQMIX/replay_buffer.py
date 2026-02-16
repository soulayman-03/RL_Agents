from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Tuple
from collections import deque
import random

import numpy as np


@dataclass(frozen=True)
class QMIXTransition:
    obs: np.ndarray  # (N, obs_dim)
    state: np.ndarray  # (state_dim,)
    actions: np.ndarray  # (N,)
    reward_team: float  # scalar team reward (normalized)
    next_obs: np.ndarray  # (N, obs_dim)
    next_state: np.ndarray  # (state_dim,)
    dones: np.ndarray  # (N,) bool (per-agent done at next state)
    done_team: bool  # terminal for the whole team (e.g., any allocation failure)
    active: np.ndarray  # (N,) bool (agent was active at current state)
    next_action_mask: np.ndarray  # (N, action_dim) bool (valid actions at next state)


class QMIXReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self._buf: Deque[QMIXTransition] = deque(maxlen=int(capacity))

    def __len__(self) -> int:
        return len(self._buf)

    def add(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        reward_team: float,
        next_obs: np.ndarray,
        next_state: np.ndarray,
        dones: np.ndarray,
        done_team: bool,
        active: np.ndarray,
        next_action_mask: np.ndarray,
    ) -> None:
        self._buf.append(
            QMIXTransition(
                obs=np.asarray(obs, dtype=np.float32),
                state=np.asarray(state, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.int64),
                reward_team=float(reward_team),
                next_obs=np.asarray(next_obs, dtype=np.float32),
                next_state=np.asarray(next_state, dtype=np.float32),
                dones=np.asarray(dones, dtype=bool),
                done_team=bool(done_team),
                active=np.asarray(active, dtype=bool),
                next_action_mask=np.asarray(next_action_mask, dtype=bool),
            )
        )

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self._buf, int(batch_size))
        obs = np.stack([t.obs for t in batch], axis=0)  # (B, N, obs_dim)
        state = np.stack([t.state for t in batch], axis=0)  # (B, state_dim)
        actions = np.stack([t.actions for t in batch], axis=0)  # (B, N)
        rewards = np.asarray([t.reward_team for t in batch], dtype=np.float32)  # (B,)
        next_obs = np.stack([t.next_obs for t in batch], axis=0)  # (B, N, obs_dim)
        next_state = np.stack([t.next_state for t in batch], axis=0)  # (B, state_dim)
        dones = np.stack([t.dones for t in batch], axis=0)  # (B, N)
        done_team = np.asarray([t.done_team for t in batch], dtype=bool)  # (B,)
        active = np.stack([t.active for t in batch], axis=0)  # (B, N)
        next_action_mask = np.stack([t.next_action_mask for t in batch], axis=0)  # (B, N, A)
        return obs, state, actions, rewards, next_obs, next_state, dones, done_team, active, next_action_mask

