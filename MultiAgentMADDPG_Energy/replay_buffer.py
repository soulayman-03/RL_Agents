from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque, Tuple

import numpy as np


@dataclass(frozen=True)
class MADDPGTransition:
    obs: np.ndarray  # (N, obs_dim)
    state: np.ndarray  # (state_dim,)
    actions: np.ndarray  # (N,) int64 (discrete)
    rewards: np.ndarray  # (N,) float32
    next_obs: np.ndarray  # (N, obs_dim)
    next_state: np.ndarray  # (state_dim,)
    dones: np.ndarray  # (N,) bool (per-agent done at next state)
    active: np.ndarray  # (N,) bool (agent was active at current state)
    action_mask: np.ndarray  # (N, action_dim) bool (valid actions at current state)
    next_action_mask: np.ndarray  # (N, action_dim) bool (valid actions at next state)


class MADDPGReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self._buf: Deque[MADDPGTransition] = deque(maxlen=int(capacity))

    def __len__(self) -> int:
        return len(self._buf)

    def add(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        next_state: np.ndarray,
        dones: np.ndarray,
        active: np.ndarray,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
    ) -> None:
        self._buf.append(
            MADDPGTransition(
                obs=np.asarray(obs, dtype=np.float32),
                state=np.asarray(state, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.int64),
                rewards=np.asarray(rewards, dtype=np.float32),
                next_obs=np.asarray(next_obs, dtype=np.float32),
                next_state=np.asarray(next_state, dtype=np.float32),
                dones=np.asarray(dones, dtype=bool),
                active=np.asarray(active, dtype=bool),
                action_mask=np.asarray(action_mask, dtype=bool),
                next_action_mask=np.asarray(next_action_mask, dtype=bool),
            )
        )

    def sample(
        self, batch_size: int
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        batch = random.sample(self._buf, int(batch_size))
        obs = np.stack([t.obs for t in batch], axis=0)  # (B, N, D)
        state = np.stack([t.state for t in batch], axis=0)  # (B, S)
        actions = np.stack([t.actions for t in batch], axis=0)  # (B, N)
        rewards = np.stack([t.rewards for t in batch], axis=0)  # (B, N)
        next_obs = np.stack([t.next_obs for t in batch], axis=0)  # (B, N, D)
        next_state = np.stack([t.next_state for t in batch], axis=0)  # (B, S)
        dones = np.stack([t.dones for t in batch], axis=0)  # (B, N)
        active = np.stack([t.active for t in batch], axis=0)  # (B, N)
        action_mask = np.stack([t.action_mask for t in batch], axis=0)  # (B, N, A)
        next_action_mask = np.stack([t.next_action_mask for t in batch], axis=0)  # (B, N, A)
        return obs, state, actions, rewards, next_obs, next_state, dones, active, action_mask, next_action_mask

