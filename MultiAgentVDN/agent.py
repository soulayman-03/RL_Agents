from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


@dataclass
class EpsilonSchedule:
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.9995

    def step(self) -> None:
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class VDNAgent:
    """Per-agent DQN module used under VDN mixing (sum)."""

    def __init__(self, obs_dim: int, action_dim: int, lr: float = 5e-4, epsilon_decay: float = 0.9995):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=float(lr))
        self.criterion = nn.SmoothL1Loss()

        self.eps = EpsilonSchedule(epsilon_decay=float(epsilon_decay))

    def act(self, obs: np.ndarray, valid_actions: Sequence[int] | None = None) -> int:
        if valid_actions is not None and len(valid_actions) == 0:
            # No valid actions. Fall back to a random action in full space; env should handle failures.
            return random.randrange(self.action_dim)

        if np.random.rand() <= self.eps.epsilon:
            if valid_actions is None:
                return random.randrange(self.action_dim)
            return int(random.choice(list(valid_actions)))

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, obs_dim)
        with torch.no_grad():
            q = self.policy_net(obs_t).squeeze(0)  # (A,)

        if valid_actions is not None:
            mask = torch.full((self.action_dim,), float("-inf"), device=self.device)
            mask[list(valid_actions)] = 0.0
            q = q + mask

        return int(torch.argmax(q).item())

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())


def stack_obs(obs_dict: dict[int, np.ndarray], agent_ids: List[int]) -> np.ndarray:
    return np.stack([obs_dict[aid] for aid in agent_ids], axis=0).astype(np.float32, copy=False)

