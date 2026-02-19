from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .networks import DiscreteActor, CentralizedCritic


@dataclass
class EpsilonSchedule:
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.9995

    def step(self) -> None:
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    tau = float(tau)
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters(), strict=True):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


class MADDPGAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        n_agents: int,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        epsilon_decay: float = 0.9995,
    ):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.n_agents = int(n_agents)
        self.joint_action_dim = self.n_agents * self.action_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = DiscreteActor(self.obs_dim, self.action_dim).to(self.device)
        self.target_actor = DiscreteActor(self.obs_dim, self.action_dim).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_actor.eval()

        self.critic = CentralizedCritic(self.state_dim, self.joint_action_dim).to(self.device)
        self.target_critic = CentralizedCritic(self.state_dim, self.joint_action_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=float(critic_lr))

        self.eps = EpsilonSchedule(epsilon_decay=float(epsilon_decay))

    def act(self, obs: np.ndarray, valid_actions: Sequence[int] | None = None) -> int:
        if valid_actions is not None and len(valid_actions) == 0:
            return random.randrange(self.action_dim)

        if np.random.rand() <= self.eps.epsilon:
            if valid_actions is None:
                return random.randrange(self.action_dim)
            return int(random.choice(list(valid_actions)))

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(obs_t).squeeze(0)  # (A,)

        if valid_actions is not None:
            mask = torch.full((self.action_dim,), float("-inf"), device=self.device)
            mask[list(valid_actions)] = 0.0
            logits = logits + mask

        return int(torch.argmax(logits).item())

    @staticmethod
    def masked_gumbel_softmax(
        logits: torch.Tensor,
        mask: torch.Tensor | None,
        tau: float,
        hard: bool = True,
    ) -> torch.Tensor:
        # logits: (B,A), mask: (B,A) bool
        if mask is not None:
            valid_counts = mask.sum(dim=1, keepdim=True)
            effective_mask = torch.where(valid_counts > 0, mask, torch.ones_like(mask))
            logits = logits.masked_fill(~effective_mask, -1e9)
        return F.gumbel_softmax(logits, tau=float(tau), hard=bool(hard), dim=-1)

