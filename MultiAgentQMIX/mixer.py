from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixer(nn.Module):
    """QMIX mixer: monotonic mixing of per-agent Qs conditioned on global state."""

    def __init__(self, n_agents: int, state_dim: int, hidden_dim: int = 32, hypernet_dim: int = 64):
        super().__init__()
        self.n_agents = int(n_agents)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)

        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_dim),
            nn.ReLU(),
            nn.Linear(hypernet_dim, self.n_agents * self.hidden_dim),
        )
        self.hyper_b1 = nn.Linear(self.state_dim, self.hidden_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_dim),
            nn.ReLU(),
            nn.Linear(hypernet_dim, self.hidden_dim * 1),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        agent_qs: (B, N)  per-agent chosen Q values
        states:   (B, S)  global state
        returns:  (B,)    Q_total
        """
        B = agent_qs.shape[0]
        agent_qs = agent_qs.view(B, 1, self.n_agents)  # (B,1,N)

        w1 = torch.abs(self.hyper_w1(states)).view(B, self.n_agents, self.hidden_dim)  # (B,N,H)
        b1 = self.hyper_b1(states).view(B, 1, self.hidden_dim)  # (B,1,H)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)  # (B,1,H)

        w2 = torch.abs(self.hyper_w2(states)).view(B, self.hidden_dim, 1)  # (B,H,1)
        b2 = self.hyper_b2(states).view(B, 1, 1)  # (B,1,1)
        q_tot = torch.bmm(hidden, w2) + b2  # (B,1,1)
        return q_tot.view(B)
