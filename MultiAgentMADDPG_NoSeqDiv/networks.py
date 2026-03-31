from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: tuple[int, ...] = (256, 256)):
        super().__init__()
        dims = (int(in_dim),) + tuple(int(d) for d in hidden_dims) + (int(out_dim),)
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiscreteActor(nn.Module):
    """Decentralized actor: outputs logits over discrete actions."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.mlp = MLP(self.obs_dim, self.action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)  # logits (B, A)


class CentralizedCritic(nn.Module):
    """Centralized critic: Q_i(state, joint_action_onehot) -> scalar."""

    def __init__(self, state_dim: int, joint_action_dim: int):
        super().__init__()
        self.state_dim = int(state_dim)
        self.joint_action_dim = int(joint_action_dim)
        self.mlp = MLP(self.state_dim + self.joint_action_dim, 1)

    def forward(self, state: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, joint_action], dim=-1)
        return self.mlp(x).squeeze(-1)  # (B,)

