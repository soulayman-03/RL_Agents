from __future__ import annotations

import os
from typing import Dict, List, Sequence

import numpy as np
import torch

from .agent import VDNAgent, stack_obs
from .replay_buffer import JointReplayBuffer


class VDNManager:
    """CTDE trainer: joint replay + VDN mixing (Q_total = sum_i Q_i)."""

    def __init__(
        self,
        agent_ids: List[int],
        obs_dim: int,
        action_dim: int,
        lr: float = 5e-4,
        gamma: float = 0.99,
        batch_size: int = 64,
        capacity: int = 50_000,
        target_update_freq: int = 1_000,
        shared_policy: bool = False,
        epsilon_decay: float = 0.9995,
    ):
        self.agent_ids = list(agent_ids)
        self.n_agents = len(self.agent_ids)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_freq = int(target_update_freq)
        self.train_step = 0

        if shared_policy:
            shared = VDNAgent(self.obs_dim, self.action_dim, lr=lr, epsilon_decay=epsilon_decay)
            self.agents = {aid: shared for aid in self.agent_ids}
        else:
            self.agents = {
                aid: VDNAgent(self.obs_dim, self.action_dim, lr=lr, epsilon_decay=epsilon_decay) for aid in self.agent_ids
            }

        self.device = next(next(iter(self.agents.values())).policy_net.parameters()).device
        self.buffer = JointReplayBuffer(capacity=capacity)

    def _unique_agents(self) -> List[VDNAgent]:
        return list({id(a): a for a in self.agents.values()}.values())

    def get_actions(
        self,
        obs_dict: Dict[int, np.ndarray],
        valid_actions: Dict[int, Sequence[int]] | None = None,
    ) -> Dict[int, int]:
        actions: Dict[int, int] = {}
        for aid in self.agent_ids:
            va = valid_actions.get(aid) if valid_actions else None
            actions[aid] = self.agents[aid].act(obs_dict[aid], va)
        return actions

    def remember(
        self,
        obs_dict: Dict[int, np.ndarray],
        actions_dict: Dict[int, int],
        rewards_dict: Dict[int, float],
        next_obs_dict: Dict[int, np.ndarray],
        dones_dict: Dict[int, bool],
        active_before_dict: Dict[int, bool],
        next_valid_actions: Dict[int, Sequence[int]] | None,
        done_team: bool,
    ) -> None:
        obs = stack_obs(obs_dict, self.agent_ids)  # (N, obs_dim)
        next_obs = stack_obs(next_obs_dict, self.agent_ids)
        actions = np.asarray([actions_dict[aid] for aid in self.agent_ids], dtype=np.int64)
        dones = np.asarray([bool(dones_dict[aid]) for aid in self.agent_ids], dtype=bool)

        # Team reward: mean of per-agent rewards (scale-stable).
        reward_team = float(sum(float(rewards_dict[aid]) for aid in self.agent_ids)) / float(self.n_agents)

        # Active mask refers to whether an agent was active at the CURRENT state (before stepping).
        active = np.asarray([not bool(active_before_dict[aid]) for aid in self.agent_ids], dtype=bool)

        # Next-state valid action mask (for masked argmax in TD targets).
        next_action_mask = np.zeros((self.n_agents, self.action_dim), dtype=bool)
        if next_valid_actions is not None:
            for i, aid in enumerate(self.agent_ids):
                va = list(next_valid_actions.get(aid, []))
                if len(va) > 0:
                    next_action_mask[i, va] = True

        self.buffer.add(obs, actions, reward_team, next_obs, dones, done_team, active, next_action_mask)

    def train(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        obs_b, actions_b, rewards_b, next_obs_b, dones_b, done_team_b, active_b, next_action_mask_b = self.buffer.sample(
            self.batch_size
        )

        obs_t = torch.as_tensor(obs_b, dtype=torch.float32, device=self.device)  # (B, N, D)
        actions_t = torch.as_tensor(actions_b, dtype=torch.int64, device=self.device)  # (B, N)
        rewards_t = torch.as_tensor(rewards_b, dtype=torch.float32, device=self.device)  # (B,)
        next_obs_t = torch.as_tensor(next_obs_b, dtype=torch.float32, device=self.device)  # (B, N, D)
        dones_next_t = torch.as_tensor(dones_b.astype(np.float32), dtype=torch.float32, device=self.device)  # (B, N)
        done_team_t = torch.as_tensor(done_team_b.astype(np.float32), dtype=torch.float32, device=self.device)  # (B,)
        active_t = torch.as_tensor(active_b.astype(np.float32), dtype=torch.float32, device=self.device)  # (B, N)
        next_action_mask_t = torch.as_tensor(next_action_mask_b, dtype=torch.bool, device=self.device)  # (B, N, A)

        # Current Q_total
        q_total = torch.zeros((obs_t.shape[0],), dtype=torch.float32, device=self.device)
        for i, aid in enumerate(self.agent_ids):
            agent = self.agents[aid]
            q_all = agent.policy_net(obs_t[:, i, :])  # (B, A)
            q_a = q_all.gather(1, actions_t[:, i].unsqueeze(1)).squeeze(1)  # (B,)
            q_total = q_total + (active_t[:, i] * q_a)

        with torch.no_grad():
            target_total = torch.zeros((obs_t.shape[0],), dtype=torch.float32, device=self.device)
            for i, aid in enumerate(self.agent_ids):
                agent = self.agents[aid]
                # Double DQN style: argmax from policy, value from target
                next_q_policy = agent.policy_net(next_obs_t[:, i, :])  # (B, A)

                # Mask invalid actions at next state.
                mask = next_action_mask_t[:, i, :]  # (B, A) bool
                valid_counts = mask.sum(dim=1, keepdim=True)  # (B, 1)
                effective_mask = torch.where(valid_counts > 0, mask, torch.ones_like(mask))
                next_q_policy = next_q_policy.masked_fill(~effective_mask, -1e9)

                next_a = torch.argmax(next_q_policy, dim=1, keepdim=True)  # (B,1)
                next_q_target = agent.target_net(next_obs_t[:, i, :]).gather(1, next_a).squeeze(1)  # (B,)
                alive_next = (1.0 - dones_next_t[:, i])  # (B,)
                target_total = target_total + (alive_next * next_q_target)

            y = rewards_t + (self.gamma * target_total * (1.0 - done_team_t))

        # Compute a scalar loss using any agent's criterion (same)
        criterion = self._unique_agents()[0].criterion
        loss = criterion(q_total, y.detach())

        # Zero grads for all unique optimizers, then backward once, then step all.
        for agent in self._unique_agents():
            agent.optimizer.zero_grad()
        loss.backward()
        for agent in self._unique_agents():
            torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), 10.0)
            agent.optimizer.step()

        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            for agent in self._unique_agents():
                agent.update_target()

        # Decay epsilon (same schedule when shared policy; stepping unique is fine)
        for agent in self._unique_agents():
            agent.eps.step()

        return float(loss.item())

    def save(self, base_path: str) -> None:
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        for aid in self.agent_ids:
            torch.save(self.agents[aid].policy_net.state_dict(), f"{base_path}_agent_{aid}.pt")

    def load(self, base_path: str) -> None:
        for aid in self.agent_ids:
            path = f"{base_path}_agent_{aid}.pt"
            if os.path.exists(path):
                self.agents[aid].policy_net.load_state_dict(torch.load(path, map_location=self.device))
                self.agents[aid].update_target()
