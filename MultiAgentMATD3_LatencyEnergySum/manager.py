from __future__ import annotations

import os
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .agent import MATD3Agent, soft_update
from .replay_buffer import MATD3ReplayBuffer


def stack_obs(obs_dict: dict[int, np.ndarray], agent_ids: List[int]) -> np.ndarray:
    return np.stack([obs_dict[aid] for aid in agent_ids], axis=0).astype(np.float32, copy=False)


class MATD3Manager:
    """CTDE trainer (discrete MATD3): decentralized actors + twin centralized critics."""

    def __init__(
        self,
        agent_ids: List[int],
        obs_dim: int,
        action_dim: int,
        state_dim: int | None = None,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        batch_size: int = 256,
        capacity: int = 50_000,
        gumbel_tau: float = 1.0,
        epsilon_decay: float = 0.9995,
        shared_policy: bool = False,
        policy_delay: int = 2,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
    ):
        self.agent_ids = list(agent_ids)
        self.n_agents = len(self.agent_ids)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim) if state_dim is not None else int(self.n_agents * self.obs_dim)

        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.gumbel_tau = float(gumbel_tau)
        self.train_step = 0

        self.policy_delay = max(1, int(policy_delay))
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)

        if shared_policy:
            shared = MATD3Agent(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                state_dim=self.state_dim,
                n_agents=self.n_agents,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                epsilon_decay=epsilon_decay,
            )
            self.agents = {aid: shared for aid in self.agent_ids}
        else:
            self.agents = {
                aid: MATD3Agent(
                    obs_dim=self.obs_dim,
                    action_dim=self.action_dim,
                    state_dim=self.state_dim,
                    n_agents=self.n_agents,
                    actor_lr=actor_lr,
                    critic_lr=critic_lr,
                    epsilon_decay=epsilon_decay,
                )
                for aid in self.agent_ids
            }

        self.device = next(next(iter(self.agents.values())).actor.parameters()).device
        self.buffer = MATD3ReplayBuffer(capacity=capacity)

    def _unique_agents(self) -> List[MATD3Agent]:
        return list({id(a): a for a in self.agents.values()}.values())

    @staticmethod
    def build_state_from_obs(obs_stack: np.ndarray) -> np.ndarray:
        return np.asarray(obs_stack, dtype=np.float32).reshape(-1)

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

    def _mask_from_valid_actions(self, valid_actions: Dict[int, Sequence[int]] | None) -> np.ndarray:
        mask = np.zeros((self.n_agents, self.action_dim), dtype=bool)
        if valid_actions is None:
            return mask
        for i, aid in enumerate(self.agent_ids):
            va = list(valid_actions.get(aid, []))
            if len(va) > 0:
                mask[i, va] = True
        return mask

    def remember(
        self,
        obs_dict: Dict[int, np.ndarray],
        actions_dict: Dict[int, int],
        rewards_dict: Dict[int, float],
        next_obs_dict: Dict[int, np.ndarray],
        dones_dict: Dict[int, bool],
        active_before_dict: Dict[int, bool],
        valid_actions: Dict[int, Sequence[int]] | None,
        next_valid_actions: Dict[int, Sequence[int]] | None,
    ) -> None:
        obs = stack_obs(obs_dict, self.agent_ids)
        next_obs = stack_obs(next_obs_dict, self.agent_ids)
        state = self.build_state_from_obs(obs)
        next_state = self.build_state_from_obs(next_obs)

        actions = np.asarray([actions_dict[aid] for aid in self.agent_ids], dtype=np.int64)
        rewards = np.asarray([rewards_dict.get(aid, 0.0) for aid in self.agent_ids], dtype=np.float32)
        dones = np.asarray([dones_dict.get(aid, False) for aid in self.agent_ids], dtype=bool)
        active = np.asarray([not bool(active_before_dict.get(aid, False)) for aid in self.agent_ids], dtype=bool)

        action_mask = self._mask_from_valid_actions(valid_actions)
        next_action_mask = self._mask_from_valid_actions(next_valid_actions)

        self.buffer.add(
            obs=obs,
            state=state,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            next_state=next_state,
            dones=dones,
            active=active,
            action_mask=action_mask,
            next_action_mask=next_action_mask,
        )

    def _noisy_target_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.policy_noise <= 0.0:
            return logits
        noise = torch.randn_like(logits) * self.policy_noise
        if self.noise_clip > 0.0:
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
        return logits + noise

    def train(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        (
            obs,
            state,
            actions,
            rewards,
            next_obs,
            next_state,
            dones_next,
            active,
            action_mask,
            next_action_mask,
        ) = self.buffer.sample(self.batch_size)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)  # (B,N,D)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)  # (B,S)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)  # (B,N)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)  # (B,N)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)  # (B,N,D)
        next_state_t = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)  # (B,S)
        dones_next_t = torch.as_tensor(dones_next, dtype=torch.float32, device=self.device)  # (B,N)
        active_t = torch.as_tensor(active, dtype=torch.float32, device=self.device)  # (B,N)

        action_mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)  # (B,N,A)
        next_action_mask_t = torch.as_tensor(next_action_mask, dtype=torch.bool, device=self.device)  # (B,N,A)

        joint_action_onehot = F.one_hot(actions_t, num_classes=self.action_dim).to(torch.float32).view(
            -1, self.n_agents * self.action_dim
        )

        with torch.no_grad():
            next_action_parts = []
            for i, aid in enumerate(self.agent_ids):
                agent = self.agents[aid]
                logits = agent.target_actor(next_obs_t[:, i, :])  # (B,A)
                logits = self._noisy_target_logits(logits)
                mask = next_action_mask_t[:, i, :]
                a = agent.masked_gumbel_softmax(logits, mask, tau=self.gumbel_tau, hard=True)
                next_action_parts.append(a)
            joint_next_action = torch.cat(next_action_parts, dim=-1)  # (B,N*A)

        critic_losses: list[torch.Tensor] = []
        actor_losses: list[torch.Tensor] = []

        for i, aid in enumerate(self.agent_ids):
            agent = self.agents[aid]
            active_i = active_t[:, i]  # (B,)
            if float(active_i.sum().item()) <= 0.0:
                continue

            q1 = agent.critic1(state_t, joint_action_onehot)
            q2 = agent.critic2(state_t, joint_action_onehot)
            with torch.no_grad():
                tq1 = agent.target_critic1(next_state_t, joint_next_action)
                tq2 = agent.target_critic2(next_state_t, joint_next_action)
                tq = torch.minimum(tq1, tq2)
                y = rewards_t[:, i] + (self.gamma * (1.0 - dones_next_t[:, i]) * tq)

            c1_loss = ((q1 - y) ** 2 * active_i).sum() / (active_i.sum() + 1e-6)
            c2_loss = ((q2 - y) ** 2 * active_i).sum() / (active_i.sum() + 1e-6)

            agent.critic1_opt.zero_grad()
            c1_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic1.parameters(), 10.0)
            agent.critic1_opt.step()

            agent.critic2_opt.zero_grad()
            c2_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic2.parameters(), 10.0)
            agent.critic2_opt.step()

            critic_losses.append((c1_loss + c2_loss).detach() * 0.5)

        do_policy = (self.train_step % self.policy_delay) == 0

        if do_policy:
            current_action_parts = []
            for i, aid in enumerate(self.agent_ids):
                agent = self.agents[aid]
                logits = agent.actor(obs_t[:, i, :])
                mask = action_mask_t[:, i, :]
                a = agent.masked_gumbel_softmax(logits, mask, tau=self.gumbel_tau, hard=True)
                current_action_parts.append(a)

            for i, aid in enumerate(self.agent_ids):
                agent = self.agents[aid]
                active_i = active_t[:, i]
                if float(active_i.sum().item()) <= 0.0:
                    continue

                parts = []
                for j in range(self.n_agents):
                    parts.append(current_action_parts[j] if j == i else current_action_parts[j].detach())
                joint_curr = torch.cat(parts, dim=-1)
                q_pi = agent.critic1(state_t, joint_curr)
                actor_loss = (-(q_pi) * active_i).sum() / (active_i.sum() + 1e-6)

                agent.actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 10.0)
                agent.actor_opt.step()
                actor_losses.append(actor_loss.detach())

            for agent in self._unique_agents():
                soft_update(agent.target_actor, agent.actor, tau=self.tau)
                soft_update(agent.target_critic1, agent.critic1, tau=self.tau)
                soft_update(agent.target_critic2, agent.critic2, tau=self.tau)
                agent.eps.step()

        self.train_step += 1

        if len(critic_losses) == 0 and len(actor_losses) == 0:
            return None

        c = torch.stack(critic_losses).mean() if len(critic_losses) else torch.tensor(0.0, device=self.device)
        a = torch.stack(actor_losses).mean() if len(actor_losses) else torch.tensor(0.0, device=self.device)
        return float((c + a).item())

    def save(self, base_path: str) -> None:
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        for aid in self.agent_ids:
            agent = self.agents[aid]
            torch.save(agent.actor.state_dict(), f"{base_path}_actor_agent_{aid}.pt")
            torch.save(agent.critic1.state_dict(), f"{base_path}_critic1_agent_{aid}.pt")
            torch.save(agent.critic2.state_dict(), f"{base_path}_critic2_agent_{aid}.pt")

    def load(self, base_path: str) -> None:
        for aid in self.agent_ids:
            agent = self.agents[aid]
            a_path = f"{base_path}_actor_agent_{aid}.pt"
            c1_path = f"{base_path}_critic1_agent_{aid}.pt"
            c2_path = f"{base_path}_critic2_agent_{aid}.pt"
            if os.path.exists(a_path):
                agent.actor.load_state_dict(torch.load(a_path, map_location=self.device))
                agent.target_actor.load_state_dict(agent.actor.state_dict())
            if os.path.exists(c1_path):
                agent.critic1.load_state_dict(torch.load(c1_path, map_location=self.device))
                agent.target_critic1.load_state_dict(agent.critic1.state_dict())
            if os.path.exists(c2_path):
                agent.critic2.load_state_dict(torch.load(c2_path, map_location=self.device))
                agent.target_critic2.load_state_dict(agent.critic2.state_dict())
