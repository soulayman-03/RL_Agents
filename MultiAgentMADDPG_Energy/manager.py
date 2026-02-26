from __future__ import annotations

import os
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .agent import MADDPGAgent, soft_update
from .replay_buffer import MADDPGReplayBuffer


def stack_obs(obs_dict: dict[int, np.ndarray], agent_ids: List[int]) -> np.ndarray:
    return np.stack([obs_dict[aid] for aid in agent_ids], axis=0).astype(np.float32, copy=False)


class MADDPGManager:
    """CTDE trainer: joint replay + centralized critics, decentralized (discrete) actors."""

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

        if shared_policy:
            shared = MADDPGAgent(
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
                aid: MADDPGAgent(
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
        self.buffer = MADDPGReplayBuffer(capacity=capacity)

    def _unique_agents(self) -> List[MADDPGAgent]:
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

    def train(self) -> dict[str, float] | None:
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

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        next_state_t = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        dones_next_t = torch.as_tensor(dones_next, dtype=torch.float32, device=self.device)
        active_t = torch.as_tensor(active, dtype=torch.float32, device=self.device)

        action_mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
        next_action_mask_t = torch.as_tensor(next_action_mask, dtype=torch.bool, device=self.device)

        joint_action_onehot = F.one_hot(actions_t, num_classes=self.action_dim).to(torch.float32).view(
            -1, self.n_agents * self.action_dim
        )

        with torch.no_grad():
            next_action_parts = []
            for i, aid in enumerate(self.agent_ids):
                agent = self.agents[aid]
                logits = agent.target_actor(next_obs_t[:, i, :])
                mask = next_action_mask_t[:, i, :]
                a = agent.masked_gumbel_softmax(logits, mask, tau=self.gumbel_tau, hard=True)
                next_action_parts.append(a)
            joint_next_action = torch.cat(next_action_parts, dim=-1)

        critic_losses: list[torch.Tensor] = []
        actor_losses: list[torch.Tensor] = []

        for i, aid in enumerate(self.agent_ids):
            agent = self.agents[aid]
            active_i = active_t[:, i]
            if float(active_i.sum().item()) <= 0.0:
                continue

            q = agent.critic(state_t, joint_action_onehot)
            with torch.no_grad():
                q_next = agent.target_critic(next_state_t, joint_next_action)
                y = rewards_t[:, i] + (self.gamma * (1.0 - dones_next_t[:, i]) * q_next)

            critic_loss = ((q - y) ** 2 * active_i).sum() / (active_i.sum() + 1e-6)
            agent.critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 10.0)
            agent.critic_opt.step()
            critic_losses.append(critic_loss.detach())

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
            q_pi = agent.critic(state_t, joint_curr)
            actor_loss = (-(q_pi) * active_i).sum() / (active_i.sum() + 1e-6)

            agent.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 10.0)
            agent.actor_opt.step()
            actor_losses.append(actor_loss.detach())

        for agent in self._unique_agents():
            soft_update(agent.target_actor, agent.actor, tau=self.tau)
            soft_update(agent.target_critic, agent.critic, tau=self.tau)
            agent.eps.step()

        self.train_step += 1

        if len(critic_losses) == 0 and len(actor_losses) == 0:
            return None

        c = torch.stack(critic_losses).mean() if len(critic_losses) else torch.tensor(0.0, device=self.device)
        a = torch.stack(actor_losses).mean() if len(actor_losses) else torch.tensor(0.0, device=self.device)
        critic = float(c.item())
        actor = float(a.item())
        return {"critic": critic, "actor": actor, "total": float(actor + critic)}

    def save(self, base_path: str) -> None:
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        for aid in self.agent_ids:
            agent = self.agents[aid]
            torch.save(agent.actor.state_dict(), f"{base_path}_actor_agent_{aid}.pt")
            torch.save(agent.critic.state_dict(), f"{base_path}_critic_agent_{aid}.pt")

    def load(self, base_path: str) -> None:
        for aid in self.agent_ids:
            agent = self.agents[aid]
            a_path = f"{base_path}_actor_agent_{aid}.pt"
            c_path = f"{base_path}_critic_agent_{aid}.pt"
            if os.path.exists(a_path):
                agent.actor.load_state_dict(torch.load(a_path, map_location=self.device))
                agent.target_actor.load_state_dict(agent.actor.state_dict())
            if os.path.exists(c_path):
                agent.critic.load_state_dict(torch.load(c_path, map_location=self.device))
                agent.target_critic.load_state_dict(agent.critic.state_dict())
