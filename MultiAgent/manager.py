import torch.nn as nn
from typing import List, Dict
import numpy as np
from MultiAgent.agent import DQNAgent

class MultiAgentManager:
    """
    Manages a collection of agents, collecting actions and handling training.
    """
    def __init__(self, agent_ids: List[int], state_dim: int, action_dim: int, lr=0.0005):
        self.agents: Dict[int, DQNAgent] = {
            aid: DQNAgent(aid, state_dim, action_dim, lr=lr) for aid in agent_ids
        }

    def get_actions(self, observations: Dict[int, np.ndarray], valid_actions: Dict[int, List[int]] = None) -> Dict[int, int]:
        """
        Collects actions from all agents for the given observations.
        """
        actions = {}
        for agent_id, obs in observations.items():
            v_actions = valid_actions.get(agent_id) if valid_actions else None
            actions[agent_id] = self.agents[agent_id].act(obs, v_actions)
        return actions

    def remember(self, observations, actions, rewards, next_observations, dones):
        """
        Stores experiences for all agents.
        """
        for agent_id in self.agents:
            if agent_id in observations and agent_id in actions:
                self.agents[agent_id].remember(
                    observations[agent_id],
                    actions[agent_id],
                    rewards[agent_id],
                    next_observations[agent_id],
                    dones[agent_id]
                )

    def train(self):
        """
        Triggers training for all agents.
        """
        for agent in self.agents.values():
            agent.replay()

    def set_epsilon(self, epsilon: float):
        for agent in self.agents.values():
            agent.epsilon = epsilon

    def save_agents(self, base_path: str):
        for agent_id, agent in self.agents.items():
            agent.save(f"{base_path}_agent_{agent_id}.pt")

    def load_agents(self, base_path: str):
        import os
        for agent_id, agent in self.agents.items():
            path = f"{base_path}_agent_{agent_id}.pt"
            if os.path.exists(path):
                agent.load(path)

    def assign_models(self, models_dict: Dict[int, nn.Module]):
        """Assigns real CNN models to specific agents."""
        for agent_id, model in models_dict.items():
            if agent_id in self.agents:
                self.agents[agent_id].assign_inference_model(model)
