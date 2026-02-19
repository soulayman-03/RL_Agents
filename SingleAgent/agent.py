import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torch.distributions import Categorical

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Hyperparameters
        self.lr = 0.0005
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995 # Much slower decay for 2000 episodes
        self.batch_size = 64 # Larger batch
        self.memory = deque(maxlen=5000) # Larger memory
        self.target_update_freq = 1000
        self.train_step = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
        # The assigned CV model (SimpleCNN, DeepCNN, etc.)
        self.inference_model = None

    def assign_inference_model(self, model):
        """Assigns a specific PyTorch model to this agent."""
        self.inference_model = model
        # Move to same device
        if self.inference_model:
            self.inference_model.to(self.device)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
        
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.update_target_network()

    def act(self, state, valid_actions=None):
        if valid_actions is None or len(valid_actions) == 0:
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_dim)
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_t)
            return torch.argmax(q_values).item()

        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t).squeeze(0)

        mask = torch.full((self.action_dim,), float("-inf"), device=self.device)
        mask[valid_actions] = 0.0
        masked_q = q_values + mask
        return torch.argmax(masked_q).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).to(self.device)
        
        # Q(s, a)
        curr_q = self.policy_net(states).gather(1, actions).squeeze(1)
        
        # Q'(s', a')
        next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + (self.gamma * next_q * (1 - dones))
        
        loss = self.criterion(curr_q, target_q.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class DeepSARSAAgent(DQNAgent):
    """
    Deep SARSA Agent: On-policy version of DQN.
    Updates using (s, a, r, s', a') where a' is the actual next action taken.
    """
    def remember(self, state, action, reward, next_state, next_action, done):
        self.memory.append((state, action, reward, next_state, next_action, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)
        next_actions = torch.LongTensor(np.array([t[4] for t in minibatch])).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(np.array([t[5] for t in minibatch])).to(self.device)
        
        # Q(s, a)
        curr_q = self.policy_net(states).gather(1, actions).squeeze(1)
        
        # SARSA: Q(s, a) = r + gamma * Q(s', a')
        # We use the target network for Q(s', a') for stability
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            next_q = next_q_values.gather(1, next_actions).squeeze(1)
            target_q = rewards + (self.gamma * next_q * (1 - dones))
        
        loss = self.criterion(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.policy_head = nn.Linear(256, action_dim)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value


class ActorCriticAgent:
    """
    Simple on-policy Actor-Critic (A2C-style) for discrete action spaces.
    Supports action masking via `valid_actions` similarly to DQNAgent.act.
    """

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Learning rates: keep critic "less confident" by using a smaller LR for value head.
        self.actor_lr = 3e-4
        self.critic_lr = self.actor_lr / 5.0
        self.gamma = 0.99
        # Entropy schedule (forces exploration early, then anneals).
        self.entropy_coef_start = 0.02
        self.entropy_coef_end = 0.001
        self.entropy_decay_steps = 2000
        self.entropy_coef = float(self.entropy_coef_start)
        self.update_calls = 0
        self.value_coef = 0.5
        # GAE(lambda) for modern Actor-Critic advantage estimation.
        self.gae_lambda = 0.95
        self.max_grad_norm = 0.5

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ActorCritic(state_dim, action_dim).to(self.device)

        # Param groups: shared + actor head at actor LR; critic head at smaller LR.
        actor_params = list(self.net.fc1.parameters()) + list(self.net.fc2.parameters()) + list(self.net.policy_head.parameters())
        critic_params = list(self.net.value_head.parameters())
        self.optimizer = optim.Adam(
            [
                {"params": actor_params, "lr": self.actor_lr},
                {"params": critic_params, "lr": self.critic_lr},
            ]
        )

        self.inference_model = None

    def set_entropy_schedule(self, start: float, end: float, decay_steps: int) -> None:
        self.entropy_coef_start = float(start)
        self.entropy_coef_end = float(end)
        self.entropy_decay_steps = max(1, int(decay_steps))
        self.update_calls = 0
        self.entropy_coef = float(self.entropy_coef_start)

    def set_gae_lambda(self, gae_lambda: float) -> None:
        self.gae_lambda = float(max(0.0, min(1.0, gae_lambda)))

    def _current_entropy_coef(self) -> float:
        if self.entropy_decay_steps <= 1:
            self.entropy_coef = float(self.entropy_coef_end)
            return self.entropy_coef
        frac = min(float(self.update_calls) / float(self.entropy_decay_steps), 1.0)
        self.entropy_coef = float(self.entropy_coef_start + frac * (self.entropy_coef_end - self.entropy_coef_start))
        return self.entropy_coef

    def assign_inference_model(self, model):
        self.inference_model = model
        if self.inference_model:
            self.inference_model.to(self.device)

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device))

    def _masked_logits(self, logits: torch.Tensor, valid_actions):
        if valid_actions is None or len(valid_actions) == 0:
            return logits
        mask = torch.full((self.action_dim,), float("-inf"), device=logits.device)
        mask[valid_actions] = 0.0
        return logits + mask

    @torch.no_grad()
    def act(self, state, valid_actions=None, deterministic: bool = True) -> int:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, _ = self.net(state_t)
        logits = logits.squeeze(0)
        logits = self._masked_logits(logits, valid_actions)

        if deterministic:
            return int(torch.argmax(logits).item())

        dist = Categorical(logits=logits)
        return int(dist.sample().item())

    def select_action(self, state, valid_actions=None):
        """
        Returns (action, log_prob, value, entropy) for training.
        """
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(state_t)
        logits = logits.squeeze(0)
        value = value.squeeze(0)

        logits = self._masked_logits(logits, valid_actions)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return int(action.item()), log_prob, value, entropy

    @torch.no_grad()
    def predict_value(self, state) -> float:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, value = self.net(state_t)
        return float(value.squeeze(0).item())

    def update(self, log_probs, values, rewards, entropies, bootstrap_value: float = 0.0, dones=None):
        """
        Compute (GAE) advantages/returns and perform one optimization step.
        Inputs are per-timestep tensors/lists collected during one rollout/episode.
        """
        if len(rewards) == 0:
            return 0.0, 0.0, 0.0

        log_probs_t = torch.stack(log_probs)
        values_t = torch.stack(values)
        entropies_t = torch.stack(entropies)

        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)

        entropy_coef = self._current_entropy_coef()

        # If dones are provided, use GAE(lambda). Otherwise, fall back to simple discounted returns.
        use_gae = dones is not None and len(dones) == len(rewards)
        if use_gae:
            dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
            next_values_t = torch.cat(
                [
                    values_t[1:],
                    torch.as_tensor([bootstrap_value], dtype=torch.float32, device=self.device),
                ]
            )
            deltas = rewards_t + self.gamma * next_values_t * (1.0 - dones_t) - values_t

            gae = torch.zeros((), dtype=torch.float32, device=self.device)
            adv_list = []
            for t in reversed(range(len(rewards))):
                gae = deltas[t] + self.gamma * self.gae_lambda * (1.0 - dones_t[t]) * gae
                adv_list.append(gae)
            adv_list.reverse()
            advantages = torch.stack(adv_list)
            returns_t = advantages + values_t
        else:
            returns = []
            R = torch.as_tensor(bootstrap_value, dtype=torch.float32, device=self.device)
            for r in reversed(rewards):
                R = torch.as_tensor(r, dtype=torch.float32, device=self.device) + self.gamma * R
                returns.append(R)
            returns.reverse()
            returns_t = torch.stack(returns)
            advantages = returns_t - values_t

        # Normalize advantage for better stability (A = (A - mean) / (std + eps))
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False)
        norm_advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        policy_loss = -(log_probs_t * norm_advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        entropy_loss = -entropies_t.mean()

        loss = policy_loss + self.value_coef * value_loss + entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.update_calls += 1
        return float(policy_loss.item()), float(value_loss.item()), float(entropies_t.mean().item())
