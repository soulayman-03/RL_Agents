import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

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
        self.gamma = 0.99
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
