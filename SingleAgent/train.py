import sys
import os

# Define paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Add parent directory to path to allow imports from rl_pdnn and split_inference
sys.path.append(PROJECT_ROOT)

import torch
import numpy as np
from agent import DQNAgent
from environment import MonoAgentIoTEnv
from utils import load_and_remap_weights, set_global_seed
from split_inference.cnn_model import SimpleCNN

def train_single_agent():
    """
    Trains a single agent to schedule its DNN layers across 5 devices.
    Uses the MultiAgentIoTEnv configured with num_agents=1.
    """
    NUM_AGENTS = 1
    NUM_DEVICES = 5
    MODEL_TYPES = ["simplecnn"]
    EPISODES = 3000
    SEED = 42
    
    print(f"=== Starting Single Agent Training (Model: {MODEL_TYPES[0]}) ===")
    
    # Initialize Environment
    set_global_seed(SEED)
    env = MonoAgentIoTEnv(num_agents=NUM_AGENTS, num_devices=NUM_DEVICES, model_types=MODEL_TYPES, seed=SEED)
    
    # Initialize Agent
    agent = DQNAgent(state_dim=env.single_state_dim, action_dim=NUM_DEVICES)
    
    # Prepare the CNN model and load pre-trained weights
    cv_model = SimpleCNN()
    weight_path = os.path.join(PROJECT_ROOT, "split_inference", "mnist_simplecnn.pth")
    
    success = load_and_remap_weights(cv_model, weight_path, "simplecnn")
    if success:
        print(f"Successfully loaded and remapped weights from {weight_path}")
    else:
        print(f"Warning: Could not load weights from {weight_path}. Using random initialization.")
    
    agent.assign_inference_model(cv_model)
    
    # Training Loop
    history_rewards = []
    history_stalls = []
    
    for e in range(EPISODES):
        # Gymnasium reset returns (obs, info)
        states, _ = env.reset()
        episode_reward = 0
        stalls = 0
        done = False
        
        step = 0
        MAX_STEPS = 100
        
        while not done and step < MAX_STEPS:
            # MultiAgentEnv expects a list of actions
            valid_actions = env.get_valid_actions(0)
            action = agent.act(states[0], valid_actions)
            actions = [action]
            
            # Gymnasium step returns (obs, reward, terminated, truncated, info)
            next_states, rewards, now_dones, truncated, _ = env.step(actions)
            
            # Record stalls (constraint violations)
            if rewards[0] == -500.0:
                stalls += 1
            
            # Store experience
            agent.remember(states[0], actions[0], rewards[0], next_states[0], now_dones[0])
            episode_reward += rewards[0]
            
            states = next_states
            done = now_dones[0]
            step += 1
            
        # Replay at end of episode for faster training
        for _ in range(5):
            agent.replay()
            
        history_rewards.append(episode_reward)
        history_stalls.append(stalls)
        
        # Log EVERY episode as requested
        respected = "YES" if stalls == 0 else f"NO ({stalls} violations)"
        print(f"Episode {e}/{EPISODES} | Reward: {episode_reward:8.2f} | Stalls: {stalls:2} | Constraints Respected: {respected} | Epsilon: {agent.epsilon:.2f}")

    # Save trained RL agent
    MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)
    
    agent.save(os.path.join(MODELS_DIR, "single_agent_simplecnn.pth"))
    np.save(os.path.join(MODELS_DIR, "single_train_history.npy"), np.array(history_rewards))
    np.save(os.path.join(MODELS_DIR, "single_stall_history.npy"), np.array(history_stalls))
    
    print("\nTraining Complete.")
    print(f"Model saved to {os.path.join(MODELS_DIR, 'single_agent_simplecnn.pth')}")

if __name__ == "__main__":
    train_single_agent()
