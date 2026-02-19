import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch

# Define paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(PROJECT_ROOT)

try:
    from .agent import DQNAgent, DeepSARSAAgent, ActorCriticAgent
    from .environment import MonoAgentIoTEnv
    from .utils import set_global_seed
except ImportError:
    from agent import DQNAgent, DeepSARSAAgent, ActorCriticAgent
    from environment import MonoAgentIoTEnv
    from utils import set_global_seed
from split_inference.cnn_model import SimpleCNN

def smooth_curve(data, window_size=50):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def _load_npy_if_exists(path):
    if not path or not os.path.exists(path):
        return None
    try:
        return np.load(path, allow_pickle=False)
    except Exception as e:
        print(f"WARNING: Failed to load {path}: {e}")
        return None

def _plot_training_reward_comparison(dqn_rewards, sarsa_rewards, ac_rewards, results_dir, window=50):
    series = []
    if dqn_rewards is not None:
        series.append(("DQN", "blue", np.asarray(dqn_rewards, dtype=float).tolist()))
    if sarsa_rewards is not None:
        series.append(("SARSA", "green", np.asarray(sarsa_rewards, dtype=float).tolist()))
    if ac_rewards is not None:
        series.append(("Actor-Critic", "purple", np.asarray(ac_rewards, dtype=float).tolist()))

    if not series:
        return

    plt.figure(figsize=(10, 6))
    for name, color, rewards in series:
        plt.plot(rewards, color=color, alpha=0.15, label=f"{name} (Raw)")

    for name, color, rewards in series:
        sm = smooth_curve(rewards, window_size=window)
        if len(rewards) >= window:
            plt.plot(range(window - 1, len(rewards)), sm, color=color, linewidth=2, label=f"{name} (MA{window})")
        else:
            plt.plot(sm, color=color, linewidth=2, label=f"{name} (Smoothed)")

    plt.title("Training Reward Comparison (DQN vs SARSA vs Actor-Critic)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_reward_comparison.png"))
    plt.close()

def evaluate_agent_detailed(agent, env, episodes=50):
    rewards = []
    latencies = []
    device_assignments = []
    
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0 # No exploration for evaluation
    
    for e in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        step = 0
        while not done and step < 100:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
                
            action = agent.act(state, valid_actions)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Track latency
            if 't_comp' in info and 't_comm' in info:
                latencies.append(info['t_comp'] + info['t_comm'])
            
            device_assignments.append(int(action))
            episode_reward += reward
            state = next_state
            done = terminated or truncated
            step += 1
            
        rewards.append(episode_reward)
        
    return {
        "avg_reward": np.mean(rewards),
        "avg_latency": np.mean(latencies) if latencies else 0,
        "assignments": device_assignments,
        "success_rate": np.mean([1 if r > -400 else 0 for r in rewards]) * 100
    }

def compare():
    # Paths
    DQN_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "single_agent_simplecnn.pth")
    SARSA_MODEL_PATHS = [
        os.path.join(SCRIPT_DIR, "models", "sarsa_agent.pth"),
        os.path.join(SCRIPT_DIR, "results", "resultSARSA", "models", "sarsa_agent.pth"),
    ]
    AC_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "single_agent_simplecnn_ac.pth")
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "comparison")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Training history (reward) paths
    DQN_TRAIN_REWARD_PATH = os.path.join(SCRIPT_DIR, "models", "single_train_history.npy")
    SARSA_TRAIN_REWARD_PATHS = [
        os.path.join(SCRIPT_DIR, "models", "sarsa_train_history.npy"),
    ]
    AC_TRAIN_REWARD_PATHS = [
        os.path.join(SCRIPT_DIR, "models", "single_ac_train_history.npy"),
        os.path.join(SCRIPT_DIR, "results", "resultAC", "train", "single_ac_train_history.npy"),
    ]

    dqn_train_rewards = _load_npy_if_exists(DQN_TRAIN_REWARD_PATH)
    sarsa_train_rewards = None
    for p in SARSA_TRAIN_REWARD_PATHS:
        sarsa_train_rewards = _load_npy_if_exists(p)
        if sarsa_train_rewards is not None:
            break

    ac_train_rewards = None
    for p in AC_TRAIN_REWARD_PATHS:
        ac_train_rewards = _load_npy_if_exists(p)
        if ac_train_rewards is not None:
            break

    if dqn_train_rewards is None:
        print(f"WARNING: DQN training history not found: {DQN_TRAIN_REWARD_PATH}")
    if sarsa_train_rewards is None:
        print(f"WARNING: SARSA training history not found in: {SARSA_TRAIN_REWARD_PATHS}")
    if ac_train_rewards is None:
        print(f"WARNING: Actor-Critic training history not found in: {AC_TRAIN_REWARD_PATHS}")

    _plot_training_reward_comparison(dqn_train_rewards, sarsa_train_rewards, ac_train_rewards, RESULTS_DIR, window=50)

    # Load environments
    set_global_seed(42)
    env = MonoAgentIoTEnv(num_agents=1, num_devices=5, model_types=["simplecnn"], seed=42)
    
    # Initialize Agents
    dqn_agent = DQNAgent(state_dim=env.single_state_dim, action_dim=5)
    if os.path.exists(DQN_MODEL_PATH):
        dqn_agent.load(DQN_MODEL_PATH)
    else:
        print(f"WARNING: DQN model not found: {DQN_MODEL_PATH}")
    
    sarsa_agent = DeepSARSAAgent(state_dim=env.single_state_dim, action_dim=5)
    sarsa_model_path = None
    for p in SARSA_MODEL_PATHS:
        if os.path.exists(p):
            sarsa_model_path = p
            break
    if sarsa_model_path is not None:
        sarsa_agent.load(sarsa_model_path)
    else:
        print(f"WARNING: SARSA model not found. Looked in: {SARSA_MODEL_PATHS}")

    ac_agent = ActorCriticAgent(state_dim=env.single_state_dim, action_dim=5)
    if os.path.exists(AC_MODEL_PATH):
        ac_agent.load(AC_MODEL_PATH)
    else:
        print(f"WARNING: Actor-Critic model not found: {AC_MODEL_PATH}")
    
    # Assign CV model
    cv_model = SimpleCNN()
    dqn_agent.assign_inference_model(cv_model)
    sarsa_agent.assign_inference_model(cv_model)
    ac_agent.assign_inference_model(cv_model)

    print("Evaluating DQN Agent...")
    dqn_results = evaluate_agent_detailed(dqn_agent, env)
    
    print("Evaluating SARSA Agent...")
    sarsa_results = evaluate_agent_detailed(sarsa_agent, env)

    print("Evaluating Actor-Critic Agent...")
    ac_results = evaluate_agent_detailed(ac_agent, env)

    # Plotting Categorized Results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(
        ["DQN", "SARSA", "AC"],
        [dqn_results["avg_latency"], sarsa_results["avg_latency"], ac_results["avg_latency"]],
        color=["blue", "green", "purple"],
    )
    plt.title("Result 1: Performance (Latency)")
    plt.ylabel("Latency (lower is better)")
    
    plt.subplot(1, 2, 2)
    # Resource Allocation: Load Balancing (Std Dev of device usage)
    dqn_counts = np.bincount(dqn_results['assignments'], minlength=5)
    sarsa_counts = np.bincount(sarsa_results['assignments'], minlength=5)
    ac_counts = np.bincount(ac_results["assignments"], minlength=5)
    dqn_std = np.std(dqn_counts)
    sarsa_std = np.std(sarsa_counts)
    ac_std = np.std(ac_counts)
    plt.bar(["DQN", "SARSA", "AC"], [dqn_std, sarsa_std, ac_std], color=["blue", "green", "purple"])
    plt.title("Result 2: Resource Balance (Std Dev)")
    plt.ylabel("Load Imbalance (lower is better)")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "categorized_analysis.png"))

    # Final Summary Report
    print("\n" + "="*50)
    print("FINAL COMPARISON REPORT: DQN VS SARSA VS ACTOR-CRITIC")
    print("="*50)
    
    print("\n[RESULT 1 - PERFORMANCE (LATENCY)]")
    print(f" - DQN Latency:   {dqn_results['avg_latency']:.4f}")
    print(f" - SARSA Latency: {sarsa_results['avg_latency']:.4f}")
    print(f" - AC Latency:    {ac_results['avg_latency']:.4f}")

    print("\n[RESULT 2 - RESOURCE ALLOCATION]")
    print(f" - DQN Load Imbalance (StdDev):   {dqn_std:.2f}")
    print(f" - SARSA Load Imbalance (StdDev): {sarsa_std:.2f}")
    print(f" - AC Load Imbalance (StdDev):    {ac_std:.2f}")

    print("\n[RESULT 3 - SUCCESS RATE (%)]")
    print(f" - DQN Success:   {dqn_results['success_rate']:.1f}%")
    print(f" - SARSA Success: {sarsa_results['success_rate']:.1f}%")
    print(f" - AC Success:    {ac_results['success_rate']:.1f}%")
    print("="*50)

if __name__ == "__main__":
    compare()
