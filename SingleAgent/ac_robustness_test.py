import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

try:
    from .agent import ActorCriticAgent
    from .environment import MonoAgentIoTEnv
    from .utils import set_global_seed
except ImportError:
    from agent import ActorCriticAgent
    from environment import MonoAgentIoTEnv
    from utils import set_global_seed


def run_test(agent, env, scenario_name, episodes=20):
    rewards = []
    stalls = []
    all_devices = []

    for _ in range(episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        ep_stalls = 0
        done = False
        step = 0

        while not done and step < 100:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            action = agent.act(state, valid_actions, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if reward == -500.0:
                ep_stalls += 1

            ep_reward += float(reward)
            all_devices.append(int(action))
            state = next_state
            step += 1

        rewards.append(ep_reward)
        stalls.append(ep_stalls)

    return {
        "scenario": scenario_name,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "success_rate": float(np.mean([1 if s == 0 else 0 for s in stalls]) * 100) if stalls else 0.0,
        "avg_stalls": float(np.mean(stalls)) if stalls else 0.0,
        "device_usage": np.bincount(np.asarray(all_devices, dtype=int), minlength=env.num_devices).tolist() if all_devices else [0] * env.num_devices,
    }


def robustness_comparison_ac():
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "resultAC", "robustness")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "single_agent_simplecnn_ac.pth")

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained AC model not found at {MODEL_PATH}. Run ac_train first.")
        return

    set_global_seed(42)
    temp_env = MonoAgentIoTEnv(num_agents=1, num_devices=5, model_types=["simplecnn"], seed=42)
    agent = ActorCriticAgent(state_dim=temp_env.single_state_dim, action_dim=5)
    agent.load(MODEL_PATH)

    results = []

    print("Testing Base Scenario...")
    env_base = MonoAgentIoTEnv(num_agents=1, num_devices=5, model_types=["simplecnn"], seed=42)
    results.append(run_test(agent, env_base, "Base"))

    print("Testing Degraded Device (Slow CPU)...")
    env_slow = MonoAgentIoTEnv(num_agents=1, num_devices=5, model_types=["simplecnn"], seed=42)
    env_slow.resource_manager.devices[0].cpu_speed *= 0.2
    results.append(run_test(agent, env_slow, "Slow CPU (D0)"))

    print("Testing Network Drop (Low Bandwidth)...")
    env_net = MonoAgentIoTEnv(num_agents=1, num_devices=5, model_types=["simplecnn"], seed=42)
    for d in env_net.resource_manager.devices.values():
        d.bandwidth *= 0.1
    results.append(run_test(agent, env_net, "Low BW"))

    print("Testing High Load...")
    env_load = MonoAgentIoTEnv(num_agents=1, num_devices=5, model_types=["simplecnn"], seed=42)
    for t in env_load.task:
        t.computation_demand *= 2.0
    results.append(run_test(agent, env_load, "High Load"))

    scenarios = [r["scenario"] for r in results]
    rewards = [r["avg_reward"] for r in results]
    success = [r["success_rate"] for r in results]

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.bar(scenarios, rewards, color="skyblue")
    plt.title("Actor-Critic Reward across Robustness Scenarios")
    plt.ylabel("Avg Reward")

    plt.subplot(2, 1, 2)
    plt.bar(scenarios, success, color="salmon")
    plt.title("Actor-Critic Success Rate (%) across Robustness Scenarios")
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, 110)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "robustness_comparison.png"))
    plt.close()

    with open(os.path.join(RESULTS_DIR, "robustness_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nAC Robustness tests complete. Results:")
    for r in results:
        print(f"- {r['scenario']}: Reward={r['avg_reward']:.2f}, Success={r['success_rate']:.1f}%")


if __name__ == "__main__":
    robustness_comparison_ac()

