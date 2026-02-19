import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Define paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Add parent directory to path
sys.path.append(PROJECT_ROOT)

try:
    from .environment import MonoAgentIoTEnv
    from .agent import DQNAgent
except ImportError:
    from environment import MonoAgentIoTEnv
    from agent import DQNAgent
from split_inference.cnn_model import SimpleCNN
try:
    from .utils import load_and_remap_weights, set_global_seed
except ImportError:
    from utils import load_and_remap_weights, set_global_seed
import torch

def plot_execution_strategy(trace, results_dir, num_devices=5):
    """
    Plots the device assignment strategy for a single successful episode.
    """
    if not trace:
        return

    devices = [t["device"] for t in trace]
    comps = [t["t_comp"] for t in trace]
    comms = [t["t_comm"] for t in trace]

    plt.figure(figsize=(10, 6))
    # Create step data
    x = np.arange(len(devices) + 1)
    y = devices + [devices[-1]]

    plt.step(x, y, where="post", color="green", linewidth=1.5)
    plt.scatter(range(len(devices)), devices, color="green", zorder=5)

    # Annotate with C and T
    for i, (d, c, t) in enumerate(zip(devices, comps, comms)):
        plt.text(i, d + 0.1, f"C:{c:.1f}\nT:{t:.1f}", ha="center", va="bottom", fontsize=8)

    plt.yticks(range(num_devices), [f"Device {i}" for i in range(num_devices)])
    plt.xticks(range(len(devices) + 1))
    plt.xlabel("Layer Index")
    plt.ylabel("Device ID")
    plt.title(f"Execution Strategy - DQN (Trace Length: {len(devices)}/{len(devices)})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "execution_strategy.png"))
    plt.close()

def plot_execution_flow(trace, max_progress, num_devices, seed, results_dir, filename):
    """Helper to plot the execution strategy trace for a single episode."""
    plt.figure(figsize=(10, 6))
    layers = [t['layer'] for t in trace]
    devices = [t['device'] for t in trace]
    
    # Add final point for visualization
    layers.append(max_progress)
    devices.append(devices[-1])
    
    plt.step(layers, devices, where='post', marker='o', color='green', label='Allocation')
    
    # Add latency annotations
    for t in trace:
        plt.text(t['layer'], t['device'] + 0.1, f"C:{t['t_comp']:.1f}\n T:{t['t_comm']:.1f}", 
                 fontsize=8, verticalalignment='bottom', horizontalalignment='center',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    plt.yticks(range(num_devices), [f"Device {d}" for d in range(num_devices)])
    plt.xticks(range(max_progress + 1))
    plt.xlim(-0.5, max_progress + 0.5)
    plt.xlabel("Layer Index")
    plt.ylabel("Device ID")
    plt.title(f"Execution Strategy - Seed {seed} (Trace Length: {len(trace)}/{max_progress})")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()

def plot_layer_latency(all_episode_traces, task, results_dir, filename):

    import numpy as np
    import matplotlib.pyplot as plt
    import os

    num_layers = len(task)
    comp = np.zeros(num_layers)
    comm = np.zeros(num_layers)
    counts = np.zeros(num_layers)

    for trace in all_episode_traces:
        for t in trace:

            # Skip invalid formats (tuples etc)
            if not isinstance(t, dict):
                continue
            if "layer" not in t:
                continue

            layer = int(t["layer"])
            comp[layer] += float(t.get("t_comp", 0))
            comm[layer] += float(t.get("t_comm", 0))
            counts[layer] += 1

    counts[counts == 0] = 1
    comp /= counts
    comm /= counts

    layer_names = [layer.name for layer in task]
    x = np.arange(num_layers)
    width = 0.4

    plt.figure(figsize=(14,5))
    plt.bar(x - width/2, comp, width, label="Computation")
    plt.bar(x + width/2, comm, width, label="Communication")

    plt.xticks(x, layer_names, rotation=60)
    plt.ylabel("Latency (s)")
    plt.xlabel("Layers")
    plt.title("Average Layer-wise Latency Breakdown")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(results_dir, filename))
    plt.close()

def evaluate_single_agent():
    NUM_EPISODES = 50
    NUM_DEVICES = 5
    MODEL_TYPE = "simplecnn"
    MODEL_TYPE = "simplecnn"
    SEEDS = [42, 43, 44, 45, 46]
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "resultDQN", "eval")
    MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "single_agent_simplecnn.pth")
    HISTORY_PATH = os.path.join(SCRIPT_DIR, "models", "single_train_history.npy")
    STALL_HISTORY_PATH = os.path.join(SCRIPT_DIR, "models", "single_stall_history.npy")
    
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    print(f"=== Starting Single Agent Evaluation ({MODEL_TYPE}) ===")
    
    # Setup Agent (weights reused across seeds)
    dummy_env = MonoAgentIoTEnv(num_agents=1, num_devices=NUM_DEVICES, model_types=[MODEL_TYPE], seed=SEEDS[0])
    agent = DQNAgent(state_dim=dummy_env.single_state_dim, action_dim=NUM_DEVICES)

    if os.path.exists(MODEL_PATH):
        agent.load(MODEL_PATH)
        print(f"Loaded trained agent from {MODEL_PATH}")
    else:
        print(f"WARNING: No trained model found at {MODEL_PATH}. Using random weights.")

    agent.epsilon = 0.0 # No exploration during evaluation

    # 2. Run Test Episodes across seeds
    all_seed_rewards = []
    all_seed_stalls = []
    all_seed_t_comp = []
    all_seed_t_comm = []
    all_seed_traces = []
    sample_trace = None
    sample_trace_full = None
    
    for seed in SEEDS:
        set_global_seed(seed)
        env = MonoAgentIoTEnv(num_agents=1, num_devices=NUM_DEVICES, model_types=[MODEL_TYPE], seed=seed)

        max_progress = len(env.task)

        test_rewards = []
        test_stalls = []
        test_t_comp = []
        test_t_comm = []

        all_episode_traces = []   # <<<<<< IMPORTANT

        print(f"Running {NUM_EPISODES} test episodes (seed={seed})...")

        for e in range(NUM_EPISODES):
            state, _ = env.reset()
            done = False

            ep_reward = 0
            ep_stalls = 0
            ep_t_comp = 0
            ep_t_comm = 0

            trace = []
            step = 0

            while not done and step < 100:

                valid_actions = env.get_valid_actions()
                action = agent.act(state, valid_actions)
                prev_progress = env.progress

                next_state, reward, terminated, truncated, info = env.step(action)

                if reward == -500.0:
                    ep_stalls += 1
                else:
                    ep_t_comp += info.get('t_comp', 0)
                    ep_t_comm += info.get('t_comm', 0)

                if env.progress > prev_progress:
                    layer = env.task[prev_progress]
                    device = env.resource_manager.devices[int(action)]

                    trace_entry = {
                        "layer": prev_progress,
                        "device": int(action),
                        "t_comp": info.get('t_comp', 0),
                        "t_comm": info.get('t_comm', 0)
                    }
                    trace.append(trace_entry)

                ep_reward += reward
                state = next_state
                done = terminated
                step += 1

            # Save metrics
            test_rewards.append(ep_reward)
            test_stalls.append(ep_stalls)
            test_t_comp.append(ep_t_comp)
            test_t_comm.append(ep_t_comm)

            all_seed_traces.append(trace)
            all_episode_traces.append(trace)

            if sample_trace_full is None and done and ep_stalls == 0 and trace:
                sample_trace_full = trace

            # Logging
            if e % 10 == 0:
                if ep_stalls > 0:
                    status = "FAILED (stall)"
                elif done:
                    status = "COMPLETED"
                else:
                    status = f"STUCK at Layer {env.progress}"

                print(f" Episode {e:2} | Reward: {ep_reward:8.2f} | "
                    f"T_comp: {ep_t_comp:6.2f} | T_comm: {ep_t_comm:6.2f} | "
                    f"Stalls: {ep_stalls:3} | {status}")

        # =========================
        # NEW PLOT : Layer Latency
        # =========================
        plot_layer_latency(
            all_episode_traces,
            env.task,
            RESULTS_DIR,
            f"layer_latency_avg_seed_{seed}.png"
        )

        # store global stats
        all_seed_rewards.extend(test_rewards)
        all_seed_stalls.extend(test_stalls)
        all_seed_t_comp.extend(test_t_comp)
        all_seed_t_comm.extend(test_t_comm)

    # 3.c Plot Device Usage Balance
    all_devices_used = []
    if sample_trace_full: # Use the full traces collected
        for ep_trace in all_seed_traces:
            for step in ep_trace:
                all_devices_used.append(step['device'])

    # 4. Plot Execution Flow (Summary Sample)
    if sample_trace_full:
        plot_execution_strategy(sample_trace_full, RESULTS_DIR, num_devices=NUM_DEVICES)
        plot_execution_flow(sample_trace_full, max_progress, NUM_DEVICES, "Sample", RESULTS_DIR, "sample_execution_flow.png")

    avg_reward = float(np.mean(all_seed_rewards)) if all_seed_rewards else 0.0
    std_reward = float(np.std(all_seed_rewards)) if all_seed_rewards else 0.0
    avg_stalls = float(np.mean(all_seed_stalls)) if all_seed_stalls else 0.0
    print(f"\nSummary: avg_reward={avg_reward:.2f} | std_reward={std_reward:.2f} | avg_stalls={avg_stalls:.2f}")
    print(f"Evaluation Complete. Results in {RESULTS_DIR}/")

if __name__ == "__main__":
    evaluate_single_agent()
