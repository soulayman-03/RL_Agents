import os
import sys
import time

import numpy as np

# Allow running both:
# - as a module:  python -m MultiAgentMADDPG.train
# - as a script:  python MultiAgentMADDPG/train.py
if __package__:
    from .environment import MultiAgentIoTEnv
    from .manager import MADDPGManager
    from .plots import plot_training_trends, plot_per_agent_training_rewards
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from MultiAgentMADDPG.environment import MultiAgentIoTEnv
    from MultiAgentMADDPG.manager import MADDPGManager
    from MultiAgentMADDPG.plots import plot_training_trends, plot_per_agent_training_rewards


def print_device_info(resource_manager):
    print("\n" + "=" * 50)
    print("DEVICE SPECIFICATIONS")
    print("=" * 50)
    for d_id, d in resource_manager.devices.items():
        print(
            f"Device {d_id}: CPU={d.cpu_speed:.2f}, RAM={d.memory_capacity:.1f}MB, "
            f"BW={d.bandwidth:.1f}Mbps, Privacy={d.privacy_clearance}"
        )
    print("=" * 50 + "\n")


def train():
    NUM_AGENTS = 3
    NUM_DEVICES = 5
    EPISODES = 5000
    # 3 large models for 3 agents
    MODEL_TYPES = ["simplecnn", "deepcnn", "miniresnet"]
    TERMINATE_ON_FAIL = True

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR = os.path.join(SCRIPT_DIR, "models")
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    env = MultiAgentIoTEnv(
        num_agents=NUM_AGENTS,
        num_devices=NUM_DEVICES,
        model_types=MODEL_TYPES,
        seed=42,
        shuffle_allocation_order=True,
    )

    manager = MADDPGManager(
        agent_ids=list(range(NUM_AGENTS)),
        obs_dim=env.single_state_dim,
        action_dim=env.num_devices,
        state_dim=NUM_AGENTS * env.single_state_dim,
        batch_size=256,
        shared_policy=False,
    )

    episode_team_rewards: list[float] = []
    episode_team_rewards_sum: list[float] = []
    losses: list[float] = []
    eps_history: list[float] = []
    episode_steps: list[int] = []
    episode_success: list[int] = []
    agent_reward_history = {i: [] for i in range(NUM_AGENTS)}
    agent_success_history = {i: [] for i in range(NUM_AGENTS)}
    total_env_steps = 0
    total_fail_episodes = 0

    def _fmt_seconds(sec: float) -> str:
        sec = max(0.0, float(sec))
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    t0 = time.time()
    print(
        "MADDPG Training (CTDE)\n"
        f"  Agents: {NUM_AGENTS} | Devices: {NUM_DEVICES} | Episodes: {EPISODES}\n"
        f"  Models: {MODEL_TYPES}\n"
        f"  SaveDir: {SAVE_DIR}\n"
        f"  Results: {RESULTS_DIR}\n"
        f"  ShuffleAllocationOrder: {True}\n"
        f"  TerminateOnFail: {TERMINATE_ON_FAIL}\n"
    )
    print_device_info(env.resource_manager)

    for ep in range(EPISODES):
        obs, _ = env.reset()
        done = {i: False for i in range(NUM_AGENTS)}  # done before step

        team_return_mean = 0.0
        team_return_sum = 0.0
        steps = 0
        ep_failed = False
        agent_ep_reward = {i: 0.0 for i in range(NUM_AGENTS)}
        agent_failed = {i: False for i in range(NUM_AGENTS)}

        while not all(done.values()):
            valid_actions = env.get_valid_actions()
            actions = manager.get_actions(obs, valid_actions)

            next_obs, rewards, next_done, truncated, infos = env.step(actions)
            steps += 1

            for aid in range(NUM_AGENTS):
                if not done.get(aid, False):
                    agent_ep_reward[aid] += float(rewards.get(aid, 0.0))

            any_fail = False
            for aid, info in infos.items():
                if isinstance(info, dict) and info.get("success") is False:
                    any_fail = True
                    agent_failed[int(aid)] = True

            if any_fail and TERMINATE_ON_FAIL:
                ep_failed = True
                next_done = {i: True for i in range(NUM_AGENTS)}

            next_valid_actions = env.get_valid_actions()
            manager.remember(
                obs_dict=obs,
                actions_dict=actions,
                rewards_dict=rewards,
                next_obs_dict=next_obs,
                dones_dict=next_done,
                active_before_dict=done,
                valid_actions=valid_actions,
                next_valid_actions=next_valid_actions,
            )

            loss = manager.train()
            if loss is not None:
                losses.append(float(loss))

            team_r = float(sum(rewards.values())) / float(NUM_AGENTS)
            team_return_mean += team_r
            team_return_sum += float(sum(rewards.values()))

            done = next_done
            obs = next_obs

        finished_steps = max(1, steps)
        episode_team_rewards.append(team_return_mean / float(finished_steps))
        episode_team_rewards_sum.append(team_return_sum)
        episode_steps.append(steps)
        episode_success.append(0 if ep_failed else 1)

        for aid in range(NUM_AGENTS):
            agent_reward_history[aid].append(agent_ep_reward[aid])
            finished = bool(env.agent_progress.get(aid, 0) >= len(env.tasks.get(aid, [])))
            agent_success_history[aid].append(1 if (not agent_failed[aid] and finished) else 0)

        total_env_steps += steps
        if ep_failed:
            total_fail_episodes += 1

        eps = manager._unique_agents()[0].eps.epsilon
        eps_history.append(float(eps))

        if (ep + 1) % 50 == 0:
            avg = float(np.mean(episode_team_rewards_sum[-50:]))
            avg_steps = float(np.mean(episode_steps[-50:]))
            avg_loss = float(np.mean(losses[-200:])) if len(losses) >= 1 else float("nan")
            elapsed = time.time() - t0
            eps_per_sec = (ep + 1) / max(elapsed, 1e-9)
            eta = (EPISODES - (ep + 1)) / max(eps_per_sec, 1e-9)
            replay_len = len(manager.buffer)

            agent_stats = []
            for aid in range(NUM_AGENTS):
                a_reward = float(np.mean(agent_reward_history[aid][-50:]))
                a_success = float(np.mean(agent_success_history[aid][-50:]) * 100.0)
                agent_stats.append(f"A{aid}: {a_reward:.1f} ({a_success:.0f}%)")
            stats_str = " | ".join(agent_stats)
            print(
                f"Ep {ep+1:4d}/{EPISODES} - Avg: {avg:.1f} - [{stats_str}] - Eps: {eps:.3f} "
                f"- Loss: {avg_loss:.4f} - Steps(50): {avg_steps:.1f} - Replay: {replay_len} "
                f"- Elapsed: {_fmt_seconds(elapsed)} - ETA: {_fmt_seconds(eta)}"
            )

    base = os.path.join(SAVE_DIR, "maddpg")
    manager.save(base)
    np.save(os.path.join(RESULTS_DIR, "team_reward_history.npy"), np.asarray(episode_team_rewards, dtype=np.float32))
    np.save(os.path.join(RESULTS_DIR, "team_reward_sum_history.npy"), np.asarray(episode_team_rewards_sum, dtype=np.float32))
    np.save(os.path.join(RESULTS_DIR, "loss_history.npy"), np.asarray(losses, dtype=np.float32))
    np.save(os.path.join(RESULTS_DIR, "epsilon_history.npy"), np.asarray(eps_history, dtype=np.float32))
    np.save(os.path.join(RESULTS_DIR, "episode_steps.npy"), np.asarray(episode_steps, dtype=np.int32))
    np.save(os.path.join(RESULTS_DIR, "episode_success.npy"), np.asarray(episode_success, dtype=np.int32))
    np.savez_compressed(
        os.path.join(RESULTS_DIR, "agent_success_history.npz"),
        **{f"agent_{i}": np.asarray(agent_success_history[i], dtype=np.int32) for i in range(NUM_AGENTS)},
    )

    plot_training_trends(
        out_path=os.path.join(RESULTS_DIR, "training_trends.png"),
        team_rewards=episode_team_rewards,
        losses=losses,
        eps_history=eps_history,
        window=50,
    )
    np.savez_compressed(
        os.path.join(RESULTS_DIR, "agent_reward_history.npz"),
        **{f"agent_{i}": np.asarray(agent_reward_history[i], dtype=np.float32) for i in range(NUM_AGENTS)},
    )
    plot_per_agent_training_rewards(
        out_path=os.path.join(RESULTS_DIR, "training_agent_rewards.png"),
        agent_reward_history=agent_reward_history,
        model_types={i: MODEL_TYPES[i] for i in range(NUM_AGENTS)},
        window=50,
    )

    total_time = time.time() - t0
    overall_succ = float(np.mean(episode_success) * 100.0) if len(episode_success) else 0.0
    print(
        "Training finished\n"
        f"  Time: {_fmt_seconds(total_time)}\n"
        f"  EnvSteps: {total_env_steps}\n"
        f"  Success: {overall_succ:.1f}% | Fail episodes: {total_fail_episodes}/{EPISODES}\n"
        f"  Models: {base}_actor_agent_*.pt and {base}_critic_agent_*.pt\n"
        f"  Plot: {os.path.join(RESULTS_DIR, 'training_trends.png')}\n"
        f"  Plot: {os.path.join(RESULTS_DIR, 'training_agent_rewards.png')}\n"
    )


if __name__ == "__main__":
    train()
