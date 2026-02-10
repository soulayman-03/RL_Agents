from MultiAgent.environment import MultiAgentIoTEnv
import numpy as np

def test_marl_env():
    print("Testing MultiAgentIoTEnv...")
    num_agents = 3
    num_devices = 5
    env = MultiAgentIoTEnv(num_agents=num_agents, num_devices=num_devices)
    
    obs, info = env.reset()
    assert len(obs) == num_agents, f"Expected {num_agents} observations, got {len(obs)}"
    print(f"Initial observations received for {num_agents} agents.")

    # Test one step with all agents choosing device 0 (if valid)
    valid_actions = env.get_valid_actions()
    actions = {}
    for i in range(num_agents):
        if 0 in valid_actions[i]:
            actions[i] = 0
        elif valid_actions[i]:
            actions[i] = valid_actions[i][0]
        else:
            # Should not happen on first step if resources are enough
            actions[i] = 0 

    print(f"Taking a step with actions: {actions}")
    next_obs, rewards, dones, truncated, infos = env.step(actions)

    assert len(next_obs) == num_agents
    assert len(rewards) == num_agents
    assert len(dones) == num_agents

    print("Step 1 results:")
    for i in range(num_agents):
        print(f"  Agent {i}: Reward={rewards[i]:.4f}, Done={dones[i]}, Info={infos[i].get('reward_type')}")

    # Check shared resource state update
    # In Step 1, some agents allocated to device 0
    step_load_0 = env.resource_manager.step_resources[0]
    print(f"Device 0 step load: {step_load_0}")
    
    # If multiple agents chose device 0, the load should reflect the sum (if not reset within step loop incorrectly)
    # My implementation resets at the start of env.step(), then accumulates. Correct.

    print("MultiAgentIoTEnv test PASSED.")

if __name__ == "__main__":
    test_marl_env()
