import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent dir to path to import environment
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in [SCRIPT_DIR, ROOT_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

from environment import MultiAgentIoTEnvLatencyEnergySum
from MultiAgentVDN.plots import plot_execution_flow, EvalEpisodeFlow

def generate_impact_plot():
    # 1. Setup environment (ensure sequential_diversity=False)
    env = MultiAgentIoTEnvLatencyEnergySum(
        num_agents=1,
        num_devices=15,
        model_types=["vgg11"],
        sequential_diversity=False
    )
    
    obs, _ = env.reset()
    done = False
    
    layer_comp = []
    layer_comm = []
    layer_names = []
    device_choices = []
    
    # Force agent 0 to use the SAME device (Device 0) for ALL layers
    device_id = 0
    
    while not done:
        actions = {0: device_id}
        next_obs, rewards, dones, _, infos = env.step(actions)
        
        info = infos[0]
        if info["success"]:
            layer_comp.append(info["t_comp"])
            # The FIRST layer ALWAYS has communication (initial loading 5.0MB)
            # Subsequent layers on the same device should have 0.0
            layer_comm.append(info["t_comm"]) 
            layer_names.append(info["layer_name"])
            device_choices.append(device_id)
        
        done = dones[0]
        
    # 2. Plotting
    x = np.arange(len(layer_names))
    width = 0.45
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, layer_comp, width, label='Computation (CPU)', color='#1f77b4', alpha=0.9)
    ax.bar(x + width/2, layer_comm, width, label='Communication (Bandwidth)', color='#ff7f0e', alpha=0.9)
    
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.set_ylabel('Latency (Seconds)')
    ax.set_title('Impact of Local Processing (Single Episode Trace)\nSame Device for all layers (Device 0)')
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, "impact_communication_demo.png")
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to: {out_path}")
    
    # 3. Plotting Execution Flow
    flow = EvalEpisodeFlow(
        agent_ids=[0],
        device_choices={0: device_choices},
        fail_step=None,
        fail_agent=None,
        model_types={0: "vgg11"}
    )
    flow_path = os.path.join(SCRIPT_DIR, "impact_execution_flow_demo.png")
    plot_execution_flow(flow_path, flow)
    print(f"Flow plot saved to: {flow_path}")
    
    # Save results as JSON
    json_data = {
        "agent_id": 0,
        "model_type": "vgg11",
        "device_strategy": "Same Device (Device 0) for all layers",
        "layers": [
            {
                "index": i,
                "name": layer_names[i],
                "device": device_choices[i],
                "t_comp": round(float(layer_comp[i]), 4),
                "t_comm": round(float(layer_comm[i]), 4)
            } for i in range(len(layer_names))
        ]
    }
    json_path = os.path.join(SCRIPT_DIR, "impact_demo_data.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    print(f"Data saved to: {json_path}")

    # Print numerical results for verification
    print("\nNumerical Verification:")
    for i, name in enumerate(layer_names):
        print(f"Layer {i} ({name}): Comp={layer_comp[i]:.4f}s, Comm={layer_comm[i]:.4f}s")

if __name__ == "__main__":
    generate_impact_plot()
