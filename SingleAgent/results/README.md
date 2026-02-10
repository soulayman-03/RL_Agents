# Evaluation Results & Technical Interpretation

This directory contains the visualizations generated during the evaluation phase. These plots provide quantitative and qualitative evidence of the agent's learning progress and decision-making logic.

## 1. Training Trends (`training_trends.png`)
- **Rewards**: Shows the convergence of the RL agent. A rising curve indicates the agent is finding lower-latency strategies.
- **Stalls**: Tracks constraint violations. Ideally, this should drop to zero quickly, meaning the agent has learned the boundaries of the hardware resources.

## 2. Reward Distribution (`reward_distribution.png`)
- **What it shows**: The consistency of the agent across multiple environment seeds.
- **Interpretation**: A tight box plot with high (less negative) rewards indicates a robust policy that adapts well to various hardware configurations.

## 3. Latency Composition (`latency_composition.png`)
- **What it shows**: The breakdown between **Compute Latency** (blue) and **Communication Latency** (red).
- **Interpretation**: 
    - If Compute dominates, the devices selected are relatively slow compared to the network speed.
    - If Communication is significant, the agent is moving large amounts of data (high Output Size layers) across the network.
    - The agent aims to balance these two to minimize the total height of the bars.

## 4. Device Usage Stats (`device_usage_stats.png`)
- **What it shows**: How many layers were assigned to each Device ID (0-4) across all test episodes.
- **Interpretation**: 
    - An even distribution suggests balanced hardware or a need to spread load.
    - A skew towards specific devices (e.g., Device 0 or 3) usually indicates those devices have superior CPU speed or specific privacy clearances required by the model.

## 5. Execution Strategy Traces (`execution_flow_seed_XX.png`)
- **Visualization**: Shows the "path" a model takes through the devices.
- **Annotations**:
    - **C**: Computation time for that layer on the chosen device.
    - **T**: Transmission time to move data from the previous device to the current one.
- **Deeper View**:
    - Vertical lines represent **Network Hops**.
    - Horizontal lines represent **Processing** on the same device.
    - The annotations help verify why the agent made a specific choice (e.g., picking a device with low C even if it adds some T).

## Summary Analysis
The current agent successfully learns to partition the `simplecnn` model. It consistently avoids OOM (Out of Memory) errors by checking `D_mem` and optimizes for the fastest execution path by comparing `L_comp / D_cpu`. Privacy constraints (Layer 0) are strictly enforced, as seen in the traces where L0 is always placed on devices with `priv=1`.
