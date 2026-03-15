# Multi-Agent MATD3 - Latency and Energy Minimization

This folder contains a specialized implementation of the MATD3 (Multi-Agent Twin Delayed Deep Deterministic Policy Gradient) algorithm designed for multi-objective optimization in IoT environments.

## Objective
The goal is to minimize the sum of total latency and total energy consumption:
**min (Ttotal + Etotal)**

## Key Features
- **Reward Function**: `reward = -(total_latency + energy_cost)`.
- **Strict Trust Constraint**: The "Trust Hard Constraint" is strictly enforced. Allocations fail immediately if a device's trust score is below the required threshold for a given layer's privacy level.
- **Energy Cost**: Calculated based on both computation demand and transmission data size.

## Files
- `environment.py`: Custom environment with the multi-objective reward and strict trust logic.
- `train.py`: Training script configured for this specific multi-objective scenario.
- `agent.py`, `manager.py`, `networks.py`, `replay_buffer.py`: MATD3 algorithm implementation.

## Usage
To start training:
```bash
python train.py --episodes 5000
```
