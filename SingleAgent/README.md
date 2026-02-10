# Single-Agent RL for DNN Partitioning in IoT

This project implements a Deep Reinforcement Learning (DRL) system to optimize the placement of Deep Neural Network (DNN) layers across a set of heterogeneous IoT devices.

## Project Structure

- `agent.py`: Implementation of the DQNAgent (Deep Q-Network).
- `environment.py`: Custom Gymnasium environment (`MonoAgentIoTEnv`) simulating the IoT resource allocation problem.
- `train.py`: Script to train the RL agent. Tracks training duration and provides detailed resource-aware logs.
- `evaluate.py`: Script to evaluate the trained agent across different environment seeds and generate performance visualizations.
- `utils.py`: Helper functions for model generation, device networking, and weight loading.
- `models/`: Directory containing saved RL models and training history.
- `results/`: Directory containing evaluation plots and performance analysis.

## Core Concepts

### 1. The Environment
The environment simulates a network of 5 IoT devices with varying capacities:
- **CPU Speed**: Influences computation latency.
- **Memory Capacity**: Limits the size of layers an agent can host.
- **Bandwidth**: Influences transmission latency when layers are moved between devices.
- **Privacy Clearance**: Ensures sensitive layers (e.g., input layers) are only processed on trusted devices.

### 2. The Agent
The agent uses a DQN algorithm to learn a strategy. Its goal is to minimize **Total Latency** (Computation + Communication) while strictly respecting all resource and privacy constraints.

## Usage

### Training
To train the agent:
```bash
python SingleAgent/train.py
```
The script will log each episode's reward, constraint status, and a detailed trace of layer-to-device mappings including device metrics (CPU, RAM, BW).

### Evaluation
To evaluate the agent and generate plots:
```bash
python SingleAgent/evaluate.py
```
This script tests the agent against multiple seeds (different hardware configurations) and generates visualizations in the `results/` folder.

## Key Constraints Respected
- **Memory**: Total RAM used on a device cannot exceed its capacity.
- **Compute**: Sequential layers on the same device must share CPU resources.
- **Diversity**: The agent learns to avoid placing consecutive layers on the same device if specific environmental constraints require it.
- **Privacy**: High-privacy layers are matched with devices having appropriate clearance.
