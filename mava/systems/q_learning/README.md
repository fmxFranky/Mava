# Q-Learning Systems

This directory contains implementations of Q-learning algorithms for multi-agent reinforcement learning.

## Available Algorithms

### Recurrent Independent Q-Learning (IQL)
- **File**: `anakin/rec_iql.py`
- **Description**: Independent Q-learning with recurrent networks for partial observability

### Recurrent QMIX
- **File**: `anakin/rec_qmix.py` 
- **Description**: QMIX algorithm with recurrent networks for value decomposition in cooperative settings

#### Enhanced Trajectory Support

The recurrent QMIX implementation now supports enhanced trajectory inputs to the Q-network:

- **Joint Trajectory**: Historical observations and actions for all agents  
  - Format: `observations: [B, N, K, *obs_dim]`, `actions: [B, N, K, *act_dim]`

Where:
- `B`: Batch size
- `N`: Number of agents
- `K`: Trajectory length (configurable via `system.traj_len`)

**Configuration**:
```yaml
system:
  traj_len: 10  # Length of historical trajectory buffer
```

**Current Status**: 
- ✅ Trajectory data is properly constructed and passed to the network during action selection
- ✅ Network interface updated to accept joint trajectory input
- ⚠️ Note: During training from replay buffer, trajectory information is not available

**Usage**: The trajectories contain observations and actions from timesteps `t-K+1:t`, ensuring proper temporal ordering for recurrent processing.

## Configuration

Each algorithm has its corresponding configuration file in `mava/configs/system/q_learning/`.

## Key Features

- **Recurrent Networks**: Handle partial observability through LSTM/GRU cells
- **Replay Buffers**: Experience replay for sample efficiency  
- **Target Networks**: Stabilized learning through periodic target updates
- **Epsilon-Greedy Exploration**: Balanced exploration-exploitation
- **Trajectory History**: Enhanced decision making through historical context (QMIX)

## Relevant papers:
* [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602)
* [Multiagent Cooperation and Competition with Deep Reinforcement Learning](https://arxiv.org/pdf/1511.08779)
* [QMIX: Monotonic Value Function Factorisation for
Deep Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/1803.11485)
