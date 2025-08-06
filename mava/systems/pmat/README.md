# Prioritized Multi-Agent Transformer (PMAT)

PMAT is an extension of Multi-Agent Transformer (MAT) that incorporates a prioritized action generation mechanism for multi-agent reinforcement learning.

## Key Features

- **Prioritized Action Generation**: Agents are ordered based on learned priority scores before action generation
- **Sequential Decision Making**: Actions are generated autoregressively following the priority order
- **Ranking Loss**: Additional loss term to optimize the action ordering policy
- **Compatible with MAT**: Extends the existing MAT architecture with minimal modifications

## Architecture

PMAT consists of three main components:

1. **Encoder**: Processes observations and generates value estimates and observation representations
2. **Scoring Block**: Computes priority scores for each agent based on their observation representations  
3. **Decoder**: Generates actions autoregressively following the priority order

## Training

The training process includes:

- Standard PPO actor-critic losses
- Ranking loss for optimizing the priority scoring network
- Sequence entropy regularization

## Usage

PMAT can be used with the same environments and configurations as MAT, with additional hyperparameters for the scoring network and ranking loss coefficient.