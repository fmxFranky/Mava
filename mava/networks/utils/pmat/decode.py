# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, Union

import chex
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import linen as nn

from mava.networks.distributions import IdentityTransformation, TanhTransformedDistribution

# General shapes legend:
# B: batch size
# N: number of agents
# O: observation dimension
# A: action dimension
# E: model embedding dimension


def discrete_parallel_act(
    decoder: nn.Module,
    obs_rep: chex.Array,  # (B, N, E)
    action: chex.Array,  # (B, N)
    action_dim: int,  # (, )
    legal_actions: chex.Array,  # (B, N, A)
    key: chex.PRNGKey,
) -> Tuple[chex.Array, chex.Array]:
    """Parallel action computation for discrete actions in PMAT."""
    B, N, _ = obs_rep.shape
    one_hot_action = jax.nn.one_hot(action, action_dim)  # (B, N, A)
    shifted_action = jnp.zeros((B, N, action_dim + 1))  # (B, N, A + 1)
    shifted_action = shifted_action.at[:, 0, 0].set(1)
    shifted_action = shifted_action.at[:, 1:, 1:].set(one_hot_action[:, :-1, :])
    logit = decoder(shifted_action, obs_rep)  # (B, N, A)

    masked_logits = jnp.where(
        legal_actions,
        logit,
        jnp.finfo(jnp.float32).min,
    )

    distribution = IdentityTransformation(distribution=tfd.Categorical(logits=masked_logits))
    action_log_prob = distribution.log_prob(action)
    entropy = distribution.entropy(seed=key)

    return action_log_prob, entropy  # (B, N), (B, N)


def continuous_parallel_act(
    decoder: nn.Module,
    obs_rep: chex.Array,  # (B, N, E)
    action: chex.Array,  # (B, N, A)
    action_dim: int,  # (, )
    legal_actions: chex.Array,  # (B, N, A)
    key: chex.PRNGKey,
) -> Tuple[chex.Array, chex.Array]:
    """Parallel action computation for continuous actions in PMAT."""
    # We don't need legal_actions for continuous actions but keep it to keep the APIs consistent.
    del legal_actions
    B, N, _ = obs_rep.shape
    shifted_action = jnp.zeros((B, N, action_dim))

    shifted_action = shifted_action.at[:, 1:, :].set(action[:, :-1, :])

    act_mean = decoder(shifted_action, obs_rep)  # (B, N, A)
    action_std = jax.nn.softplus(decoder.log_std)

    distribution = tfd.Normal(loc=act_mean, scale=action_std)
    distribution = tfd.Independent(
        TanhTransformedDistribution(distribution),
        reinterpreted_batch_ndims=1,
    )
    action_log_prob = distribution.log_prob(action)
    entropy = distribution.entropy(seed=key)

    return action_log_prob, entropy  # (B, N), (B, N)


def discrete_autoregressive_act(
    decoder: nn.Module,
    obs_rep: chex.Array,  # (B, N, E)
    action_dim: int,  # (, )
    legal_actions: chex.Array,  # (B, N, A)
    key: chex.PRNGKey,
    deterministic: bool = False,
) -> Tuple[chex.Array, chex.Array]:
    """Autoregressive action sampling for discrete actions in PMAT.

    This follows the prioritized ordering determined by the scoring network.
    """
    B, N, _ = obs_rep.shape
    shifted_action = jnp.zeros((B, N, action_dim + 1))
    shifted_action = shifted_action.at[:, 0, 0].set(1)
    output_action = jnp.zeros((B, N))
    output_action_log = jnp.zeros_like(output_action)

    for i in range(N):
        logit = decoder(shifted_action, obs_rep)[:, i, :]  # (B, A)
        masked_logits = jnp.where(
            legal_actions[:, i, :],
            logit,
            jnp.finfo(jnp.float32).min,
        )
        key, sample_key = jax.random.split(key)

        distribution = IdentityTransformation(distribution=tfd.Categorical(logits=masked_logits))

        if deterministic:
            action = jnp.argmax(masked_logits, axis=-1)
        else:
            action = distribution.sample(seed=sample_key)  # (B, )

        action_log = distribution.log_prob(action)  # (B, )

        output_action = output_action.at[:, i].set(action)
        output_action_log = output_action_log.at[:, i].set(action_log)

        # Adds all except the last action to shifted_actions, as it is out of range
        shifted_action = shifted_action.at[:, i + 1, 1:].set(
            jax.nn.one_hot(action, action_dim), mode="drop"
        )

    return output_action.astype(jnp.int32), output_action_log  # (B, N), (B, N)


def continuous_autoregressive_act(
    decoder: nn.Module,
    obs_rep: chex.Array,  # (B, N, E)
    action_dim: int,  # (, )
    legal_actions: Union[chex.Array, None],
    key: chex.PRNGKey,
    deterministic: bool = False,
) -> Tuple[chex.Array, chex.Array]:
    """Autoregressive action sampling for continuous actions in PMAT.

    This follows the prioritized ordering determined by the scoring network.
    """
    # We don't need legal_actions for continuous actions but keep it to keep the APIs consistent.
    del legal_actions
    B, N, _ = obs_rep.shape
    shifted_action = jnp.zeros((B, N, action_dim))
    output_action = jnp.zeros((B, N, action_dim))
    output_action_log = jnp.zeros((B, N))

    for i in range(N):
        act_mean = decoder(shifted_action, obs_rep)[:, i, :]  # (B, A)
        action_std = jax.nn.softplus(decoder.log_std)

        key, sample_key = jax.random.split(key)

        distribution = tfd.Normal(loc=act_mean, scale=action_std)
        distribution = tfd.Independent(
            TanhTransformedDistribution(distribution),
            reinterpreted_batch_ndims=1,
        )

        if deterministic:
            action = act_mean
        else:
            action = distribution.sample(seed=sample_key)  # (B, A)

        action_log = distribution.log_prob(action)  # (B,)

        output_action = output_action.at[:, i, :].set(action)
        output_action_log = output_action_log.at[:, i].set(action_log)

        # Adds all except the last action to shifted_actions, as it is out of range
        shifted_action = shifted_action.at[:, i + 1, :].set(action, mode="drop")

    return output_action, output_action_log  # (B, N, A), (B, N)


def sample_sequence_by_score(
    scores: chex.Array,  # (B, N)
    key: chex.PRNGKey,
    deterministic: bool = False,
) -> Tuple[chex.Array, chex.Array]:
    """Sample action sequence based on agent priority scores.

    Args:
        scores: Agent priority scores (B, N)
        key: Random key for sampling
        deterministic: Whether to use deterministic (greedy) sampling

    Returns:
        sampled_seq: Sampled sequence of agent indices (B, N)
        sampled_seq_log_prob: Log probability of the sampled sequence (B,)
    """
    batch_size, seq_length = scores.shape

    sampled_seq = jnp.zeros((batch_size, seq_length), dtype=jnp.int32)
    sampled_seq_log_prob = jnp.zeros(batch_size, dtype=jnp.float32)

    remaining = jnp.ones((batch_size, seq_length), dtype=jnp.bool_)
    current_scores = scores

    for item in range(seq_length):
        # Mask unavailable agents
        masked_scores = jnp.where(remaining, current_scores, 0.0)
        probabilities = masked_scores / (jnp.sum(masked_scores, axis=1, keepdims=True) + 1e-8)

        if deterministic:
            sampled_indices = jnp.argmax(probabilities, axis=-1)
        else:
            key, sample_key = jax.random.split(key)
            sampled_indices = jax.random.categorical(sample_key, jnp.log(probabilities + 1e-8))

        sampled_seq = sampled_seq.at[:, item].set(sampled_indices)

        # Update log probabilities
        batch_indices = jnp.arange(batch_size)
        selected_probs = probabilities[batch_indices, sampled_indices]
        sampled_seq_log_prob += jnp.log(selected_probs + 1e-8)

        # Update remaining mask
        remaining = remaining.at[batch_indices, sampled_indices].set(False)

    return sampled_seq, sampled_seq_log_prob


def compute_sequence_log_prob(scores: chex.Array, seq: chex.Array) -> chex.Array:
    """Compute log probability of a given sequence based on priority scores.

    Args:
        scores: Agent priority scores (B, N)
        seq: Given sequence of agent indices (B, N)

    Returns:
        seq_log_probs: Log probability of the sequence (B,)
    """
    batch_size, seq_length = scores.shape

    remaining = jnp.ones((batch_size, seq_length), dtype=jnp.bool_)
    current_scores = scores
    seq_log_probs = jnp.zeros(batch_size, dtype=jnp.float32)

    for item in range(seq_length):
        # Mask unavailable agents
        masked_scores = jnp.where(remaining, current_scores, 0.0)
        probabilities = masked_scores / (jnp.sum(masked_scores, axis=1, keepdims=True) + 1e-8)

        # Get selected agent for this step
        selected_agents = seq[:, item].astype(jnp.int32)
        batch_indices = jnp.arange(batch_size)

        # Add log probability
        selected_probs = probabilities[batch_indices, selected_agents]
        seq_log_probs += jnp.log(selected_probs + 1e-8)

        # Update remaining mask
        remaining = remaining.at[batch_indices, selected_agents].set(False)

    return seq_log_probs


def reindex_tensor(tensor: chex.Array, indices: chex.Array) -> chex.Array:
    """Reindex tensor based on given indices.

    Args:
        tensor: Input tensor (B, N, ...)
        indices: Reindexing indices (B, N)

    Returns:
        reindexed_tensor: Reindexed tensor (B, N, ...)
    """
    batch_size = tensor.shape[0]
    batch_indices = jnp.arange(batch_size)[:, None]
    return tensor[batch_indices, indices]


def calculate_restore_index(sampled_seq: chex.Array) -> chex.Array:
    """Calculate indices to restore original agent ordering.

    Args:
        sampled_seq: Sampled sequence of agent indices (B, N)

    Returns:
        restore_index: Indices to restore original order (B, N)
    """
    batch_size, seq_length = sampled_seq.shape
    restore_index = jnp.zeros_like(sampled_seq)

    for i in range(seq_length):
        restore_index = restore_index.at[jnp.arange(batch_size), sampled_seq[:, i]].set(i)

    return restore_index
