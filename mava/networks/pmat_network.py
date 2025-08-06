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

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import orthogonal

from mava.networks.attention import SelfAttention
from mava.networks.torsos import SwiGLU
from mava.networks.utils.pmat.decode import (
    continuous_autoregressive_act,
    continuous_parallel_act,
    discrete_autoregressive_act,
    discrete_parallel_act,
)
from mava.systems.pmat.types import PMATNetworkConfig
from mava.types import MavaObservation
from mava.utils.network_utils import _CONTINUOUS, _DISCRETE


def _make_mlp(embed_dim: int, use_swiglu: bool) -> nn.Module:
    if use_swiglu:
        return SwiGLU(embed_dim, embed_dim)

    return nn.Sequential(
        [
            nn.Dense(embed_dim, kernel_init=orthogonal(jnp.sqrt(2))),
            nn.gelu,
            nn.Dense(embed_dim, kernel_init=orthogonal(0.01)),
        ],
    )


class ScoringBlock(nn.Module):
    """Scoring block for PMAT to compute agent priorities."""
    embed_dim: int
    hidden_dim: int = 64
    n_layers: int = 2

    def setup(self) -> None:
        layers = []
        layers.append(nn.LayerNorm())
        layers.append(nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2))))
        layers.append(nn.gelu)
        layers.append(nn.LayerNorm())

        # Add additional hidden layers
        for _ in range(self.n_layers - 1):
            layers.extend([
                nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.gelu,
                nn.LayerNorm(),
            ])

        # Final layer to output a single score
        layers.append(nn.Dense(1, kernel_init=orthogonal(0.01)))

        self.mlp = nn.Sequential(layers)

    def __call__(self, obs_rep: chex.Array) -> chex.Array:
        # Output raw scores, sigmoid transformation happens in sampling functions
        raw_scores = self.mlp(obs_rep)
        # Apply sigmoid and scale to [1, 10] range like the original implementation
        scores = 9 * nn.sigmoid(raw_scores) + 1
        return jnp.squeeze(scores, axis=-1)


class EncodeBlock(nn.Module):
    n_agent: int
    net_config: PMATNetworkConfig
    masked: bool = False

    def setup(self) -> None:
        ln = nn.RMSNorm if self.net_config.use_rmsnorm else nn.LayerNorm
        self.ln1 = ln()
        self.ln2 = ln()

        self.attn = SelfAttention(
            self.net_config.embed_dim, self.net_config.n_head, self.n_agent, self.masked
        )

        self.mlp = _make_mlp(self.net_config.embed_dim, self.net_config.use_swiglu)

    def __call__(self, x: chex.Array) -> chex.Array:
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class Encoder(nn.Module):
    action_dim: int
    n_agent: int
    net_config: PMATNetworkConfig

    def setup(self) -> None:
        ln = nn.RMSNorm if self.net_config.use_rmsnorm else nn.LayerNorm

        self.obs_encoder = nn.Sequential(
            [
                ln(),
                nn.Dense(self.net_config.embed_dim, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.gelu,
            ],
        )
        self.ln = ln()
        self.blocks = nn.Sequential(
            [
                EncodeBlock(
                    self.n_agent,
                    self.net_config,
                )
                for _ in range(self.net_config.n_block)
            ]
        )
        self.head = nn.Sequential(
            [
                nn.Dense(self.net_config.embed_dim, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.gelu,
                ln(),
                nn.Dense(1, kernel_init=orthogonal(0.01)),
            ],
        )

    def __call__(self, obs: chex.Array) -> Tuple[chex.Array, chex.Array]:
        obs_embeddings = self.obs_encoder(obs)
        x = obs_embeddings

        rep = self.blocks(self.ln(x))
        value = self.head(rep)

        return jnp.squeeze(value, axis=-1), rep


class DecodeBlock(nn.Module):
    n_agent: int
    net_config: PMATNetworkConfig
    masked: bool = True

    def setup(self) -> None:
        ln = nn.RMSNorm if self.net_config.use_rmsnorm else nn.LayerNorm
        self.ln1 = ln()
        self.ln2 = ln()
        self.ln3 = ln()

        self.attn1 = SelfAttention(
            self.net_config.embed_dim, self.net_config.n_head, self.n_agent, self.masked
        )
        self.attn2 = SelfAttention(
            self.net_config.embed_dim, self.net_config.n_head, self.n_agent, self.masked
        )

        self.mlp = _make_mlp(self.net_config.embed_dim, self.net_config.use_swiglu)

    def __call__(self, x: chex.Array, rep_enc: chex.Array) -> chex.Array:
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x


class Decoder(nn.Module):
    action_dim: int
    n_agent: int
    action_space_type: str
    net_config: PMATNetworkConfig

    def setup(self) -> None:
        ln = nn.RMSNorm if self.net_config.use_rmsnorm else nn.LayerNorm

        use_bias = self.action_space_type == _CONTINUOUS
        self.action_encoder = nn.Sequential(
            [
                nn.Dense(
                    self.net_config.embed_dim,
                    use_bias=use_bias,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                ),
                nn.gelu,
            ],
        )

        # Always initialize log_std but set to None for discrete action spaces
        # This ensures the attribute exists but signals it should not be used.
        self.log_std = (
            self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            if self.action_space_type == _CONTINUOUS
            else None
        )

        self.obs_encoder = nn.Sequential(
            [
                ln(),
                nn.Dense(self.net_config.embed_dim, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.gelu,
            ],
        )
        self.ln = ln()
        self.blocks = [
            DecodeBlock(
                self.n_agent,
                self.net_config,
                name=f"cross_attention_block_{block_id}",
            )
            for block_id in range(self.net_config.n_block)
        ]
        self.head = nn.Sequential(
            [
                nn.Dense(self.net_config.embed_dim, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.gelu,
                ln(),
                nn.Dense(self.action_dim, kernel_init=orthogonal(0.01)),
            ],
        )

    def __call__(self, action: chex.Array, obs_rep: chex.Array) -> chex.Array:
        action_embeddings = self.action_encoder(action)
        x = self.ln(action_embeddings)

        # Need to loop here because the input and output of the blocks are different.
        # Blocks take an action embedding and observation encoding as input but only give the cross
        # attention output as output.
        for block in self.blocks:
            x = block(x, obs_rep)
        logit = self.head(x)

        return logit


class PrioritizedMultiAgentTransformer(nn.Module):
    """Prioritized Multi-Agent Transformer (PMAT) network."""
    action_dim: int
    n_agent: int
    net_config: PMATNetworkConfig
    action_space_type: str = _DISCRETE

    # General shape names:
    # B: batch size
    # N: number of agents
    # O: observation dimension
    # A: action dimension
    # E: model embedding dimension

    def setup(self) -> None:
        if self.action_space_type not in [_DISCRETE, _CONTINUOUS]:
            raise ValueError(f"Invalid action space type: {self.action_space_type}")

        self.encoder = Encoder(
            self.action_dim,
            self.n_agent,
            self.net_config,
        )
        self.decoder = Decoder(
            self.action_dim,
            self.n_agent,
            self.action_space_type,
            self.net_config,
        )
        self.scorer = ScoringBlock(
            embed_dim=self.net_config.embed_dim,
            hidden_dim=self.net_config.scoring_hidden_dim,
            n_layers=self.net_config.scoring_n_layers,
        )

        if self.action_space_type == _DISCRETE:
            self.act_function = discrete_autoregressive_act
            self.train_function = discrete_parallel_act
        elif self.action_space_type == _CONTINUOUS:
            self.act_function = continuous_autoregressive_act
            self.train_function = continuous_parallel_act
        else:
            raise ValueError(f"Invalid action space type: {self.action_space_type}")

    def __call__(
        self,
        observation: MavaObservation,  # (B, N, ...)
        action: chex.Array,  # (B, N, A)
        seq: chex.Array,  # (B, N) - action sequence
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        value, obs_rep = self.encoder(observation.agents_view)

        # Compute agent priorities
        rep_scores = self.scorer(obs_rep)  # (B, N)

        # Compute sequence log probability and entropy
        seq_log_prob = self._compute_sequence_log_prob(rep_scores, seq)
        seq_entropy = self._compute_sequence_entropy(rep_scores)

        action_log, entropy = self.train_function(
            decoder=self.decoder,
            obs_rep=obs_rep,
            action=action,
            action_dim=self.action_dim,
            legal_actions=observation.action_mask,
            key=key,
        )

        return action_log, value, entropy, seq_log_prob, seq_entropy

    def get_actions(
        self,
        observation: MavaObservation,  # (B, N, ...)
        key: chex.PRNGKey,
        deterministic: bool = False,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        value, obs_rep = self.encoder(observation.agents_view)

        # Compute agent priorities and generate action sequence
        rep_scores = self.scorer(obs_rep)  # (B, N)
        sampled_seq, sampled_seq_log_prob = self._sample_sequence_by_score(
            rep_scores, key, deterministic
        )
        sampled_seq_entropy = self._compute_sequence_entropy(rep_scores)

        # Reorder observations based on the sampled sequence
        reindexed_obs_rep = self._reindex_tensor(obs_rep, sampled_seq)
        reindexed_legal_actions = None
        if observation.action_mask is not None:
            reindexed_legal_actions = self._reindex_tensor(observation.action_mask, sampled_seq)

        key, action_key = jax.random.split(key)
        output_action, output_action_log = self.act_function(
            decoder=self.decoder,
            obs_rep=reindexed_obs_rep,
            action_dim=self.action_dim,
            legal_actions=reindexed_legal_actions,
            key=action_key,
        )

        # Restore original order
        restore_index = self._calculate_restore_index(sampled_seq)
        restored_action = self._reindex_tensor(output_action, restore_index)
        restored_action_log = self._reindex_tensor(output_action_log, restore_index)

        return restored_action, restored_action_log, value, sampled_seq_log_prob, sampled_seq

    def _sample_sequence_by_score(
        self, scores: chex.Array, key: chex.PRNGKey, deterministic: bool = False
    ) -> Tuple[chex.Array, chex.Array]:
        """Sample action sequence based on agent scores."""
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

    def _compute_sequence_log_prob(self, scores: chex.Array, seq: chex.Array) -> chex.Array:
        """Compute log probability of a given sequence."""
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

    def _compute_sequence_entropy(self, scores: chex.Array) -> chex.Array:
        """Compute entropy of the sequence distribution based on agent scores."""
        batch_size, seq_length = scores.shape
        
        remaining = jnp.ones((batch_size, seq_length), dtype=jnp.bool_)
        current_scores = scores
        total_entropy = jnp.zeros(batch_size, dtype=jnp.float32)
        
        for item in range(seq_length):
            # Mask unavailable agents
            masked_scores = jnp.where(remaining, current_scores, 0.0)
            probabilities = masked_scores / (jnp.sum(masked_scores, axis=1, keepdims=True) + 1e-8)
            
            # Compute entropy: -sum(p * log(p))
            log_probs = jnp.log(probabilities + 1e-8)
            step_entropy = -jnp.sum(probabilities * log_probs, axis=1)
            total_entropy += step_entropy
            
            # For entropy calculation, we use the most likely next agent
            # This gives us the entropy of the actual policy distribution
            selected_agents = jnp.argmax(probabilities, axis=1)
            batch_indices = jnp.arange(batch_size)
            remaining = remaining.at[batch_indices, selected_agents].set(False)
        
        return total_entropy

    def _reindex_tensor(self, tensor: chex.Array, indices: chex.Array) -> chex.Array:
        """Reindex tensor based on given indices."""
        batch_size = tensor.shape[0]
        batch_indices = jnp.arange(batch_size)[:, None]
        return tensor[batch_indices, indices]

    def _calculate_restore_index(self, sampled_seq: chex.Array) -> chex.Array:
        """Calculate indices to restore original order."""
        batch_size, seq_length = sampled_seq.shape
        restore_index = jnp.zeros_like(sampled_seq)

        for i in range(seq_length):
            restore_index = restore_index.at[jnp.arange(batch_size), sampled_seq[:, i]].set(i)

        return restore_index