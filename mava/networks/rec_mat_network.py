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
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import orthogonal
from typing_extensions import NamedTuple

from mava.networks.attention import SelfAttention
from mava.networks.torsos import SwiGLU
from mava.networks.utils.mat.rec_decode import (
    continuous_autoregressive_act_with_hidden,
    continuous_parallel_act_with_hidden,
    discrete_autoregressive_act_with_hidden,
    discrete_parallel_act_with_hidden,
)
from mava.systems.mat.types import MATNetworkConfig
from mava.types import MavaObservation
from mava.utils.network_utils import _CONTINUOUS, _DISCRETE


# Define hidden states for recurrent MAT
class MATHiddenStates(NamedTuple):
    """Hidden states for recurrent MAT."""

    encoder: chex.Array  # (B, embed_dim) - CLS token for encoder
    decoder_self: chex.Array  # (B, embed_dim) - CLS token for decoder self-attention
    decoder_cross: chex.Array  # (B, embed_dim) - CLS token for decoder cross-attention


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


class RecurrentEncodeBlock(nn.Module):
    """Recurrent encoder block with hidden state support."""

    n_agent: int
    net_config: MATNetworkConfig
    masked: bool = False

    def setup(self) -> None:
        ln = nn.RMSNorm if self.net_config.use_rmsnorm else nn.LayerNorm
        self.ln1 = ln()
        self.ln2 = ln()

        self.attn = SelfAttention(
            self.net_config.embed_dim, self.net_config.n_head, self.n_agent + 1, self.masked
        )  # +1 for hidden state token

        self.mlp = _make_mlp(self.net_config.embed_dim, self.net_config.use_swiglu)

    def __call__(self, x: chex.Array, hidden_state: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Forward pass with hidden state as CLS token."""
        # Concatenate hidden state as CLS token at the beginning
        # x: (B, N, embed_dim), hidden_state: (B, embed_dim)
        hidden_state = hidden_state[:, jnp.newaxis, :]  # (B, 1, embed_dim)
        x_with_cls = jnp.concatenate([hidden_state, x], axis=1)  # (B, N+1, embed_dim)

        # Apply attention
        attended = self.attn(x_with_cls, x_with_cls, x_with_cls)
        x_with_cls = self.ln1(x_with_cls + attended)

        # Apply MLP
        mlp_out = self.mlp(x_with_cls)
        x_with_cls = self.ln2(x_with_cls + mlp_out)

        # Split back to hidden state and agent features
        new_hidden_state = x_with_cls[:, 0, :]  # (B, embed_dim)
        new_x = x_with_cls[:, 1:, :]  # (B, N, embed_dim)

        return new_x, new_hidden_state


class RecurrentEncoder(nn.Module):
    """Recurrent encoder with hidden state support."""

    action_dim: int
    n_agent: int
    net_config: MATNetworkConfig

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
        self.blocks = [
            RecurrentEncodeBlock(
                self.n_agent,
                self.net_config,
                name=f"encoder_block_{block_id}",
            )
            for block_id in range(self.net_config.n_block)
        ]
        self.head = nn.Sequential(
            [
                nn.Dense(self.net_config.embed_dim, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.gelu,
                ln(),
                nn.Dense(1, kernel_init=orthogonal(0.01)),
            ],
        )

    def __call__(
        self, obs: chex.Array, hidden_state: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Forward pass with hidden state."""
        obs_embeddings = self.obs_encoder(obs)
        x = obs_embeddings
        x = self.ln(x)

        # Apply blocks with hidden state
        current_hidden = hidden_state
        for block in self.blocks:
            x, current_hidden = block(x, current_hidden)

        rep = x
        value = self.head(rep)

        return jnp.squeeze(value, axis=-1), rep, current_hidden


class RecurrentDecodeBlock(nn.Module):
    """Recurrent decoder block with hidden state support."""

    n_agent: int
    net_config: MATNetworkConfig
    masked: bool = True

    def setup(self) -> None:
        ln = nn.RMSNorm if self.net_config.use_rmsnorm else nn.LayerNorm
        self.ln1 = ln()
        self.ln2 = ln()
        self.ln3 = ln()

        # Self-attention with hidden state
        self.attn1 = SelfAttention(
            self.net_config.embed_dim, self.net_config.n_head, self.n_agent + 1, self.masked
        )
        # Cross-attention with hidden state
        self.attn2 = SelfAttention(
            self.net_config.embed_dim, self.net_config.n_head, self.n_agent + 1, self.masked
        )

        self.mlp = _make_mlp(self.net_config.embed_dim, self.net_config.use_swiglu)

    def __call__(
        self, x: chex.Array, rep_enc: chex.Array, hidden_self: chex.Array, hidden_cross: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Forward pass with two hidden states."""
        # Self-attention with hidden state
        hidden_self = hidden_self[:, jnp.newaxis, :]  # (B, 1, embed_dim)
        x_with_cls_self = jnp.concatenate([hidden_self, x], axis=1)  # (B, N+1, embed_dim)

        attended_self = self.attn1(x_with_cls_self, x_with_cls_self, x_with_cls_self)
        x_with_cls_self = self.ln1(x_with_cls_self + attended_self)

        new_hidden_self = x_with_cls_self[:, 0, :]  # (B, embed_dim)
        x_self = x_with_cls_self[:, 1:, :]  # (B, N, embed_dim)

        # Cross-attention with hidden state
        hidden_cross = hidden_cross[:, jnp.newaxis, :]  # (B, 1, embed_dim)
        rep_enc_with_cls = jnp.concatenate([hidden_cross, rep_enc], axis=1)  # (B, N+1, embed_dim)
        x_with_cls_cross = jnp.concatenate([hidden_cross, x_self], axis=1)  # (B, N+1, embed_dim)

        attended_cross = self.attn2(
            key=x_with_cls_cross, value=x_with_cls_cross, query=rep_enc_with_cls
        )
        rep_enc_with_cls = self.ln2(rep_enc_with_cls + attended_cross)

        new_hidden_cross = rep_enc_with_cls[:, 0, :]  # (B, embed_dim)
        x_cross = rep_enc_with_cls[:, 1:, :]  # (B, N, embed_dim)

        # MLP
        x_cross = self.ln3(x_cross + self.mlp(x_cross))

        return x_cross, new_hidden_self, new_hidden_cross


class RecurrentDecoder(nn.Module):
    """Recurrent decoder with hidden state support."""

    action_dim: int
    n_agent: int
    action_space_type: str
    net_config: MATNetworkConfig

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
            RecurrentDecodeBlock(
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

    def __call__(
        self, action: chex.Array, obs_rep: chex.Array, hidden_states: Tuple[chex.Array, chex.Array]
    ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
        """Forward pass with hidden states."""
        action_embeddings = self.action_encoder(action)
        x = self.ln(action_embeddings)

        hidden_self, hidden_cross = hidden_states

        # Apply blocks with hidden states
        for block in self.blocks:
            x, hidden_self, hidden_cross = block(x, obs_rep, hidden_self, hidden_cross)

        logit = self.head(x)
        return logit, (hidden_self, hidden_cross)


class RecurrentMultiAgentTransformer(nn.Module):
    """Recurrent Multi-Agent Transformer with hidden states."""

    action_dim: int
    n_agent: int
    net_config: MATNetworkConfig
    action_space_type: str = _DISCRETE

    def setup(self) -> None:
        if self.action_space_type not in [_DISCRETE, _CONTINUOUS]:
            raise ValueError(f"Invalid action space type: {self.action_space_type}")

        self.encoder = RecurrentEncoder(
            self.action_dim,
            self.n_agent,
            self.net_config,
        )
        self.decoder = RecurrentDecoder(
            self.action_dim,
            self.n_agent,
            self.action_space_type,
            self.net_config,
        )

        if self.action_space_type == _DISCRETE:
            self.act_function = discrete_autoregressive_act_with_hidden
            self.train_function = discrete_parallel_act_with_hidden
        elif self.action_space_type == _CONTINUOUS:
            self.act_function = continuous_autoregressive_act_with_hidden
            self.train_function = continuous_parallel_act_with_hidden
        else:
            raise ValueError(f"Invalid action space type: {self.action_space_type}")

    def __call__(
        self,
        observation: MavaObservation,
        action: chex.Array,
        hidden_states: MATHiddenStates,
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Training forward pass."""
        value, obs_rep, new_enc_hidden = self.encoder(
            observation.agents_view, hidden_states.encoder
        )

        action_log, entropy = self.train_function(
            decoder=self.decoder,
            obs_rep=obs_rep,
            action=action,
            action_dim=self.action_dim,
            legal_actions=observation.action_mask,
            hidden_states=(hidden_states.decoder_self, hidden_states.decoder_cross),
            key=key,
        )

        return action_log, value, entropy

    def get_actions(
        self,
        observation: MavaObservation,
        hidden_states: MATHiddenStates,
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, MATHiddenStates]:
        """Inference forward pass."""
        value, obs_rep, new_enc_hidden = self.encoder(
            observation.agents_view, hidden_states.encoder
        )

        output_action, output_action_log = self.act_function(
            decoder=self.decoder,
            obs_rep=obs_rep,
            action_dim=self.action_dim,
            legal_actions=observation.action_mask,
            hidden_states=(hidden_states.decoder_self, hidden_states.decoder_cross),
            key=key,
        )

        # For now, we'll just return the encoder hidden state as new states
        # The decoder hidden states would be updated in the act_function
        new_hidden_states = MATHiddenStates(
            encoder=new_enc_hidden,
            decoder_self=hidden_states.decoder_self,  # Updated in act_function
            decoder_cross=hidden_states.decoder_cross,  # Updated in act_function
        )

        return output_action, output_action_log, value, new_hidden_states
