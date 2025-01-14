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
from mava.networks.utils.mat.decode import (
    continuous_autoregressive_act,
    continuous_parallel_act,
    discrete_autoregressive_act,
    discrete_parallel_act,
)
from mava.systems.mat.types import MATNetworkConfig
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


class EncodeBlock(nn.Module):
    n_agent: int
    net_config: MATNetworkConfig
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
    net_config: MATNetworkConfig
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


class MultiAgentTransformer(nn.Module):
    action_dim: int
    n_agent: int
    net_config: MATNetworkConfig
    action_space_type: str = _DISCRETE
    use_byol: bool = False

    # General shape names:
    # B: batch size
    # N: number of agents
    # O: observation dimension
    # A: action dimension
    # E: model embedding dimension

    def setup(self) -> None:
        if self.action_space_type not in [_DISCRETE, _CONTINUOUS]:
            raise ValueError(f"Invalid action space type: {self.action_space_type}")

        # Initialize main network
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
        self.dynamic_model = DynamicModel(
            self.action_dim,
            self.n_agent,
            self.net_config,
            self.action_space_type,
        )

        # Initialize BYOL related networks only when use_byol is True
        if self.use_byol:
            # BYOL projection head
            self.projector = nn.Sequential(
                [
                    nn.Dense(self.net_config.embed_dim * 4, kernel_init=orthogonal(jnp.sqrt(2))),
                    nn.gelu,
                    nn.LayerNorm(),
                    nn.Dense(self.net_config.embed_dim, kernel_init=orthogonal(jnp.sqrt(2))),
                ]
            )

            # BYOL prediction head
            self.predictor = nn.Sequential(
                [
                    nn.Dense(self.net_config.embed_dim * 4, kernel_init=orthogonal(jnp.sqrt(2))),
                    nn.gelu,
                    nn.LayerNorm(),
                    nn.Dense(self.net_config.embed_dim, kernel_init=orthogonal(jnp.sqrt(2))),
                ]
            )

            # Initialize target networks
            self.target_encoder = Encoder(
                self.action_dim,
                self.n_agent,
                self.net_config,
            )
            self.target_projector = nn.Sequential(
                [
                    nn.Dense(self.net_config.embed_dim * 4, kernel_init=orthogonal(jnp.sqrt(2))),
                    nn.gelu,
                    nn.LayerNorm(),
                    nn.Dense(self.net_config.embed_dim, kernel_init=orthogonal(jnp.sqrt(2))),
                ]
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
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        # Calculate policy and value
        value, obs_rep = self.encoder(observation.agents_view)
        action_log, entropy = self.train_function(
            decoder=self.decoder,
            obs_rep=obs_rep,
            action=action,
            action_dim=self.action_dim,
            legal_actions=observation.action_mask,
            key=key,
        )
        return action_log, value, entropy

    def init_params(
        self,
        observation: MavaObservation,  # (B, N, ...)
        next_observation: MavaObservation,  # (B, N, ...)
        action: chex.Array,  # (B, N, A)
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        # Calculate policy and value
        value, obs_rep = self.encoder(observation.agents_view)
        action_log, entropy = self.train_function(
            decoder=self.decoder,
            obs_rep=obs_rep,
            action=action,
            action_dim=self.action_dim,
            legal_actions=observation.action_mask,
            key=key,
        )
        # Calculate BYOL representations
        if self.use_byol:
            obs_rep, next_obs_rep, target_next_obs_rep = self.get_representations(
                observation, next_observation, action
            )
        return 

    def get_actions(
        self,
        observation: MavaObservation,  # (B, N, ...)
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        value, obs_rep = self.encoder(observation.agents_view)
        output_action, output_action_log = self.act_function(
            decoder=self.decoder,
            obs_rep=obs_rep,
            action_dim=self.action_dim,
            legal_actions=observation.action_mask,
            key=key,
        )
        return output_action, output_action_log, value

    def get_representations(
        self,
        observation: MavaObservation,  # (B, N, ...)
        next_observation: MavaObservation,  # (B, N, ...)
        action: chex.Array,  # (B, N, A)
    ) -> Tuple[chex.Array, chex.Array]:
        """Compute BYOL representations for current and next observations."""

        # Get current state representation
        _, obs_rep = self.encoder(observation.agents_view)

        # Calculate predicted next state representation
        next_obs_rep = self.predictor(self.projector(self.dynamic_model(action, obs_rep)))

        # Calculate target network's next state representation
        _, target_next_obs_rep = jax.lax.stop_gradient(
            self.target_encoder(next_observation.agents_view)
        )
        target_next_obs_rep = jax.lax.stop_gradient(self.target_projector(target_next_obs_rep))
        return obs_rep, next_obs_rep, target_next_obs_rep

    def get_scores(
        self,
        obs_rep: chex.Array,
        action: chex.Array,
        next_obs_rep: chex.Array,
        target_next_obs_rep: chex.Array,
        rng_key: chex.PRNGKey,
    ) -> chex.Array:
        """Calculate intrinsic reward loss using discriminator."""
        batch_size = obs_rep.shape[0]
        # random select half of the batch to generate fake samples
        false_batch_idx = jax.random.choice(
            rng_key, batch_size, shape=(batch_size // 2,), replace=False
        )
        # create target next_obs_rep, replace fake samples with target_next_obs_rep
        z_next_target = next_obs_rep.at[false_batch_idx].set(target_next_obs_rep[false_batch_idx])
        # create labels, fake samples are 0, real samples are 1
        labels = jnp.ones((batch_size, self.n_agent), dtype=jnp.int32)
        labels = labels.at[false_batch_idx].set(0)
        # calculate logits using discriminator
        logits = self.discriminator(obs_rep, action, z_next_target)
        return logits, labels


class DynamicModel(nn.Module):
    action_dim: int
    n_agent: int
    net_config: MATNetworkConfig
    action_space_type: str

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
                ln(),
            ],
        )
        self.ln = ln()
        # Create a projection layer to map concatenated 2D dimensions back to D dimensions
        self.proj = nn.Dense(
            self.net_config.embed_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
        )
        # Create EncodeBlock to process projected representation
        self.encode_block = EncodeBlock(
            self.n_agent,
            self.net_config,
            masked=False,
        )

    def __call__(self, action: chex.Array, rep_enc: chex.Array) -> chex.Array:
        # action shape: [B, N, A]
        # rep_enc shape: [B, N, D]
        # Process action
        if self.action_space_type == _DISCRETE:
            processed_action = jax.nn.one_hot(action, self.action_dim)
        else:
            processed_action = action
        # embedding action
        action_embeddings = self.action_encoder(processed_action)  # [B, N, D]
        # Concatenate action embeddings and state representations along feature dimension
        combined = jnp.concatenate([action_embeddings, rep_enc], axis=-1)  # [B, N, 2D]
        # Apply layer normalization
        normalized = self.ln(combined)
        # Project 2D dimensions back to D dimensions
        projected = self.proj(normalized)  # [B, N, D]
        # Process through EncodeBlock
        next_obs_rep = self.encode_block(projected)

        return next_obs_rep


class Discriminator(nn.Module):
    action_dim: int
    n_agent: int
    net_config: MATNetworkConfig
    action_space_type: str

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
                ln(),
            ],
        )
        self.ln = ln()
        # Create EncodeBlock to process representations
        self.encode_block = EncodeBlock(
            self.n_agent,
            self.net_config,
            masked=False,
        )
        self.classifier = nn.Sequential(
            [
                nn.Dense(self.net_config.embed_dim, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.LayerNorm(),
                nn.tanh,
                nn.Dense(self.net_config.embed_dim, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.elu,
                nn.Dense(2, kernel_init=orthogonal(0.01)),
            ]
        )

    def __call__(
        self, obs_rep: chex.Array, action: chex.Array, next_obs_rep: chex.Array
    ) -> chex.Array:
        # action shape: [B, N, A]
        # obs_rep/next_obs_rep shape: [B, N, D]
        # Process action
        if self.action_space_type == _DISCRETE:
            processed_action = jax.nn.one_hot(action, self.action_dim)
        else:
            processed_action = action
        # embedding action
        action_embeddings = self.action_encoder(processed_action)
        # Concatenate along sequence dimension
        x = jnp.concatenate([obs_rep, action_embeddings, next_obs_rep], axis=1)
        # Process through EncodeBlock
        x = self.encode_block(x)[:, : self.n_agent, :]
        logits = self.classifier(x)

        return logits
