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
import optax
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
    aux_pred_mode: str = "forward"
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
        # Initialize BYOL related networks only when use_byol is True
        if self.use_byol:
            self.dynamic_model = DynamicModel(
                self.action_dim,
                self.n_agent,
                self.net_config,
                self.action_space_type,
            )
            self.backward_dynamic_model = BackwardDynamicModel(
                self.action_dim,
                self.n_agent,
                self.net_config,
                self.action_space_type,
            )
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
            if self.aux_pred_mode == "forward":
                dm_rep = self.dynamic_model(action, obs_rep)
            elif self.aux_pred_mode == "backward":
                dm_rep = self.backward_dynamic_model(action, obs_rep)
            else:
                dm_rep = self.dynamic_model(action, obs_rep)
                dm_rep = self.backward_dynamic_model(action, dm_rep)
            dm_rep = self.projector(dm_rep)
            dm_rep = self.predictor(dm_rep)
            _, target_dm_rep = self.target_encoder(observation.agents_view)
            target_dm_rep = self.target_projector(target_dm_rep)
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

    def get_forward_representations(
        self,
        seq_observation: MavaObservation,  # (B, K+1, N, ...)  # sequence of observations
        seq_action: chex.Array,  # (B, K, N, A)  # sequence of actions
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Compute representations for current and K-step future observations.

        Following SPR and PlayVirtual paper:
        1. Get current state representation from encoder
        2. Use dynamics model to predict next K steps iteratively
        3. Get target representations from target encoder for supervision

        Args:
            seq_observation: Sequence of observations with shape (B, K+1, N, ...)
            seq_action: Sequence of actions with shape (B, K, N, A)

        Returns:
            pred_obs_reps: Predicted future observation representations for K steps
            target_obs_reps: Target observation representations from target encoder
        """
        # Get current state representation from first observation
        _, obs_rep = self.encoder(seq_observation.agents_view[:, 0])  # (B, N, E)

        # Transpose actions to put time dimension first
        # From (B, K, N, A) to (K, B, N, A)
        if self.action_space_type == _DISCRETE:
            seq_action_t = jnp.transpose(seq_action[:, :-1], (1, 0, 2))
        else:
            seq_action_t = jnp.transpose(seq_action[:, :-1], (1, 0, 2, 3))

        # Define scan function for K-step prediction
        def _prediction_step(
            prev_rep: chex.Array, action: chex.Array
        ) -> Tuple[chex.Array, chex.Array]:
            """Single step prediction using dynamics model.

            Args:
                prev_rep: Previous step's representation (output from last DM call), shape (B, N, E)
                action: Current step's action, shape (B, N, A)

            Returns:
                (next_rep, next_rep): (carry for next step, output for current step)
            """
            next_rep = self.dynamic_model(action, prev_rep)  # use previous rep to predict next
            return next_rep, next_rep  # carry forward next_rep as prev_rep for next step

        # Predict next K steps using scan
        # At each step k:
        # - prev_rep is the representation from step k-1 (or obs_rep for k=0)
        # - action is seq_action_t[k], shape (B, N, A)
        # - output next_rep will be used as prev_rep for step k+1
        _, pred_obs_reps = jax.lax.scan(
            _prediction_step,
            obs_rep,  # initial prev_rep (B, N, E)
            seq_action_t,  # actions sequence (K, B, N, A)
        )  # pred_obs_reps shape: (K, B, N, E)

        # Transpose predictions to match desired shape (B, K, N, E)
        pred_obs_reps = jnp.transpose(pred_obs_reps, (1, 0, 2, 3))

        # Get target representations using target encoder
        # Process each future observation through target encoder
        _, target_obs_reps = jax.vmap(lambda x: self.target_encoder(x), in_axes=1, out_axes=1)(
            seq_observation.agents_view[:, 1:]
        )  # (B, K, N, E)

        # Stop gradient for target representations
        target_obs_reps = jax.lax.stop_gradient(target_obs_reps)

        # Project predictions and targets to BYOL space
        pred_obs_reps = jax.vmap(
            lambda x: self.predictor(self.projector(x)), in_axes=1, out_axes=1
        )(pred_obs_reps)  # (B, K, N, E)

        target_obs_reps = jax.vmap(lambda x: self.target_projector(x), in_axes=1, out_axes=1)(
            target_obs_reps
        )  # (B, K, N, E)
        target_obs_reps = jax.lax.stop_gradient(target_obs_reps)

        return pred_obs_reps, target_obs_reps

    def get_backward_representations(
        self,
        seq_observation: MavaObservation,  # (B, K+1, N, ...)  # sequence of observations
        seq_action: chex.Array,  # (B, K, N, A)  # sequence of actions
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Compute representations for K-step and previous observations.

        Following SPR and PlayVirtual paper:
        1. Get K-step state representation from encoder
        2. Use backward dynamics model to predict previous K steps iteratively
        3. Get target representations from target encoder for supervision

        Args:
            seq_observation: Sequence of observations with shape (B, K+1, N, ...)
            seq_action: Sequence of actions with shape (B, K, N, A)

        Returns:
            obs_rep_k: K-step observation representation
            pred_obs_reps: Predicted previous observation representations for K steps
            target_obs_reps: Target observation representations from target encoder
        """
        # Get K-step state representation from last observation
        _, obs_rep_k = self.encoder(seq_observation.agents_view[:, -1])  # (B, N, E)

        # Transpose actions to put time dimension first and reverse the sequence
        # From (B, K, N, A) to (K, B, N, A)
        if self.action_space_type == _DISCRETE:
            seq_action_t = jnp.transpose(seq_action[:, :-1], (1, 0, 2))
        else:
            seq_action_t = jnp.transpose(seq_action[:, :-1], (1, 0, 2, 3))
        # Reverse the sequence to predict backwards
        seq_action_t = jnp.flip(seq_action_t, axis=0)

        # Define scan function for K-step backward prediction
        def _backward_prediction_step(
            prev_rep: chex.Array, action: chex.Array
        ) -> Tuple[chex.Array, chex.Array]:
            """Single step backward prediction using dynamics model.

            Args:
                prev_rep: Previous step's representation (output from last DM call), shape (B, N, E)
                action: Current step's action, shape (B, N, A)

            Returns:
                (prev_rep, prev_rep): (carry for next step, output for current step)
            """
            prev_rep = self.backward_dynamic_model(
                action, prev_rep
            )  # use current rep to predict previous
            return prev_rep, prev_rep  # carry forward prev_rep for next step

        # Predict previous K steps using scan
        # At each step k:
        # - prev_rep is the representation from step k+1 (or obs_rep_k for k=K)
        # - action is seq_action_t[k], shape (B, N, A)
        # - output prev_rep will be used as prev_rep for step k-1
        _, pred_obs_reps = jax.lax.scan(
            _backward_prediction_step,
            obs_rep_k,  # initial prev_rep (B, N, E)
            seq_action_t,  # actions sequence (K, B, N, A)
        )  # pred_obs_reps shape: (K, B, N, E)

        # Transpose predictions to match desired shape (B, K, N, E)
        pred_obs_reps = jnp.transpose(pred_obs_reps, (1, 0, 2, 3))
        # Flip back the sequence to match the original order
        pred_obs_reps = jnp.flip(pred_obs_reps, axis=1)

        # Get target representations using target encoder
        # Process each previous observation through target encoder
        _, target_obs_reps = jax.vmap(lambda x: self.target_encoder(x), in_axes=1, out_axes=1)(
            seq_observation.agents_view[:, :-1]
        )  # (B, K, N, E)

        # Stop gradient for target representations
        target_obs_reps = jax.lax.stop_gradient(target_obs_reps)

        # Project predictions and targets to BYOL space
        pred_obs_reps = jax.vmap(
            lambda x: self.predictor(self.projector(x)), in_axes=1, out_axes=1
        )(pred_obs_reps)  # (B, K, N, E)

        target_obs_reps = jax.vmap(lambda x: self.target_projector(x), in_axes=1, out_axes=1)(
            target_obs_reps
        )  # (B, K, N, E)
        target_obs_reps = jax.lax.stop_gradient(target_obs_reps)

        return pred_obs_reps, target_obs_reps

    def get_cycle_representations(
        self,
        seq_observation: MavaObservation,  # (B, K+1, N, ...)  # sequence of observations
        seq_action: chex.Array,  # (B, K, N, A)  # sequence of actions
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """Compute both forward and backward representations in a cycle.

        The process follows:
        1. Forward phase:
           - Get current state representation from encoder
           - Use dynamics model to predict next K steps iteratively
           - Get target representations from target encoder
        2. Backward phase:
           - Use the last predicted representation from forward phase
           - Use backward dynamics model to predict previous K steps iteratively
           - Get target representations from target encoder

        Args:
            seq_observation: Sequence of observations with shape (B, K+1, N, ...)
            seq_action: Sequence of actions with shape (B, K, N, A)

        Returns:
            forward_pred_obs_reps: Forward predicted observation representations for K steps
            forward_target_obs_reps: Forward target observation representations
            backward_pred_obs_reps: Backward predicted observation representations for K steps
            backward_target_obs_reps: Backward target observation representations
        """
        # Forward phase
        forward_pred_obs_reps, forward_target_obs_reps = self.get_forward_representations(
            seq_observation, seq_action
        )

        # Get the last predicted representation from forward phase
        # This will be used as the starting point for backward prediction
        obs_rep_k = forward_pred_obs_reps[:, -1]  # (B, N, E)

        # Transpose actions to put time dimension first and reverse the sequence
        # From (B, K, N, A) to (K, B, N, A)
        if self.action_space_type == _DISCRETE:
            seq_action_t = jnp.transpose(seq_action[:, :-1], (1, 0, 2))
        else:
            seq_action_t = jnp.transpose(seq_action[:, :-1], (1, 0, 2, 3))
        # Reverse the sequence to predict backwards
        seq_action_t = jnp.flip(seq_action_t, axis=0)

        # Define scan function for K-step backward prediction
        def _backward_prediction_step(
            prev_rep: chex.Array, action: chex.Array
        ) -> Tuple[chex.Array, chex.Array]:
            """Single step backward prediction using dynamics model.

            Args:
                prev_rep: Previous step's representation (output from last DM call), shape (B, N, E)
                action: Current step's action, shape (B, N, A)

            Returns:
                (prev_rep, prev_rep): (carry for next step, output for current step)
            """
            prev_rep = self.backward_dynamic_model(
                action, prev_rep
            )  # use current rep to predict previous
            return prev_rep, prev_rep  # carry forward prev_rep for next step

        # Predict previous K steps using scan
        # At each step k:
        # - prev_rep is the representation from step k+1 (or obs_rep_k for k=K)
        # - action is seq_action_t[k], shape (B, N, A)
        # - output prev_rep will be used as prev_rep for step k-1
        _, backward_pred_obs_reps = jax.lax.scan(
            _backward_prediction_step,
            obs_rep_k,  # initial prev_rep (B, N, E)
            seq_action_t,  # actions sequence (K, B, N, A)
        )  # backward_pred_obs_reps shape: (K, B, N, E)

        # Transpose predictions to match desired shape (B, K, N, E)
        backward_pred_obs_reps = jnp.transpose(backward_pred_obs_reps, (1, 0, 2, 3))
        # Flip back the sequence to match the original order
        backward_pred_obs_reps = jnp.flip(backward_pred_obs_reps, axis=1)

        # Get target representations using target encoder
        # Process each previous observation through target encoder
        _, backward_target_obs_reps = jax.vmap(
            lambda x: self.target_encoder(x), in_axes=1, out_axes=1
        )(seq_observation.agents_view[:, :-1])  # (B, K, N, E)

        # Stop gradient for target representations
        backward_target_obs_reps = jax.lax.stop_gradient(backward_target_obs_reps)

        # Project predictions and targets to BYOL space
        backward_pred_obs_reps = jax.vmap(
            lambda x: self.predictor(self.projector(x)), in_axes=1, out_axes=1
        )(backward_pred_obs_reps)  # (B, K, N, E)

        backward_target_obs_reps = jax.vmap(
            lambda x: self.target_projector(x), in_axes=1, out_axes=1
        )(backward_target_obs_reps)  # (B, K, N, E)
        backward_target_obs_reps = jax.lax.stop_gradient(backward_target_obs_reps)

        return (
            forward_pred_obs_reps,
            forward_target_obs_reps,
            backward_pred_obs_reps,
            backward_target_obs_reps,
        )

    def get_cycleward_representations(
        self,
        seq_observation: MavaObservation,  # (B, K+1, N, ...)  # sequence of observations
        seq_action: chex.Array,  # (B, K, N, A)  # sequence of actions
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """Compute both forward and backward representations using their respective functions.

        The process follows:
        1. Forward phase (using get_forward_representations):
           - Get current state representation from encoder
           - Use dynamics model to predict next K steps iteratively
           - Get target representations from target encoder
        2. Backward phase (using get_backward_representations):
           - Get K-step state representation from encoder
           - Use backward dynamics model to predict previous K steps iteratively
           - Get target representations from target encoder

        Args:
            seq_observation: Sequence of observations with shape (B, K+1, N, ...)
            seq_action: Sequence of actions with shape (B, K, N, A)

        Returns:
            forward_pred_obs_reps: Forward predicted observation representations for K steps
            forward_target_obs_reps: Forward target observation representations
            backward_pred_obs_reps: Backward predicted observation representations for K steps
            backward_target_obs_reps: Backward target observation representations
        """
        # Forward phase
        forward_pred_obs_reps, forward_target_obs_reps = self.get_forward_representations(
            seq_observation, seq_action
        )

        # Backward phase
        backward_pred_obs_reps, backward_target_obs_reps = self.get_backward_representations(
            seq_observation, seq_action
        )

        return (
            forward_pred_obs_reps,
            forward_target_obs_reps,
            backward_pred_obs_reps,
            backward_target_obs_reps,
        )

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


class BackwardDynamicModel(nn.Module):
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
        prev_obs_rep = self.encode_block(projected)

        return prev_obs_rep


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
