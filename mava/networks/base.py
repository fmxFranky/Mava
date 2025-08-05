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

import functools
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import linen as nn
from flax.linen.initializers import orthogonal

from mava.networks.attention import SelfAttention
from mava.networks.distributions import MaskedEpsGreedyDistribution
from mava.networks.mat_network import _make_mlp, MATNetworkConfig
from mava.networks.s2mp_network import S2MPNetwork
from mava.networks.torsos import MLPTorso
from mava.types import (
    JointTrajectory,
    Observation,
    ObservationGlobalState,
    RNNGlobalObservation,
    RNNObservation,
)


import numpy as np

if TYPE_CHECKING:
    from mava.types import JointTrajectory


class FeedForwardActor(nn.Module):
    """Feed Forward Actor Network."""

    torso: nn.Module
    action_head: nn.Module

    @nn.compact
    def __call__(self, observation: Observation) -> tfd.Distribution:
        """Forward pass."""
        obs_embedding = self.torso(observation.agents_view)

        return self.action_head(obs_embedding, observation.action_mask)


class FeedForwardValueNet(nn.Module):
    """Feedforward Value Network. Returns the value of an observation."""

    torso: nn.Module
    centralised_critic: bool = False

    @nn.compact
    def __call__(self, observation: Union[Observation, ObservationGlobalState]) -> chex.Array:
        """Forward pass."""
        if self.centralised_critic:
            if not isinstance(observation, ObservationGlobalState):
                raise ValueError("Global state must be provided to the centralised critic.")
            # Get global state in the case of a centralised critic.
            observation = observation.global_state
        else:
            # Get single agent view in the case of a decentralised critic.
            observation = observation.agents_view

        critic_output = self.torso(observation)
        critic_output = nn.Dense(1, kernel_init=orthogonal(1.0))(critic_output)

        return jnp.squeeze(critic_output, axis=-1)


class FeedForwardQNet(nn.Module):
    """Feedforward Q Network. Returns the value of an observation-action pair."""

    torso: nn.Module
    centralised_critic: bool = False

    def setup(self) -> None:
        self.critic = nn.Dense(1, kernel_init=orthogonal(1.0))

    def __call__(
        self,
        observation: Union[Observation, ObservationGlobalState],
        action: chex.Array,
    ) -> chex.Array:
        if self.centralised_critic:
            if not isinstance(observation, ObservationGlobalState):
                raise ValueError("Global state must be provided to the centralised critic.")
            # Get global state in the case of a centralised critic.
            observation = observation.global_state
        else:
            # Get single agent view in the case of a decentralised critic.
            observation = observation.agents_view

        x = jnp.concatenate([observation, action], axis=-1)
        x = self.torso(x)
        y = self.critic(x)

        return jnp.squeeze(y, axis=-1)


class ScannedRNN(nn.Module):
    hidden_state_dim: int = 128

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry: chex.Array, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, :, jnp.newaxis],
            self.initialize_carry((ins.shape[0], ins.shape[1]), self.hidden_state_dim),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[-1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size: Sequence[int], hidden_size: int) -> chex.Array:
        """Initializes the carry state."""
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (*batch_size, hidden_size))


# We need a per agent ScannedRNN for the HAPPO actors since we vmap over agents
class ScannedRNNPerAgent(nn.Module):
    hidden_state_dim: int = 128

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry: chex.Array, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, jnp.newaxis],
            # only a single agent, so we don't have an agent batch dim anymore
            self.initialize_carry((ins.shape[0]), self.hidden_state_dim),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[-1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int) -> chex.Array:
        """Initializes the carry state."""
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class SelfWithoutAttention(nn.Module):
    """Self-attention network without masking."""

    input_dim: int
    heads: int
    output_dim: int

    @nn.compact
    def __call__(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """前向传播。"""
        query_layer = nn.Dense(self.input_dim * self.heads, name="query")
        key_layer = nn.Dense(self.input_dim * self.heads, name="key")
        value_layer = nn.Dense(self.input_dim * self.heads, name="value")

        queries = query_layer(x)
        keys = key_layer(x)
        values = value_layer(x)

        # 计算注意力分数
        scores = jnp.matmul(queries, jnp.swapaxes(keys, -2, -1)) / jnp.sqrt(self.input_dim)
        attention = nn.softmax(scores, axis=-1)
        weighted = jnp.matmul(attention, values)

        return attention, weighted


class RecurrentActor(nn.Module):
    """Recurrent Actor Network."""

    pre_torso: nn.Module
    post_torso: nn.Module
    action_head: nn.Module
    pred_torso: S2MPNetwork
    hidden_state_dim: int = 128
    traj_len: int = 10  # Length of trajectory history K
    use_ma2e_fusion: bool = False
    scan_fn: nn.Module = ScannedRNN

    def setup(self) -> None:
        self.rnn = self.scan_fn(self.hidden_state_dim)
        if self.use_ma2e_fusion:
            self.self_obs_attention = SelfWithoutAttention(
                input_dim=self.pred_torso.obs_dim, heads=1, output_dim=self.pred_torso.obs_dim
            )
            self.fusion_fc = nn.Dense(self.hidden_state_dim // 2, kernel_init=orthogonal(1.0))

    def __call__(
        self,
        policy_hidden_state: Union[chex.Array, Sequence[chex.Array]],
        observation_done: RNNObservation,
        joint_trajectory: Optional["JointTrajectory"] = None,
        key: Optional[chex.PRNGKey] = None,
    ) -> Tuple[chex.Array, tfd.Distribution]:
        """Forward pass."""
        observation, done = observation_done

        # Handle both single hidden state and list of hidden states for backward compatibility
        if isinstance(policy_hidden_state, (list, tuple)):
            policy_hidden_state = policy_hidden_state[0]

        bs1 = observation.agents_view.shape[0]
        bs2 = observation.agents_view.shape[1]

        if self.use_ma2e_fusion:
            obs_embedding = self.pre_torso(observation.agents_view)
            batched_jt = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), joint_trajectory)
            s2mp_agent_mask = observation.agents_view[:, :, : self.pred_torso.n_agents].reshape(
                -1, self.pred_torso.n_agents
            )
            s2mp_agents_view = self.get_predictions(
                batched_jt.observations, batched_jt.actions, s2mp_agent_mask
            )
            agent_idx = s2mp_agent_mask.argmax(axis=1).mean().astype(jnp.int32)
            s2mp_agents_view = s2mp_agents_view.at[:, agent_idx, -1, :].set(
                observation.agents_view.reshape(bs1 * bs2, -1)
            )
            _, fusion_embedding = self.self_obs_attention(s2mp_agents_view)
            fusion_embedding = fusion_embedding[:, agent_idx, -1, :].reshape(bs1, bs2, -1)
            fusion_embedding = self.fusion_fc(fusion_embedding)
            policy_embedding = jnp.concatenate([obs_embedding, fusion_embedding], axis=-1)

        else:
            policy_embedding = self.pre_torso(observation.agents_view)

        policy_rnn_input = (policy_embedding, done)
        # print(f"[train]policy_embedding: {policy_embedding.shape}")
        # print(f"[train]policy_hidden_state: {policy_hidden_state.shape}")
        policy_hidden_state, policy_embedding = self.rnn(policy_hidden_state, policy_rnn_input)
        policy_embedding = self.post_torso(policy_embedding)
        pi = self.action_head(policy_embedding, observation.action_mask)

        return policy_hidden_state, pi

    def get_actions(
        self,
        policy_hidden_state: Union[chex.Array, Sequence[chex.Array]],
        observation_done: RNNObservation,
        joint_trajectory: Optional["JointTrajectory"] = None,
        key: Optional[chex.PRNGKey] = None,
    ) -> Tuple[chex.Array, tfd.Distribution]:
        """Forward pass."""
        observation, done = observation_done

        # Handle both single hidden state and list of hidden states for backward compatibility
        if isinstance(policy_hidden_state, (list, tuple)):
            policy_hidden_state = policy_hidden_state[0]

        if self.use_ma2e_fusion:
            obs_embedding = self.pre_torso(observation.agents_view)
            if obs_embedding.ndim == 2:
                bs = observation.agents_view.shape[0:1]
            else:
                bs = observation.agents_view.shape[0:2]
            if joint_trajectory.observations.ndim == 4:
                batched_jt = jax.tree.map(lambda x: x.reshape(-1, *x.shape[1:]), joint_trajectory)
            else:
                batched_jt = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), joint_trajectory)
            s2mp_agent_mask = observation.agents_view[:, :, : self.pred_torso.n_agents].reshape(
                -1, self.pred_torso.n_agents
            )
            s2mp_agents_view = self.get_predictions(
                batched_jt.observations, batched_jt.actions, s2mp_agent_mask
            )
            agent_idx = s2mp_agent_mask.argmax(axis=1).mean().astype(jnp.int32)
            s2mp_agents_view = s2mp_agents_view.at[:, agent_idx, -1, :].set(
                observation.agents_view.reshape(-1, *observation.agents_view.shape[2:])
            )
            _, fusion_embedding = self.self_obs_attention(s2mp_agents_view)

            fusion_embedding = fusion_embedding[:, agent_idx, -1, :].reshape(*bs, -1)

            fusion_embedding = self.fusion_fc(fusion_embedding)
            policy_embedding = jnp.concatenate([obs_embedding, fusion_embedding], axis=-1)
        else:
            policy_embedding = self.pre_torso(observation.agents_view)

        policy_rnn_input = (policy_embedding, done)
        policy_hidden_state, policy_embedding = self.rnn(policy_hidden_state, policy_rnn_input)
        policy_embedding = self.post_torso(policy_embedding)
        pi = self.action_head(policy_embedding, observation.action_mask)

        return policy_hidden_state, pi

    def get_predictions(
        self,
        obs_seq: chex.Array,
        action_seq: chex.Array,
        agent_mask: chex.Array,
    ) -> Tuple[chex.Array, tfd.Distribution]:
        """Forward pass."""
        out = self.pred_torso(obs_seq, action_seq, agent_mask)
        return out





class RecurrentEncodeBlock(nn.Module):
    """Recurrent encoder block with hidden state support."""

    n_agent: int
    embed_dim: int
    n_head: int
    use_rmsnorm: bool = True
    use_swiglu: bool = True
    masked: bool = False

    def setup(self) -> None:
        ln = nn.RMSNorm if self.use_rmsnorm else nn.LayerNorm
        self.ln1 = ln()
        self.ln2 = ln()

        self.attn = SelfAttention(
            self.embed_dim, self.n_head, self.n_agent * 2 + 1, self.masked
        )  # +1 for hidden state token

        self.mlp = _make_mlp(self.embed_dim, self.use_swiglu)

    def __call__(
        self, x: chex.Array, hidden_state: chex.Array, cls_token: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        x_with_cls = jnp.concatenate([cls_token, hidden_state, x], axis=1)  # (B, N+1, embed_dim)

        # Apply attention
        attended = self.attn(x_with_cls, x_with_cls, x_with_cls)
        x_with_cls = self.ln1(x_with_cls + attended)

        # Apply MLP
        mlp_out = self.mlp(x_with_cls)
        x_with_cls = self.ln2(x_with_cls + mlp_out)

        # Split back to hidden state and agent features
        new_hidden_state = x_with_cls[:, 1 : self.n_agent + 1, :]  # (B, N, embed_dim)
        new_x = x_with_cls[:, self.n_agent + 1 :, :]  # (B, N, embed_dim)

        return new_x, new_hidden_state


class RecurrentValueNet(nn.Module):
    """Recurrent Critic Network."""

    pre_torso: nn.Module
    post_torso: nn.Module
    centralised_critic: bool = False
    hidden_state_dim: int = 128
    traj_len: int = 10  # Length of trajectory history K
    use_transformer_torso: bool = False
    n_agent: int = None

    def setup(self) -> None:
        if self.use_transformer_torso:
            self.rnn_torso = RecurrentEncodeBlock(
                n_agent=self.n_agent,
                embed_dim=self.hidden_state_dim,
                n_head=1,
                use_rmsnorm=False,
                use_swiglu=False,
                masked=False,
            )
            self.cls_token = self.param(
                "cls_token", nn.initializers.normal(stddev=0.02), (1, self.hidden_state_dim)
            )
            self.fc = nn.Dense(self.hidden_state_dim, kernel_init=orthogonal(1.0))
        else:
            self.rnn_torso = ScannedRNN(self.hidden_state_dim)
        self.value_head = nn.Dense(1, kernel_init=orthogonal(1.0))

    @nn.compact
    def __call__(
        self,
        value_net_hidden_state: Union[chex.Array, Sequence[chex.Array]],
        observation_done: Union[RNNObservation, RNNGlobalObservation],
        joint_trajectory: Optional["JointTrajectory"],
    ) -> Tuple[chex.Array, chex.Array]:
        """Forward pass."""
        observation, done = observation_done
        if isinstance(value_net_hidden_state, (list, tuple)):
            value_net_hidden_state = value_net_hidden_state[0]

        if self.centralised_critic and not isinstance(observation, ObservationGlobalState):
            raise ValueError("Global state must be provided to the centralised critic.")

        value_embedding = self.pre_torso(
            observation.global_state if self.centralised_critic else observation.agents_view
        )

        if not self.use_transformer_torso:
            value_rnn_input = (value_embedding, done)
            value_net_hidden_state, value_embedding = self.rnn_torso(
                value_net_hidden_state, value_rnn_input
            )

        else:
            bs1, bs2 = observation.agents_view.shape[0:2]
            value_embedding = value_embedding.reshape(-1, *value_embedding.shape[2:])
            ori_ndim = value_net_hidden_state.ndim
            if ori_ndim == 4:
                value_net_hidden_state = value_net_hidden_state.reshape(
                    -1, *value_net_hidden_state.shape[2:]
                )
            if self.centralised_critic:
                cls_token = self.fc(observation.global_state)
                cls_token = cls_token.reshape(-1, *cls_token.shape[2:]).mean(axis=1, keepdims=True)
            else:
                cls_token = jnp.repeat(self.cls_token, value_embedding.shape[0], axis=0)
            value_embedding, value_net_hidden_state = self.rnn_torso(
                value_embedding, value_net_hidden_state, cls_token
            )
            if ori_ndim == 4:
                value_net_hidden_state = value_net_hidden_state.reshape(bs1, bs2, self.n_agent, -1)
            value_embedding = value_embedding.reshape(bs1, bs2, self.n_agent, -1)

        value = self.post_torso(value_embedding)
        value = self.value_head(value)

        return value_net_hidden_state, jnp.squeeze(value, axis=-1)


class RecQNetwork(nn.Module):
    """Recurrent Q-Network."""

    pre_torso: nn.Module
    post_torso: nn.Module
    num_actions: int
    hidden_state_dim: int = 128
    traj_len: int = 10  # Length of trajectory history K

    @nn.compact
    def get_q_values(
        self,
        hidden_state: chex.Array,
        observations_resets: RNNObservation,
        joint_trajectory: Optional["JointTrajectory"] = None,
    ) -> chex.Array:
        """Forward pass to obtain q values."""
        obs, resets = observations_resets

        # TODO: Process joint_trajectory for enhanced Q-value computation
        # Currently joint_trajectory is passed but not used in the computation
        # joint_trajectory contains: observations [B, N, K, *obs_dim], actions [B, N, K, *act_dim]
        # The last timestep observation in trajectories should match current obs.agents_view

        embedding = self.pre_torso(obs.agents_view)

        rnn_input = (embedding, resets)
        hidden_state, embedding = ScannedRNN(self.hidden_state_dim)(hidden_state, rnn_input)

        embedding = self.post_torso(embedding)

        q_values = nn.Dense(self.num_actions, kernel_init=orthogonal(0.01))(embedding)

        return hidden_state, q_values

    def __call__(
        self,
        hidden_state: chex.Array,
        observations_resets: RNNObservation,
        eps: float = 0,
        joint_trajectory: Optional["JointTrajectory"] = None,
    ) -> chex.Array:
        """Forward pass with additional construction of epsilon-greedy distribution.
        When epsilon is not specified, we assume a greedy approach.
        """
        obs, _ = observations_resets
        hidden_state, q_values = self.get_q_values(
            hidden_state, observations_resets, joint_trajectory
        )
        eps_greedy_dist = MaskedEpsGreedyDistribution(q_values, eps, obs.action_mask)

        return hidden_state, eps_greedy_dist


class QMixingNetwork(nn.Module):
    num_actions: int
    num_agents: int
    hyper_hidden_dim: int = 64
    embed_dim: int = 32
    norm_env_states: bool = True

    def setup(self) -> None:
        self.hyper_w1: MLPTorso = MLPTorso(
            (self.hyper_hidden_dim, self.embed_dim * self.num_agents),
            activate_final=False,
        )

        self.hyper_b1: MLPTorso = MLPTorso(
            (self.embed_dim,),
            activate_final=False,
        )

        self.hyper_w2: MLPTorso = MLPTorso(
            (self.hyper_hidden_dim, self.embed_dim),
            activate_final=False,
        )

        self.hyper_b2: MLPTorso = MLPTorso(
            (self.embed_dim, 1),
            activate_final=False,
        )

        self.layer_norm: nn.Module = nn.LayerNorm()

    @nn.compact
    def __call__(
        self,
        agent_qs: chex.Array,
        env_global_state: chex.Array,
    ) -> chex.Array:
        B, T = agent_qs.shape[:2]  # batch size

        agent_qs = jnp.reshape(agent_qs, (B, T, 1, self.num_agents))

        if self.norm_env_states:
            states = self.layer_norm(env_global_state)
        else:
            states = env_global_state

        # First layer
        w1 = jnp.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        w1 = jnp.reshape(w1, (B, T, self.num_agents, self.embed_dim))
        b1 = jnp.reshape(b1, (B, T, 1, self.embed_dim))

        # Matrix multiplication
        hidden = nn.elu(jnp.matmul(agent_qs, w1) + b1)

        # Second layer
        w2 = jnp.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)

        w2 = jnp.reshape(w2, (B, T, self.embed_dim, 1))
        b2 = jnp.reshape(b2, (B, T, 1, 1))

        # Compute final output
        y = jnp.matmul(hidden, w2) + b2

        # Reshape
        q_tot = jnp.reshape(y, (B, T, 1))

        return q_tot
