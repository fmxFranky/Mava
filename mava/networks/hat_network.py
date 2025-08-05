import typing
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from flax.linen import initializers
from flax.linen.initializers import orthogonal
from typing_extensions import NamedTuple

from mava.networks.rec_mat_network import (
    MATHiddenStates,
    RecurrentDecoder,
    RecurrentEncoder,
    continuous_autoregressive_act_with_hidden,
    continuous_parallel_act_with_hidden,
    discrete_autoregressive_act_with_hidden,
    discrete_parallel_act_with_hidden,
)
from mava.networks.s2mp_network import S2MPNetwork
from mava.systems.mat.types import MATNetworkConfig
from mava.types import JointTrajectory, MavaObservation, RNNObservation
from mava.utils.network_utils import _CONTINUOUS, _DISCRETE

# class HeterogeneousAgentTransformer(nn.Module):


class RecurrentHATActor(nn.Module):
    """Recurrent Actor Network."""

    pred_torso: S2MPNetwork
    hidden_state_dim: int = 128
    traj_len: int = 10  # Length of trajectory history K
    action_dim: int = 1
    action_space_type: str = "discrete"
    n_agents: int = 1
    n_encoder_block: int = 1
    n_encoder_head: int = 1
    n_decoder_block: int = 1
    n_decoder_head: int = 1

    def setup(self) -> None:
        self.encoder = RecurrentEncoder(
            action_dim=self.action_dim,
            n_agent=self.n_agents,
            net_config=MATNetworkConfig(
                n_block=self.n_encoder_block,
                n_head=self.n_encoder_head,
                embed_dim=self.hidden_state_dim,
                use_rmsnorm=False,
                use_swiglu=False,
            ),
        )
        self.decoder = RecurrentDecoder(
            action_dim=self.action_dim,
            n_agent=self.n_agents,
            action_space_type=self.action_space_type,
            net_config=MATNetworkConfig(
                n_block=self.n_decoder_block,
                n_head=self.n_decoder_head,
                embed_dim=self.hidden_state_dim,
                use_rmsnorm=False,
                use_swiglu=False,
            ),
        )
        if self.action_space_type == "discrete":
            self.act_function = discrete_autoregressive_act_with_hidden
            self.train_function = discrete_parallel_act_with_hidden
        else:
            self.act_function = continuous_autoregressive_act_with_hidden
            self.train_function = continuous_parallel_act_with_hidden

    def __call__(
        self,
        policy_hidden_state: Union[chex.Array, Sequence[chex.Array]],
        observation_done: RNNObservation,
        joint_trajectory: Optional["JointTrajectory"] = None,
        key: Optional[chex.PRNGKey] = None,
    ) -> Tuple[chex.Array, tfd.Distribution]:
        """Forward pass."""
        observation, done = observation_done
        bs1 = observation.agents_view.shape[0]
        bs2 = observation.agents_view.shape[1]
        n_agents = self.n_agents

        encoder_hidden_state = policy_hidden_state[0].reshape(bs1 * bs2, -1)
        decoder_self_hidden_state = policy_hidden_state[1].reshape(bs1 * bs2, -1)
        decoder_cross_hidden_state = policy_hidden_state[2].reshape(bs1 * bs2, -1)

        batched_jt = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), joint_trajectory)
        s2mp_agent_mask = observation.agents_view[:, :, :n_agents].reshape(-1, n_agents)
        s2mp_agents_view = self.get_predictions(
            batched_jt.observations, batched_jt.actions, s2mp_agent_mask
        )
        agent_idx = s2mp_agent_mask.argmax(axis=1).mean().astype(jnp.int32)
        s2mp_agents_view = s2mp_agents_view.at[:, agent_idx, -1, :].set(
            observation.agents_view.reshape(bs1 * bs2, -1)
        )

        value, obs_rep, new_encoder_hidden = self.encoder(
            s2mp_agents_view[:, :, -1, :], encoder_hidden_state
        )

        # key, order_key = jax.random.split(key)
        shuffled_agent_order = jnp.argsort(
            jnp.where(
                s2mp_agent_mask[0].astype(jnp.bool_),
                n_agents,
                jnp.arange(n_agents),
            )
        )

        reordered_action = batched_jt.last_actions[:, shuffled_agent_order]
        reordered_obs_rep = obs_rep[:, shuffled_agent_order, :]

        action_log, entropy = self.train_function(
            decoder=self.decoder,
            obs_rep=reordered_obs_rep,
            action=reordered_action,
            action_dim=self.action_dim,
            legal_actions=(
                batched_jt.last_action_masks[:, shuffled_agent_order]
                if batched_jt.last_action_masks is not None
                else None
            ),
            hidden_states=(decoder_self_hidden_state, decoder_cross_hidden_state),
            key=key,
        )
        obs_rep = obs_rep.reshape(bs1, bs2, -1)
        action_log = action_log.reshape(bs1, bs2, *action_log.shape[1:])
        return s2mp_agents_view[:, :, -1, :], obs_rep, action_log, shuffled_agent_order

    def get_actions(
        self,
        policy_hidden_state: Union[chex.Array, Sequence[chex.Array]],
        observation_done: RNNObservation,
        joint_trajectory: Optional["JointTrajectory"] = None,
        key: Optional[chex.PRNGKey] = None,
    ) -> Tuple[chex.Array, tfd.Distribution]:
        """Forward pass."""
        observation, done = observation_done

        encoder_hidden_state = policy_hidden_state[0]
        decoder_self_hidden_state = policy_hidden_state[1]
        decoder_cross_hidden_state = policy_hidden_state[2]

        n_agents = self.n_agents

        if observation.agents_view.ndim == 2:
            bs = observation.agents_view.shape[0:1]
        else:
            bs = observation.agents_view.shape[0:2]
        if joint_trajectory.observations.ndim == 4:
            batched_jt = jax.tree.map(lambda x: x.reshape(-1, *x.shape[1:]), joint_trajectory)
        else:
            batched_jt = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), joint_trajectory)
        if joint_trajectory.observations.ndim == 4:
            batched_jt = jax.tree.map(lambda x: x.reshape(-1, *x.shape[1:]), joint_trajectory)
        else:
            batched_jt = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), joint_trajectory)
        s2mp_agent_mask = observation.agents_view[:, :, :n_agents].reshape(-1, n_agents)
        s2mp_agents_view = self.get_predictions(
            batched_jt.observations, batched_jt.actions, s2mp_agent_mask
        )
        agent_idx = s2mp_agent_mask.argmax(axis=1).mean().astype(jnp.int32)
        s2mp_agents_view = s2mp_agents_view.at[:, agent_idx, -1, :].set(
            observation.agents_view.reshape(-1, *observation.agents_view.shape[2:])
        )

        value, obs_rep, new_encoder_hidden = self.encoder(
            s2mp_agents_view[:, :, -1, :], encoder_hidden_state
        )
        shuffled_agent_order = jnp.arange(n_agents, dtype=jnp.int32)
        key, subkey = jax.random.split(key)

        shuffled_agent_order = jnp.argsort(
            jnp.where(
                s2mp_agent_mask[0].astype(jnp.bool_),
                n_agents,
                jax.random.randint(subkey, (n_agents,), 0, n_agents),
            )
        )
        reordered_obs_rep = obs_rep[:, shuffled_agent_order]

        legal_actions = observation.action_mask.reshape(-1, *observation.action_mask.shape[2:])
        legal_actions = jnp.repeat(legal_actions[:, jnp.newaxis, :], n_agents, axis=1)
        legal_actions = legal_actions.at[:, :-1, :].set(True)

        output_action, output_action_log, new_dec_hidden = self.act_function(
            decoder=self.decoder,
            obs_rep=reordered_obs_rep,
            action_dim=self.action_dim,
            legal_actions=(legal_actions if observation.action_mask is not None else None),
            hidden_states=(decoder_self_hidden_state, decoder_cross_hidden_state),
            key=key,
        )

        output_action = output_action.reshape(*bs, *output_action.shape[1:])
        output_action_log = output_action_log.reshape(*bs, *output_action_log.shape[1:])
        new_hidden_states = [new_encoder_hidden, new_dec_hidden[0], new_dec_hidden[1]]

        return new_hidden_states, output_action, output_action_log

    def get_predictions(
        self,
        obs_seq: chex.Array,
        action_seq: chex.Array,
        agent_mask: chex.Array,
    ) -> Tuple[chex.Array, tfd.Distribution]:
        """Forward pass."""
        out = self.pred_torso(obs_seq, action_seq, agent_mask)
        return out


HATNetworkConfig = MATNetworkConfig
HATActorApply = Callable
HATLearnerState = dict
HATTransition = dict
HATParams = dict
