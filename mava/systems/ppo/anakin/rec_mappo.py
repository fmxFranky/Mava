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

import copy
import time
from typing import Any, Callable, Dict, Tuple

import chex
import flashbax as fbx
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
from colorama import Fore, Style
from flashbax.vault import Vault
from flax.core.frozen_dict import FrozenDict
from jax import tree
from omegaconf import DictConfig, OmegaConf

from mava.evaluator import (
    get_eval_fn_with_traj,
    get_num_eval_envs,
    make_rec_eval_act_fn_with_traj,
)
from mava.networks import RecurrentActor as Actor
from mava.networks import RecurrentValueNet as Critic
from mava.networks import ScannedRNN
from mava.systems.ppo.types import (
    HiddenStates,
    OptStates,
    Params,
    RNNLearnerState,
    RNNPPOTransition,
    TrajectoryState,
)
from mava.types import (
    ExperimentOutput,
    JointTrajectory,
    MarlEnv,
    MavaObservation,
    Metrics,
    RecActorApply,
    RecCriticApply,
)
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.multistep import calculate_gae
from mava.utils.network_utils import get_action_head
from mava.utils.training import make_learning_rate
from mava.wrappers.episode_metrics import get_final_step_metrics

# Define the learner function type that returns experience data
StoreExpLearnerFn = Callable[
    [RNNLearnerState], Tuple[ExperimentOutput[RNNLearnerState], RNNPPOTransition]
]


def update_trajectory_state(
    trajectory_state: TrajectoryState,
    current_obs: MavaObservation,
    current_action: chex.Array,
) -> TrajectoryState:
    """Update trajectory state with new observation and action."""
    # Shift history and add new data
    new_obs_history = jnp.roll(trajectory_state.obs_history, -1, axis=2)
    # Set the last position to the current observations
    # current_obs is Observation or ObservationGlobalState, use .agents_view
    new_obs_history = new_obs_history.at[:, :, -1].set(current_obs.agents_view)

    new_action_history = jnp.roll(trajectory_state.action_history, -1, axis=2)
    new_action_history = new_action_history.at[:, :, -1].set(current_action)

    # Update buffer index (capped at traj_len)
    new_buffer_idx = jnp.minimum(trajectory_state.buffer_idx + 1, new_obs_history.shape[2])
    new_buffer_full = trajectory_state.buffer_full | (new_buffer_idx == new_obs_history.shape[2])

    return TrajectoryState(
        obs_history=new_obs_history,
        action_history=new_action_history,
        buffer_idx=new_buffer_idx,
        buffer_full=new_buffer_full,
    )


def construct_joint_trajectory_for_timestep(
    traj_batch: RNNPPOTransition, timestep_idx: int, config: DictConfig
) -> JointTrajectory:
    """Construct joint trajectory for a specific timestep with proper history.

    This function constructs the trajectory history for timestep_idx,
    where the history includes [timestep_idx-traj_len+1, ..., timestep_idx].
    """
    traj_len = config.system.get("traj_len", 5)
    T, B, N = traj_batch.obs.agents_view.shape[:3]
    obs_shape = traj_batch.obs.agents_view.shape[3:]

    # For timestep_idx, we want history [timestep_idx-traj_len+1, ..., timestep_idx]
    start_idx = max(0, timestep_idx - traj_len + 1)
    end_idx = timestep_idx + 1

    # Extract the available history
    obs_slice = traj_batch.obs.agents_view[start_idx:end_idx]  # [actual_len, B, N, *obs_dim]
    action_slice = traj_batch.action[start_idx:end_idx]  # [actual_len, B, N] or [actual_len, B]

    actual_len = obs_slice.shape[0]

    # Handle action dimensions - expand if needed
    if action_slice.ndim == 2:  # [actual_len, B] -> expand to [actual_len, B, N]
        action_slice = jnp.broadcast_to(action_slice[:, :, jnp.newaxis], (actual_len, B, N))

    # Pad if necessary (when timestep_idx < traj_len-1)
    if actual_len < traj_len:
        pad_len = traj_len - actual_len
        obs_pad = jnp.zeros((pad_len, B, N, *obs_shape))
        action_pad = jnp.zeros((pad_len, B, N), dtype=action_slice.dtype)

        obs_hist = jnp.concatenate([obs_pad, obs_slice], axis=0)  # [traj_len, B, N, *obs_dim]
        action_hist = jnp.concatenate([action_pad, action_slice], axis=0)  # [traj_len, B, N]
    else:
        obs_hist = obs_slice[-traj_len:]  # Take last traj_len steps
        action_hist = action_slice[-traj_len:]

    # Transpose to [B, N, traj_len, *dims] format expected by networks
    obs_hist = obs_hist.transpose(1, 2, 0, *range(3, obs_hist.ndim))  # [B, N, traj_len, *obs_dim]
    action_hist = action_hist.transpose(1, 2, 0)  # [B, N, traj_len]

    return JointTrajectory(
        observations=obs_hist,  # [B, N, traj_len, *obs_dim]
        actions=action_hist,  # [B, N, traj_len]
    )


def get_learner_fn(
    env: MarlEnv,
    apply_fns: Tuple[RecActorApply, RecCriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> StoreExpLearnerFn:
    """Get the learner function."""
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    def _update_step(learner_state: RNNLearnerState, _: Any) -> Tuple[RNNLearnerState, Tuple]:
        """A single update of the network.

        This function steps the environment and records the trajectory batch for
        training. It then calculates advantages and targets based on the recorded
        trajectory and updates the actor and critic networks based on the calculated
        losses.

        Args:
        ----
            learner_state (NamedTuple):
                - params (Params): The current model parameters.
                - opt_states (OptStates): The current optimizer states.
                - key (PRNGKey): The random number generator state.
                - env_state (State): The environment state.
                - last_timestep (TimeStep): The last timestep in the current trajectory.
                - last_done (bool): Whether the last timestep was a terminal state.
                - hstates (HiddenStates): The hidden state of the policy and critic RNN.
            _ (Any): The current metrics info.

        """

        def _env_step(
            learner_state: RNNLearnerState, _: Any
        ) -> Tuple[RNNLearnerState, Tuple[RNNPPOTransition, Metrics]]:
            """Step the environment."""
            (
                params,
                opt_states,
                key,
                env_state,
                last_timestep,
                last_done,
                last_hstates,
                trajectory_state,
            ) = learner_state

            key, policy_key = jax.random.split(key)

            # Add a batch dimension to the observation.
            batched_observation = tree.map(lambda x: x[jnp.newaxis, :], last_timestep.observation)
            ac_in = (batched_observation, last_done[jnp.newaxis, :])

            # Construct joint trajectory directly from trajectory state
            joint_traj = JointTrajectory(
                observations=trajectory_state.obs_history[jnp.newaxis, ...],
                actions=trajectory_state.action_history[jnp.newaxis, ...],
            )

            # Run the network with trajectory information
            policy_hidden_state, actor_policy = actor_apply_fn(
                params.actor_params, last_hstates.policy_hidden_state, ac_in, joint_traj
            )
            critic_hidden_state, value = critic_apply_fn(
                params.critic_params, last_hstates.critic_hidden_state, ac_in, joint_traj
            )

            # Sample action from the policy and squeeze out the batch dimension.
            action = actor_policy.sample(seed=policy_key)
            log_prob = actor_policy.log_prob(action)

            action, log_prob, value = action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)

            # Update trajectory state with new observation and action
            new_trajectory_state = update_trajectory_state(
                trajectory_state, last_timestep.observation, action
            )

            # Step the environment.
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            done = timestep.last().repeat(env.num_agents).reshape(config.arch.num_envs, -1)
            hstates = HiddenStates(policy_hidden_state, critic_hidden_state)
            transition = RNNPPOTransition(
                last_done,
                action,
                value,
                timestep.reward,
                log_prob,
                last_timestep.observation,
                last_hstates,
            )
            learner_state = RNNLearnerState(
                params, opt_states, key, env_state, timestep, done, hstates, new_trajectory_state
            )
            metrics = timestep.extras["episode_metrics"] | timestep.extras["env_metrics"]
            return learner_state, (transition, metrics)

        # Step environment for rollout length
        learner_state, (traj_batch, episode_metrics) = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        # Calculate advantage
        params, opt_states, key, env_state, last_timestep, last_done, hstates, trajectory_state = (
            learner_state
        )

        # Add a batch dimension to the observation.
        batched_last_observation = tree.map(lambda x: x[jnp.newaxis, :], last_timestep.observation)
        ac_in = (batched_last_observation, last_done[jnp.newaxis, :])

        # Construct joint trajectory for last value computation with proper per-timestep history
        # Use the current trajectory_state which contains the latest history
        joint_traj_last = JointTrajectory(
            observations=trajectory_state.obs_history,  # [B, N, traj_len, *obs_dim]
            actions=trajectory_state.action_history,  # [B, N, traj_len, *action_dim]
        )

        # Run the network.
        _, last_val = critic_apply_fn(
            params.critic_params, hstates.critic_hidden_state, ac_in, joint_traj_last
        )

        # Squeeze out the batch dimension and mask out the value of terminal states.
        last_val = last_val.squeeze(0)

        advantages, targets = calculate_gae(
            traj_batch, last_val, last_done, config.system.gamma, config.system.gae_lambda
        )

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""
                params, opt_states, key = train_state
                traj_batch, advantages, targets = batch_info

                def _actor_loss_fn(
                    actor_params: FrozenDict,
                    traj_batch: RNNPPOTransition,
                    gae: chex.Array,
                    key: chex.PRNGKey,
                ) -> Tuple:
                    """Calculate the actor loss."""
                    # Rerun network
                    obs_and_done = (traj_batch.obs, traj_batch.done)

                    # Construct real joint trajectory for the last timestep in the sequence
                    # This provides the most complete historical context for the network
                    T = traj_batch.obs.agents_view.shape[0]  # sequence length
                    last_timestep_idx = T - 1

                    joint_traj = construct_joint_trajectory_for_timestep(
                        traj_batch, last_timestep_idx, config
                    )

                    _, actor_policy = actor_apply_fn(
                        actor_params,
                        traj_batch.hstates.policy_hidden_state[0],
                        obs_and_done,
                        joint_traj,  # Real joint trajectory with proper historical context
                    )
                    log_prob = actor_policy.log_prob(traj_batch.action)

                    # Calculate actor loss
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    # Nomalise advantage at minibatch level
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    actor_loss1 = ratio * gae
                    actor_loss2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config.system.clip_eps,
                            1.0 + config.system.clip_eps,
                        )
                        * gae
                    )
                    actor_loss = -jnp.minimum(actor_loss1, actor_loss2)
                    actor_loss = actor_loss.mean()
                    # The seed will be used in the TanhTransformedDistribution:
                    entropy = actor_policy.entropy(seed=key).mean()

                    total_loss = actor_loss - config.system.ent_coef * entropy
                    return total_loss, (actor_loss, entropy)

                def _critic_loss_fn(
                    critic_params: FrozenDict,
                    traj_batch: RNNPPOTransition,
                    targets: chex.Array,
                ) -> Tuple:
                    """Calculate the critic loss."""
                    # Rerun network
                    obs_and_done = (traj_batch.obs, traj_batch.done)

                    # Construct real joint trajectory for the last timestep in the sequence
                    # This provides the most complete historical context for the network
                    T = traj_batch.obs.agents_view.shape[0]  # sequence length
                    last_timestep_idx = T - 1

                    joint_traj = construct_joint_trajectory_for_timestep(
                        traj_batch, last_timestep_idx, config
                    )

                    _, value = critic_apply_fn(
                        critic_params,
                        traj_batch.hstates.critic_hidden_state[0],
                        obs_and_done,
                        joint_traj,  # Real joint trajectory with proper historical context
                    )

                    # Clipped MSE loss
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                        -config.system.clip_eps, config.system.clip_eps
                    )
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                    total_loss = config.system.vf_coef * value_loss
                    return total_loss, value_loss

                # Calculate actor loss
                key, entropy_key = jax.random.split(key)
                actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                actor_loss_info, actor_grads = actor_grad_fn(
                    params.actor_params,
                    traj_batch,
                    advantages,
                    entropy_key,
                )

                # Calculate critic loss
                critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                value_loss_info, critic_grads = critic_grad_fn(
                    params.critic_params, traj_batch, targets
                )

                # Compute the parallel mean (pmean) over the batch.
                # This pmean could be a regular mean as the batch axis is on the same device.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="batch"
                )
                # pmean over devices.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="device"
                )

                critic_grads, value_loss_info = jax.lax.pmean(
                    (critic_grads, value_loss_info), axis_name="batch"
                )
                # pmean over devices.
                critic_grads, value_loss_info = jax.lax.pmean(
                    (critic_grads, value_loss_info), axis_name="device"
                )

                # Update params and optimiser state
                actor_updates, actor_new_opt_state = actor_update_fn(
                    actor_grads, opt_states.actor_opt_state
                )
                actor_new_params = optax.apply_updates(params.actor_params, actor_updates)

                critic_updates, critic_new_opt_state = critic_update_fn(
                    critic_grads, opt_states.critic_opt_state
                )
                critic_new_params = optax.apply_updates(params.critic_params, critic_updates)

                new_params = Params(actor_new_params, critic_new_params)
                new_opt_state = OptStates(actor_new_opt_state, critic_new_opt_state)

                actor_loss, (_, entropy) = actor_loss_info
                value_loss, unscaled_value_loss = value_loss_info

                total_loss = actor_loss + value_loss
                loss_info = {
                    "total_loss": total_loss,
                    "value_loss": unscaled_value_loss,
                    "actor_loss": actor_loss,
                    "entropy": entropy,
                }

                return (new_params, new_opt_state, entropy_key), loss_info

            params, opt_states, traj_batch, advantages, targets, key = update_state
            key, shuffle_key, entropy_key = jax.random.split(key, 3)

            # Shuffle minibatches
            batch = (traj_batch, advantages, targets)
            num_recurrent_chunks = (
                config.system.rollout_length // config.system.recurrent_chunk_size
            )
            batch = tree.map(
                lambda x: x.reshape(
                    config.system.recurrent_chunk_size,
                    config.arch.num_envs * num_recurrent_chunks,
                    *x.shape[2:],
                ),
                batch,
            )
            permutation = jax.random.permutation(
                shuffle_key, config.arch.num_envs * num_recurrent_chunks
            )
            shuffled_batch = tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
            reshaped_batch = tree.map(
                lambda x: jnp.reshape(
                    x, (x.shape[0], config.system.num_minibatches, -1, *x.shape[2:])
                ),
                shuffled_batch,
            )
            minibatches = tree.map(lambda x: jnp.swapaxes(x, 1, 0), reshaped_batch)

            # Update minibatches
            (params, opt_states, entropy_key), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states, entropy_key), minibatches
            )

            update_state = (
                params,
                opt_states,
                traj_batch,
                advantages,
                targets,
                key,
            )
            return update_state, loss_info

        update_state = (
            params,
            opt_states,
            traj_batch,
            advantages,
            targets,
            key,
        )

        # Update epochs
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.ppo_epochs
        )

        params, opt_states, traj_batch, advantages, targets, key = update_state
        learner_state = RNNLearnerState(
            params,
            opt_states,
            key,
            env_state,
            last_timestep,
            last_done,
            hstates,
            trajectory_state,
        )
        return learner_state, (episode_metrics, loss_info, traj_batch)

    def learner_fn(
        learner_state: RNNLearnerState,
    ) -> Tuple[ExperimentOutput[RNNLearnerState], RNNPPOTransition]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
        ----
            learner_state (NamedTuple):
                - params (Params): The initial model parameters.
                - opt_states (OptStates): The initial optimizer states.
                - key (chex.PRNGKey): The random number generator state.
                - env_state (LogEnvState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.
                - dones (bool): Whether the initial timestep was a terminal state.
                - hstates (HiddenStates): The hidden state of the policy and critic RNN.

        """
        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        learner_state, (episode_info, loss_info, traj_batch) = jax.lax.scan(
            batched_update_step, learner_state, None, config.system.num_updates_per_eval
        )
        return (
            ExperimentOutput(
                learner_state=learner_state,
                episode_metrics=episode_info,
                train_metrics=loss_info,
            ),
            traj_batch,
        )

    return learner_fn


def learner_setup(
    env: MarlEnv, keys: chex.Array, config: DictConfig
) -> Tuple[StoreExpLearnerFn, Actor, RNNLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Determine number of devices: default to 1 unless override via config
    n_devices = min(config.arch.get("n_devices", 1), len(jax.devices()))

    # Get number of agents.
    num_agents = env.num_agents
    config.system.num_agents = num_agents

    # Set trajectory length, default to 10 if not specified
    traj_len = getattr(config.system, "traj_len", 10)

    # PRNG keys.
    key, actor_net_key, critic_net_key = keys

    # Define network and optimiser.
    actor_pre_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_post_torso = hydra.utils.instantiate(config.network.actor_network.post_torso)
    action_head, _ = get_action_head(env.action_spec)
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=env.action_dim)
    critic_pre_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    critic_post_torso = hydra.utils.instantiate(config.network.critic_network.post_torso)

    actor_network = Actor(
        pre_torso=actor_pre_torso,
        post_torso=actor_post_torso,
        action_head=actor_action_head,
        hidden_state_dim=config.network.hidden_state_dim,
        traj_len=traj_len,
    )
    critic_network = Critic(
        pre_torso=critic_pre_torso,
        post_torso=critic_post_torso,
        hidden_state_dim=config.network.hidden_state_dim,
        centralised_critic=True,
        traj_len=traj_len,
    )

    actor_lr = make_learning_rate(config.system.actor_lr, config)
    critic_lr = make_learning_rate(config.system.critic_lr, config)

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )

    # Initialise observation with obs of all agents.
    init_obs = env.observation_spec.generate_value()
    init_obs = tree.map(
        lambda x: jnp.repeat(x[jnp.newaxis, ...], config.arch.num_envs, axis=0),
        init_obs,
    )
    init_obs = tree.map(lambda x: x[jnp.newaxis, ...], init_obs)
    init_done = jnp.zeros((1, config.arch.num_envs, num_agents), dtype=bool)
    init_obs_done = (init_obs, init_done)

    # Initialise hidden state.
    init_policy_hstate = ScannedRNN.initialize_carry(
        (config.arch.num_envs, num_agents), config.network.hidden_state_dim
    )
    init_critic_hstate = ScannedRNN.initialize_carry(
        (config.arch.num_envs, num_agents), config.network.hidden_state_dim
    )

    # initialise params and optimiser state.
    actor_params = actor_network.init(actor_net_key, init_policy_hstate, init_obs_done, None)
    actor_opt_state = actor_optim.init(actor_params)
    critic_params = critic_network.init(critic_net_key, init_critic_hstate, init_obs_done, None)
    critic_opt_state = critic_optim.init(critic_params)

    # Get network apply functions and optimiser updates.
    apply_fns = (actor_network.apply, critic_network.apply)
    update_fns = (actor_optim.update, critic_optim.update)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fns, config)
    # Map learner over specified devices
    devices = jax.local_devices()[:n_devices]
    learn = jax.pmap(learn, axis_name="device", devices=devices)

    # Pack params and initial states.
    params = Params(actor_params, critic_params)
    hstates = HiddenStates(init_policy_hstate, init_critic_hstate)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, restored_hstates = loaded_checkpoint.restore_params(
            input_params=params, restore_hstates=True, THiddenState=HiddenStates
        )
        # Update the params and hstates
        params = restored_params
        hstates = restored_hstates if restored_hstates else hstates

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.system.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_keys),
    )
    reshape_states = lambda x: x.reshape(
        (n_devices, config.system.update_batch_size, config.arch.num_envs, *x.shape[1:])
    )
    # (devices, update batch size, num_envs, ...)
    env_states = tree.map(reshape_states, env_states)
    timesteps = tree.map(reshape_states, timesteps)

    # Define params to be replicated across devices and batches.
    dones = jnp.zeros(
        (config.arch.num_envs, num_agents),
        dtype=bool,
    )

    # Initialize trajectory state
    # Get observation dims from first timestep
    sample_obs = timesteps.observation
    if hasattr(sample_obs, "agents_view"):
        obs_dims = sample_obs.agents_view.shape[1:]
    else:
        obs_dims = sample_obs.shape[1:]

    # Infer action shape and dtype from action_spec
    sample_action = env.action_spec.generate_value()  # shape: (num_agents, *act_dim)
    action_shape = sample_action.shape[1:]
    action_dtype = jnp.array(sample_action).dtype

    # Initialize trajectory buffers with zeros
    # obs_history: [num_envs, num_agents, traj_len, *obs_dims]
    init_obs_history = jnp.zeros((config.arch.num_envs, num_agents, traj_len, *obs_dims))
    # action_history: [num_envs, num_agents, traj_len, *action_shape]
    init_action_history = jnp.zeros(
        (config.arch.num_envs, num_agents, traj_len, *action_shape),
        dtype=action_dtype,
    )

    trajectory_state = TrajectoryState(
        obs_history=init_obs_history,
        action_history=init_action_history,
        buffer_idx=0,
        buffer_full=False,
    )

    key, step_keys = jax.random.split(key)
    opt_states = OptStates(actor_opt_state, critic_opt_state)
    replicate_learner = (params, opt_states, hstates, step_keys, dones, trajectory_state)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size, *jnp.shape(x)))
    replicate_learner = tree.map(broadcast, replicate_learner)

    # Duplicate learner across specified devices
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=devices)

    # Initialise learner state.
    params, opt_states, hstates, step_keys, dones, trajectory_state = replicate_learner
    init_learner_state = RNNLearnerState(
        params=params,
        opt_states=opt_states,
        key=step_keys,
        env_state=env_states,
        timestep=timesteps,
        dones=dones,
        hstates=hstates,
        trajectory_state=trajectory_state,
    )
    return learn, actor_network, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    _config.logger.system_name = "rec_mappo"
    config = copy.deepcopy(_config)

    # Vault configuration
    save_vault = getattr(config.system, "save_vault", False)
    vault_name = getattr(config.system, "vault_name", "rec_mappo")
    vault_uid = getattr(config.system, "vault_uid", None)
    vault_save_interval = getattr(config.system, "vault_save_interval", 5)

    # Determine number of devices: default to 1 unless override via config
    n_devices = min(config.arch.get("n_devices", 1), len(jax.devices()))

    # Set recurrent chunk size.
    if config.system.recurrent_chunk_size is None:
        config.system.recurrent_chunk_size = config.system.rollout_length
    else:
        assert config.system.rollout_length % config.system.recurrent_chunk_size == 0, (
            "Rollout length must be divisible by recurrent chunk size."
        )

        assert config.arch.num_envs % config.system.num_minibatches == 0, (
            "Number of envs must be divisibile by number of minibatches."
        )

    # Create the enviroments for train and eval.
    env, eval_env = environments.make(config=config, add_global_state=True)

    # PRNG keys.
    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=4
    )

    # Setup learner.
    learn, actor_network, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key), config
    )

    # Setup evaluator.
    # One key per device for evaluation.
    eval_keys = jax.random.split(key_e, n_devices)
    eval_act_fn = make_rec_eval_act_fn_with_traj(actor_network.apply, config)
    evaluator = get_eval_fn_with_traj(eval_env, eval_act_fn, config, absolute_metric=False)

    # Calculate total timesteps.
    config = check_total_timesteps(config)
    assert config.system.num_updates > config.arch.num_evaluation, (
        "Number of updates per evaluation must be less than total number of updates."
    )

    # Calculate number of updates per evaluation.
    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        n_devices
        * config.system.num_updates_per_eval
        * config.system.rollout_length
        * config.system.update_batch_size
        * config.arch.num_envs
    )
    # Logger setup
    logger = MavaLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.logger.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )

    # Set up vault for experience storage if enabled
    buffer_state = None
    vault = None
    buffer_add = None
    if save_vault:
        # Get observation dims from environment spec
        sample_obs = env.observation_spec.generate_value()
        obs_shape = (
            sample_obs.agents_view.shape[1:]
            if hasattr(sample_obs, "agents_view")
            else sample_obs.shape[1:]
        )

        # Get global state shape
        global_state_shape = (
            sample_obs.global_state.shape
            if hasattr(sample_obs, "global_state")
            else (1,)  # fallback if no global state
        )

        # Get action spec to determine correct action shape and dtype
        sample_action = env.action_spec.generate_value()
        action_shape = sample_action.shape
        action_dtype = jnp.array(sample_action).dtype

        # Set up dummy transition based on actual environment specs
        dummy_flashbax_transition = {
            "done": jnp.zeros((config.system.num_agents,), dtype=bool),
            "action": jnp.zeros(action_shape, dtype=action_dtype),
            "reward": jnp.zeros((config.system.num_agents,), dtype=jnp.float32),
            "observation": jnp.zeros(
                (config.system.num_agents, *obs_shape),
                dtype=jnp.float32,
            ),
            "global_state": jnp.zeros(global_state_shape, dtype=jnp.float32),
            "legal_action_mask": jnp.zeros(
                (config.system.num_agents, env.action_dim),
                dtype=bool,
            ),
        }

        buffer = fbx.make_flat_buffer(
            max_length=int(5e5),  # Max number of transitions to store
            min_length=int(1),
            sample_batch_size=1,
            add_sequences=True,
            add_batch_size=(
                n_devices
                * config.system.num_updates_per_eval
                * config.system.update_batch_size
                * config.arch.num_envs
            ),
        )
        buffer_state = buffer.init(dummy_flashbax_transition)
        buffer_add = jax.jit(buffer.add, donate_argnums=(0))

        # Create vault
        vault = Vault(
            vault_name=vault_name,
            experience_structure=buffer_state.experience,
            vault_uid=vault_uid,
            metadata=OmegaConf.to_container(config, resolve=True),
        )

        # Shape legend:
        # D: Number of devices
        # NU: Number of updates per evaluation
        # UB: Update batch size
        # T: Time steps per rollout
        # NE: Number of environments

        @jax.jit
        def _reshape_experience(experience: Dict[str, chex.Array]) -> Dict[str, chex.Array]:
            """Reshape experience to match buffer."""
            # Swap the T and NE axes (D, NU, UB, T, NE, ...) -> (D, NU, UB, NE, T, ...)
            experience = tree.map(lambda x: x.swapaxes(3, 4), experience)
            # Merge 4 leading dimensions into 1. (D, NU, UB, NE, T ...) -> (D * NU * UB * NE, T, ...)
            experience = tree.map(lambda x: x.reshape(-1, *x.shape[4:]), experience)
            return experience

    # Create an initial hidden state and trajectory history used for resetting memory for evaluation
    eval_batch_size = get_num_eval_envs(config, absolute_metric=False)
    eval_hs = ScannedRNN.initialize_carry(
        (n_devices, eval_batch_size, config.system.num_agents),
        config.network.hidden_state_dim,
    )

    # Initialize trajectory history for evaluation - make sure structure matches what eval_act_fn expects
    traj_len = getattr(config.system, "traj_len", 10)
    sample_obs = env.observation_spec.generate_value()
    obs_shape = (
        sample_obs.agents_view.shape[1:]
        if hasattr(sample_obs, "agents_view")
        else sample_obs.shape[1:]
    )
    sample_action = env.action_spec.generate_value()
    action_shape = sample_action.shape[1:]
    action_dtype = jnp.array(sample_action).dtype

    # Initialize empty trajectory history for evaluation
    eval_traj_history = {
        "obs_history": jnp.zeros(
            (n_devices, eval_batch_size, config.system.num_agents, traj_len, *obs_shape)
        ),
        "action_history": jnp.zeros(
            (n_devices, eval_batch_size, config.system.num_agents, traj_len, *action_shape),
            dtype=action_dtype,
        ),
    }
    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
    best_params = None
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()
        learner_output, experience_to_store = learn(learner_state)

        # Record data into the vault if enabled
        if save_vault:
            # Pack transition
            flashbax_transition = _reshape_experience(
                {
                    # (D, NU, UB, T, NE, ...)
                    "done": experience_to_store.done,
                    "action": experience_to_store.action,
                    "reward": experience_to_store.reward,
                    "observation": experience_to_store.obs.agents_view,
                    "global_state": experience_to_store.obs.global_state,
                    "legal_action_mask": experience_to_store.obs.action_mask,
                }
            )
            # Add to fbx buffer
            buffer_state = buffer_add(buffer_state, flashbax_transition)

            # Save buffer into vault
            if eval_step % vault_save_interval == 0:
                write_length = vault.write(buffer_state)
                print(f"(Wrote {write_length}) Vault index = {vault.vault_index}")

        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Separately log timesteps, actoring metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        trained_params = unreplicate_batch_dim(learner_state.params.actor_params)
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)
        # Evaluate.
        eval_metrics = evaluator(
            trained_params,
            eval_keys,
            {"hidden_state": eval_hs, "trajectory_history": eval_traj_history},
        )
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)
        episode_return = jnp.mean(eval_metrics["episode_return"])

        if save_checkpoint:
            # Save checkpoint of learner state
            checkpointer.save(
                timestep=steps_per_rollout * (eval_step + 1),
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Final write to vault for any remaining data
    if save_vault and vault is not None:
        vault.write(buffer_state)

    # Record the performance for the final evaluation run.
    eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))

    # Measure absolute metric.
    if config.arch.absolute_metric:
        eval_batch_size = get_num_eval_envs(config, absolute_metric=True)
        eval_hs = ScannedRNN.initialize_carry(
            (n_devices, eval_batch_size, config.system.num_agents),
            config.network.hidden_state_dim,
        )

        # Initialize trajectory history for absolute metric evaluation
        eval_traj_history_abs = {
            "obs_history": jnp.zeros(
                (n_devices, eval_batch_size, config.system.num_agents, traj_len, *obs_shape)
            ),
            "action_history": jnp.zeros(
                (n_devices, eval_batch_size, config.system.num_agents, traj_len, *action_shape),
                dtype=action_dtype,
            ),
        }

        abs_metric_evaluator = get_eval_fn_with_traj(
            eval_env, eval_act_fn, config, absolute_metric=True
        )
        eval_keys = jax.random.split(key, n_devices)

        eval_metrics = abs_metric_evaluator(
            best_params,
            eval_keys,
            {"hidden_state": eval_hs, "trajectory_history": eval_traj_history_abs},
        )

        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()

    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    config_name="rec_mappo.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}Recurrent MAPPO experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
