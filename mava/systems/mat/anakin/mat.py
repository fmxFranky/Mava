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
from functools import partial
from typing import Any, Dict, Tuple

import chex
import flashbax as fbx
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
from colorama import Fore, Style
from flashbax.buffers.trajectory_buffer import TrajectoryBuffer, TrajectoryBufferSample
from flax.core.frozen_dict import FrozenDict
from jax import tree
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from mava.evaluator import ActorState, get_eval_fn
from mava.networks.mat_network import MultiAgentTransformer
from mava.systems.mat.types import (
    ActorApply,
    BufferState,
    ByolApply,
    LearnerApply,
    LearnerState,
    OffPolicyTransition,
    PPOTransition,
)
from mava.types import (
    ExperimentOutput,
    LearnerFn,
    MarlEnv,
    Metrics,
    TimeStep,
)
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.jax_utils import merge_leading_dims, unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.network_utils import get_action_head
from mava.utils.training import make_learning_rate
from mava.wrappers.episode_metrics import get_final_step_metrics


def get_learner_fn(
    env: MarlEnv,
    apply_fns: Tuple[ActorApply, LearnerApply, ByolApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    rb: TrajectoryBuffer,
    config: DictConfig,
) -> LearnerFn[LearnerState]:
    """Get the learner function."""

    # Get apply and update functions for actor and critic networks.
    actor_action_select_fn, actor_apply_fn, byol_apply_fn = apply_fns
    mat_update_fn, byol_update_fn = update_fns  # Unpack two update functions

    def _update_step(learner_state: LearnerState, _: Any) -> Tuple[LearnerState, Tuple]:
        """A single update of the network.

        This function steps the environment and records the trajectory batch for
        training. It then calculates advantages and targets based on the recorded
        trajectory and updates the actor and critic networks based on the calculated
        losses.

        Args:
            learner_state (NamedTuple):
                - params: The current model parameters.
                - opt_state: The optimizer state
                - key: The random number generator state.
                - env_state: The environment state.
                - last_timestep: The last timestep in the current trajectory.
                - buffer_state: The replay buffer state.
            _ (Any): The current metrics info.
        """

        def _env_step(
            learner_state: LearnerState, _: Any
        ) -> Tuple[LearnerState, Tuple[PPOTransition, Metrics]]:
            """Step the environment."""
            params, opt_state, key, env_state, last_timestep, buffer_state = learner_state

            # Select action
            key, policy_key = jax.random.split(key)
            action, log_prob, value = actor_action_select_fn(  # type: ignore
                params,
                last_timestep.observation,
                policy_key,
            )
            # Step environment
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            done = timestep.last().repeat(env.num_agents).reshape(config.arch.num_envs, -1)

            # Create and store off-policy transition
            off_policy_transition = OffPolicyTransition(
                action=action,
                obs=last_timestep.observation,
                reward=timestep.reward,
                terminal=done,
                next_obs=timestep.extras["real_next_obs"],
            )
            # Add dummy time dim for buffer
            off_policy_transition = tree.map(
                lambda x: x[:, jnp.newaxis, ...], off_policy_transition
            )
            next_buffer_state = rb.add(buffer_state, off_policy_transition)

            transition = PPOTransition(
                done,
                action,
                value,
                timestep.reward,
                log_prob,
                last_timestep.observation,
                timestep.extras["real_next_obs"],
            )
            learner_state = LearnerState(
                params, opt_state, key, env_state, timestep, next_buffer_state
            )
            return learner_state, (transition, timestep.extras["episode_metrics"])

        # Step environment for rollout length
        learner_state, (traj_batch, episode_metrics) = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        # Calculate advantage
        params, opt_state, key, env_state, last_timestep, buffer_state = learner_state

        key, last_val_key = jax.random.split(key)
        _, _, last_val = actor_action_select_fn(  # type: ignore
            params,
            last_timestep.observation,
            last_val_key,
        )

        def _calculate_gae(
            traj_batch: PPOTransition, last_val: chex.Array
        ) -> Tuple[chex.Array, chex.Array]:
            """Calculate the GAE."""

            def _get_advantages(gae_and_next_value: Tuple, transition: PPOTransition) -> Tuple:
                """Calculate the GAE for a single transition."""
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                gamma = config.system.gamma
                delta = reward + gamma * next_value * (1 - done) - value
                gae = delta + gamma * config.system.gae_lambda * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val)

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""
                params, opt_states, key = train_state
                mat_opt_state, byol_opt_state = opt_states
                traj_batch, advantages, targets = batch_info

                def _mat_loss_fn(
                    params: FrozenDict,
                    traj_batch: PPOTransition,
                    gae: chex.Array,
                    value_targets: chex.Array,
                    entropy_key: chex.PRNGKey,
                ) -> Tuple:
                    """Calculate the MAT loss (policy + value)."""
                    # Rerun network
                    log_prob, value, entropy = actor_apply_fn(
                        params,
                        traj_batch.obs,
                        traj_batch.action,
                        entropy_key,
                    )

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
                    entropy = entropy.mean()

                    # Clipped MSE loss
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                        -config.system.clip_eps, config.system.clip_eps
                    )
                    value_losses = jnp.square(value - value_targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - value_targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                    total_loss = (
                        actor_loss
                        - config.system.ent_coef * entropy
                        + config.system.vf_coef * value_loss
                    )
                    return total_loss, (actor_loss, entropy, value_loss)

                def _byol_loss_fn(
                    params: FrozenDict,
                    buffer_state: BufferState,
                    key: chex.PRNGKey,
                ) -> Tuple:
                    """Calculate the BYOL loss using sampled sequences from replay buffer."""
                    if not config.system.use_byol:
                        return 0.0, {}

                    # Sample sequence from replay buffer
                    sampled_data = rb.sample(buffer_state, key).experience

                    # Prepare sequence inputs for get_forward_representations
                    seq_obs = sampled_data.obs  # (B, T, N, ...)
                    seq_action = sampled_data.action  # (B, T-1, N, A)

                    # Get representations using forward prediction
                    obs_rep, pred_obs_reps, target_obs_reps = byol_apply_fn(
                        params,
                        seq_obs,
                        seq_action,
                    )

                    # Calculate MSE loss between predicted and target representations
                    byol_loss = jnp.mean(
                        jnp.sum(jnp.square(pred_obs_reps - target_obs_reps), axis=-1)
                    )

                    return byol_loss, {"byol_loss": byol_loss}

                # Calculate MAT loss
                key, mat_key = jax.random.split(key)
                mat_grad_fn = jax.value_and_grad(_mat_loss_fn, has_aux=True)
                (mat_loss, mat_loss_info), mat_grads = mat_grad_fn(
                    params,
                    traj_batch,
                    advantages,
                    targets,
                    mat_key,
                )

                # Calculate BYOL loss
                key, byol_key = jax.random.split(key)
                byol_grad_fn = jax.value_and_grad(_byol_loss_fn, has_aux=True)
                (byol_loss, byol_loss_info), byol_grads = byol_grad_fn(
                    params,
                    buffer_state,
                    byol_key,
                )

                # Mean over devices and batch
                mat_grads, mat_loss_info = jax.lax.pmean(
                    (mat_grads, mat_loss_info), axis_name="device"
                )
                mat_grads, mat_loss_info = jax.lax.pmean(
                    (mat_grads, mat_loss_info), axis_name="batch"
                )

                byol_grads, byol_loss_info = jax.lax.pmean(
                    (byol_grads, byol_loss_info), axis_name="device"
                )
                byol_grads, byol_loss_info = jax.lax.pmean(
                    (byol_grads, byol_loss_info), axis_name="batch"
                )

                # Update params with MAT optimizer
                mat_updates, new_mat_opt_state = mat_update_fn(mat_grads, mat_opt_state)
                new_params = optax.apply_updates(params, mat_updates)

                # Update params with BYOL optimizer if enabled
                if config.system.use_byol:
                    byol_updates, new_byol_opt_state = byol_update_fn(byol_grads, byol_opt_state)
                    new_params = optax.apply_updates(new_params, byol_updates)

                    # Update target networks using EMA
                    if isinstance(new_params, FrozenDict):
                        new_params = new_params.unfreeze()
                        is_frozen = True
                    else:
                        new_params = dict(new_params)  # Create a new dict copy
                        is_frozen = False

                    if "encoder" in new_params and "target_encoder" in new_params:
                        new_params["target_encoder"] = optax.incremental_update(
                            new_params["encoder"],
                            new_params["target_encoder"],
                            config.system.target_network_ema,
                        )
                    if "projector" in new_params and "target_projector" in new_params:
                        new_params["target_projector"] = optax.incremental_update(
                            new_params["projector"],
                            new_params["target_projector"],
                            config.system.target_network_ema,
                        )

                    if is_frozen:
                        new_params = FrozenDict(new_params)
                else:
                    new_byol_opt_state = byol_opt_state

                # Combine loss info
                actor_loss, entropy, value_loss = mat_loss_info
                loss_info = {
                    "total_loss": mat_loss + config.system.byol_coef * byol_loss,
                    "value_loss": value_loss,
                    "actor_loss": actor_loss,
                    "entropy": entropy,
                }
                if config.system.use_byol:
                    loss_info.update(byol_loss_info)

                return (new_params, (new_mat_opt_state, new_byol_opt_state), key), loss_info

            params, opt_state, traj_batch, advantages, targets, key = update_state
            key, batch_shuffle_key, agent_shuffle_key, entropy_key = jax.random.split(key, 4)

            # Shuffle minibatches
            batch_size = config.system.rollout_length * config.arch.num_envs
            permutation = jax.random.permutation(batch_shuffle_key, batch_size)

            batch = (traj_batch, advantages, targets)
            batch = tree.map(lambda x: merge_leading_dims(x, 2), batch)
            shuffled_batch = tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)

            # Shuffle along the agent dimension as well
            permutation = jax.random.permutation(agent_shuffle_key, config.system.num_agents)
            shuffled_batch = tree.map(lambda x: jnp.take(x, permutation, axis=1), shuffled_batch)

            minibatches = tree.map(
                lambda x: jnp.reshape(x, (config.system.num_minibatches, -1, *x.shape[1:])),
                shuffled_batch,
            )

            # Update minibatches
            (params, opt_state, entropy_key), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_state, entropy_key), minibatches
            )

            update_state = params, opt_state, traj_batch, advantages, targets, key
            return update_state, loss_info

        update_state = params, opt_state, traj_batch, advantages, targets, key

        # Update epochs
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.ppo_epochs
        )

        params, opt_state, traj_batch, advantages, targets, key = update_state
        learner_state = LearnerState(params, opt_state, key, env_state, last_timestep, buffer_state)

        return learner_state, (episode_metrics, loss_info)

    def learner_fn(learner_state: LearnerState) -> ExperimentOutput[LearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
            learner_state (NamedTuple):
                - params: The initial model parameters.
                - opt_state: The initial optimiser state.
                - key: The random number generator state.
                - env_state: The environment state.
                - timesteps: The initial timestep in the initial trajectory.
        """

        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, config.system.num_updates_per_eval
        )
        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def learner_setup(
    env: MarlEnv, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[LearnerState], Any, LearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of agents.
    config.system.num_agents = env.num_agents

    # PRNG keys.
    key, actor_net_key = keys

    # Initialise observation: Obs for all agents.
    init_obs = env.observation_spec.generate_value()
    init_x = tree.map(lambda x: x[None, ...], init_obs)

    _, action_space_type = get_action_head(env.action_spec)

    if action_space_type == "discrete":
        init_action = jnp.zeros((1, config.system.num_agents), dtype=jnp.int32)
    elif action_space_type == "continuous":
        init_action = jnp.zeros((1, config.system.num_agents, env.action_dim), dtype=jnp.float32)
    else:
        raise ValueError("Invalid action space type")

    # Define network and optimiser.
    actor_network = MultiAgentTransformer(
        action_dim=env.action_dim,
        n_agent=config.system.num_agents,
        net_config=config.network,
        action_space_type=action_space_type,
        use_byol=config.system.use_byol,
    )

    # MAT optimizer (for encoder and decoder)
    mat_lr = make_learning_rate(config.system.actor_lr, config)
    mat_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(mat_lr, eps=1e-5),
    )

    # BYOL optimizer (for encoder, dynamic_model, projector, predictor)
    byol_lr = make_learning_rate(config.system.fdm_lr, config)  # Use same learning rate
    byol_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(byol_lr, eps=1e-5),
    )

    # Initialise actor params and optimisers state.
    params = actor_network.init(
        actor_net_key, init_x, init_x, init_action, jax.random.PRNGKey(0), method="init_params"
    )
    print(
        actor_network.tabulate(
            actor_net_key,
            init_x,
            init_x,
            init_action,
            jax.random.PRNGKey(0),
            method="init_params",
            depth=1,
        )
    )

    if config.system.use_byol:
        assert "target_encoder" in params["params"] and "target_projector" in params["params"], (
            "BYOL requires target_encoder and target_projector in params"
        )
        params["params"]["target_encoder"] = copy.deepcopy(params["params"]["encoder"])
        params["params"]["target_projector"] = copy.deepcopy(params["params"]["projector"])

    # Initialize states for both optimizers
    mat_opt_state = mat_optim.init(params)
    byol_opt_state = byol_optim.init(params)

    # Initialize replay buffer
    rb = fbx.make_trajectory_buffer(
        sample_sequence_length=config.system.fdm_seq_len,
        period=config.system.fdm_period,  # sample any unique trajectory
        add_batch_size=config.arch.num_envs,
        sample_batch_size=config.system.fdm_batch_size,
        max_length_time_axis=config.system.fdm_max_length,
        min_length_time_axis=config.system.fdm_min_length,
    )

    # Initialize buffer state with dummy transition
    init_transition = OffPolicyTransition(
        action=init_action[0],
        obs=init_obs,
        reward=jnp.zeros((env.num_agents,), dtype=float),
        terminal=jnp.zeros((env.num_agents,), dtype=bool),
        next_obs=init_obs,
    )
    buffer_state = rb.init(init_transition)

    # Pack apply and update functions.
    apply_fns = (
        partial(actor_network.apply, method="get_actions"),  # action selection
        actor_network.apply,  # policy and value computation
        partial(actor_network.apply, method="get_forward_representations"),  # BYOL computation
    )

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, (mat_optim.update, byol_optim.update), rb, config)
    learn = jax.pmap(learn, axis_name="device")

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.system.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_keys),
    )
    reshape_states = lambda x: x.reshape(
        (n_devices, config.system.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    # (devices, update batch size, num_envs, ...)
    env_states = tree.map(reshape_states, env_states)
    timesteps = tree.map(reshape_states, timesteps)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        # Update the params
        params = restored_params

    # Define params to be replicated across devices and batches.
    key, step_keys = jax.random.split(key)
    replicate_learner = (params, (mat_opt_state, byol_opt_state), step_keys)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size, *x.shape))
    replicate_learner = tree.map(broadcast, replicate_learner)

    # Duplicate buffer state for update_batch_size
    buffer_state = tree.map(broadcast, buffer_state)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())
    buffer_state = flax.jax_utils.replicate(buffer_state, devices=jax.devices())

    # Initialise learner state.
    params, opt_states, step_keys = replicate_learner
    init_learner_state = LearnerState(
        params, opt_states, step_keys, env_states, timesteps, buffer_state
    )

    return learn, actor_network, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    _config.logger.system_name = "mat"
    config = copy.deepcopy(_config)

    n_devices = len(jax.devices())

    # Create the enviroments for train and eval.
    env, eval_env = environments.make(config)

    # PRNG keys.
    key, key_e, actor_net_key = jax.random.split(jax.random.PRNGKey(config.system.seed), num=3)

    # Setup learner.
    learn, actor_network, learner_state = learner_setup(env, (key, actor_net_key), config)

    eval_keys = jax.random.split(key_e, n_devices)

    def eval_act_fn(
        params: FrozenDict,
        timestep: TimeStep,
        key: chex.PRNGKey,
        actor_state: ActorState,
    ) -> Tuple[chex.Array, ActorState]:
        """The acting function that get's passed to the evaluator.
        Given that the MAT network has a `get_actions` method we define this eval_act_fn
        accordingly.
        """

        del actor_state  # Unused since the system doesn't have memory over time.
        output_action, _, _ = actor_network.apply(  # type: ignore
            params,
            timestep.observation,
            key,
            method="get_actions",
        )
        return output_action, {}

    evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)

    # Calculate total timesteps.
    config = check_total_timesteps(config)
    assert config.system.num_updates > config.arch.num_evaluation, (
        "Number of updates per evaluation must be less than total number of updates."
    )

    assert config.arch.num_envs % config.system.num_minibatches == 0, (
        "Number of envs must be divisibile by number of minibatches."
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
    cfg: Dict = OmegaConf.to_container(config, resolve=True)
    cfg["arch"]["devices"] = jax.devices()
    pprint(cfg)

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.logger.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )

    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
    best_params = None
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()

        learner_output = learn(learner_state)
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

        trained_params = unreplicate_batch_dim(learner_state.params)
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        # Evaluate.
        eval_metrics = evaluator(trained_params, eval_keys, {})
        jax.block_until_ready(eval_metrics)
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

    # Record the performance for the final evaluation run.
    eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))

    # Measure absolute metric.
    if config.arch.absolute_metric:
        abs_metric_evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)
        eval_keys = jax.random.split(key, n_devices)

        eval_metrics = abs_metric_evaluator(best_params, eval_keys, {})
        jax.block_until_ready(eval_metrics)

        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()

    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    config_name="mat.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    eval_performance = run_experiment(cfg)
    jax.block_until_ready(eval_performance)
    print(f"{Fore.CYAN}{Style.BRIGHT}MAT experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
