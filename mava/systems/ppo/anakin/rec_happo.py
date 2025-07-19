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
from gymnasium.spaces import Discrete, MultiDiscrete
from jax import tree
from jumanji.specs import DiscreteArray, MultiDiscreteArray  # Local import to avoid circular deps
from omegaconf import DictConfig, OmegaConf
from optax._src.base import OptState

from mava.evaluator import (
    get_eval_fn_with_traj,
    get_num_eval_envs,
    make_rec_eval_act_fn_with_traj,
)
from mava.networks import RecurrentActor as Actor
from mava.networks import RecurrentValueNet as Critic
from mava.networks.base import ScannedRNN, ScannedRNNPerAgent
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


def construct_joint_trajectory_for_agents(
    trajectory_state: TrajectoryState, config: DictConfig
) -> JointTrajectory:
    """Construct joint trajectory for all agents from trajectory state.

    Args:
        trajectory_state: Current trajectory state with obs_history [B, N, traj_len, *obs_dim]
                         and action_history [B, N, traj_len, *action_dim]
        config: Configuration containing traj_len

    Returns:
        JointTrajectory with shape [N, 1, B, N, traj_len, *dims] for vmapped agent processing
        Each agent sees the same complete multi-agent trajectory data
    """
    traj_len = config.system.get("traj_len", 10)
    B, N = trajectory_state.obs_history.shape[:2]

    # trajectory_state.obs_history: [B, N, traj_len, *obs_dim]
    # trajectory_state.action_history: [B, N, traj_len, *action_dim]

    # For HAPPO, each agent should see the same complete joint trajectory
    # Add batch dimension and repeat for each agent
    # [B, N, traj_len, *obs_dim] -> [1, B, N, traj_len, *obs_dim] -> [N, 1, B, N, traj_len, *obs_dim]
    joint_obs = trajectory_state.obs_history[jnp.newaxis, ...]  # [1, B, N, traj_len, *obs_dim]
    joint_obs = jnp.repeat(joint_obs, N, axis=0)  # [N, 1, B, N, traj_len, *obs_dim]

    joint_actions = trajectory_state.action_history[
        jnp.newaxis, ...
    ]  # [1, B, N, traj_len, *action_dim]
    joint_actions = jnp.repeat(joint_actions, N, axis=0)  # [N, 1, B, N, traj_len, *action_dim]

    return JointTrajectory(
        observations=joint_obs,  # [N, 1, B, N, traj_len, *obs_dim]
        actions=joint_actions,  # [N, 1, B, N, traj_len, *action_dim]
    )


def construct_joint_trajectory_window_slide(
    traj_batch: RNNPPOTransition, config: DictConfig
) -> JointTrajectory:
    """Construct joint trajectory using window slide for minibatch training.

    For each timestep t in the rollout, constructs trajectory history of length traj_len
    ending at timestep t: [t-traj_len+1, ..., t]

    Args:
        traj_batch: Trajectory batch with shape [T, B, N, *dims]
        config: Configuration containing traj_len

    Returns:
        JointTrajectory with shape [T, B, N, traj_len, *dims]
    """
    traj_len = config.system.get("traj_len", 10)
    T, B, N = traj_batch.obs.agents_view.shape[:3]
    obs_shape = traj_batch.obs.agents_view.shape[3:]

    # Handle action dimensions - expand if needed
    action_slice = traj_batch.action
    if action_slice.ndim == 2:  # [T, B] -> expand to [T, B, N]
        action_slice = jnp.broadcast_to(action_slice[:, :, jnp.newaxis], (T, B, N))

    # Efficiently create window slides using JAX operations
    # Create indices for each window: for timestep t, indices are [max(0, t-traj_len+1), ..., t]
    timesteps = jnp.arange(T)  # [0, 1, 2, ..., T-1]

    # For each timestep t, create window indices [t-traj_len+1, ..., t]
    # Shape: [T, traj_len]
    window_offsets = jnp.arange(traj_len)[jnp.newaxis, :]  # [1, traj_len]: [0, 1, ..., traj_len-1]
    window_indices = timesteps[:, jnp.newaxis] - traj_len + 1 + window_offsets  # [T, traj_len]

    # Clip indices to valid range [0, T-1] and handle padding
    window_indices = jnp.clip(window_indices, 0, T - 1)

    # Create mask for valid indices (those that are >= current_timestep - traj_len + 1)
    valid_mask = window_indices >= (timesteps[:, jnp.newaxis] - traj_len + 1)

    # ------------------------------------------------------------------
    # Zero-out cross-episode history: we only want transitions from the
    # current episode.  Compute for every timestep how many steps have
    # elapsed since the last done (per env, per agent). If window offset
    # exceeds this number, we mark it invalid.
    # ------------------------------------------------------------------
    done_flags = traj_batch.done  # shape [T, B, N]

    def _scan_fn(carry, d):
        # carry is steps since last done
        new_carry = jnp.where(d, 1, carry + 1)
        return new_carry, new_carry

    init_steps = jnp.zeros((B, N), dtype=jnp.int32)
    episode_steps, _ = jax.lax.scan(_scan_fn, init_steps, done_flags)
    # episode_steps[t] = 1 at first step after reset, then 2,3,...

    # Expand to [T, traj_len, B, N] for comparison with window offsets
    ep_steps_exp = episode_steps[:, jnp.newaxis, :, :]  # [T,1,B,N]
    k_offsets = window_offsets[:, :, jnp.newaxis, jnp.newaxis]  # [T, traj_len,1,1]
    cross_ep_mask = k_offsets < ep_steps_exp  # True where within episode

    # Combine masks
    valid_mask = valid_mask & cross_ep_mask

    # Extract windows using advanced indexing
    obs_windows = traj_batch.obs.agents_view[window_indices]  # [T, traj_len, B, N, *obs_dim]
    action_windows = action_slice[window_indices]  # [T, traj_len, B, N]

    # Apply padding mask - set invalid positions to zero
    obs_pad_value = jnp.zeros((B, N) + obs_shape)
    action_pad_value = jnp.zeros((B, N), dtype=action_slice.dtype)

    # Create mask shapes that are compatible with observations and actions
    obs_mask_shape = (valid_mask.shape[0], valid_mask.shape[1]) + (1, 1) + (1,) * len(obs_shape)
    action_mask_shape = (valid_mask.shape[0], valid_mask.shape[1]) + (1, 1)

    obs_mask = valid_mask.reshape(obs_mask_shape)
    action_mask = valid_mask.reshape(action_mask_shape)

    obs_windows = jnp.where(obs_mask, obs_windows, obs_pad_value)
    action_windows = jnp.where(action_mask, action_windows, action_pad_value)

    # Transpose to [T, B, N, traj_len, *dims] format
    # obs_windows: [T, traj_len, B, N, *obs_dim] -> [T, B, N, traj_len, *obs_dim]
    perm_obs = (0, 2, 3, 1) + tuple(range(4, obs_windows.ndim))
    obs_windows = obs_windows.transpose(perm_obs)
    # action_windows: [T, traj_len, B, N] -> [T, B, N, traj_len]
    action_windows = action_windows.transpose(0, 2, 3, 1)

    return JointTrajectory(
        observations=obs_windows,  # [T, B, N, traj_len, *obs_dim]
        actions=action_windows,  # [T, B, N, traj_len]
    )


def construct_dummy_joint_trajectory(
    obs_shape: Tuple,
    action_shape: Tuple,
    batch_size: int,
    num_agents: int,
    traj_len: int,
    action_dtype: jnp.dtype = jnp.int32,
) -> JointTrajectory:
    """Construct dummy joint trajectory for initialization and evaluation.

    Args:
        obs_shape: Shape of observation excluding batch and agent dimensions
        action_shape: Shape of action excluding batch and agent dimensions
        batch_size: Batch size
        num_agents: Number of agents
        traj_len: Trajectory length
        action_dtype: Data type for actions

    Returns:
        Dummy JointTrajectory with appropriate shapes
    """
    dummy_obs = jnp.zeros((batch_size, num_agents, traj_len, *obs_shape))
    dummy_actions = jnp.zeros((batch_size, num_agents, traj_len, *action_shape), dtype=action_dtype)

    return JointTrajectory(
        observations=dummy_obs,
        actions=dummy_actions,
    )


def construct_training_joint_trajectory(
    traj_batch: RNNPPOTransition, env_num_agents: int, config: DictConfig
) -> JointTrajectory:
    """Construct joint trajectory for training using each timestep's own observation.

    This creates a more realistic joint trajectory where each timestep uses its actual
    observation repeated across the trajectory length, rather than using dummy data.
    """
    obs_shape_full = traj_batch.obs.agents_view.shape
    T = obs_shape_full[0]  # recurrent_chunk_size
    B = obs_shape_full[1]  # minibatch_size

    # Check if we have agent dimension
    if len(obs_shape_full) >= 3 and obs_shape_full[2] == env_num_agents:
        # Case: [T, B, N, *obs_dim] - we have per-agent observations
        N = obs_shape_full[2]
        obs_shape = obs_shape_full[3:]
        obs_data = traj_batch.obs.agents_view  # [T, B, N, *obs_dim]

    else:
        # Case: [T, B, *obs_dim] - single agent case, expand to multi-agent
        N = env_num_agents
        obs_shape = obs_shape_full[2:]
        obs_data = jnp.broadcast_to(
            traj_batch.obs.agents_view[:, :, jnp.newaxis, :], (T, B, N, *obs_shape)
        )

    traj_len = config.system.get("traj_len", 10)

    # For each timestep, repeat its observation across traj_len
    # IMPORTANT: Keep agent-specific observations distinct
    joint_obs = jnp.broadcast_to(
        obs_data[:, :, :, jnp.newaxis, ...],  # [T, B, N, 1, *obs_dim]
        (T, B, N, traj_len, *obs_shape),
    )

    # Handle actions similarly
    action_shape_full = traj_batch.action.shape
    action_data = traj_batch.action

    if len(action_shape_full) == 2:  # [T, B] discrete actions
        action_data = jnp.broadcast_to(
            action_data[:, :, jnp.newaxis],  # [T, B, 1] -> [T, B, N]
            (T, B, N),
        )
        joint_actions = jnp.broadcast_to(
            action_data[:, :, :, jnp.newaxis],  # [T, B, N, 1]
            (T, B, N, traj_len),
        )
    elif len(action_shape_full) == 3 and action_shape_full[2] == env_num_agents:
        # [T, B, N] multi-agent discrete actions
        joint_actions = jnp.broadcast_to(
            action_data[:, :, :, jnp.newaxis],  # [T, B, N, 1]
            (T, B, N, traj_len),
        )
    else:  # [T, B, N, *action_dim] multi-agent continuous actions
        joint_actions = jnp.broadcast_to(
            action_data[:, :, :, jnp.newaxis, ...],  # [T, B, N, 1, *action_dim]
            (T, B, N, traj_len, *action_data.shape[3:]),
        )

    return JointTrajectory(
        observations=joint_obs,  # [T, B, N, traj_len, *obs_dim]
        actions=joint_actions,  # [T, B, N, traj_len] or [T, B, N, traj_len, *action_dim]
    )


def get_learner_fn(
    env: MarlEnv,
    apply_fns: Tuple[Tuple[RecActorApply, RecActorApply], RecCriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> StoreExpLearnerFn:
    """Get the learner function."""

    (actor_train_apply_fn, actor_exec_apply_fn), critic_apply_fn = apply_fns
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
                - hstates (HiddenStates): The current hidden states of the RNN.
            _ (Any): The current metrics info.

        """

        def _env_step(
            learner_state: RNNLearnerState, _: Any
        ) -> Tuple[RNNLearnerState, RNNPPOTransition]:
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

            # Allocate action tensor based on discrete vs continuous action spaces
            is_discrete = isinstance(
                env.action_spec,
                (DiscreteArray, MultiDiscreteArray, Discrete, MultiDiscrete),
            )
            if is_discrete:
                # Discrete action: single integer per agent
                actions = jnp.zeros((config.arch.num_envs, env.num_agents), dtype=jnp.int32)
            else:
                # Continuous action: vector per agent
                actions = jnp.zeros(
                    (config.arch.num_envs, env.num_agents, env.action_dim), dtype=jnp.float32
                )

            log_probs = jnp.zeros((config.arch.num_envs, env.num_agents))
            policy_hidden_states = jnp.zeros_like(
                last_hstates.policy_hidden_state, dtype=jnp.float32
            )

            # Add a batch dimension to the observation.
            batched_observation = tree.map(lambda x: x[jnp.newaxis, :], last_timestep.observation)
            ac_in = (
                batched_observation,
                last_done[jnp.newaxis, :],
            )

            # Update trajectory state first with current observation to ensure consistency
            # Shift history and add current observation
            updated_obs_history = jnp.roll(trajectory_state.obs_history, -1, axis=2)

            updated_obs_history = updated_obs_history.at[:, :, -1].set(
                last_timestep.observation.agents_view
            )

            # For actions, we'll use the previous action history as we don't have current action yet
            # This will be updated after network computation
            updated_action_history = trajectory_state.action_history

            # Create updated trajectory state with current observation
            updated_trajectory_state = TrajectoryState(
                obs_history=updated_obs_history,
                action_history=updated_action_history,
                buffer_idx=trajectory_state.buffer_idx,
                buffer_full=trajectory_state.buffer_full,
            )

            # Construct joint trajectory from updated trajectory state for all agents
            joint_traj_all_agents = construct_joint_trajectory_for_agents(
                updated_trajectory_state, config
            )

            # ---------------- Vectorised per-agent actor forward pass ----------------
            # Split RNG for each agent once
            key, policy_key = jax.random.split(key)
            agent_keys = jax.random.split(policy_key, env.num_agents)

            # Prepare per-agent observations and done flags:  (N, 1, B, ...)
            obs_agents = tree.map(
                lambda x: jnp.transpose(x, (2, 0, 1) + tuple(range(3, x.ndim))),  # (N, 1, B, ...)
                batched_observation,
            )
            done_expand = last_done[jnp.newaxis, ...]  # (1, B, N)
            done_agents = jnp.transpose(done_expand, (2, 0, 1))  # (N, 1, B)

            # Vectorised apply + sampling with per-agent joint trajectory
            def _per_agent_apply(p, h, o, d, k, jt):
                new_h, pi = actor_exec_apply_fn(p, [h], (o, d), jt)
                act = pi.sample(seed=k)
                lp = pi.log_prob(act)
                # squeeze out the leading time dimension (0) we added (size=1)
                return new_h, act.squeeze(0), lp.squeeze(0)

            vmapped_apply = jax.vmap(_per_agent_apply, in_axes=(0, 1, 0, 0, 0, 0))

            # Run vmapped apply
            new_h_states, agent_actions, agent_log_probs = vmapped_apply(
                params.actor_params,
                last_hstates.policy_hidden_state,  # [B, N, H]  -> vmap agent axis=1
                obs_agents,
                done_agents,
                agent_keys,
                joint_traj_all_agents,  # [N, 1, B, N, traj_len, *dims]
            )

            # Reshape back to (B, N, ...)
            policy_hidden_states = new_h_states.transpose(1, 0, 2)  # (B, N, H)
            actions = agent_actions.transpose(1, 0, *range(2, agent_actions.ndim))  # (B, N, ...)
            log_probs = agent_log_probs.transpose(1, 0)  # (B, N)

            # We can keep the critic as is.
            # Construct joint trajectory for critic (single trajectory, not per-agent)
            critic_joint_traj = JointTrajectory(
                observations=trajectory_state.obs_history[
                    jnp.newaxis, ...
                ],  # [1, B, N, traj_len, *obs_dim]
                actions=trajectory_state.action_history[
                    jnp.newaxis, ...
                ],  # [1, B, N, traj_len, *action_dim]
            )
            critic_hidden_state, value = critic_apply_fn(
                params.critic_params, last_hstates.critic_hidden_state, ac_in, critic_joint_traj
            )

            value = value.squeeze(0)

            # Step the environment.
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, actions)

            # Update trajectory state with new observation and action
            # Shift history and add new data (this time include the actions we just computed)
            new_action_history = jnp.roll(updated_trajectory_state.action_history, -1, axis=2)
            new_action_history = new_action_history.at[:, :, -1].set(actions)

            # The observation history is already updated with current observation
            new_obs_history = updated_trajectory_state.obs_history

            # Update buffer index (capped at traj_len)
            traj_len = getattr(config.system, "traj_len", 10)
            new_buffer_idx = jnp.minimum(trajectory_state.buffer_idx + 1, traj_len)
            new_buffer_full = trajectory_state.buffer_full | (new_buffer_idx == traj_len)

            new_trajectory_state = TrajectoryState(
                obs_history=new_obs_history,
                action_history=new_action_history,
                buffer_idx=new_buffer_idx,
                buffer_full=new_buffer_full,
            )

            # log episode return and length
            # Duplicate info over agents since we need to be able to slice per agent in the
            # traj_batch later on in the trainer.
            done = timestep.last().repeat(env.num_agents).reshape(config.arch.num_envs, -1)

            hstates = HiddenStates(policy_hidden_states, critic_hidden_state)
            transition = RNNPPOTransition(
                last_done,
                actions,
                value,
                timestep.reward,
                log_probs,
                last_timestep.observation,
                last_hstates,
            )
            learner_state = RNNLearnerState(
                params, opt_states, key, env_state, timestep, done, hstates, new_trajectory_state
            )
            return learner_state, transition

        # INITIALISE RNN STATE
        initial_hstates = learner_state.hstates

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        # CALCULATE ADVANTAGE
        (
            params,
            opt_states,
            key,
            env_state,
            last_timestep,
            last_done,
            hstates,
            trajectory_state,
        ) = learner_state

        # Add a batch dimension to the observation.
        batched_last_observation = tree.map(lambda x: x[jnp.newaxis, :], last_timestep.observation)
        ac_in = (
            batched_last_observation,
            last_done[jnp.newaxis, :],
        )

        # Construct joint trajectory for last value computation with proper per-timestep history
        # Use the current trajectory_state which contains the latest history
        joint_traj_last = JointTrajectory(
            observations=trajectory_state.obs_history[
                jnp.newaxis, ...
            ],  # [1, B, N, traj_len, *obs_dim]
            actions=trajectory_state.action_history[
                jnp.newaxis, ...
            ],  # [1, B, N, traj_len, *action_dim]
        )

        # Run the network.
        _, last_val = critic_apply_fn(
            params.critic_params, hstates.critic_hidden_state, ac_in, joint_traj_last
        )
        # Squeeze out the batch dimension and mask out the value of terminal states.
        last_val = last_val.squeeze(0)
        last_val = jnp.where(last_done, jnp.zeros_like(last_val), last_val)

        advantages, targets = calculate_gae(
            traj_batch, last_val, last_done, config.system.gamma, config.system.gae_lambda
        )

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""

                # UNPACK TRAIN STATE AND BATCH INFO
                params, opt_states, key = train_state
                traj_batch, advantages, targets = batch_info

                # Before defining actor_grad_fn, precompute full joint trajectory with all agents.
                full_joint_traj = construct_training_joint_trajectory(
                    traj_batch, env.num_agents, config
                )

                def _actor_loss_fn(
                    actor_params: FrozenDict,
                    actor_opt_state: OptState,
                    traj_batch: RNNPPOTransition,
                    gae: chex.Array,
                    key: chex.PRNGKey,
                ) -> Tuple:
                    """Calculate the actor loss using full multi-agent joint trajectory."""
                    # RERUN NETWORK with single-agent obs/done as before
                    obs_and_done = (traj_batch.obs, traj_batch.done)

                    # Use precomputed full_joint_traj (contains all agents' data)
                    _, actor_policy = actor_train_apply_fn(
                        actor_params,
                        [traj_batch.hstates.policy_hidden_state[0]],
                        obs_and_done,
                        full_joint_traj,
                    )
                    log_prob = actor_policy.log_prob(traj_batch.action)

                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config.system.clip_eps,
                            1.0 + config.system.clip_eps,
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    # The seed will be used in the TanhTransformedDistribution:
                    entropy = actor_policy.entropy(seed=key).mean()

                    total_loss = loss_actor - config.system.ent_coef * entropy
                    return total_loss, (loss_actor, entropy)

                def _critic_loss_fn(
                    critic_params: FrozenDict,
                    critic_opt_state: OptState,
                    traj_batch: RNNPPOTransition,
                    targets: chex.Array,
                ) -> Tuple:
                    """Calculate the critic loss."""
                    # RERUN NETWORK
                    obs_and_done = (traj_batch.obs, traj_batch.done)

                    # Construct joint trajectory for minibatch training using actual timestep data
                    joint_traj = construct_training_joint_trajectory(
                        traj_batch, env.num_agents, config
                    )

                    _, value = critic_apply_fn(
                        critic_params,
                        traj_batch.hstates.critic_hidden_state[0],
                        obs_and_done,
                        joint_traj,  # Joint trajectory with proper shape for minibatch
                    )

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                        -config.system.clip_eps, config.system.clip_eps
                    )
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                    total_loss = config.system.vf_coef * value_loss
                    return total_loss, (value_loss)

                # CALCULATE ACTOR LOSS
                # we assume advantages are the same over all agents.
                advantages_single_agent = advantages[:, :, 0]
                actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                agents_params = params.actor_params
                agent_opt_states = opt_states.actor_opt_state

                key, shuffle_key = jax.random.split(key, 2)
                shuffled_agents = jax.random.permutation(shuffle_key, env.num_agents)

                def _agent_update(carry, agent_idx):
                    """Update actor params/opt_state for one agent."""
                    agents_params, agent_opt_states, adv_sa, inner_key = carry

                    inner_key, entropy_key = jax.random.split(inner_key)

                    agent_params = tree.map(lambda x: x[agent_idx], agents_params)
                    agent_traj = tree.map(lambda x: x[:, :, agent_idx], traj_batch)
                    agent_opt_state = tree.map(lambda x: x[agent_idx], agent_opt_states)

                    # Actor loss & gradients
                    (actor_loss_info_per_agent, actor_grads_per_agent) = actor_grad_fn(
                        agent_params,
                        agent_opt_state,
                        agent_traj,
                        adv_sa,
                        entropy_key,
                    )

                    actor_grads_per_agent, _ = jax.lax.pmean(
                        (actor_grads_per_agent, actor_loss_info_per_agent), axis_name="batch"
                    )
                    actor_grads_per_agent, _ = jax.lax.pmean(
                        (actor_grads_per_agent, actor_loss_info_per_agent), axis_name="device"
                    )

                    # Apply gradients
                    actor_updates, actor_new_opt_state = actor_update_fn(
                        actor_grads_per_agent, agent_opt_state
                    )
                    actor_new_params = optax.apply_updates(agent_params, actor_updates)

                    # Write back into full structures
                    agents_params = tree.map(
                        lambda x, y: x.at[agent_idx].set(y), agents_params, actor_new_params
                    )
                    agent_opt_states = tree.map(
                        lambda x, y: x.at[agent_idx].set(y), agent_opt_states, actor_new_opt_state
                    )

                    # Update advantage via importance sampling
                    agent_obs_and_done = (agent_traj.obs, agent_traj.done)

                    # For agent update, we have two choices:
                    # 1. Use the FULL joint trajectory with ALL agent observations (HAPPO style)
                    # 2. Use per-agent trajectory that matches the agent_obs_and_done

                    # Let's use approach 1: Full joint trajectory for proper HAPPO
                    # Use the shared full_joint_traj; remove redundant construction.
                    agent_joint_traj = full_joint_traj

                    _, actor_policy = actor_train_apply_fn(
                        actor_new_params,
                        [agent_traj.hstates.policy_hidden_state[0]],
                        agent_obs_and_done,  # Single agent obs/done for this specific agent
                        agent_joint_traj,  # Full multi-agent joint trajectory
                    )
                    log_prob = actor_policy.log_prob(agent_traj.action)
                    ratio = jnp.exp(log_prob - agent_traj.log_prob)
                    adv_sa *= ratio

                    new_carry = (agents_params, agent_opt_states, adv_sa, inner_key)
                    return new_carry, None

                # Scan over agents in shuffled order
                init_carry = (agents_params, agent_opt_states, advantages_single_agent, key)
                (agents_params, agent_opt_states, advantages_single_agent, key), _ = jax.lax.scan(
                    _agent_update, init_carry, shuffled_agents
                )

                # CALCULATE CRITIC LOSS
                critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                critic_loss_info, critic_grads = critic_grad_fn(
                    params.critic_params, opt_states.critic_opt_state, traj_batch, targets
                )

                # Compute the parallel mean (pmean) over the batch.
                # This calculation is inspired by the Anakin architecture demo notebook.
                # available at https://tinyurl.com/26tdzs5x
                # This pmean could be a regular mean as the batch axis is on the same device.

                critic_grads, critic_loss_info = jax.lax.pmean(
                    (critic_grads, critic_loss_info), axis_name="batch"
                )
                # pmean over devices.
                critic_grads, critic_loss_info = jax.lax.pmean(
                    (critic_grads, critic_loss_info), axis_name="device"
                )

                # UPDATE CRITIC PARAMS AND OPTIMISER STATE
                critic_updates, critic_new_opt_state = critic_update_fn(
                    critic_grads, opt_states.critic_opt_state
                )
                critic_new_params = optax.apply_updates(params.critic_params, critic_updates)

                new_params = Params(agents_params, critic_new_params)
                new_opt_state = OptStates(agent_opt_states, critic_new_opt_state)

                # PACK LOSS INFO
                return (new_params, new_opt_state, key), {"total_loss": critic_loss_info[1]}

            params, opt_states, init_hstates, traj_batch, advantages, targets, key = update_state
            key, shuffle_key, entropy_key = jax.random.split(key, 3)

            # SHUFFLE MINIBATCHES
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

            # UPDATE MINIBATCHES
            (params, opt_states, entropy_key), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states, entropy_key), minibatches
            )

            update_state = (
                params,
                opt_states,
                init_hstates,
                traj_batch,
                advantages,
                targets,
                key,
            )
            return update_state, loss_info

        init_hstates = tree.map(lambda x: x[None, :], initial_hstates)
        update_state = (
            params,
            opt_states,
            init_hstates,
            traj_batch,
            advantages,
            targets,
            key,
        )

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.ppo_epochs
        )

        params, opt_states, _, traj_batch, advantages, targets, key = update_state
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
        metric = last_timestep.extras["episode_metrics"] | last_timestep.extras["env_metrics"]
        return learner_state, (metric, loss_info, traj_batch)

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
                - hstates (HiddenStates): The initial hidden states of the RNN.

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

    # PRNG keys.
    key, actor_net_key, critic_net_key = keys
    actor_net_keys = jax.random.split(actor_net_key, num_agents)

    # Define network and optimisers.
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
        traj_len=getattr(config.system, "traj_len", 10),
        scan_fn=ScannedRNNPerAgent,
    )
    critic_network = Critic(
        pre_torso=critic_pre_torso,
        post_torso=critic_post_torso,
        hidden_state_dim=config.network.hidden_state_dim,
        centralised_critic=True,
        traj_len=getattr(config.system, "traj_len", 10),
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
    init_x = (init_obs, init_done)

    # Initialise hidden states.
    init_policy_hstate = ScannedRNNPerAgent.initialize_carry(
        config.arch.num_envs, config.network.hidden_state_dim
    )
    # # Duplicate the hidden states across the number of agents.
    init_policy_hstate = jnp.repeat(init_policy_hstate[:, jnp.newaxis, :], num_agents, axis=1)
    init_critic_hstate = ScannedRNN.initialize_carry(
        (config.arch.num_envs, num_agents), config.network.hidden_state_dim
    )

    # initialise params and optimiser state.

    # Create dummy joint trajectory for initialization
    sample_obs = env.observation_spec.generate_value()
    obs_shape = (
        sample_obs.agents_view.shape[1:]
        if hasattr(sample_obs, "agents_view")
        else sample_obs.shape[1:]
    )
    sample_action = env.action_spec.generate_value()
    action_shape = sample_action.shape[1:]
    action_dtype = jnp.array(sample_action).dtype
    traj_len = config.system.get("traj_len", 10)

    # Create dummy joint trajectory for each agent
    # Each agent needs the joint trajectory with proper batch dimension
    dummy_joint_traj_single = construct_dummy_joint_trajectory(
        obs_shape=obs_shape,
        action_shape=action_shape,
        batch_size=1,  # batch size for single call
        num_agents=num_agents,
        traj_len=traj_len,
        action_dtype=action_dtype,
    )

    # Expand for num_envs to match expected format: (1, num_envs, num_agents, traj_len, *dims)
    dummy_joint_traj_expanded = JointTrajectory(
        observations=jnp.broadcast_to(
            dummy_joint_traj_single.observations[jnp.newaxis, ...],  # Add leading dimension
            (1, config.arch.num_envs, num_agents, traj_len, *obs_shape),
        ),
        actions=jnp.broadcast_to(
            dummy_joint_traj_single.actions[jnp.newaxis, ...],  # Add leading dimension
            (1, config.arch.num_envs, num_agents, traj_len, *action_shape),
        ),
    )

    # actor net keys has agent dim at 0,
    # init_policy_hstate has agent dim at 1,
    # init_x has agent dims at (2, 2)
    # For joint_trajectory, we don't vmap it since each agent should see the same full trajectory
    actor_params = jax.vmap(actor_network.init, in_axes=(0, 1, (2, 2), None))(
        actor_net_keys, init_policy_hstate, init_x, dummy_joint_traj_expanded
    )
    actor_opt_state = jax.vmap(actor_optim.init)(actor_params)

    # We leave the critic as is since it is centralised.
    critic_params = critic_network.init(
        critic_net_key, init_critic_hstate, init_x, dummy_joint_traj_expanded
    )
    critic_opt_state = critic_optim.init(critic_params)

    # Get network apply functions and optimiser updates.
    # Use __call__ for training and get_actions for execution/evaluation
    actor_train_apply_fn = actor_network.apply
    actor_exec_apply_fn = lambda params, *args, **kwargs: actor_network.apply(
        params, *args, method=actor_network.get_actions, **kwargs
    )
    apply_fns = ((actor_train_apply_fn, actor_exec_apply_fn), critic_network.apply)
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
    traj_len = getattr(config.system, "traj_len", 10)
    # Get observation dims from first timestep
    sample_obs = env.observation_spec.generate_value()
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
    _config.logger.system_name = "rec_happo"
    config = copy.deepcopy(_config)

    # Vault configuration
    save_vault = getattr(config.system, "save_vault", False)
    vault_name = getattr(config.system, "vault_name", "rec_happo")
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

    # Determine action type and dimension for HAPPO evaluator
    from gymnasium.spaces import Discrete, MultiDiscrete
    from jumanji.specs import DiscreteArray, MultiDiscreteArray

    is_discrete = isinstance(
        env.action_spec, (DiscreteArray, MultiDiscreteArray, Discrete, MultiDiscrete)
    )
    action_type = "discrete" if is_discrete else "continuous"
    action_dim = None if is_discrete else env.action_dim

    # Create actor apply function for execution/evaluation using get_actions method
    actor_exec_apply_fn = lambda params, *args, **kwargs: actor_network.apply(
        params, *args, method=actor_network.get_actions, **kwargs
    )
    eval_act_fn = make_rec_eval_act_fn_with_traj(
        actor_exec_apply_fn, config, is_happo=True, action_type=action_type, action_dim=action_dim
    )
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
    # Single-device hidden state: [batch, agents, hidden_dim]
    eval_hs_single = ScannedRNNPerAgent.initialize_carry(
        eval_batch_size,
        config.network.hidden_state_dim,
    )
    eval_hs_single = jnp.repeat(eval_hs_single[:, jnp.newaxis, :], env.num_agents, axis=1)
    # Broadcast across devices: [n_devices, batch, agents, hidden_dim]
    eval_hs = jnp.broadcast_to(eval_hs_single[jnp.newaxis, ...], (n_devices, *eval_hs_single.shape))

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
            (n_devices, eval_batch_size, env.num_agents, traj_len, *obs_shape)
        ),
        "action_history": jnp.zeros(
            (n_devices, eval_batch_size, env.num_agents, traj_len, *action_shape),
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
        start_time = time.time()

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
        start_time = time.time()

        eval_batch_size = get_num_eval_envs(config, absolute_metric=True)
        # Single-device hidden state for absolute eval
        eval_hs_single = ScannedRNNPerAgent.initialize_carry(
            eval_batch_size,
            config.network.hidden_state_dim,
        )
        eval_hs_single = jnp.repeat(eval_hs_single[:, jnp.newaxis, :], env.num_agents, axis=1)
        # Broadcast across devices
        eval_hs = jnp.broadcast_to(
            eval_hs_single[jnp.newaxis, ...], (n_devices, *eval_hs_single.shape)
        )

        # Initialize trajectory history for absolute metric evaluation
        eval_traj_history_abs = {
            "obs_history": jnp.zeros(
                (n_devices, eval_batch_size, env.num_agents, traj_len, *obs_shape)
            ),
            "action_history": jnp.zeros(
                (n_devices, eval_batch_size, env.num_agents, traj_len, *action_shape),
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
    config_name="rec_happo.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}Recurrent HAPPO experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
