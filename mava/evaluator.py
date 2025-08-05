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

import math
import time
import warnings
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
from chex import Array, PRNGKey
from flax.core.frozen_dict import FrozenDict
from jax import tree
from jumanji.types import TimeStep
from omegaconf import DictConfig
from typing_extensions import TypeAlias

from mava.types import (
    Action,
    ActorApply,
    JointTrajectory,
    MarlEnv,
    Metrics,
    Observation,
    ObservationGlobalState,
    RecActorApply,
    State,
)
from mava.wrappers.gym import GymToJumanji

# Optional extras that are passed out of the actor and then into the actor in the next step
ActorState: TypeAlias = Dict[str, Any]
# Type of the carry for the _env_step function in the evaluator
_EvalEnvStepState: TypeAlias = Tuple[State, TimeStep, PRNGKey, ActorState]
# The function signature for the mava evaluation function (returned by `get_eval_fn`).
EvalFn: TypeAlias = Callable[[FrozenDict, PRNGKey, ActorState], Metrics]


class EvalActFn(Protocol):
    """The API for the acting function that is passed to the `EvalFn`.

    A get_action function must conform to this API in order to be used with Mava's evaluator.
    See `make_ff_eval_act_fn` and `make_rec_eval_act_fn` as examples.
    """

    def __call__(
        self,
        params: FrozenDict,
        timestep: TimeStep[Union[Observation, ObservationGlobalState]],
        key: PRNGKey,
        actor_state: ActorState,
    ) -> Tuple[Array, ActorState]: ...


def get_num_eval_envs(config: DictConfig, absolute_metric: bool) -> int:
    """Returns the number of vmapped envs/batch size during evaluation."""
    # Determine number of devices for evaluation: use configured arch.n_devices or default to 1
    n_devices = config.arch.get("n_devices", 1) if config.arch.architecture_name == "anakin" else 1
    n_parallel_envs = config.arch.num_envs * n_devices

    if absolute_metric:
        eval_episodes = config.arch.num_absolute_metric_eval_episodes
    else:
        eval_episodes = config.arch.num_eval_episodes

    if eval_episodes <= n_parallel_envs:
        return math.ceil(eval_episodes / n_devices)  # type: ignore
    else:
        return config.arch.num_envs  # type: ignore


def get_eval_fn(
    env: MarlEnv, act_fn: EvalActFn, config: DictConfig, absolute_metric: bool
) -> EvalFn:
    """Creates a function that can be used to evaluate agents on a given environment.

    Args:
    ----
        env: an environment that conforms to the mava environment spec.
        act_fn: a function that takes in params, timestep, key and optionally a state
                and returns actions and optionally a state (see `EvalActFn`).
        config: the system config.
        absolute_metric: whether or not this evaluator calculates the absolute_metric.
                This determines how many evaluation episodes it does.
    """
    # Determine number of devices for evaluation: use configured arch.n_devices or default to 1
    n_devices = config.arch.get("n_devices", 1)
    eval_episodes = (
        config.arch.num_absolute_metric_eval_episodes
        if absolute_metric
        else config.arch.num_eval_episodes
    )
    n_vmapped_envs = get_num_eval_envs(config, absolute_metric)
    n_parallel_envs = n_vmapped_envs * n_devices
    episode_loops = math.ceil(eval_episodes / n_parallel_envs)

    # Warnings if num eval episodes is not divisible by num parallel envs.
    if eval_episodes % n_parallel_envs != 0:
        warnings.warn(
            f"Number of evaluation episodes ({eval_episodes}) is not divisible by `num_envs` * "
            f"`num_devices` ({n_parallel_envs} * {n_devices}). Some extra evaluations will be "
            f"executed. New number of evaluation episodes = {episode_loops * n_parallel_envs}",
            stacklevel=2,
        )

    def eval_fn(params: FrozenDict, key: PRNGKey, init_act_state: ActorState) -> Metrics:
        """Evaluates the given params on an environment and returns relevent metrics.

        Metrics are collected by the `RecordEpisodeMetrics` wrapper: episode return and length,
        also win rate for environments that support it.

        Returns: Dict[str, Array] - dictionary of metric name to metric values for each episode.
        """

        def _env_step(eval_state: _EvalEnvStepState, _: Any) -> Tuple[_EvalEnvStepState, TimeStep]:
            """Performs a single environment step"""
            env_state, ts, key, actor_state = eval_state

            key, act_key = jax.random.split(key)
            action, actor_state = act_fn(params, ts, act_key, actor_state)
            env_state, ts = jax.vmap(env.step)(env_state, action)

            return (env_state, ts, key, actor_state), ts

        def _episode(key: PRNGKey, _: Any) -> Tuple[PRNGKey, Metrics]:
            """Simulates `num_envs` episodes."""
            key, reset_key = jax.random.split(key)
            reset_keys = jax.random.split(reset_key, n_vmapped_envs)
            env_state, ts = jax.vmap(env.reset)(reset_keys)

            step_state = env_state, ts, key, init_act_state
            _, timesteps = jax.lax.scan(_env_step, step_state, jnp.arange(env.time_limit + 1))

            metrics = timesteps.extras["episode_metrics"] | timesteps.extras["env_metrics"]

            # find the first instance of done to get the metrics at that timestep, we don't
            # care about subsequent steps because we only the results from the first episode
            done_idx = jnp.argmax(timesteps.last(), axis=0)
            metrics = tree.map(lambda m: m[done_idx, jnp.arange(n_vmapped_envs)], metrics)

            return key, metrics

        # This loop is important because we don't want too many parallel envs.
        # So in evaluation we have num_envs parallel envs and loop enough times
        # so that we do at least `eval_episodes` number of episodes.
        _, metrics = jax.lax.scan(_episode, key, xs=None, length=episode_loops)
        metrics = tree.map(lambda x: x.reshape(-1), metrics)  # flatten metrics
        return metrics

    def timed_eval_fn(params: FrozenDict, key: PRNGKey, init_act_state: ActorState) -> Metrics:
        """Wrapper around eval function to time it and add in steps per second metric."""
        start_time = time.time()

        # Map evaluation function over specified devices
        devices = jax.local_devices()[: config.arch.get("n_devices", 1)]
        metrics = jax.pmap(eval_fn, devices=devices)(params, key, init_act_state)
        metrics = jax.block_until_ready(metrics)

        end_time = time.time()
        total_timesteps = jnp.sum(metrics["episode_length"])
        metrics["steps_per_second"] = total_timesteps / (end_time - start_time)
        return metrics

    return timed_eval_fn


def make_ff_eval_act_fn(actor_apply_fn: ActorApply, config: DictConfig) -> EvalActFn:
    """Makes an act function that conforms to the evaluator API given a standard
    feed forward mava actor network."""

    def eval_act_fn(
        params: FrozenDict, timestep: TimeStep, key: PRNGKey, actor_state: ActorState
    ) -> Tuple[Action, Dict]:
        pi = actor_apply_fn(params, timestep.observation)
        action = pi.mode() if config.arch.evaluation_greedy else pi.sample(seed=key)
        return action, {}

    return eval_act_fn


def make_rec_eval_act_fn(actor_apply_fn: RecActorApply, config: DictConfig) -> EvalActFn:
    """Makes an act function that conforms to the evaluator API given a standard
    recurrent mava actor network."""

    _hidden_state = "hidden_state"

    def eval_act_fn(
        params: FrozenDict, timestep: TimeStep, key: PRNGKey, actor_state: ActorState
    ) -> Tuple[Action, Dict]:
        hidden_state = actor_state[_hidden_state]

        n_agents = timestep.observation.agents_view.shape[1]
        last_done = timestep.last()[:, jnp.newaxis].repeat(n_agents, axis=-1)
        ac_in = (timestep.observation, last_done)
        ac_in = tree.map(lambda x: x[jnp.newaxis], ac_in)  # add batch dim to obs

        hidden_state, pi = actor_apply_fn(params, hidden_state, ac_in, None, key=key)
        action = pi.mode() if config.arch.evaluation_greedy else pi.sample(seed=key)
        return action.squeeze(0), {_hidden_state: hidden_state}

    return eval_act_fn


def make_rec_eval_act_fn_with_traj(
    actor_apply_fn: RecActorApply,
    config: DictConfig,
    is_happo: bool = False,
    action_type: str = None,
    action_dim: int = None,
) -> EvalActFn:
    """Makes an act function that conforms to the evaluator API given a standard
    recurrent mava actor network with trajectory support.

    In evaluation, following CTDE principle, we create a joint trajectory where only
    the current agent's information is real and others are zero-filled.
    """

    _hidden_state = "hidden_state"
    _trajectory_history = "trajectory_history"

    def eval_act_fn(
        params: FrozenDict, timestep: TimeStep, key: PRNGKey, actor_state: ActorState
    ) -> Tuple[Action, Dict]:
        hidden_state = actor_state[_hidden_state]
        traj_history = actor_state.get(_trajectory_history, None)

        n_envs = timestep.observation.agents_view.shape[0]
        n_agents = timestep.observation.agents_view.shape[1]
        last_done = timestep.last()[:, jnp.newaxis].repeat(n_agents, axis=-1)
        ac_in = (timestep.observation, last_done)
        ac_in = tree.map(lambda x: x[jnp.newaxis], ac_in)  # add batch dim to obs

        if is_happo:
            # Build provisional obs_history that already includes CURRENT observation so that
            # joint_trajectory passed to the network matches observation.agents_view exactly.
            traj_len = config.system.get("traj_len", 10)
            n_envs = timestep.observation.agents_view.shape[0]
            n_agents = timestep.observation.agents_view.shape[1]
            obs_shape = timestep.observation.agents_view.shape[2:]

            if traj_history is None:
                # initialise history with zeros then set last position to current observation
                obs_history_tmp = jnp.zeros((n_envs, n_agents, traj_len, *obs_shape))
            else:
                # start from previous history and roll
                obs_history_tmp = jnp.roll(traj_history["obs_history"], -1, axis=2)
            obs_history_tmp = obs_history_tmp.at[:, :, -1].set(timestep.observation.agents_view)

            # For actions we don't know current action yet, keep previous (or zeros)
            if traj_history is None:
                action_history_tmp = jnp.zeros((n_envs, n_agents, traj_len))
            else:
                action_history_tmp = traj_history["action_history"]

            # Build FULL joint trajectory (no masking) so every agent sees all agents' data
            joint_traj = JointTrajectory(
                observations=obs_history_tmp[
                    jnp.newaxis, ...
                ],  # [1, n_envs, n_agents, traj_len, *obs_dim]
                actions=action_history_tmp[
                    jnp.newaxis, ...
                ],  # [1, n_envs, n_agents, traj_len, *action_dim]
                last_actions=None,  # Not available during evaluation
                last_action_masks=None,  # Not available during evaluation - only current agent's mask available
            )

            # Allocate action tensor based on declared action_type and action_dim
            if action_type == "discrete":
                action = jnp.zeros((1, n_envs, n_agents), dtype=jnp.int32)
            else:
                action = jnp.zeros((1, n_envs, n_agents, action_dim), dtype=jnp.float32)
            # Initialize new hidden state
            new_hidden_state = jnp.zeros_like(hidden_state, dtype=jnp.float32)
            for agent in range(n_agents):
                key, policy_key = jax.random.split(key)
                single_agent_ac_in = tree.map(lambda x, agent=agent: x[:, :, agent], ac_in)
                agent_params = tree.map(lambda x, agent=agent: x[agent], params)
                agent_hstates = tree.map(lambda x, agent=agent: x[:, agent, :], hidden_state)
                # Run the network for this agent WITH correct joint trajectory
                agent_policy_hidden_state, agent_actor_policy = actor_apply_fn(
                    agent_params, agent_hstates, single_agent_ac_in, joint_traj, key=policy_key
                )
                new_hidden_state = new_hidden_state.at[:, agent].set(agent_policy_hidden_state)
                # Select action
                if config.arch.evaluation_greedy:
                    action_per_agent = agent_actor_policy.mode()
                else:
                    action_per_agent = agent_actor_policy.sample(seed=policy_key)
                action = action.at[:, :, agent].set(action_per_agent.squeeze(0))
            # After all agents, update trajectory history with chosen action
            new_traj_history = update_eval_trajectory_history(
                traj_history, timestep.observation, action.squeeze(0), config
            )
            return action.squeeze(0), {
                _hidden_state: new_hidden_state,
                _trajectory_history: new_traj_history,
            }
        else:
            # standard CTDE evaluation
            joint_traj = construct_ctde_joint_trajectory(timestep.observation, traj_history, config)
            hidden_state, pi = actor_apply_fn(params, hidden_state, ac_in, joint_traj, key=key)
            action = pi.mode() if config.arch.evaluation_greedy else pi.sample(seed=key)
            new_traj_history = update_eval_trajectory_history(
                traj_history, timestep.observation, action.squeeze(0), config
            )
            return action.squeeze(0), {
                _hidden_state: hidden_state,
                _trajectory_history: new_traj_history,
            }

    return eval_act_fn


def construct_ctde_joint_trajectory(
    observation: Union["Observation", "ObservationGlobalState"],
    traj_history: Optional[Dict],
    config: DictConfig,
) -> "JointTrajectory":
    """Construct a CTDE-compliant joint trajectory for evaluation.

    For each agent, only its own observations and actions are real, others are zero-filled.
    This maintains the joint trajectory shape while respecting CTDE constraints.
    """
    traj_len = config.system.get("traj_len", 10)
    n_envs = observation.agents_view.shape[0]
    n_agents = observation.agents_view.shape[1]
    obs_shape = observation.agents_view.shape[2:]

    # Initialize with zeros if no history
    if traj_history is None:
        obs_history = jnp.zeros((n_envs, n_agents, traj_len, *obs_shape))
        # Default to scalar actions - will be corrected when first action comes in
        action_history = jnp.zeros((n_envs, n_agents, traj_len))
    else:
        obs_history = traj_history["obs_history"]
        action_history = traj_history["action_history"]

    # Apply CTDE masking using vectorized operations
    # Create an identity matrix to mask each agent's own trajectory
    # Shape: (n_agents, n_agents) - diagonal matrix
    agent_mask = jnp.eye(n_agents)

    # Expand mask to match trajectory dimensions
    # For obs: add dimensions for traj_len and obs_shape
    obs_mask_shape = (n_agents, n_agents, 1, *[1 for _ in obs_shape])
    obs_mask = agent_mask.reshape(n_agents, n_agents, 1, 1).reshape(obs_mask_shape)

    # For actions: add dimensions for traj_len and action dimensions
    action_extra_dims = len(action_history.shape) - 3  # exclude (n_envs, n_agents, traj_len)
    action_mask_shape = (n_agents, n_agents, 1, *[1 for _ in range(action_extra_dims)])
    action_mask = agent_mask.reshape(n_agents, n_agents, 1).reshape(action_mask_shape)

    # Vectorized CTDE masking: each agent can only see its own trajectory
    # Use einsum to apply per-agent masking efficiently
    # obs_history: (n_envs, n_agents, traj_len, *obs_shape)
    # obs_mask: (n_agents, n_agents, 1, 1, ...)
    # Result: (n_envs, n_agents, traj_len, *obs_shape) where each agent only sees its own data
    ctde_obs_history = jnp.einsum("ij...,ej...->ei...", obs_mask, obs_history)
    ctde_action_history = jnp.einsum("ij...,ej...->ei...", action_mask, action_history)

    # Add batch dimension to match RecurrentActor's expected format
    # Shape: (n_envs, n_agents, traj_len, *dims) -> (1, n_envs, n_agents, traj_len, *dims)
    ctde_obs_history = ctde_obs_history[jnp.newaxis, ...]
    ctde_action_history = ctde_action_history[jnp.newaxis, ...]

    return JointTrajectory(
        observations=ctde_obs_history,  # (1, n_envs, n_agents, traj_len, *obs_shape)
        actions=ctde_action_history,  # (1, n_envs, n_agents, traj_len, *action_shape)
        last_actions=None,  # Not available during evaluation
        last_action_masks=None,  # Not available during evaluation - only partial agent masks available
    )


def update_eval_trajectory_history(
    traj_history: Optional[Dict],
    observation: Union["Observation", "ObservationGlobalState"],
    action: chex.Array,
    config: DictConfig,
) -> Dict:
    """Update trajectory history for evaluation, maintaining CTDE constraints."""

    traj_len = config.system.get("traj_len", 10)
    n_envs = observation.agents_view.shape[0]
    n_agents = observation.agents_view.shape[1]
    obs_shape = observation.agents_view.shape[2:]

    if traj_history is None:
        # Initialize history with current observation/action
        obs_history = jnp.zeros((n_envs, n_agents, traj_len, *obs_shape))

        # Determine action history shape from actual action
        if len(action.shape) == 2:
            # Discrete actions: (n_envs, n_agents)
            action_history = jnp.zeros((n_envs, n_agents, traj_len))
        else:
            # Continuous actions: (n_envs, n_agents, action_dim)
            action_dim = action.shape[2]
            action_history = jnp.zeros((n_envs, n_agents, traj_len, action_dim))

        # Set the last position to current observation/action
        obs_history = obs_history.at[:, :, -1].set(observation.agents_view)
        action_history = action_history.at[:, :, -1].set(action)
    else:
        obs_history = traj_history["obs_history"]
        action_history = traj_history["action_history"]

        # Shift history and add new data
        obs_history = jnp.roll(obs_history, -1, axis=2)
        action_history = jnp.roll(action_history, -1, axis=2)

        # Add current observation/action to the last position
        obs_history = obs_history.at[:, :, -1].set(observation.agents_view)
        action_history = action_history.at[:, :, -1].set(action)

    return {
        "obs_history": obs_history,
        "action_history": action_history,
    }


def get_sebulba_eval_fn(
    env_maker: Callable[[int, int], GymToJumanji],
    act_fn: EvalActFn,
    config: DictConfig,
    np_rng: np.random.Generator,
    absolute_metric: bool,
) -> Tuple[EvalFn, Any]:
    """Creates a function that can be used to evaluate agents on a given environment.

    Args:
    ----
        env_maker: A function to create the environment instances.
        act_fn: A function that takes in params, timestep, key and optionally a state
                and returns actions and optionally a state (see `EvalActFn`).
        config: The system config.
        np_rng: Random number generator for seeding environment.
        absolute_metric: Whether or not this evaluator calculates the absolute_metric.
                This determines how many evaluation episodes it does.
    """
    n_devices = jax.device_count()
    eval_episodes = (
        config.arch.num_absolute_metric_eval_episodes
        if absolute_metric
        else config.arch.num_eval_episodes
    )

    n_parallel_envs = min(eval_episodes, config.arch.num_envs)
    episode_loops = math.ceil(eval_episodes / n_parallel_envs)
    env = env_maker(config, n_parallel_envs)

    act_fn = jax.jit(
        act_fn, device=jax.local_devices()[config.arch.actor_device_ids[0]]
    )  # Evaluate using the first actor device

    # Warnings if num eval episodes is not divisible by num parallel envs.
    if eval_episodes % n_parallel_envs != 0:
        warnings.warn(
            f"Number of evaluation episodes ({eval_episodes}) is not divisible by `num_envs` * "
            f"`num_devices` ({n_parallel_envs} * {n_devices}). Some extra evaluations will be "
            f"executed. New number of evaluation episodes = {episode_loops * n_parallel_envs}",
            stacklevel=2,
        )

    def eval_fn(params: FrozenDict, key: PRNGKey, init_act_state: ActorState) -> Metrics:
        """Evaluates the given params on an environment and returns relevent metrics.

        Metrics are collected by the `RecordEpisodeMetrics` wrapper: episode return and length,
        also win rate for environments that support it.

        Returns: Dict[str, Array] - dictionary of metric name to metric values for each episode.
        """

        def _episode(key: PRNGKey) -> Tuple[PRNGKey, Metrics]:
            """Simulates `num_envs` episodes."""

            # Generate a list of random seeds within the 32-bit integer range, using a seeded RNG.
            seeds = np_rng.integers(np.iinfo(np.int32).max, size=n_parallel_envs).tolist()
            ts = env.reset(seed=seeds)

            timesteps_array = [ts]

            actor_state = init_act_state
            finished_eps = ts.last()

            while not finished_eps.all():
                key, act_key = jax.random.split(key)
                action, actor_state = act_fn(params, ts, act_key, actor_state)
                cpu_action = jax.device_get(action)
                ts = env.step(cpu_action)
                timesteps_array.append(ts)

                finished_eps = np.logical_or(finished_eps, ts.last())

            timesteps = jax.tree.map(lambda *x: np.stack(x), *timesteps_array)

            metrics = timesteps.extras["episode_metrics"]
            if config.env.log_win_rate:
                metrics["won_episode"] = timesteps.extras["won_episode"]

            # Find the first instance of done to get the metrics at that timestep.
            done_idx = np.argmax(timesteps.last(), axis=0)
            metrics = tree.map(lambda m: m[done_idx, np.arange(n_parallel_envs)], metrics)
            del metrics["is_terminal_step"]  # uneeded for logging

            return key, metrics

        # This loop is important because we don't want too many parallel envs.
        # So in evaluation we have num_envs parallel envs and loop enough times
        # so that we do at least `eval_episodes` number of episodes.
        metrics_array = []
        for _ in range(episode_loops):
            key, metric = _episode(key)
            metrics_array.append(metric)

        # flatten metrics
        metrics: Metrics = tree.map(lambda *x: np.array(x).reshape(-1), *metrics_array)
        return metrics

    def timed_eval_fn(params: FrozenDict, key: PRNGKey, init_act_state: ActorState) -> Metrics:
        """Wrapper around eval function to time it and add in steps per second metric."""
        start_time = time.time()

        metrics = eval_fn(params, key, init_act_state)

        end_time = time.time()
        total_timesteps = jnp.sum(metrics["episode_length"])
        metrics["steps_per_second"] = total_timesteps / (end_time - start_time)
        return metrics

    return timed_eval_fn, env


def get_eval_fn_with_traj(
    env: "MarlEnv", act_fn: EvalActFn, config: DictConfig, absolute_metric: bool
) -> "EvalFn":
    """Creates a function that can be used to evaluate agents on a given environment with trajectory support.

    This version maintains trajectory history during evaluation while respecting CTDE constraints.
    """
    import time

    from jax import tree

    n_devices = config.arch.get("n_devices", 1)
    eval_episodes = (
        config.arch.num_absolute_metric_eval_episodes
        if absolute_metric
        else config.arch.num_eval_episodes
    )

    episode_loops = int(eval_episodes // (config.arch.num_envs * n_devices))
    n_vmapped_envs = config.arch.num_envs

    # Create named tuple for eval state that includes trajectory history
    _EvalEnvStepState = tuple[Any, Any, Any, ActorState]

    def eval_fn(params: FrozenDict, key: PRNGKey, init_act_state: ActorState) -> "Metrics":
        """Evaluates the given params on an environment and returns relevant metrics."""

        def _env_step(eval_state: _EvalEnvStepState, _: Any) -> Tuple[_EvalEnvStepState, Any]:
            """Performs a single environment step with trajectory tracking"""
            env_state, ts, key, actor_state = eval_state

            key, act_key = jax.random.split(key)
            action, actor_state = act_fn(params, ts, act_key, actor_state)
            env_state, ts = jax.vmap(env.step)(env_state, action)

            return (env_state, ts, key, actor_state), ts

        def _episode(key: PRNGKey, _: Any) -> Tuple[PRNGKey, "Metrics"]:
            """Simulates `num_envs` episodes with trajectory tracking."""
            key, reset_key = jax.random.split(key)
            reset_keys = jax.random.split(reset_key, n_vmapped_envs)
            env_state, ts = jax.vmap(env.reset)(reset_keys)

            step_state = env_state, ts, key, init_act_state
            _, timesteps = jax.lax.scan(_env_step, step_state, jnp.arange(env.time_limit + 1))

            metrics = timesteps.extras["episode_metrics"] | timesteps.extras["env_metrics"]

            # find the first instance of done to get the metrics at that timestep
            done_idx = jnp.argmax(timesteps.last(), axis=0)
            metrics = tree.map(lambda m: m[done_idx, jnp.arange(n_vmapped_envs)], metrics)

            return key, metrics

        _, metrics = jax.lax.scan(_episode, key, xs=None, length=episode_loops)
        metrics = tree.map(lambda x: x.reshape(-1), metrics)  # flatten metrics
        return metrics

    def timed_eval_fn(params: FrozenDict, key: PRNGKey, init_act_state: ActorState) -> "Metrics":
        """Wrapper around eval function to time it and add in steps per second metric."""
        start_time = time.time()

        # Map evaluation function over specified devices
        devices = jax.local_devices()[:n_devices]
        metrics = jax.pmap(eval_fn, devices=devices)(params, key, init_act_state)
        metrics = jax.block_until_ready(metrics)

        end_time = time.time()
        total_timesteps = jnp.sum(metrics["episode_length"])
        metrics["steps_per_second"] = total_timesteps / (end_time - start_time)
        return metrics

    return timed_eval_fn
