#!/usr/bin/env python3
"""
Test script for rec_mappo.py trajectory support.
Verifies update_trajectory_state and construct_trajectories functions.
"""

import jax.numpy as jnp

from mava.systems.ppo.anakin.rec_mappo import (
    construct_trajectories,
    update_trajectory_state,
)
from mava.systems.ppo.types import TrajectoryState
from mava.types import Observation


def main() -> None:
    # Test parameters
    batch_size = 1
    num_agents = 3
    traj_len = 4
    obs_dim = 5

    # Create a dummy full trajectory state with distinct values
    obs_history = jnp.arange(batch_size * num_agents * traj_len * obs_dim, dtype=jnp.float32)
    obs_history = obs_history.reshape((batch_size, num_agents, traj_len, obs_dim))
    action_history = jnp.arange(batch_size * num_agents * traj_len, dtype=jnp.int32)
    action_history = action_history.reshape((batch_size, num_agents, traj_len))

    trajectory_state = TrajectoryState(
        obs_history=obs_history,
        action_history=action_history,
        buffer_idx=traj_len,
        buffer_full=True,
    )

    # Construct trajectories and verify correctness
    # Create a dummy current observation matching the last timestep
    last_obs = obs_history[0, :, -1, :]
    action_mask = jnp.zeros((num_agents,), dtype=bool)
    current_obs = Observation(agents_view=last_obs, action_mask=action_mask)

    agent_idx = 2  # test a specific agent
    individual_traj, joint_traj = construct_trajectories(
        trajectory_state, current_obs, agent_idx=agent_idx
    )

    # individual_traj.observations should match obs_history for agent_idx
    expected_ind = obs_history[0, agent_idx, :, :]
    assert jnp.array_equal(individual_traj.observations[0], expected_ind), (
        "Individual trajectory observations do not match expected values"
    )
    # joint_traj.observations should match full obs_history
    assert jnp.array_equal(joint_traj.observations[0], obs_history[0]), (
        "Joint trajectory observations do not match expected values"
    )

    print("construct_trajectories test passed!")

    # Test update_trajectory_state
    # Initialize an empty trajectory state
    init_obs = jnp.zeros((batch_size, num_agents, traj_len, obs_dim), dtype=jnp.float32)
    init_actions = jnp.zeros((batch_size, num_agents, traj_len), dtype=jnp.int32)
    init_state = TrajectoryState(
        obs_history=init_obs,
        action_history=init_actions,
        buffer_idx=0,
        buffer_full=False,
    )

    # Dummy new observation and action
    new_obs = jnp.arange(num_agents * obs_dim, dtype=jnp.float32).reshape((num_agents, obs_dim))
    new_action = jnp.arange(num_agents, dtype=jnp.int32)
    current_obs2 = Observation(agents_view=new_obs, action_mask=action_mask)

    updated_state = update_trajectory_state(init_state, current_obs2, new_action)

    # The last entry in updated_state should match the new obs and action
    assert jnp.array_equal(updated_state.obs_history[0, :, -1, :], new_obs), (
        "Updated obs_history last timestep does not match new observation"
    )
    assert jnp.array_equal(updated_state.action_history[0, :, -1], new_action), (
        "Updated action_history last timestep does not match new action"
    )

    print("update_trajectory_state test passed!")


if __name__ == "__main__":
    main()
