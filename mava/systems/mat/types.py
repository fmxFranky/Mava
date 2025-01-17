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

from typing import Callable, Tuple

import chex
import optax
from chex import Array, PRNGKey
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from flax.core.frozen_dict import FrozenDict
from jumanji.types import TimeStep
from typing_extensions import NamedTuple, TypeAlias

from mava.types import Action, Done, MavaObservation, Observation, State, Value


class PPOTransition(NamedTuple):
    """Transition tuple for MAT."""

    done: Done
    action: Action
    value: Value
    reward: chex.Array
    log_prob: chex.Array
    obs: Observation
    next_obs: Observation


class OffPolicyTransition(NamedTuple):
    """Transition tuple for MAT."""

    action: Action
    obs: Observation
    reward: chex.Array
    terminal: chex.Array
    next_obs: Observation


BufferState: TypeAlias = TrajectoryBufferState[OffPolicyTransition]


class LearnerState(NamedTuple):
    """The state of the learner."""

    params: FrozenDict
    opt_states: Tuple[optax.OptState, optax.OptState]  # (mat_opt_state, byol_opt_state)
    key: chex.PRNGKey
    env_state: State
    last_timestep: TimeStep
    buffer_state: BufferState  # Add buffer state to learner state


class MATNetworkConfig(NamedTuple):
    """Configuration for the MAT network."""

    n_block: int
    n_head: int
    embed_dim: int
    use_swiglu: bool
    use_rmsnorm: bool


ActorApply = Callable[
    [FrozenDict, MavaObservation, PRNGKey],
    Tuple[Array, Array, Array],
]

LearnerApply = Callable[
    [FrozenDict, MavaObservation, MavaObservation, Array, PRNGKey],
    Tuple[Array, Array, Array, Array, Array],
]

ByolApply = Callable[
    [FrozenDict, MavaObservation, MavaObservation, Array],
    Tuple[Array, Array],
]
