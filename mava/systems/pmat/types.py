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
from chex import Array, PRNGKey
from flax.core.frozen_dict import FrozenDict
from jumanji.types import TimeStep
from optax._src.base import OptState
from typing_extensions import NamedTuple

from mava.types import MavaObservation, State


class LearnerState(NamedTuple):
    """State of the PMAT learner."""

    params: FrozenDict
    opt_state: OptState
    key: chex.PRNGKey
    env_state: State
    timestep: TimeStep


class PMATNetworkConfig(NamedTuple):
    """Configuration for the PMAT network."""

    n_block: int
    n_head: int
    embed_dim: int
    use_swiglu: bool
    use_rmsnorm: bool
    # PMAT-specific configuration
    scoring_hidden_dim: int = 64
    scoring_n_layers: int = 2
    ranking_loss_coef: float = 1.0


class PMATTransition(NamedTuple):
    """PMAT transition data structure extending PPO transition."""

    done: chex.Array
    action: chex.Array
    value: chex.Array
    reward: chex.Array
    log_prob: chex.Array
    obs: chex.Array
    # PMAT-specific fields
    seq_log_prob: chex.Array  # Log probability of action sequence
    seq: chex.Array  # Action sequence (priority ordering)


ActorApply = Callable[
    [FrozenDict, MavaObservation, PRNGKey],
    Tuple[Array, Array, Array, Array, Array],  # action, log_prob, value, seq_log_prob, seq
]
LearnerApply = Callable[
    [FrozenDict, MavaObservation, Array, Array, PRNGKey],
    Tuple[
        Array, Array, Array, Array, Array
    ],  # action_log, value, entropy, seq_log_prob, seq_entropy
]
