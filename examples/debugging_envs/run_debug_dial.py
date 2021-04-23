# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
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

"""Example running DIAL"""

# import importlib
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import dm_env
import numpy as np
import sonnet as snt
import tensorflow as tf
from absl import app, flags
from acme import types
from acme.specs import EnvironmentSpec
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.utils.loggers.tf_summary import TFSummaryLogger

from mava import specs as mava_specs
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems.tf import dial
from mava.utils.debugging.environments import switch_game
from mava.wrappers.debugging_envs import SwitchGameWrapper

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", 100, "Number of training episodes to run for.")

flags.DEFINE_integer(
    "num_episodes_per_eval",
    10,
    "Number of training episodes to run between evaluation " "episodes.",
)


class DIAL_policy(snt.RNNCore):
    def __init__(
        self,
        action_spec: EnvironmentSpec,
        message_spec: EnvironmentSpec,
        gru_hidden_size: int,
        gru_layers: int,
        task_mlp_size: Sequence,
        message_in_mlp_size: Sequence,
        message_out_mlp_size: Sequence,
        output_mlp_size: Sequence,
        name: str = None,
    ):
        super(DIAL_policy, self).__init__(name=name)
        self._action_spec = action_spec
        self._action_dim = np.prod(self._action_spec.shape, dtype=int)
        self._message_spec = message_spec
        self._message_dim = np.prod(self._message_spec.shape, dtype=int)

        self._gru_hidden_size = gru_hidden_size

        self.task_mlp = networks.LayerNormMLP(
            task_mlp_size,
            activate_final=True,
        )

        self.message_in_mlp = networks.LayerNormMLP(
            message_in_mlp_size,
            activate_final=True,
        )

        self.message_in_mlp = networks.LayerNormMLP(
            message_out_mlp_size,
            activate_final=True,
        )

        self.gru = snt.GRU(gru_hidden_size)

        self.output_mlp = snt.Sequential(
            [
                networks.LayerNormMLP(output_mlp_size, activate_final=True),
                networks.NearZeroInitializedLinear(self._action_dim),
                networks.TanhToSpec(self._action_spec),
            ]
        )

        self.message_out_mlp = snt.Sequential(
            [
                networks.LayerNormMLP(message_out_mlp_size, activate_final=True),
                networks.NearZeroInitializedLinear(self._message_dim),
                networks.TanhToSpec(self._message_spec),
            ]
        )

        self._gru_layers = gru_layers

    def initial_state(self, batch_size: int = 1) -> snt.Module:
        return self.gru.initial_state(batch_size)

    def __call__(
        self,
        x: snt.Module,
        state: Optional[snt.Module] = None,
        message: Optional[snt.Module] = None,
    ) -> snt.Module:
        if state is None:
            state = self.initial_state()
        if message is None:
            message = tf.zeros(self._message_dim, dtype=tf.float32)

        x_task = self.task_mlp(tf.dtypes.cast(x, tf.float32))
        x_message = self.message_in_mlp(message)
        x = tf.concat([x_task, x_message], axis=1)

        x, state = self.gru(x, state)

        x_output = self.output_mlp(x[:, -1])
        x_message = self.message_out_mlp(x[:, -1])

        return (x_output, x_message), state


def make_environment(
    env_name: str = "switch",
    num_agents: int = 3,
) -> dm_env.Environment:
    """Creates a SwitchGame environment."""
    env_module = switch_game.MultiAgentSwitchGame(num_agents=num_agents)
    environment = SwitchGameWrapper(env_module)
    return environment


# TODO Kevin: Define message head node correctly
def make_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    policy_network_gru_hidden_sizes: Union[Dict[str, int], int] = 128,
    policy_network_gru_layers: Union[Dict[str, int], int] = 2,
    policy_network_task_mlp_sizes: Union[Dict[str, Sequence], Sequence] = (128,),
    policy_network_message_in_mlp_sizes: Union[Dict[str, Sequence], Sequence] = (128,),
    policy_network_message_out_mlp_sizes: Union[Dict[str, Sequence], Sequence] = (128,),
    policy_network_output_mlp_sizes: Union[Dict[str, Sequence], Sequence] = (128,),
    message_size: int = 1,
    shared_weights: bool = True,
    sigma: float = 0.3,
) -> Mapping[str, types.TensorTransformation]:
    """Creates networks used by the agents."""
    specs = environment_spec.get_agent_specs()
    extra_specs = environment_spec.get_extra_specs()

    # Create agent_type specs
    if shared_weights:
        type_specs = {key.split("_")[0]: specs[key] for key in specs.keys()}
        specs = type_specs

    if isinstance(policy_network_gru_hidden_sizes, int):
        policy_network_gru_hidden_sizes = {
            key: policy_network_gru_hidden_sizes for key in specs.keys()
        }

    if isinstance(policy_network_gru_layers, int):
        policy_network_gru_layers = {
            key: policy_network_gru_layers for key in specs.keys()
        }

    if isinstance(policy_network_task_mlp_sizes, Sequence):
        policy_network_task_mlp_sizes = {
            key: policy_network_task_mlp_sizes for key in specs.keys()
        }

    if isinstance(policy_network_message_in_mlp_sizes, Sequence):
        policy_network_message_in_mlp_sizes = {
            key: policy_network_message_in_mlp_sizes for key in specs.keys()
        }

    if isinstance(policy_network_message_out_mlp_sizes, Sequence):
        policy_network_message_out_mlp_sizes = {
            key: policy_network_message_out_mlp_sizes for key in specs.keys()
        }

    if isinstance(policy_network_output_mlp_sizes, Sequence):
        policy_network_output_mlp_sizes = {
            key: policy_network_output_mlp_sizes for key in specs.keys()
        }

    observation_networks = {}
    policy_networks = {}
    behavior_networks = {}

    for key in specs.keys():

        # Get total number of action dimensions from action and message spec.
        num_dimensions = np.prod(specs[key].actions.shape, dtype=int)
        num_dimensions += np.prod(message_size, dtype=int)

        # Create the shared observation network
        observation_network = tf2_utils.to_sonnet_module(tf.identity)

        # Create the policy network.
        policy_network = DIAL_policy(
            action_spec=specs[key].actions,
            message_spec=extra_specs["messages"],
            gru_hidden_size=policy_network_gru_hidden_sizes[key],
            gru_layers=policy_network_gru_layers[key],
            task_mlp_size=policy_network_task_mlp_sizes[key],
            message_in_mlp_size=policy_network_message_in_mlp_sizes[key],
            message_out_mlp_size=policy_network_message_out_mlp_sizes[key],
            output_mlp_size=policy_network_output_mlp_sizes[key],
        )

        # Create the behavior policy.
        behavior_network = snt.Sequential(
            [
                observation_network,
                policy_network,
                networks.ClippedGaussian(sigma),
                networks.ClipToSpec(specs[key].actions),
            ]
        )

        observation_networks[key] = observation_network
        policy_networks[key] = policy_network
        behavior_networks[key] = behavior_network

    return {
        "policies": policy_networks,
        "observations": observation_networks,
        "behaviors": behavior_networks,
    }


def main(_: Any) -> None:
    # Create an environment, grab the spec, and use it to create networks.
    environment = make_environment()
    environment_spec = mava_specs.MAEnvironmentSpec(environment)
    system_networks = make_networks(environment_spec)

    # create tf loggers
    logs_dir = "logs"
    system_logger = TFSummaryLogger(f"{logs_dir}/system")
    train_logger = TFSummaryLogger(f"{logs_dir}/train_loop")
    eval_logger = TFSummaryLogger(f"{logs_dir}/eval_loop")

    # Construct the agent.
    system = dial.DIAL(
        environment_spec=environment_spec,
        networks=system_networks["policies"],
        observation_networks=system_networks[
            "observations"
        ],  # pytype: disable=wrong-arg-types
        behavior_networks=system_networks["behaviors"],
        logger=system_logger,
        checkpoint=False,
    )

    # Create the environment loop used for training.
    train_loop = ParallelEnvironmentLoop(
        environment, system, logger=train_logger, label="train_loop"
    )

    # Create the evaluation policy.
    # NOTE: assumes weight sharing
    specs = environment_spec.get_agent_specs()
    type_specs = {key.split("_")[0]: specs[key] for key in specs.keys()}
    specs = type_specs
    eval_policies = {
        key: snt.Sequential(
            [
                system_networks["policies"][key],
            ]
        )
        for key in specs.keys()
    }

    # Create the evaluation actor and loop.
    eval_actor = dial.DIALExecutor(
        policy_networks=eval_policies,
        communication_module=system._communication_module,
    )
    eval_env = make_environment()
    eval_loop = ParallelEnvironmentLoop(
        eval_env, eval_actor, logger=eval_logger, label="eval_loop"
    )

    for _ in range(FLAGS.num_episodes // FLAGS.num_episodes_per_eval):
        # train_loop.run(num_episodes=FLAGS.num_episodes_per_eval)
        train_loop.run(num_episodes=1)
        return
        eval_loop.run(num_episodes=1)


if __name__ == "__main__":
    app.run(main)
