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

"""Value Decomposition system implementation."""

from typing import Callable, Dict, List, Optional, Type, Union, Mapping

import dm_env
import reverb
import sonnet as snt
from acme import specs as acme_specs

import mava
from mava import specs as mava_specs
from mava.components.tf.architectures import (
    DecentralisedValueActor,
)
from mava.types import EpsilonScheduler
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems.tf.madqn.execution import MADQNRecurrentExecutor
from mava.systems.tf.value_decomposition.training import ValueDecompositionRecurrentTrainer
from mava.systems.tf.variable_sources import VariableSource as MavaVariableSource
from mava.utils import enums
from mava.utils.loggers import MavaLogger
from mava.systems.tf.madqn import MADQN


class ValueDecomposition(MADQN):
    """Value Decomposition systems."""

    """TODO: Implement faster adders to speed up training times when
    using multiple trainers with non-shared weights."""

    def __init__(  # noqa
        self,
        environment_factory: Callable[[bool], dm_env.Environment],
        network_factory: Callable[[acme_specs.BoundedArray], Dict[str, snt.Module]],
        mixer: snt.Module,
        exploration_scheduler_fn: Union[
            EpsilonScheduler,
            Mapping[str, EpsilonScheduler],
            Mapping[str, Mapping[str, EpsilonScheduler]],
        ],
        logger_factory: Callable[[str], MavaLogger] = None,
        architecture: Type[
            DecentralisedValueActor
        ] = DecentralisedValueActor,
        trainer_fn: Type[ValueDecompositionRecurrentTrainer] = ValueDecompositionRecurrentTrainer,
        executor_fn: Type[MADQNRecurrentExecutor] = MADQNRecurrentExecutor,
        num_executors: int = 1,
        trainer_networks: Union[
            Dict[str, List], enums.Trainer
        ] = enums.Trainer.single_trainer,
        network_sampling_setup: Union[
            List, enums.NetworkSampler
        ] = enums.NetworkSampler.fixed_agent_networks,
        shared_weights: bool = True,
        environment_spec: mava_specs.MAEnvironmentSpec = None,
        discount: float = 0.99,
        batch_size: int = 32,
        prefetch_size: int = 4,
        target_averaging: bool = False,
        target_update_period: int = 200,
        target_update_rate: Optional[float] = None,
        executor_variable_update_period: int = 200,
        min_replay_size: int = 100,
        max_replay_size: int = 5000,
        samples_per_insert: Optional[float] = 32.0,
        optimizer: Union[
            snt.Optimizer, Dict[str, snt.Optimizer]
        ] = snt.optimizers.Adam(learning_rate=1e-4),
        mixer_optimizer: snt.Optimizer = snt.optimizers.Adam(learning_rate=1e-4),
        sequence_length: int = 20,
        period: int = 10,
        max_gradient_norm: float = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
        checkpoint_minute_interval: int = 5,
        logger_config: Dict = {},
        train_loop_fn: Callable = ParallelEnvironmentLoop,
        eval_loop_fn: Callable = ParallelEnvironmentLoop,
        train_loop_fn_kwargs: Dict = {},
        eval_loop_fn_kwargs: Dict = {},
        termination_condition: Optional[Dict[str, int]] = None,
        evaluator_interval: Optional[dict] = None,
        learning_rate_scheduler_fn: Optional[Dict[str, Callable[[int], None]]] = None,
    ):
        """Initialise the system
        Args:
            environment_factory: function to
                instantiate an environment.
            network_factory: function to instantiate system networks.
            logger_factory: function to
                instantiate a system logger.
            architecture:
                system architecture, e.g. decentralised or centralised.
            trainer_fn: training type
                associated with executor and architecture, e.g. centralised training.
            executor_fn: executor type, e.g.
                feedforward or recurrent.
            num_executors: number of executor processes to run in
                parallel..
            environment_spec: description of
                the action, observation spaces etc. for each agent in the system.
            trainer_networks: networks each
                trainer trains on.
            network_sampling_setup: List of networks that are randomly
                sampled from by the executors at the start of an environment run.
                enums.NetworkSampler settings:
                fixed_agent_networks: Keeps the networks
                used by each agent fixed throughout training.
                random_agent_networks: Creates N network policies, where N is the
                number of agents. Randomly select policies from this sets for each
                agent at the start of a episode. This sampling is done with
                replacement so the same policy can be selected for more than one
                agent for a given episode.
                Custom list: Alternatively one can specify a custom nested list,
                with network keys in, that will be used by the executors at
                the start of each episode to sample networks for each agent.
            shared_weights: whether agents should share weights or not.
                When network_sampling_setup are provided the value of shared_weights is
                ignored.
            discount: discount factor to use for TD updates.
            batch_size: sample batch size for updates.
            prefetch_size: size to prefetch from replay.
            target_averaging: whether to use polyak averaging for
                target network updates.
            target_update_period: number of steps before target
                networks are updated.
            target_update_rate: update rate when using
                averaging.
            executor_variable_update_period: number of steps before
                updating executor variables from the variable source.
            min_replay_size: minimum replay size before updating.
            max_replay_size: maximum replay size.
            samples_per_insert: number of samples to take
                from replay for every insert that is made.
            policy_optimizer: optimizer(s) for updating policy networks.
            critic_optimizer: optimizer for updating critic
                networks.
            n_step: number of steps to include prior to boostrapping.
            sequence_length: recurrent sequence rollout length.
            period: Consecutive starting points for overlapping
                rollouts across a sequence.
            max_gradient_norm: maximum allowed norm for gradients
                before clipping is applied.
            checkpoint: whether to checkpoint models.
            checkpoint_minute_interval: The number of minutes to wait between
                checkpoints.
            checkpoint_subpath: subdirectory specifying where to store
                checkpoints.
            logger_config: additional configuration settings for the
                logger factory.
            train_loop_fn: function to instantiate a train loop.
            eval_loop_fn: function to instantiate an evaluation
                loop.
            train_loop_fn_kwargs: possible keyword arguments to send
                to the training loop.
            eval_loop_fn_kwargs: possible keyword arguments to send to
            the evaluation loop.
            termination_condition: An optional terminal condition can be
                provided that stops the program once the condition is
                satisfied. Available options include specifying maximum
                values for trainer_steps, trainer_walltime, evaluator_steps,
                evaluator_episodes, executor_episodes or executor_steps.
                E.g. termination_condition = {'trainer_steps': 100000}.
            learning_rate_scheduler_fn: dict with two functions/classes (one for the
                policy and one for the critic optimizer), that takes in a trainer
                step t and returns the current learning rate,
                e.g. {"policy": policy_lr_schedule ,"critic": critic_lr_schedule}.
                See
                examples/debugging/simple_spread/feedforward/decentralised/run_maddpg_lr_schedule.py
                for an example.
            evaluator_interval: An optional condition that is used to
                evaluate/test system performance after [evaluator_interval]
                condition has been met. If None, evaluation will
                happen at every timestep.
                E.g. to evaluate a system after every 100 executor episodes,
                evaluator_interval = {"executor_episodes": 100}.
        """
        super().__init__(
            environment_factory=environment_factory,
            network_factory=network_factory,
            exploration_scheduler_fn=exploration_scheduler_fn,
            logger_factory=logger_factory,
            architecture=architecture,
            trainer_fn=trainer_fn,
            executor_fn=executor_fn,
            num_executors=num_executors,
            trainer_networks=trainer_networks,
            network_sampling_setup=network_sampling_setup,
            shared_weights=shared_weights,
            environment_spec=environment_spec,
            discount=discount,
            batch_size=batch_size,
            prefetch_size=prefetch_size,
            target_averaging=target_averaging,
            target_update_period=target_update_period,
            target_update_rate=target_update_rate,
            executor_variable_update_period=executor_variable_update_period,
            min_replay_size=min_replay_size,
            max_replay_size=max_replay_size,
            samples_per_insert=samples_per_insert,
            optimizer=optimizer,
            sequence_length=sequence_length,
            period=period,
            max_gradient_norm=max_gradient_norm,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
            checkpoint_minute_interval=checkpoint_minute_interval,
            logger_config=logger_config,
            train_loop_fn=train_loop_fn,
            eval_loop_fn=eval_loop_fn,
            train_loop_fn_kwargs=train_loop_fn_kwargs,
            eval_loop_fn_kwargs=eval_loop_fn_kwargs,
            termination_condition=termination_condition,
            evaluator_interval=evaluator_interval,
            learning_rate_scheduler_fn=learning_rate_scheduler_fn,
        ) 

        self._mixer = mixer
        self._mixer_optimizer = mixer_optimizer

    def trainer(
        self,
        trainer_id: str,
        replay: reverb.Client,
        variable_source: MavaVariableSource,
    ) -> mava.core.Trainer:
        """System trainer
        Args:
            trainer_id: Id of the trainer being created.
            replay: replay data table to pull data from.
            variable_source: variable server for updating
                network variables.
        Returns:
            system trainer.
        """

        print("##########")
        print("1")
        # create logger
        trainer_logger_config = {}
        if self._logger_config and "trainer" in self._logger_config:
            trainer_logger_config = self._logger_config["trainer"]
        trainer_logger = self._logger_factory(  # type: ignore
            trainer_id, **trainer_logger_config
        )

        print("2")
        # Create the system
        networks = self.create_system()

        print("3")
        dataset = self._builder.make_dataset_iterator(replay, trainer_id)

        print("4")
        trainer = self._builder.make_trainer(
            networks=networks,
            trainer_networks=self._trainer_networks[trainer_id],
            trainer_table_entry=self._table_network_config[trainer_id],
            dataset=dataset,
            logger=trainer_logger,
            variable_source=variable_source,
        )

        print("5")
        trainer.setup_mixer(self._mixer, self._mixer_optimizer)

        return trainer