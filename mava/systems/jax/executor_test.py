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

"""Tests for executor class for Jax-based Mava systems"""

from types import SimpleNamespace

from mava.utils.environments import debugging_utils
import functools
import numpy as np
import pytest

from mava.components.jax.executing.executor import DefaultExecutor
from mava.systems.jax import Executor
from mava.systems.jax.system import System
from mava.testing.building import mocks
from mava.systems.jax import mappo

# class TestSystem(System):
#     def design(self) -> SimpleNamespace:
#         """Mock system design with zero components.

#         Returns:
#             system callback components
#         """
#         components = SimpleNamespace(
#             data_server=mocks.MockDataServer,
#             data_server_adder=mocks.MockAdderSignature,
#             parameter_server=mocks.MockParameterServer,
#             parameter_client=mocks.MockParameterClient,
#             logger=mocks.MockLogger,
#             executor=DefaultExecutor,
#             executor_adder=mocks.MockAdder,
#             executor_environment_loop=mocks.MockExecutorEnvironmentLoop,
#             trainer=mocks.MockTrainer,
#             trainer_dataset=mocks.MockTrainerDataset,
#             distributor=mocks.MockDistributor,
#         )
#         return components

# # Executor example
# system = TestSystem()

# # Environment.
# environment_factory = functools.partial(
#     debugging_utils.make_environment,
#     env_name="simple_spread",
#     action_space="discrete",
# )

# # Networks.
# network_factory = mappo.make_default_networks

# # Build the system
# system.build(
#     environment_factory=environment_factory,
#     network_factory=network_factory
#     ),


# (
#     data_server,
#     parameter_server,
#     executor,
#     evaluator,
#     trainer,
# ) = test_system._builder.attr.system_build

# assert type(executor) == Executor

# print("executor: ", executor)

# # Executor example
# exit()


# @pytest.fixture
# def test_system() -> System:
#     """Dummy system with zero components."""
#     return TestSystem()


# def test_executor(
#     test_system: System,
# ) -> None:
#     """Test if the parameter server instantiates processes as expected."""
#     test_system.build()
#     (
#         data_server,
#         parameter_server,
#         executor,
#         evaluator,
#         trainer,
#     ) = test_system._builder.attr.system_build
#     assert type(parameter_server) == ParameterServer

#     step_var = parameter_server.get_parameters("trainer_steps")
#     assert type(step_var) == np.ndarray
#     assert step_var[0] == 0

#     parameter_server.set_parameters({"trainer_steps": np.ones(1, dtype=np.int32)})
#     assert parameter_server.get_parameters("trainer_steps")[0] == 1

#     parameter_server.add_to_parameters({"trainer_steps": np.ones(1, dtype=np.int32)})
#     assert parameter_server.get_parameters("trainer_steps")[0] == 2

