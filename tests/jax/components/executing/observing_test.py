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

"""Tests for FeedforwardExecutorObserve class for Jax-based Mava systems"""


from types import SimpleNamespace
from typing import Dict

import numpy as np
import pytest
from acme.types import NestedArray
from dm_env import StepType, TimeStep

from mava.components.jax.executing.observing import (
    ExecutorObserveConfig,
    FeedforwardExecutorObserve,
)
from mava.systems.jax.executor import Executor
from mava.types import OLT
from mava.utils.sort_utils import sort_str_num


class MockAdder:
    def __init__(self) -> None:
        self.parm = "empty"

    def add_first(
        self, timestep: TimeStep, extras: Dict[str, NestedArray] = ...  # type: ignore
    ) -> None:
        self.parm = "after_add_first"

    def add(
        self,
        actions: Dict[str, NestedArray],
        next_timestep: TimeStep,
        next_extras: Dict[str, NestedArray] = ...,  # type: ignore
    ) -> None:
        self.parm = "after_add"


class MockExecutorParameterClient:
    def __init__(self) -> None:
        self.parm = False

    def get_async(self) -> None:
        self.parm = True


@pytest.fixture
def mock_executor_without_adder() -> Executor:
    """Mock executor component without adder"""
    extras={
        "agent_0": np.array([0]),
        "agent_1": np.array([1]),
        "agent_2": np.array([2])
    }
    store = SimpleNamespace(is_evaluator=None, observations={}, adder=None, extras=extras)
    return Executor(store=store)


class MockExecutor(Executor):
    def __init__(self, *args: object) -> None:
        # agent_net_keys
        agent_net_keys = {
            "agent_0": "network_agent_0",
            "agent_1": "network_agent_1",
            "agent_2": "network_agent_2",
        }
        # network_int_keys_extras
        network_int_keys_extras = None
        # network_sampling_setup
        network_sampling_setup = [
            ["network_agent_0"],
            ["network_agent_1"],
            ["network_agent_2"],
        ]
        # net_keys_to_ids
        all_samples = []
        for sample in network_sampling_setup:
            all_samples.extend(sample)
        unique_net_keys = list(sort_str_num(list(set(all_samples))))
        net_keys_to_ids = {net_key: i for i, net_key in enumerate(unique_net_keys)}
        # timestep
        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation=OLT(
                observation=[0.1, 0.3, 0.7], legal_actions=[1], terminal=[0.0]
            ),
        )
        # extras
        extras = {}  # type: ignore
        # Aadder
        adder = MockAdder()
        # actions_info
        actions_info = {
            "agent_0": "action_info_agent_0",
            "agent_1": "action_info_agent_1",
            "agent_2": "action_info_agent_2",
        }
        # policies_info
        policies_info = {
            "agent_0": "policy_info_agent_0",
            "agent_1": "policy_info_agent_1",
            "agent_2": "policy_info_agent_2",
        }
        # executor_parameter_client
        executor_parameter_client = MockExecutorParameterClient()
        # Store
        store = SimpleNamespace(
            is_evaluator=None,
            observations={},
            policy={},
            agent_net_keys=agent_net_keys,
            network_int_keys_extras=network_int_keys_extras,
            network_sampling_setup=network_sampling_setup,
            net_keys_to_ids=net_keys_to_ids,
            timestep=timestep,
            extras=extras,
            adder=adder,
            next_extras=extras,
            next_timestep=timestep,
            actions_info=actions_info,
            policies_info=policies_info,
            executor_parameter_client=executor_parameter_client,
        )
        self.store = store


@pytest.fixture
def mock_executor() -> MockExecutor:
    """Mock executor component."""
    return MockExecutor()


@pytest.fixture
def feedforward_executor_observe() -> FeedforwardExecutorObserve:
    """FeedforwardExecutorObserve.

    Returns:
        FeedforwardExecutorObserve
    """
    return FeedforwardExecutorObserve()


def test_on_execution_observe_first_without_adder(
    feedforward_executor_observe: FeedforwardExecutorObserve,
    mock_executor_without_adder: Executor,
) -> None:
    """Test entering executor without store.adder

    Args:
        feedforward_executor_observe: FeedForwardExecutorObserve,
        mock_executor_without_adder: Executor
    """
    feedforward_executor_observe.on_execution_observe_first(
        executor=mock_executor_without_adder
    )

    assert mock_executor_without_adder.store.extras== {
        "agent_0": np.array([0]),
        "agent_1": np.array([1]),
        "agent_2": np.array([2])
    }
    assert not hasattr(mock_executor_without_adder.store, "network_int_keys_extras")
    assert not hasattr(mock_executor_without_adder.store, "agent_net_keys")


def test_on_execution_observe_first(
    feedforward_executor_observe: FeedforwardExecutorObserve,
    mock_executor: MockExecutor,
) -> None:
    """Test on_execution_observe_first method from FeedForwardExecutorObserve

    Args:
        feedforward_executor_observe: FeedForwardExecutorObserve,
        mock_executor: Executor
    """
    feedforward_executor_observe.on_execution_observe_first(executor=mock_executor)

    for agent, net in mock_executor.store.agent_net_keys.items():
        assert type(agent) == str
        assert type(net) == str
        assert agent in ["agent_0", "agent_1", "agent_2"]
        assert net in ["network_agent_0", "network_agent_1", "network_agent_2"]

    agents = sort_str_num(list(mock_executor.store.agent_net_keys.keys()))
    for agent, value in mock_executor.store.network_int_keys_extras.items():
        assert type(agent) == str
        assert type(value) == np.ndarray
        assert agent in agents
        assert value in mock_executor.store.net_keys_to_ids.values()

    assert (
        mock_executor.store.extras["network_int_keys"]
        == mock_executor.store.network_int_keys_extras
    )
    assert mock_executor.store.adder.parm == "after_add_first"


def test_on_execution_observe_without_adder(
    feedforward_executor_observe: FeedforwardExecutorObserve,
    mock_executor_without_adder: Executor,
) -> None:
    """Test entering executor without store.adder

    Args:
        feedforward_executor_observe: FeedForwardExecutorObserve,
        mock_executor_without_adder: Executor
    """
    feedforward_executor_observe.on_execution_observe(
        executor=mock_executor_without_adder
    )

    assert not hasattr(mock_executor_without_adder.store, "next_extras")
    assert not hasattr(mock_executor_without_adder.store.adder, "add")


def test_on_execution_observe(
    feedforward_executor_observe: FeedforwardExecutorObserve,
    mock_executor: MockExecutor,
) -> None:
    """Test on_execution_observe method from FeedForwardExecutorObserve

    Args:
        feedforward_executor_observe: FeedForwardExecutorObserve,
        mock_executor: Executor
    """
    feedforward_executor_observe.on_execution_observe(executor=mock_executor)

    for agent in mock_executor.store.policies_info.keys():
        assert mock_executor.store.next_extras["policy_info"][
            agent
        ] == "policy_info_" + str(agent)
    assert (
        mock_executor.store.next_extras["network_int_keys"]
        == mock_executor.store.network_int_keys_extras
    )
    assert mock_executor.store.adder.parm == "after_add"


def test_on_execution_update(
    feedforward_executor_observe: FeedforwardExecutorObserve,
    mock_executor: MockExecutor,
) -> None:
    """Test on_execution_update method from FeedForwardExecutorObserve

    Args:
        feedforward_executor_observe: FeedForwardExecutorObserve,
        mock_executor: Executor
    """
    feedforward_executor_observe.on_execution_update(executor=mock_executor)

    assert mock_executor.store.executor_parameter_client.parm == True
