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

import os
import typing
import warnings
from datetime import datetime
from pydoc import locate
from typing import Any, Dict, Optional, Union

import absl.logging as absl_logging
import orbax.checkpoint
from chex import Numeric
from flax.core.frozen_dict import FrozenDict
from jax.tree_util import tree_map
from omegaconf import DictConfig

from mava.types import HiddenStates, LearnerState, Params, RNNLearnerState

CHECKPOINTER_VERSION = 0.1


class Checkpointer:
    """Model checkpointer for saving and restoring the `learner_state`."""

    def __init__(
        self,
        model_name: str,
        config: Optional[Dict] = None,
        rel_dir: str = "checkpoints",
        timestamp_override: Optional[str] = None,
        save_interval_steps: int = 1,
        max_to_keep: Optional[int] = 1,
        keep_period: Optional[int] = None,
    ):
        """Initialise the checkpointer tool

        Args:
            model_name (str): Name of the model to be saved.
            config (Optional[Dict], optional):
                For storing model metadata. Defaults to None.
            rel_dir (str, optional):
                Relative directory of checkpoints. Defaults to "checkpoints".
            timestamp_override (Optional[str], optional):
                Set the dir name within rel_dir/model_name/<...>.
                Defaults to None, which means the timestamp is used.
            save_interval_steps (int, optional):
                The interval at which checkpoints should be saved. Defaults to 1.
            max_to_keep (Optional[int], optional):
                Maximum number of checkpoints to keep. Defaults to 1.
            keep_period (Optional[int], optional):
                If set, will not delete any checkpoint where
                checkpoint_step % keep_period == 0. Defaults to None.
        """

        # Sharding info will be read from file, rather than from 'RestoreArgs'
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="Couldn't find sharding info under RestoreArgs",
        )

        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        timestamp_str = (
            timestamp_override if timestamp_override else datetime.now().strftime("%Y%m%d%H%M%S")
        )

        options = orbax.checkpoint.CheckpointManagerOptions(
            create=True,
            best_fn=lambda x: x["episode_return"],
            best_mode="max",
            save_interval_steps=save_interval_steps,
            max_to_keep=max_to_keep,
            keep_period=keep_period,
        )

        def get_json_ready(obj: Any) -> Any:
            if not isinstance(obj, (bool, str, int, float, type(None))):
                return str(obj)
            else:
                return obj

        config_json_ready = tree_map(get_json_ready, config)

        self._manager = orbax.checkpoint.CheckpointManager(
            directory=os.path.join(os.getcwd(), rel_dir, model_name, timestamp_str),
            checkpointers=orbax_checkpointer,
            options=options,
            metadata={
                "checkpointer_version": CHECKPOINTER_VERSION,
                **(config_json_ready if config_json_ready is not None else {}),
            },
        )

        # Don't log checkpointing messages (at INFO level)
        absl_logging.set_verbosity(absl_logging.WARNING)

    def save(
        self,
        timestep: int,
        unreplicated_learner_state: Union[LearnerState, RNNLearnerState],
        episode_return: Numeric = 0.0,
    ) -> bool:
        """Save the learner state.

        Args:
            timestep (int):
                timestep at which the state is being saved.
            unreplicated_learner_state (Union[LearnerState, RNNLearnerState]):
                a Mava LearnerState (must be unreplicated)
            episode_return (Numeric, optional):
                Optional value to determine whether this is the 'best' model to save.
                Defaults to 0.0.

        Returns:
            bool: whether the saving was successful.
        """
        model_save_success: bool = self._manager.save(
            step=timestep,
            items={
                "learner_state": unreplicated_learner_state,
                "type": str(type(unreplicated_learner_state)),
            },
            # TODO: Currently we only log the episode return,
            #       but perhaps we should log other metrics.
            metrics={"episode_return": episode_return},
        )
        return model_save_success

    def restore_learner_state(
        self,
        unreplicated_input_learner_state: Union[LearnerState, RNNLearnerState],
        n: Optional[int] = None,
        restore_params: bool = True,
        restore_hstates: bool = True,
    ) -> Union[LearnerState, RNNLearnerState]:
        """Restore the learner state.

        Args:
            n (Optional[int], optional):
                Specific timestep for restoration (of course, only if that timestep exists).
                Defaults to None, in which case the latest step will be used.

        Returns:
            Union[LearnerState, RNNLearnerState]: the restored learner state
        """
        # Restore the checkpoint
        restored_checkpoint = self._manager.restore(n if n else self._manager.latest_step())

        # Dictionary of the restored learner state
        restored_learner_state_raw = restored_checkpoint["learner_state"]

        # Restore the learner state type and check it matches the input type
        restored_learner_state_type = locate(restored_checkpoint["type"])
        assert restored_learner_state_type == type(unreplicated_input_learner_state)

        # We base the new learner state on the input learner state
        new_learner_state = unreplicated_input_learner_state

        if restore_params:
            new_learner_state = new_learner_state._replace(
                params=Params(**FrozenDict(restored_learner_state_raw["params"])),
            )

        if restore_hstates and restored_learner_state_type == RNNLearnerState:
            new_learner_state = typing.cast(RNNLearnerState, new_learner_state)  # for mypy
            new_learner_state = new_learner_state._replace(
                hstates=HiddenStates(**FrozenDict(restored_learner_state_raw["hstates"])),
            )
        return new_learner_state

    def get_cfg(self) -> DictConfig:
        """Return the metadata of the checkpoint.

        Returns:
            DictConfig: metadata of the checkpoint.
        """
        return DictConfig(self._manager.metadata())
