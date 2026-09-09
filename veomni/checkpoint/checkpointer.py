# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

from ..utils.logging import get_logger
from ..utils.registry import Registry


logger = get_logger(__name__)


CHECKPOINTER_REGISTRY = Registry("checkpointer")
CHECKPOINT_TO_STATE_DICT_REGISTRY = Registry("checkpoint_to_state_dict")


def build_checkpointer(ckpt_manager: str, dist_backend: str):
    return CHECKPOINTER_REGISTRY[ckpt_manager](dist_backend)


def ckpt_to_state_dict(
    save_checkpoint_path: Union[str, os.PathLike],
    ckpt_manager: str = "dcp",
    **kwargs,
) -> Dict[str, Any]:
    """
    Interface to convert a checkpoint to a state_dict.
    Supported checkpoint managers:
        - dcp

    Args:
        save_checkpoint_path: Path to the checkpoint.
        ckpt_manager: Checkpoint manager.
    Returns:
        state_dict: State dict.
    """
    return CHECKPOINT_TO_STATE_DICT_REGISTRY[ckpt_manager](save_checkpoint_path, **kwargs)


class CheckpointerBase(ABC):
    """Base class for checkpointer"""

    @abstractmethod
    def save(
        cls,
        path: str,
        state: Dict[str, Any],
        save_async: Optional[bool],
        global_steps: Optional[int],
        trainable_only: bool = False,
        save_to_lowest_rank: bool = False,
        parallel_state=None,
        stage_dir: Optional[str] = None,
    ):
        """Persist training state to ``path``.

        Args:
            path: Destination the checkpoint is written to.
            state: Objects to persist; must contain ``model``.
            save_async: Return before the write completes, leaving it to a
                background thread. Backends without async support ignore this.
            global_steps: Step number; when given, the checkpoint goes into a
                per-step subdirectory of ``path``.
            trainable_only: Persist only parameters with ``requires_grad``.
            save_to_lowest_rank: Concentrate replicated shards on the lowest rank
                that holds them instead of spreading the writes.
            parallel_state: Parallelism layout the state was sharded under.
            stage_dir: Write under this directory and copy to ``path`` afterwards,
                for a destination far slower than local disk.
        """
        return

    @abstractmethod
    def load(
        cls,
        path: str,
        state: Dict[str, Any],
        trainable_only: bool = False,
        parallel_state=None,
    ):
        return

    @classmethod
    def wait_for_pending_save(cls) -> None:
        """Block until any in-flight async save completes.

        Default: no-op for backends that do not support async saves.
        Backends with async support (e.g. DCP) override this.
        """
        return


@CHECKPOINTER_REGISTRY.register("dcp")
def dcp_checkpointer(dist_backend: str):
    if dist_backend not in ["ddp", "fsdp2"]:
        raise ValueError(
            f"Unsupported distributed backend: {dist_backend} for DCP checkpoint manager, supported modes are: ddp, fsdp2"
        )
    from .dcp_checkpointer import DistributedCheckpointer

    return DistributedCheckpointer


@CHECKPOINT_TO_STATE_DICT_REGISTRY.register("dcp")
def dcp_ckpt_to_state_dict(save_checkpoint_path: Union[str, os.PathLike], **kwargs):
    from .dcp_checkpointer import dcp_to_torch_state_dict

    return dcp_to_torch_state_dict(save_checkpoint_path)
