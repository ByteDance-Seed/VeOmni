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
    return CHECKPOINT_TO_STATE_DICT_REGISTRY[ckpt_manager](save_checkpoint_path)


class CheckpointerBase(ABC):
    """Base class for checkpointer"""

    @abstractmethod
    def save(
        cls,
        path: str,
        state: Dict[str, Any],
        save_async: Optional[bool],
        global_steps: Optional[int],
    ):
        return

    @abstractmethod
    def load(
        cls,
        path: str,
        state: Dict[str, Any],
    ):
        return


@CHECKPOINTER_REGISTRY.register("dcp")
def dcp_checkpointer(dist_backend: str):
    from ..utils.import_utils import is_torch_version_greater_than

    if not is_torch_version_greater_than("2.4"):
        raise ValueError("DCP checkpoint manager requires torch version >= 2.4")
    if dist_backend not in ["ddp", "fsdp1", "fsdp2"]:
        raise ValueError(
            f"Unsupported distributed backend: {dist_backend} for DCP checkpoint manager, supported modes are: ddp, fsdp1, fsdp2"
        )
    from .dcp_checkpointer import DistributedCheckpointer

    return DistributedCheckpointer


@CHECKPOINT_TO_STATE_DICT_REGISTRY.register("dcp")
def dcp_ckpt_to_state_dict(save_checkpoint_path: Union[str, os.PathLike], **kwargs):
    from ..utils.import_utils import is_torch_version_greater_than

    if not is_torch_version_greater_than("2.4"):
        raise ValueError("DCP checkpoint manager requires torch version >= 2.4")
    from .dcp_checkpointer import dcp_to_torch_state_dict

    return dcp_to_torch_state_dict(save_checkpoint_path)
