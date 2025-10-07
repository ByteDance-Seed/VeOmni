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
from abc import ABC
from typing import Any, Dict, Union

from ..utils.import_utils import is_torch_version_greater_than, is_veomni_patch_available
from ..utils.logging import get_logger


if is_torch_version_greater_than("2.4"):
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
else:
    STATE_DICT_TYPE = ABC

logger = get_logger(__name__)

_MODEL_DIR = "model"


if is_veomni_patch_available():
    # for internal use only
    from veomni_patch.checkpoint.format_utils import omnistore_ckpt_to_state_dict
else:

    def omnistore_ckpt_to_state_dict(*args, **kwargs):
        raise ValueError("omnistore_ckpt_to_state_dict is not available, please install veomni_patch")


def ckpt_to_state_dict(
    save_checkpoint_path: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    ckpt_manager: str = "omnistore",
) -> Dict[str, Any]:
    """
    Interface to convert a checkpoint to a state_dict.
    Supported checkpoint managers:
        - omnistore
        - dcp
        - native

    Args:
        save_checkpoint_path: Path to the checkpoint.
        output_dir: Path to the output directory.
        ckpt_manager: Checkpoint manager.
    Returns:
        state_dict: State dict.
    """
    if ckpt_manager == "omnistore":
        state_dict = omnistore_ckpt_to_state_dict(save_checkpoint_path, output_dir)
    elif ckpt_manager == "bytecheckpoint":
        state_dict = bytecheckpoint_ckpt_to_state_dict(save_checkpoint_path, output_dir)
    elif ckpt_manager == "dcp":
        state_dict = dcp_to_torch_state_dict(save_checkpoint_path)
    else:
        raise ValueError(f"Unknown checkpoint manager: {ckpt_manager}")
    return state_dict


def bytecheckpoint_ckpt_to_state_dict(
    save_checkpoint_path: Union[str, os.PathLike], output_dir: Union[str, os.PathLike]
):
    """
    Given a directory containing an Bytecheckpoint checkpoint, this function will convert it into a
    Torch state_dict.
    Args:
        save_checkpoint_path: Directory containing the Bytecheckpoint checkpoint.
        output_dir: Directory to save the converted checkpoint.
    """

    from bytecheckpoint.utilities.ckpt_format.merge_tool import bytecheckpoint_ckpt_to_pytorch_ckpt

    state_dict = bytecheckpoint_ckpt_to_pytorch_ckpt(
        save_path=save_checkpoint_path,
        output_path=output_dir,
        framework="fsdp",
        model_only=True,
        return_dict=True,
    )
    return state_dict["model"]


def dcp_to_torch_state_dict(save_checkpoint_path: Union[str, os.PathLike]) -> STATE_DICT_TYPE:
    """
    Given a directory containing a DCP checkpoint, this function will convert it into a
    Torch state_dict.

    Args:
        save_checkpoint_path: Directory containing the DCP checkpoint.

    .. warning::
        To avoid OOM, it's recommended to only run this function on a single rank.
    """

    # Load the state_dict from the DCP checkpoint
    state_dict: STATE_DICT_TYPE = {}

    _load_state_dict(
        state_dict,
        storage_reader=FileSystemReader(save_checkpoint_path),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    if "state" in state_dict:
        # this happens when the model state dicts are flatten during saving
        state_dict = state_dict["state"]

    return state_dict["model"]
