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
from typing import Any, Dict

import torch
import torch.distributed as dist

from ..utils.import_utils import is_torch_version_greater_than
from ..utils.logging import get_logger


if is_torch_version_greater_than("2.4"):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import (
        FileSystemReader,
        FileSystemWriter,
    )
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        get_optimizer_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
    )
    from torch.distributed.checkpoint.stateful import Stateful
else:
    Stateful = ABC

logger = get_logger(__name__)

_EXTRA_STATE_FORMAT = "extra_state_rank_{}.pt"
_MODEL_DIR = "model"
_OPTIMIZER_DIR = "optimizer"
_EXTRA_STATE_DIR = "extra_state"


class ModelState(Stateful):
    """
    A wrapper around a model to make it stateful.
    Args:
        model (Model): model to wrap.
    """

    def __init__(self, model):
        self.model = model

    def state_dict(self):
        model_state_dict = get_model_state_dict(model=self.model)
        return {"model": model_state_dict}

    def load_state_dict(self, state_dict):
        set_model_state_dict(model=self.model, model_state_dict=state_dict["model"])


class OptimizerState(Stateful):
    """
    A wrapper around an optimizer to make it stateful.

    Args:
        model (Model): model to wrap.
        optimizer (Optimizer): optimizer to wrap.
    """

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        optimizer_state_dict = get_optimizer_state_dict(model=self.model, optimizers=self.optimizer)
        return {"optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        set_optimizer_state_dict(model=self.model, optimizers=self.optimizer, optim_state_dict=state_dict["optim"])


def build_checkpointer(
    dist_backend: str = "fsdp1",
    ckpt_manager: str = "bytecheckpoint",
):
    """
    create a checkpointer manager with given mode.
    Args:
        dist_backend (str, optional): checkpoint mode. Defaults to "fsdp1".
            fsdp1: FSDP1 checkpoint from bytecheckpoint
            fsdp2-vescale: FSDP2 checkpoint from bytecheckpoint
            fsdp2: FSDP2 checkpoint from bytecheckpoint
            ddp: DDP checkpoint from bytecheckpoint
            dcp: DCP checkpoint from torch.distributed.checkpoint
        ckpt_manager (str, optional): checkpoint manager. Defaults to "bytecheckpoint".
            bytecheckpoint: bytecheckpoint checkpoint manager
            dcp: torch dcp checkpoint manager
    Raises:
        ValueError: if ckpt_manager is not supported

    Returns:
        Checkpointer: checkpointer with given mode.
    """

    if ckpt_manager == "bytecheckpoint":
        if dist_backend == "ddp":
            from bytecheckpoint import DDPCheckpointer as Checkpointer
        elif dist_backend == "fsdp1":
            from bytecheckpoint import FSDPCheckpointer as Checkpointer
        elif dist_backend == "fsdp2-vescale":
            from bytecheckpoint import VeScaleCheckpointer as Checkpointer
        elif dist_backend == "fsdp2":
            from bytecheckpoint import FSDP2Checkpointer as Checkpointer
    elif ckpt_manager == "dcp":
        if not is_torch_version_greater_than("2.4"):
            raise ValueError("DCP checkpoint manager requires torch version >= 2.4")
        if dist_backend not in ["ddp", "fsdp1", "fsdp2"]:
            raise ValueError(
                f"Unsupported distributed backend: {dist_backend} for DCP checkpoint manager, supported modes are: ddp, fsdp1, fsdp2"
            )
        Checkpointer = DistributedCheckpointer
    else:
        raise ValueError(
            f"Unknown checkpoint manager: {ckpt_manager}, supported modes are: bytecheckpoint, dcp, native"
        )

    return Checkpointer


class CheckpointerBase(ABC):
    """Base class for checkpointer"""

    @abstractmethod
    def save(
        cls,
        path: str,
        state: Dict[str, Any],
    ):
        return

    @abstractmethod
    def load(
        cls,
        path: str,
        state: Dict[str, Any],
    ):
        return


class DistributedCheckpointer(CheckpointerBase):
    """
    Distributed checkpointer for torch.distributed.checkpoint
    """

    @classmethod
    def save(
        cls,
        path: str,
        state: Dict[str, Any],
        global_steps: int = None,
        save_async=False,
    ) -> None:
        """
        save training state to distributed checkpoint

        args:
            path: path to save checkpoint
            state: state to save
            global_steps: global steps
            save_async: whether to save asynchronously
        return:
            None
        """

        checkpoint_dir = f"{path}/global_step_{global_steps}" if global_steps else path
        os.makedirs(checkpoint_dir, exist_ok=True)

        if "model" not in state:
            raise ValueError("Model must be provided to save a distributed checkpoint.")

        if save_async:
            model_dir = os.path.join(checkpoint_dir, _MODEL_DIR)
            dcp.async_save(
                state_dict={"state": ModelState(state["model"])},
                storage_writer=FileSystemWriter(
                    model_dir,
                    thread_count=16,
                    single_file_per_rank=True,
                    sync_files=False,
                ),
            )
            if "optimizer" in state:
                optimizer_dir = os.path.join(checkpoint_dir, _OPTIMIZER_DIR)
                dcp.async_save(
                    state_dict={"state": OptimizerState(model=state["model"], optimizer=state["optimizer"])},
                    storage_writer=FileSystemWriter(
                        optimizer_dir,
                        thread_count=16,
                        single_file_per_rank=True,
                        sync_files=False,
                    ),
                )
        else:
            model_dir = os.path.join(checkpoint_dir, _MODEL_DIR)

            dcp.save(
                state_dict={"state": ModelState(state["model"])},
                storage_writer=FileSystemWriter(
                    model_dir,
                    thread_count=16,
                    single_file_per_rank=True,
                    sync_files=False,
                ),
            )
            if "optimizer" in state:
                optimizer_dir = os.path.join(checkpoint_dir, _OPTIMIZER_DIR)
                dcp.save(
                    state_dict={"state": OptimizerState(model=state["model"], optimizer=state["optimizer"])},
                    storage_writer=FileSystemWriter(
                        optimizer_dir,
                        thread_count=16,
                        single_file_per_rank=True,
                        sync_files=False,
                    ),
                )

        if "extra_state" in state:
            extra_state_dir = os.path.join(checkpoint_dir, _EXTRA_STATE_DIR)
            os.makedirs(extra_state_dir, exist_ok=True)
            extra_state_path = os.path.join(extra_state_dir, _EXTRA_STATE_FORMAT.format(dist.get_rank()))
            torch.save(
                state["extra_state"],
                extra_state_path,
            )

        logger.info_rank0(f"Saved checkpoint to {checkpoint_dir}")

    @classmethod
    def load(
        cls,
        path: str,
        state: Dict[str, Any],
        process_group=None,
    ) -> Dict[str, Any]:
        """
        load training state from distributed checkpoint
        args:
            path: path to load checkpoint
            state: state to load, "model" are required,  "optimizer" and "extra_state" are optional

        return:
            state: state loaded
        """
        checkpoint_dir = path

        if state is None:
            raise ValueError("State dict must be provided to load a distributed checkpoint.")

        if "model" not in state:
            raise ValueError("Model must be provided to load a distributed checkpoint.")

        if "optimizer" in state:
            model_dir = os.path.join(checkpoint_dir, _MODEL_DIR)
            dcp.load(
                state_dict={"state": ModelState(state["model"])},
                storage_reader=FileSystemReader(model_dir),
                process_group=process_group,
            )

            optimizer_dir = os.path.join(checkpoint_dir, _OPTIMIZER_DIR)
            dcp.load(
                state_dict={"state": OptimizerState(model=state["model"], optimizer=state["optimizer"])},
                storage_reader=FileSystemReader(optimizer_dir),
                process_group=process_group,
            )
        else:
            model_dir = os.path.join(checkpoint_dir, _MODEL_DIR)
            dcp.load(
                state_dict={"state": ModelState(state["model"])},
                storage_reader=FileSystemReader(model_dir),
                process_group=process_group,
            )

        if "extra_state" in state:
            extra_state_dir = os.path.join(checkpoint_dir, _EXTRA_STATE_DIR)
            os.makedirs(extra_state_dir, exist_ok=True)
            extra_state_path = os.path.join(extra_state_dir, _EXTRA_STATE_FORMAT.format(dist.get_rank()))
            state["extra_state"] = torch.load(
                extra_state_path,
            )

        logger.info_rank0(f"Loaded checkpoint from {checkpoint_dir}")

        return state
