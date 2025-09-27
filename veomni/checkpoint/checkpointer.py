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
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, DTensor, Shard

from ..distributed.parallel_state import get_parallel_state
from ..utils.checkpoint_utils import _GLOBAL_STEP_PREFIX
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
_EXTRA_STATE_DIR = "extra_state"


class ModelState(Stateful):
    """
    A wrapper around a model to make it stateful.
    Args:
        model (Model): model to wrap.
    """

    def __init__(self, model):
        self.model = model

        # Determine whether this is EP+FSDP2 case
        # If so, we need to restore EP-dim before saving to DCP
        # For FSDP1, it is implemented by FSDPExtension and state_dict hooks
        # which is aumatically triggered by get_model_state_dict
        self.parallel_state = get_parallel_state()
        self.ep_fqn2spec_info = getattr(self.model, "_fqn2spec_info", None)
        self.should_ep_aware = self.ep_fqn2spec_info is not None and self.parallel_state.dp_mode == "fsdp2"

    @torch.no_grad()
    def state_dict(self):
        model_state_dict = get_model_state_dict(model=self.model)
        if self.should_ep_aware:
            logger.info_rank0(
                "Getting model state_dict from ModelState wrapper, would restore EP dim for Experts module"
            )
            model_state_dict = self.get_state_dict_with_ep_dim(model_state_dict)

        return {"model": model_state_dict}

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        """
        perform the reverse operation for state_dict()
        need to drop EP-dim when loading from DCP checkpoints
        so that EP-FSDP would not be confused
        """
        model_state_dict = state_dict["model"]
        if self.should_ep_aware:
            model_state_dict = self.get_state_dict_without_ep_dim(model_state_dict)

        set_model_state_dict(model=self.model, model_state_dict=model_state_dict)

    def get_state_dict_with_ep_dim(self, state_dict):
        ep_fqn2spec_info = self.ep_fqn2spec_info
        assert ep_fqn2spec_info is not None, "if fqn2spec_info is None it should not be patch"

        ep_mesh = self.parallel_state.ep_fsdp_device_mesh["ep"]
        assert ep_mesh is not None

        global_device_mesh = self.parallel_state.ep_fsdp_device_mesh
        assert global_device_mesh.ndim == 2

        keys = list(state_dict.keys())
        for name in sorted(keys):
            if name in ep_fqn2spec_info and isinstance(ep_fqn2spec_info[name].placement, Shard):
                cur_spec_info = ep_fqn2spec_info[name]
                tensor = state_dict[name]
                tensor = self._restore_ep_dim(tensor, cur_spec_info.ep_fsdp_mesh)
                state_dict[name] = tensor

        return state_dict

    def get_state_dict_without_ep_dim(self, state_dict):
        fqn2spec_info = getattr(self.model, "_fqn2spec_info", None)
        assert fqn2spec_info is not None, "if fqn2spec_info is None it should not be patch"

        ep_mesh = self.parallel_state.ep_fsdp_device_mesh["ep"]
        assert ep_mesh is not None

        global_device_mesh = self.parallel_state.ep_fsdp_device_mesh
        assert global_device_mesh.ndim == 2

        keys = list(state_dict.keys())
        for name in sorted(keys):
            if name in fqn2spec_info and isinstance(fqn2spec_info[name].placement, Shard):
                cur_spec_info = fqn2spec_info[name]
                tensor = state_dict[name]
                tensor = self._drop_ep_dim(tensor, cur_spec_info.ep_fsdp_mesh)
                state_dict[name] = tensor

        return state_dict

    def _restore_ep_dim(self, orgin_tensor: torch.Tensor, device_mesh: DeviceMesh):
        """
        Restore EP dim so that DCP can be aware about EP ranks

        args:
            orgin_tensor (torch.Tensor): The orgin tensor.
            device_mesh (DeviceMesh): The ep device mesh.
            shard (Shard): The shard info, default Shard(0).

        """
        assert device_mesh.ndim == 2, f"global_mesh.ndim must be 2, got {device_mesh.ndim}"
        ep_mesh = device_mesh["ep"]

        if orgin_tensor.__class__.__name__ == "DTensor":
            # EP+FSDP2
            dtensor = DTensor.from_local(
                orgin_tensor._local_tensor, device_mesh=device_mesh, placements=[Shard(0), Shard(1)]
            )
        elif orgin_tensor.__class__.__name__ == "Tensor":
            # If there is no FSDP
            dtensor = DTensor.from_local(orgin_tensor, device_mesh=ep_mesh, placements=[Shard(0)])

        return dtensor

    def _drop_ep_dim(self, loaded_tensor: torch.Tensor, device_mesh: DeviceMesh):
        """
        Drop EP dims after loading from DCP so that EP-FSDP would not be confused
        """
        assert device_mesh.ndim == 2, f"global_mesh.ndim must be 2, got {device_mesh.ndim}"
        ep_fsdp_mesh = device_mesh["ep_fsdp"]

        if len(loaded_tensor.placements) == 2:
            tensor_to_put = DTensor.from_local(
                loaded_tensor._local_tensor, device_mesh=ep_fsdp_mesh, placements=[Shard(1)]
            )
        elif len(loaded_tensor.placements) == 1:
            tensor_to_put = loaded_tensor.to_local()
        else:
            raise RuntimeError(
                f"Expect EP paramters from checkpoints to be DTensor with 1-dim (no FSDP) or 2-dim (EP+FSDP), got {loaded_tensor}"
            )

        return tensor_to_put


class OptimizerState(Stateful):
    """
    A wrapper around an optimizer to make it stateful.

    Args:
        optimizer (Optimizer): optimizer to wrap.
    """

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # MultiOptimizer is only used for EP+FSDP2 case for now,
        # and it knows how to produce a merged, flattened dict already
        if getattr(self.optimizer, "_is_multi_optimizer", False):
            return {"optim": self.optimizer.state_dict()}

        # Single torch optimizer
        sd = get_optimizer_state_dict(model=self.model, optimizers=self.optimizer)
        return {"optim": sd}

    def load_state_dict(self, state_dict):
        optim_state = state_dict["optim"]

        # Delegate to MultiOptimizer (it will split/filter correctly)
        if getattr(self.optimizer, "_is_multi_optimizer", False):
            self.optimizer.load_state_dict(optim_state)
            return

        # Single torch optimizer
        set_optimizer_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            optim_state_dict=optim_state,
        )


def build_checkpointer(
    dist_backend: str = "fsdp1",
    ckpt_manager: str = "omnistore",
):
    """
    create a checkpointer manager with given mode.
    Args:
        dist_backend (str, optional): checkpoint mode. Defaults to "fsdp1".
            fsdp1: FSDP1 checkpointer
            fsdp2: FSDP2 checkpointer
            ddp: DDP checkpointer
            dcp: DCP checkpoint from torch.distributed.checkpoint
            native: native checkpoint from torch.save
        ckpt_manager (str, optional): checkpoint manager. Defaults to "bytecheckpoint".
            omnistore: omnistore checkpoint manager
            bytecheckpoint: byted checkpoint manager
            dcp: torch dcp checkpoint manager
    Raises:
        ValueError: if ckpt_manager is not supported

    Returns:
        Checkpointer: checkpointer with given mode.
    """

    if ckpt_manager == "omnistore":
        if dist_backend == "ddp":
            from omnistore import DDPCheckpointer as Checkpointer
        elif dist_backend == "fsdp1":
            from omnistore import FSDPCheckpointer as Checkpointer
        elif dist_backend == "fsdp2":
            from omnistore import FSDP2Checkpointer as Checkpointer
    elif ckpt_manager == "bytecheckpoint":
        if dist_backend == "ddp":
            from bytecheckpoint import DDPCheckpointer as Checkpointer
        elif dist_backend == "fsdp1":
            from bytecheckpoint import FSDPCheckpointer as Checkpointer
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
        raise ValueError(f"Unknown checkpoint manager: {ckpt_manager}, supported modes are: omnistore, dcp, native")

    return Checkpointer


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


class DistributedCheckpointer(CheckpointerBase):
    """
    Distributed checkpointer for torch.distributed.checkpoint
    """

    dcp_save_future: Optional[Any] = None
    # Dedicated process group for async saves (created on first use)
    _async_process_group: Optional[Any] = None

    @classmethod
    def save(
        cls,
        path: str,
        state: Dict[str, Any],
        save_async: bool = False,
        global_steps: int = None,
    ) -> None:
        """
        save training state to distributed checkpoint

        args:
            path: path to save checkpoint
            state: state to save
            global_steps: global steps
        return:
            None
        """

        checkpoint_dir = f"{path}/{_GLOBAL_STEP_PREFIX}{global_steps}" if global_steps else path
        os.makedirs(checkpoint_dir, exist_ok=True)

        if "model" not in state:
            raise ValueError("Model must be provided to save a distributed checkpoint.")

        save_state = {"model": ModelState(state["model"])}
        if "optimizer" in state:
            save_state["optimizer"] = OptimizerState(model=state["model"], optimizer=state["optimizer"])  # type: ignore[index]

        if save_async:
            # Lazily create a dedicated Gloo process group for async DCP saves
            if cls._async_process_group is None:
                cls._async_process_group = dist.new_group(backend="gloo")

            if cls.dcp_save_future is not None:
                logger.info_rank0("waiting for previous DCP saving session to end...")
                cls.dcp_save_future.result()
                cls.dcp_save_future = None

            cls.dcp_save_future = dcp.async_save(
                state_dict=save_state,
                storage_writer=FileSystemWriter(
                    checkpoint_dir,
                    thread_count=16,
                    single_file_per_rank=True,
                    sync_files=False,
                ),
                process_group=cls._async_process_group,
            )
        else:
            dcp.save(
                state_dict=save_state,
                storage_writer=FileSystemWriter(
                    checkpoint_dir,
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

        load_state = {"model": ModelState(state["model"])}
        if "optimizer" in state:
            load_state["optimizer"] = OptimizerState(model=state["model"], optimizer=state["optimizer"])  # type: ignore[index]

        dcp.load(
            state_dict=load_state,
            storage_reader=FileSystemReader(checkpoint_dir),
            process_group=process_group,
        )
        # Note: further per-param DTensor alignment and device fixes happen inside OptimizerState.load_state_dict

        if "extra_state" in state:
            extra_state_dir = os.path.join(checkpoint_dir, _EXTRA_STATE_DIR)
            os.makedirs(extra_state_dir, exist_ok=True)
            extra_state_path = os.path.join(extra_state_dir, _EXTRA_STATE_FORMAT.format(dist.get_rank()))
            state["extra_state"] = torch.load(extra_state_path, weights_only=False)

        logger.info_rank0(f"Loaded checkpoint from {checkpoint_dir}")

        return state
