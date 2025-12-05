import gc
import os
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed._tensor import DeviceMesh, DTensor, Shard
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
)
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint.stateful import Stateful

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from ..utils.checkpoint_utils import _GLOBAL_STEP_PREFIX
from ..utils.device import empty_cache, synchronize
from .checkpointer import CheckpointerBase


logger = logging.get_logger(__name__)

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
            model_state_dict = self.get_state_dict_with_ep_dim_preprocess(model_state_dict, "restore")

        return model_state_dict

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        """
        perform the reverse operation for state_dict()
        need to drop EP-dim when loading from DCP checkpoints
        so that EP-FSDP would not be confused
        """
        model_state_dict = state_dict
        if self.should_ep_aware:
            model_state_dict = self.get_state_dict_with_ep_dim_preprocess(model_state_dict, "drop")

        set_model_state_dict(model=self.model, model_state_dict=model_state_dict)

    def get_state_dict_with_ep_dim_preprocess(self, state_dict, action):
        ep_fqn2spec_info = self.ep_fqn2spec_info
        assert ep_fqn2spec_info is not None, "if fqn2spec_info is None it should not be patch"

        ep_mesh = self.parallel_state.ep_fsdp_device_mesh["ep"]
        assert ep_mesh is not None

        global_device_mesh = self.parallel_state.ep_fsdp_device_mesh
        assert global_device_mesh.ndim == 2

        assert action in ["restore", "drop"]

        keys = list(state_dict.keys())
        for name in sorted(keys):
            if name in ep_fqn2spec_info and isinstance(ep_fqn2spec_info[name].placement, Shard):
                cur_spec_info = ep_fqn2spec_info[name]
                tensor = state_dict[name]
                if action == "drop":
                    tensor = drop_ep_dim(tensor, cur_spec_info.ep_fsdp_mesh)
                else:
                    tensor = restore_ep_dim(tensor, cur_spec_info.ep_fsdp_mesh)
                state_dict[name] = tensor

        return state_dict


class OptimizerState(Stateful):
    """
    A wrapper around an optimizer to make it stateful.

    Args:
        optimizer (Optimizer): optimizer to wrap.
    """

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        # Similar to ModelState, OptimizerState also need to be EP+FSDP2 aware
        self.parallel_state = get_parallel_state()
        self.ep_fqn2spec_info = getattr(self.model, "_fqn2spec_info", None)
        self.should_ep_aware = self.ep_fqn2spec_info is not None and self.parallel_state.dp_mode == "fsdp2"

    def state_dict(self):
        if self.should_ep_aware:
            logger.info_rank0(
                "Getting optimizer state_dict from OptimizerState wrapper, would restore EP dim for Experts module"
            )
            # MultiOptimizer is only used for EP+FSDP2 case for now,
            # and it knows how to produce a merged, flattened dict already
            assert self.optimizer._is_multi_optimizer, "EP is enabled but optimizer is not a MultiOptimizer instance"
            vanilla_optim_sd = self.optimizer.state_dict()
            optim_sd_with_ep_dim = self.get_state_dict_with_ep_dim_preprocess(vanilla_optim_sd, "restore")
            return optim_sd_with_ep_dim

        # Single torch optimizer
        sd = get_optimizer_state_dict(model=self.model, optimizers=self.optimizer)
        return sd

    def load_state_dict(self, state_dict):
        optim_state_from_dcp_load = state_dict
        if self.should_ep_aware:
            # we need to drop EP dim before loading them into optimizers
            optim_state_without_ep_dim = self.get_state_dict_with_ep_dim_preprocess(optim_state_from_dcp_load, "drop")
            # Delegate to MultiOptimizer (it will split/filter correctly)
            self.optimizer.load_state_dict(optim_state_without_ep_dim)
            return

        # Single torch optimizer
        set_optimizer_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            optim_state_dict=optim_state_from_dcp_load,
        )

    def get_state_dict_with_ep_dim_preprocess(self, state_dict, action):
        ep_fqn2spec_info = self.ep_fqn2spec_info
        assert ep_fqn2spec_info is not None, "if fqn2spec_info is None it should not be patch"

        ep_mesh = self.parallel_state.ep_fsdp_device_mesh["ep"]
        assert ep_mesh is not None

        global_device_mesh = self.parallel_state.ep_fsdp_device_mesh
        assert global_device_mesh.ndim == 2

        assert action in ["drop", "restore"]

        keys = list(state_dict.keys())
        ep_keys = list(ep_fqn2spec_info.keys())

        for name in sorted(keys):
            # Find EP spec whose FQN appears in the state_dict key
            # e.g. name = "state.model.layers.0.mlp.experts.gate_proj.step"
            #      ep_key = "model.layers.0.mlp.experts.gate_proj"
            matches = [ep_key for ep_key in ep_keys if ep_key in name]
            if not matches:
                # ignore non-ep tensor
                continue

            # each tensor in the state dict should only belong to one EP entry
            assert len(matches) == 1, f"Ambiguous EP spec match for state key '{name}': {matches}"

            ep_key = matches[0]
            cur_spec_info = ep_fqn2spec_info[ep_key]

            # skip non-ep params which has Replicate placement in model spec info
            if not isinstance(cur_spec_info.placement, Shard):
                continue

            tensor = state_dict[name]
            if not torch.is_tensor(tensor):
                # we skip param-group hyperparams like `param_groups.model.layers.0.mlp.experts.down_proj.amsgrad`
                continue
            # Skip scalars (0-D tensors) – cannot be sharded on dim 0
            if tensor.ndim == 0:
                continue

            if action == "drop":
                tensor = drop_ep_dim(tensor, cur_spec_info.ep_fsdp_mesh)
            elif action == "restore":
                tensor = restore_ep_dim(tensor, cur_spec_info.ep_fsdp_mesh)
            state_dict[name] = tensor

        return state_dict


def drop_ep_dim(loaded_tensor: torch.Tensor, device_mesh: DeviceMesh):
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


def restore_ep_dim(orgin_tensor: torch.Tensor, device_mesh: DeviceMesh):
    """
    Restore EP dim so that DCP can be aware about EP ranks

    args:
        orgin_tensor (torch.Tensor): The orgin tensor.
        device_mesh (DeviceMesh): The ep device mesh.
        shard (Shard): The shard info, default Shard(0).

    """
    assert device_mesh.ndim == 2, f"global_mesh.ndim must be 2, got {device_mesh.ndim}"
    ep_mesh = device_mesh["ep"]

    if isinstance(orgin_tensor, DTensor):
        # EP+FSDP2
        dtensor = DTensor.from_local(
            orgin_tensor._local_tensor, device_mesh=device_mesh, placements=[Shard(0), Shard(1)]
        )
    elif torch.is_tensor(orgin_tensor):
        # If there is no FSDP but only EP
        dtensor = DTensor.from_local(orgin_tensor, device_mesh=ep_mesh, placements=[Shard(0)])
    else:
        raise RuntimeError(f"origin_tensor - {orgin_tensor} is not a tensor!")

    return dtensor


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
        storage_writer: Optional[FileSystemWriter] = None,
    ) -> None:
        """
        save training state to distributed checkpoint

        args:
            path: path to save checkpoint
            state: state to save
            save_async: whether to save asynchronously
            global_steps: global steps
            storage_writer: storage writer backend for dcp.save and dcp.async_save. If None, will use FileSystemWriter
        return:
            None
        """

        checkpoint_dir = f"{path}/{_GLOBAL_STEP_PREFIX}{global_steps}" if global_steps else path
        os.makedirs(checkpoint_dir, exist_ok=True)

        # saving extra_state first to gurantee that every saved model/optimizer ckpts have their extra_state saved before them
        if "extra_state" in state:
            extra_state_dir = os.path.join(checkpoint_dir, _EXTRA_STATE_DIR)
            os.makedirs(extra_state_dir, exist_ok=True)
            extra_state_path = os.path.join(extra_state_dir, _EXTRA_STATE_FORMAT.format(dist.get_rank()))
            torch.save(
                state["extra_state"],
                extra_state_path,
            )

        if "model" not in state:
            raise ValueError("Model must be provided to save a distributed checkpoint.")

        save_state = {"model": ModelState(state["model"])}
        if "optimizer" in state:
            save_state["optimizer"] = OptimizerState(model=state["model"], optimizer=state["optimizer"])  # type: ignore[index]

        if storage_writer is None:
            storage_writer = FileSystemWriter(
                checkpoint_dir,
                thread_count=16,
                single_file_per_rank=True,
                sync_files=False,
            )

        if save_async:
            # Lazily create a dedicated Gloo process group for async DCP saves
            if cls._async_process_group is None:
                cls._async_process_group = dist.new_group(backend="gloo")

            if cls.dcp_save_future is not None:
                logger.info(f"[RANK {dist.get_rank()}] waiting for previous DCP saving session to end...")
                cls.dcp_save_future.result()
                cls.dcp_save_future = None
                # block until all the ranks resolve their previous dcp async saving
                dist.barrier()

            cls.dcp_save_future = dcp.async_save(
                state_dict=save_state,
                storage_writer=storage_writer,
                process_group=cls._async_process_group,
            )
        else:
            dcp.save(
                state_dict=save_state,
                storage_writer=storage_writer,
            )
            if dist.is_initialized():
                dist.barrier()
            gc.collect()
            empty_cache()
            synchronize()

        logger.info_rank0(f"Saved checkpoint to {checkpoint_dir}")

    @classmethod
    def load(
        cls,
        path: str,
        state: Dict[str, Any],
        process_group=None,
        storage_reader: Optional[FileSystemReader] = None,
    ) -> Dict[str, Any]:
        """
        load training state from distributed checkpoint
        args:
            path: path to load checkpoint
            state: state to load, "model" are required,  "optimizer" and "extra_state" are optional
            process_group: process group for loading checkpoint
            storage_reader: storage reader backend for dcp.load. If None, will use FileSystemReader

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

        if storage_reader is None:
            storage_reader = FileSystemReader(checkpoint_dir)

        dcp.load(
            state_dict=load_state,
            storage_reader=storage_reader,
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
