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


import gc
import os
from collections import OrderedDict
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed._tensor import DeviceMesh, DTensor, Shard
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load,
)
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE, Metadata
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from ..utils.checkpoint_utils import _GLOBAL_STEP_PREFIX
from ..utils.device import empty_cache, synchronize
from .checkpointer import CheckpointerBase


logger = logging.get_logger(__name__)

_EXTRA_STATE_FORMAT = "extra_state_rank_{}.pt"
_EXTRA_STATE_DIR = "extra_state"


def _validate_extra_parallel_meshes(parallel_state) -> None:
    """Fail-fast precondition for ExtraParallel state dict preprocessing.

    At least one ExtraParallel mesh must be non-None, and at least one
    of those meshes must be 2D (the ExtraParallel + FSDP composition).
    """
    extra_parallel_mesh = {
        para: parallel_state.extra_parallel_fsdp_device_mesh[para][para]
        if parallel_state.extra_parallel_fsdp_device_mesh[para] is not None
        else None
        for para in parallel_state.extra_parallel_names
    }
    assert any(m is not None for m in extra_parallel_mesh.values()), (
        "At least one extra_parallel mesh should be not None"
    )
    assert any(
        parallel_state.extra_parallel_fsdp_device_mesh[para] is not None
        and parallel_state.extra_parallel_fsdp_device_mesh[para].ndim == 2
        for para in parallel_state.extra_parallel_names
    ), "At least one extra_parallel fsdp_device_mesh should be not None"


def _apply_extra_parallel_dim(
    state_dict: Dict[str, Any],
    extra_parallel_fqn2spec_info: Dict[str, Any],
    parallel_state,
    action: str,
    *,
    key_match: str,
) -> Dict[str, Any]:
    """Drop or restore the ExtraParallel dimension on each tensor in a state dict.

    Shared by ``ModelState`` and ``OptimizerState``.  The only meaningful
    difference between the two callers is how state-dict keys map to
    ExtraParallel FQNs:

    * ``"exact"`` (model): the state-dict key IS the FQN,
      e.g. ``"model.layers.0.mlp.experts.gate_proj"``.
    * ``"substring"`` (optimizer): the state-dict key contains the FQN
      with extra prefix/suffix, e.g.
      ``"state.model.layers.0.mlp.experts.gate_proj.exp_avg"``.

    Non-tensor values and 0-D tensors are skipped unconditionally — they
    appear only in optimizer state dicts (param-group hyperparams, scalar
    ``step`` tensors); model state dicts never contain them, so the guard
    is a safe no-op there.
    """
    assert action in ("drop", "restore"), f"action must be 'drop' or 'restore', got {action!r}"
    assert key_match in ("exact", "substring"), f"key_match must be 'exact' or 'substring', got {key_match!r}"
    assert extra_parallel_fqn2spec_info is not None, "fqn2spec_info must not be None"

    _validate_extra_parallel_meshes(parallel_state)

    extra_parallel_keys = list(extra_parallel_fqn2spec_info.keys()) if key_match == "substring" else None

    for name in sorted(state_dict.keys()):
        if key_match == "exact":
            if name not in extra_parallel_fqn2spec_info:
                continue
            spec_info = extra_parallel_fqn2spec_info[name]
        else:  # "substring"
            matches = [k for k in extra_parallel_keys if k in name]
            if not matches:
                continue
            assert len(matches) == 1, f"Ambiguous ExtraParallel spec match for state key '{name}': {matches}"
            spec_info = extra_parallel_fqn2spec_info[matches[0]]

        if not isinstance(spec_info.placement, Shard):
            continue

        tensor = state_dict[name]
        if not torch.is_tensor(tensor):
            continue
        if tensor.ndim == 0:
            continue

        assert spec_info.para_fsdp_mesh is not None, f"ExtraParallel spec {name} must have an ExtraParallel FSDP mesh"

        fsdp_submesh = spec_info.para_fsdp_mesh[f"{spec_info.para_name}_fsdp"]
        if action == "drop":
            tensor = drop_extra_parallel_dim(tensor, fsdp_submesh)
        else:
            tensor = restore_extra_parallel_dim(tensor, spec_info.para_fsdp_mesh, fsdp_submesh)
        state_dict[name] = tensor

    return state_dict


class ModelState(Stateful):
    """A wrapper around a model to make it stateful.

    Args:
        model: model to wrap.
        trainable_only: when ``True`` the state_dict only contains parameters with
            ``requires_grad=True`` (uses ``StateDictOptions(ignore_frozen_params=True)``).
            This is the LoRA / PEFT path: frozen base weights are skipped on save and
            ``set_model_state_dict`` runs in ``strict=False`` mode on load so the
            (already populated from ``model_path``) base params are left untouched.
    """

    def __init__(self, model, trainable_only: bool = False):
        self.model = model
        self.trainable_only = trainable_only

        # Determine whether this is ExtraParallel+FSDP2 case
        # If so, we need to restore Para(e.g. EP)-dim before saving to DCP
        self.parallel_state = get_parallel_state()
        self.extra_parallel_fqn2spec_info = getattr(self.model, "_fqn2spec_info", None)
        self.should_extra_parallel_aware = (
            self.extra_parallel_fqn2spec_info is not None and self.parallel_state.dp_mode == "fsdp2"
        )

    @torch.no_grad()
    def state_dict(self):
        options = StateDictOptions(ignore_frozen_params=True) if self.trainable_only else None
        model_state_dict = get_model_state_dict(model=self.model, options=options)
        if self.should_extra_parallel_aware:
            logger.info_rank0(
                "Getting model state_dict from ModelState wrapper, would restore ExtraParallel dim for ExtraParallel (e.g. Experts/Embeds) module"
            )
            # As fsdp+extra parallel and pure extra parallel have different placements, e.g. [Shard(0), Shard(1)] and [Shard(0)],
            # restoring state dict should be extra parallel aware.
            model_state_dict = self.get_state_dict_with_extra_parallel_dim_preprocess(model_state_dict, "restore")

        return model_state_dict

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        """
        perform the reverse operation for state_dict()
        need to drop ExtraParallel-dim when loading from DCP checkpoints
        so that ExtraParallel-FSDP would not be confused
        """
        model_state_dict = state_dict
        if self.should_extra_parallel_aware:
            model_state_dict = self.get_state_dict_with_extra_parallel_dim_preprocess(model_state_dict, "drop")

        options = StateDictOptions(strict=False) if self.trainable_only else None
        set_model_state_dict(model=self.model, model_state_dict=model_state_dict, options=options)

    def get_state_dict_with_extra_parallel_dim_preprocess(self, state_dict, action):
        return _apply_extra_parallel_dim(
            state_dict,
            self.extra_parallel_fqn2spec_info,
            self.parallel_state,
            action,
            key_match="exact",
        )


class OptimizerState(Stateful):
    """A wrapper around an optimizer to make it stateful.

    On save, only optimizer state that actually exists is persisted — params
    that never received a gradient (e.g. unused MoE experts, frozen LoRA
    base weights) are simply absent from the checkpoint.

    On load, ``allow_partial_load=True`` is passed to the DCP load planner
    so missing optimizer entries are skipped.  For a fresh optimizer (the
    normal resume path), ``set_optimizer_state_dict`` internally calls
    ``_init_optim_state`` which pre-fills zero/default state for every
    param; DCP then overwrites the entries that exist in the checkpoint.
    Params absent from the checkpoint keep their default-initialised state,
    equivalent to what AdamW would create on the next ``step()`` call.

    Note: ``allow_partial_load`` is set globally on the DCP planner (it
    cannot be scoped to optimizer-only).  Model-weight integrity is still
    enforced by ``set_model_state_dict(strict=True)`` inside
    ``ModelState.load_state_dict`` for non-LoRA loads.
    """

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.parallel_state = get_parallel_state()
        self.extra_parallel_fqn2spec_info = getattr(self.model, "_fqn2spec_info", None)
        self.should_extra_parallel_aware = (
            self.extra_parallel_fqn2spec_info is not None and self.parallel_state.dp_mode == "fsdp2"
        )

    def state_dict(self):
        if self.should_extra_parallel_aware:
            logger.info_rank0(
                "Getting optimizer state_dict from OptimizerState wrapper, would restore ExtraParallel dim for Experts module"
            )
            assert self.optimizer._is_multi_optimizer, (
                "ExtraParallel is enabled but optimizer is not a MultiOptimizer instance"
            )
            vanilla_optim_sd = self.optimizer.state_dict()
            optim_sd_with_extra_parallel_dim = self.get_state_dict_with_extra_parallel_dim_preprocess(
                vanilla_optim_sd, "restore"
            )
            return optim_sd_with_extra_parallel_dim

        return get_optimizer_state_dict(model=self.model, optimizers=self.optimizer)

    def load_state_dict(self, state_dict):
        optim_state_from_dcp_load = state_dict
        if self.should_extra_parallel_aware:
            # we need to drop ExtraParallel dim before loading them into optimizers
            optim_state_without_extra_parallel_dim = self.get_state_dict_with_extra_parallel_dim_preprocess(
                optim_state_from_dcp_load, "drop"
            )
            # Delegate to MultiOptimizer (it will split/filter correctly)
            self.optimizer.load_state_dict(optim_state_without_extra_parallel_dim)
            return

        # Single torch optimizer
        set_optimizer_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            optim_state_dict=optim_state_from_dcp_load,
        )

    def get_state_dict_with_extra_parallel_dim_preprocess(self, state_dict, action):
        return _apply_extra_parallel_dim(
            state_dict,
            self.extra_parallel_fqn2spec_info,
            self.parallel_state,
            action,
            key_match="substring",
        )


def drop_extra_parallel_dim(loaded_tensor: torch.Tensor, device_mesh: DeviceMesh):
    """
    Drop ExtraParallel dims after loading from DCP so that ExtraParallel-FSDP would not be confused
    """

    if len(loaded_tensor.placements) == 2:
        tensor_to_put = DTensor.from_local(loaded_tensor._local_tensor, device_mesh=device_mesh, placements=[Shard(1)])
    elif len(loaded_tensor.placements) == 1:
        tensor_to_put = loaded_tensor.to_local()
    else:
        raise RuntimeError(
            f"Expect ExtraParallel paramters from checkpoints to be DTensor with 1-dim (no FSDP) or 2-dim (ExtraParallel+FSDP), got {loaded_tensor}"
        )

    return tensor_to_put


def restore_extra_parallel_dim(
    orgin_tensor: torch.Tensor, fsdp_mesh: DeviceMesh, extra_parallel_fsdp_mesh: DeviceMesh
):
    """
    Restore ExtraParallel dim so that DCP can be aware about ExtraParallel ranks

    args:
        orgin_tensor (torch.Tensor): The orgin tensor.
        fsdp_mesh (DeviceMesh): The extra_parallel fsdp device mesh.
        shard (Shard): The shard info, default Shard(0).

    """
    assert fsdp_mesh.ndim == 2, f"global_mesh.ndim must be 2, got {fsdp_mesh.ndim}"

    if isinstance(orgin_tensor, DTensor):
        # ExtraParallel+FSDP2
        dtensor = DTensor.from_local(
            orgin_tensor._local_tensor, device_mesh=fsdp_mesh, placements=[Shard(0), Shard(1)]
        )
    elif torch.is_tensor(orgin_tensor):
        # If there is no FSDP but only ExtraParallel
        dtensor = DTensor.from_local(orgin_tensor, device_mesh=extra_parallel_fsdp_mesh, placements=[Shard(0)])
    else:
        raise RuntimeError(f"origin_tensor - {orgin_tensor} is not a tensor!")

    return dtensor


class DistributedCheckpointer(CheckpointerBase):
    """
    Distributed checkpointer for torch.distributed.checkpoint
    """

    save_future: Optional[Any] = None
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
        trainable_only: bool = False,
    ) -> None:
        """
        save training state to distributed checkpoint

        args:
            path: path to save checkpoint
            state: state to save
            save_async: whether to save asynchronously
            global_steps: global steps
            storage_writer: storage writer backend for dcp.save and dcp.async_save. If None, will use FileSystemWriter
            trainable_only: when True, only persist parameters with ``requires_grad=True``
                (LoRA / PEFT path). Frozen base weights are skipped on save and must be
                re-materialised from ``model.model_path`` at resume time. The optimizer
                state is already trainable-only by construction (the optimizer is built
                from ``filter(lambda p: p.requires_grad, ...)``), so this flag only
                affects the model state dump.
        return:
            None
        """
        if "model" not in state:
            raise ValueError("Model must be provided to save a distributed checkpoint.")

        checkpoint_dir = f"{path}/{_GLOBAL_STEP_PREFIX}{global_steps}" if global_steps else path
        cls._create_checkpoint_dir(checkpoint_dir)

        # saving extra_state first to gurantee that every saved model/optimizer ckpts have their extra_state saved before them
        cls._save_extra_state(checkpoint_dir=checkpoint_dir, state=state)

        save_state = {"model": ModelState(state["model"], trainable_only=trainable_only)}
        if "optimizer" in state:
            save_state["optimizer"] = OptimizerState(model=state["model"], optimizer=state["optimizer"])

        if storage_writer is None:
            storage_writer = cls._create_storage_writer(checkpoint_dir)

        cls.execute_save(save_state=save_state, storage_writer=storage_writer, save_async=save_async)

        logger.info_rank0(f"Saved checkpoint to {checkpoint_dir}")

    @classmethod
    def load(
        cls,
        path: str,
        state: Dict[str, Any],
        process_group=None,
        storage_reader: Optional[FileSystemReader] = None,
        trainable_only: bool = False,
    ) -> Dict[str, Any]:
        """
        load training state from distributed checkpoint
        args:
            path: path to load checkpoint
            state: state to load, "model" are required,  "optimizer" and "extra_state" are optional
            process_group: process group for loading checkpoint
            storage_reader: storage reader backend for dcp.load. If None, will use FileSystemReader
            trainable_only: when True, ``set_model_state_dict`` runs in non-strict
                mode (``StateDictOptions(strict=False)``). Use this for LoRA / PEFT
                resumes where the DCP only contains trainable adapter weights and the
                frozen base must come from ``model.model_path``. Safe to enable when
                the DCP is full (extra strictness is just dropped).

        return:
            state: state loaded
        """
        checkpoint_dir = path

        if state is None:
            raise ValueError("State dict must be provided to load a distributed checkpoint.")

        if "model" not in state:
            raise ValueError("Model must be provided to load a distributed checkpoint.")

        load_state = {"model": ModelState(state["model"], trainable_only=trainable_only)}
        if "optimizer" in state:
            load_state["optimizer"] = OptimizerState(model=state["model"], optimizer=state["optimizer"])  # type: ignore[index]

        if storage_reader is None:
            storage_reader = cls._create_storage_reader(checkpoint_dir)

        dcp.load(
            state_dict=load_state,
            storage_reader=storage_reader,
            process_group=process_group,
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )

        cls._load_extra_state(checkpoint_dir=checkpoint_dir, state=state)

        logger.info_rank0(f"Loaded checkpoint from {checkpoint_dir}")

        return state

    @classmethod
    def wait_for_pending_save(cls) -> None:
        """Block until any pending async DCP save completes.

        Safe to call when no save is pending (no-op).  Re-raises any
        exception from the pending save after logging which rank saw it.
        After completion, all ranks synchronize via a barrier so callers
        can safely begin a new collective operation.

        This is the single entrypoint for all async-save coordination —
        prefer calling this over poking ``save_future`` directly.
        """
        if cls.save_future is None:
            return
        rank = dist.get_rank() if dist.is_initialized() else 0
        try:
            logger.info(f"[RANK {rank}] waiting for previous DCP saving session to end...")
            cls.save_future.result()
        except Exception:
            logger.error(f"[RANK {rank}] previous async DCP save raised; propagating", exc_info=True)
            raise
        finally:
            cls.save_future = None
        if dist.is_initialized():
            dist.barrier()

    @classmethod
    def execute_save(
        cls,
        save_state: Dict[str, Any],
        storage_writer: FileSystemWriter,
        save_async: bool,
    ) -> None:
        """Execute DCP save with optional async support."""
        if save_async:
            # Lazily create a dedicated Gloo process group for async DCP saves
            if cls._async_process_group is None:
                cls._async_process_group = dist.new_group(backend="gloo")

            cls.wait_for_pending_save()

            cls.save_future = dcp.async_save(
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

    # Private helper methods
    @classmethod
    def _create_checkpoint_dir(cls, checkpoint_dir: str) -> None:
        """Create checkpoint directory."""
        os.makedirs(checkpoint_dir, exist_ok=True)

    @classmethod
    def _create_storage_reader(cls, checkpoint_dir: str) -> FileSystemReader:
        """Create storage reader for DCP."""
        return FileSystemReader(checkpoint_dir)

    @classmethod
    def _create_storage_writer(cls, checkpoint_dir: str) -> FileSystemWriter:
        """Create storage writer for DCP."""
        return FileSystemWriter(
            checkpoint_dir,
            thread_count=16,
            single_file_per_rank=True,
            sync_files=False,
        )

    @classmethod
    def _save_extra_state(cls, checkpoint_dir: str, state: Dict[str, Any]) -> None:
        """Save extra_state to checkpoint directory."""
        if "extra_state" not in state:
            logger.warning_rank0("extra_state not found in state, skipping extra_state save")
            return

        extra_state_dir = os.path.join(checkpoint_dir, _EXTRA_STATE_DIR)
        os.makedirs(extra_state_dir, exist_ok=True)
        extra_state_path = os.path.join(extra_state_dir, _EXTRA_STATE_FORMAT.format(dist.get_rank()))
        torch.save(
            state["extra_state"],
            extra_state_path,
        )

    @classmethod
    def _load_extra_state(cls, checkpoint_dir: str, state: Dict[str, Any]) -> None:
        """Load extra_state from checkpoint directory."""
        if "extra_state" not in state:
            logger.warning_rank0("extra_state not found in state, skipping extra_state load")
            return

        extra_state_dir = os.path.join(checkpoint_dir, _EXTRA_STATE_DIR)
        os.makedirs(extra_state_dir, exist_ok=True)
        extra_state_path = os.path.join(extra_state_dir, _EXTRA_STATE_FORMAT.format(dist.get_rank()))
        state["extra_state"] = torch.load(extra_state_path, weights_only=False)


def get_dtype_size(dtype: torch.dtype) -> int:
    """Return size in bytes for a given dtype."""
    size_map = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int64: 8,
        torch.int32: 4,
        torch.int16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.bool: 1,
    }
    return size_map.get(dtype, 4)


def _normalize_key(key: str) -> Optional[str]:
    """
    Convert DCP key to HuggingFace format. Returns None for non-model weights.

    Conversion rules:
    - "model.model.*" -> "model.*" (remove first "model." prefix)
    - "model.lm_head.weight" -> "lm_head.weight" (special case)
    - Other "model.*" keys -> log warning and strip "model." prefix
    """
    if not key.startswith("model."):
        return None

    if key.startswith("model.model."):
        # Standard case: model.model.* -> model.*
        return key[6:]  # Remove first "model." prefix
    elif key == "model.lm_head.weight":
        # Special case: model.lm_head.weight -> lm_head.weight
        return "lm_head.weight"
    else:
        # Other keys with single "model." prefix - log and strip prefix
        logger.warning(
            f"Found key with single 'model.' prefix that doesn't match expected patterns: '{key}'. "
            f"Converting to '{key[6:]}' by stripping 'model.' prefix."
        )
        return key[6:]


def _get_sharding_plan(
    checkpoint_path: Union[str, os.PathLike],
    shard_size: int = None,
    save_dtype: Optional[Union[str, torch.dtype]] = None,
):
    """
    Create sharding plan from checkpoint metadata without loading weights.

    Returns:
        shards: List of {hf_key: dcp_key} dicts per shard
        total_size: Total size in bytes
        all_dcp_keys: All valid DCP model keys
    """
    reader = FileSystemReader(checkpoint_path)
    metadata = reader.read_metadata()

    if not isinstance(metadata, Metadata):
        raise ValueError(f"Invalid metadata format in {checkpoint_path}")

    # Collect model tensors and calculate sizes
    tensor_infos = []
    all_dcp_keys = []

    for key, tensor_meta in metadata.state_dict_metadata.items():
        hf_key = _normalize_key(key)
        if hf_key:
            # Determine dtype for size calculation
            if save_dtype:
                dtype = getattr(torch, save_dtype) if isinstance(save_dtype, str) else save_dtype
            else:
                if not hasattr(tensor_meta.properties, "dtype"):
                    raise ValueError(
                        f"Cannot determine dtype for tensor '{key}': metadata does not contain dtype information"
                    )
                dtype = tensor_meta.properties.dtype

            # Calculate tensor size in bytes
            numel = 1
            for dim in tensor_meta.size:
                numel *= dim

            byte_size = numel * get_dtype_size(dtype)

            tensor_infos.append({"dcp_key": key, "hf_key": hf_key, "size": byte_size, "metadata": tensor_meta})
            all_dcp_keys.append(key)

    # Sort by key name for deterministic output
    tensor_infos.sort(key=lambda x: x["hf_key"])

    # Pack tensors into shards
    shards = []
    current_shard = {}
    current_shard_size = 0
    total_size = 0

    for info in tensor_infos:
        size = info["size"]
        total_size += size

        # Start new shard if adding this tensor exceeds shard_size (unless current shard is empty)
        if shard_size is not None and current_shard and (current_shard_size + size > shard_size):
            shards.append(current_shard)
            current_shard = {}
            current_shard_size = 0

        current_shard[info["hf_key"]] = info["dcp_key"]
        current_shard_size += size

    if current_shard:
        shards.append(current_shard)
    if shard_size is None:
        assert len(shards) == 1, "Shard size None should result in a single shard"
        shards = shards[0]
    return shards, total_size, all_dcp_keys


def _process_shard(
    shard_keys: Dict[str, str],
    checkpoint_path: str,
    save_dtype: Optional[Union[str, torch.dtype]] = None,
) -> str:
    reader = FileSystemReader(checkpoint_path)
    metadata = reader.read_metadata()

    state_dict = OrderedDict()
    dcp_keys_to_load = list(shard_keys.values())

    for dcp_key in dcp_keys_to_load:
        tensor_metadata = metadata.state_dict_metadata[dcp_key]
        if not hasattr(tensor_metadata.properties, "dtype"):
            raise ValueError(
                f"Cannot determine dtype for tensor '{dcp_key}': metadata does not contain dtype information"
            )
        state_dict[dcp_key] = torch.empty(
            tensor_metadata.size,
            dtype=tensor_metadata.properties.dtype,
        )

    # Load partial checkpoint
    load(
        state_dict,
        checkpoint_id=checkpoint_path,
        storage_reader=FileSystemReader(checkpoint_path),
        no_dist=True,
    )

    # Cast and rename tensors
    processed_dict = OrderedDict()
    target_dtype = None
    if save_dtype:
        target_dtype = getattr(torch, save_dtype) if isinstance(save_dtype, str) else save_dtype

    for hf_key, dcp_key in shard_keys.items():
        tensor = state_dict[dcp_key]

        if hasattr(tensor, "full_tensor"):
            tensor = tensor.full_tensor()

        if target_dtype:
            tensor = tensor.to(dtype=target_dtype)

        # Explicitly move to CPU and detach to avoid memory retention
        processed_dict[hf_key] = tensor.cpu().detach().clone()
        # Delete the original tensor immediately
        del tensor

    # Clean up state_dict and force garbage collection
    del state_dict
    del metadata
    del reader
    gc.collect()
    empty_cache()
    return processed_dict


def dcp_to_torch_state_dict(save_checkpoint_path: Union[str, os.PathLike]) -> STATE_DICT_TYPE:
    """
    Given a directory containing a DCP checkpoint, this function will convert it into a
    Torch state_dict.

    Args:
        save_checkpoint_path: Directory containing the DCP checkpoint.

    .. warning::
        To avoid OOM, it's recommended to only run this function on a single rank.
    """
    shard, _, _ = _get_sharding_plan(save_checkpoint_path)

    processed_dict = _process_shard(shard, save_checkpoint_path)

    return processed_dict
