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
import json
import os
import re
import shutil
import struct
import time
from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Set

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from veomni.checkpoint import ckpt_to_state_dict, layout
from veomni.models import save_model_assets, save_model_weights
from veomni.models.module_utils import _save_state_dict
from veomni.utils import helper
from veomni.utils.device import IS_NPU_AVAILABLE, synchronize
from veomni.utils.import_utils import is_torch_version_greater_than


logger = helper.create_logger(__name__)


def _attach_weight_conversions_from_mro(model: torch.nn.Module) -> None:
    """Recover class-name converters after FSDP2 wraps the module class.

    ``fully_shard`` builds a subclass whose ``__name__`` no longer matches the
    registered mapping. ``model_type`` fallback is wrong for Bagel: eager and
    accelerated share ``bagel_qwen2_mot``, but only accelerated exports fused
    QKV through converters. Walk the MRO until a class-name mapping hits.
    """
    if getattr(model, "_weight_conversions", None) is not None:
        return

    from transformers.conversion_mapping import get_checkpoint_conversion_mapping

    for cls in type(model).__mro__:
        conversions = get_checkpoint_conversion_mapping(cls.__name__)
        if conversions is not None:
            model._weight_conversions = list(conversions)
            return


def _revert_model_weight_conversions_for_hf(
    model: Optional[torch.nn.Module],
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if model is None or not getattr(model, "_export_hf_checkpoint_with_weight_conversions", False):
        return state_dict

    from transformers.core_model_loading import revert_weight_conversion

    _attach_weight_conversions_from_mro(model)
    return revert_weight_conversion(model, state_dict)


@torch.no_grad()
def get_model_save_state(
    model: torch.nn.Module,
    fqn_to_index_mapping: Optional[Dict[str, int]],
    parallel_state=None,
) -> Dict[str, torch.Tensor]:
    """Build a flat state dict suitable for HuggingFace safetensors saving.

    1. Extracts a flat state dict via ``ModelState`` (FQNs match HF weight_map keys).
    2. Casts float32 tensors to bfloat16 on copies (original model dtypes are preserved).
    3. Filters out tied weights not present in ``fqn_to_index_mapping``.
    """
    from veomni.checkpoint.dcp_checkpointer import ModelState

    # Use flat state dict so DCP FQNs match the original HF weight_map keys
    # (e.g. "model.embed_tokens.weight" instead of "model.model.embed_tokens.weight")
    save_state = ModelState(model, parallel_state=parallel_state).state_dict()
    save_state = _revert_model_weight_conversions_for_hf(model, save_state)

    # Convert float32 tensors to bfloat16 on a copy of the state dict,
    # so the original model parameters remain unchanged.
    converted_state = {}
    for k, v in save_state.items():
        if v.dtype == torch.float32:
            logger.info_rank0(f"Converting {k} from {v.dtype} to torch.bfloat16")
            converted_state[k] = v.to(torch.bfloat16)
        else:
            converted_state[k] = v
    save_state = converted_state

    # Remove tied weights not present in the HF weight_map
    # (e.g. lm_head.weight is tied to model.embed_tokens.weight via tie_word_embeddings)
    if fqn_to_index_mapping is not None:
        filtered_state = {}
        for k, v in save_state.items():
            if k in fqn_to_index_mapping:
                filtered_state[k] = v
            else:
                logger.info_rank0(f"Skipping weight not in HF weight_map: {k}")
        save_state = filtered_state
    else:
        logger.info_rank0(
            f"fqn_to_index_mapping is None, splitting the weights into safetensors files of up to "
            f"{_MAX_SHARD_BYTES / 1e9:.0f} GB."
        )

    return save_state


_MAX_SHARD_BYTES = 5_000_000_000


def _size_balanced_index_mapping(save_state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """1-based output file index per FQN, packing tensors in order into files of up to ``_MAX_SHARD_BYTES``.

    Sizes are those of the full (unsharded) tensors, so every rank derives the same mapping.
    """
    mapping, index, file_bytes = {}, 1, 0
    for fqn, tensor in save_state.items():
        size = tensor.numel() * tensor.element_size()
        if file_bytes and file_bytes + size > _MAX_SHARD_BYTES:
            index, file_bytes = index + 1, 0
        mapping[fqn] = index
        file_bytes += size
    return mapping


# One rank writes each output file, and one tensor cannot span files: a single
# tens-of-GB tensor through an HDFS mount (~50 MB/s a stream) runs far past the
# 10 min NCCL watchdog the other ranks would otherwise wait under.
_CONSOLIDATION_TIMEOUT = timedelta(hours=4)
_consolidation_group = None


def _consolidation_process_group():
    global _consolidation_group
    if _consolidation_group is None:
        _consolidation_group = dist.new_group(backend="gloo", timeout=_CONSOLIDATION_TIMEOUT)
    return _consolidation_group


def _whole_rows(tensor: torch.Tensor) -> bool:
    """True when every shard of ``tensor`` is a run of complete dim-0 rows.

    An FSDP-wrapped table is also split along dim 1, so this holds only under
    ``extra_parallel_only``. ``_StridedShard`` subclasses ``Shard`` but its
    local rows are not contiguous.
    """
    from torch.distributed.tensor import DTensor, Replicate, Shard

    if not isinstance(tensor, DTensor):
        return True
    return all(isinstance(p, Replicate) or (type(p) is Shard and p.dim == 0) for p in tensor.placements)


def _deferred_merge_fqns(model: torch.nn.Module, save_state: Dict[str, torch.Tensor]) -> Set[str]:
    """Row-sharded ``emb`` tensors too large for one output file.

    Consolidating one means a single rank streaming the whole table into one
    file while every other rank waits. Their per-rank shards are kept instead,
    and :func:`merge_pending_hf_shards` writes them out off the training job.
    Only whole-row shards qualify: the merge appends them in row order.
    """
    from torch.distributed.tensor import Shard

    spec_info = getattr(model, "_fqn2spec_info", None) or {}
    deferred = set()
    for fqn, tensor in save_state.items():
        spec = spec_info.get(fqn)
        # A parameter the plan does not name still gets a spec, placed Replicate.
        if spec is None or spec.para_name != "emb" or spec.placement != Shard(0):
            continue
        if tensor.numel() * tensor.element_size() > _MAX_SHARD_BYTES and _whole_rows(tensor):
            deferred.add(fqn)
    return deferred


def _isolate_deferred(fqn_to_index_mapping: Dict[str, int], deferred: Set[str]) -> Dict[str, int]:
    """Give each deferred tensor a file of its own, then number the files ``1..N`` in order."""
    mapping = dict(fqn_to_index_mapping)
    next_index = max(mapping.values()) + 1
    for fqn in sorted(deferred):
        mapping[fqn] = next_index
        next_index += 1
    renumbered = {index: i for i, index in enumerate(sorted(set(mapping.values())), start=1)}
    return {fqn: renumbered[index] for fqn, index in mapping.items()}


_SHARD_FILE = re.compile(r"^shard-\d+-model-(?P<index>\d+)-of-\d+\.safetensors$")


def _consolidate_all_but_deferred(
    sharded_path: str,
    save_path: str,
    fqn_to_index_mapping: Dict[str, int],
    deferred: Dict[str, Dict],
    total_size: int,
) -> None:
    """Consolidate every output file but the deferred ones; leave those as per-rank shards.

    Ranks split the files as ``consolidate_safetensors_files_on_every_rank``
    does. Rank 0 then writes the marker naming the files still to be merged
    (with each tensor's shape and size, which the merge checks against), drops
    the other files' shards, and writes the index for the whole export.
    """
    from torch.distributed.checkpoint._consolidate_hf_safetensors import _consolidate_safetensors_files
    from torch.distributed.checkpoint._hf_utils import _gen_file_name

    distributed = dist.is_initialized()
    rank, world_size = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    num_files = max(fqn_to_index_mapping.values())
    file_of = {fqn: _gen_file_name(index, num_files) for fqn, index in fqn_to_index_mapping.items()}
    mine = {
        fqn: file_of[fqn]
        for fqn, index in fqn_to_index_mapping.items()
        if fqn not in deferred and index % world_size == rank
    }
    if mine:
        _consolidate_safetensors_files(sharded_path, save_path, mine, num_threads=1)
    if distributed:
        dist.barrier(group=_consolidation_process_group())
    if rank != 0:
        return

    # Marker first: until it exists nothing tells a loader or the merge that the
    # index names files that are not there yet.
    pending = {
        "sharded_dir": layout.HF_SHARDED_DIRNAME,
        "tensors": {fqn: {"file": file_of[fqn], **deferred[fqn]} for fqn in sorted(deferred)},
    }
    with open(os.path.join(save_path, layout.HF_PENDING_MERGE_FILENAME), "w") as f:
        json.dump(pending, f, indent=2)
    deferred_indices = {fqn_to_index_mapping[fqn] for fqn in deferred}
    for name in os.listdir(sharded_path):
        match = _SHARD_FILE.match(name)
        if match is None or int(match.group("index")) not in deferred_indices:
            os.remove(os.path.join(sharded_path, name))
    index = {"metadata": {"total_size": total_size}, "weight_map": file_of}
    with open(os.path.join(save_path, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    logger.info(
        f"Left {len(deferred)} row-sharded tensor(s) unmerged under {sharded_path}: "
        f"run `python scripts/merge_hf_export.py {save_path}` before loading this export."
    )


_COPY_CHUNK_BYTES = 64 << 20


def _read_safetensors_header(path: str) -> tuple[dict, int]:
    """``(header, data_start)`` of a safetensors file."""
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(length)), 8 + length


def _row_shards(sharded_path: str, fqn: str, shape: List[int], nbytes: int) -> tuple[str, List[tuple]]:
    """``(dtype, [(path, start, end), ...])`` of ``fqn``'s shards in row order, checked to tile it exactly."""
    from torch.distributed.checkpoint._hf_utils import _get_dcp_custom_metadata

    found = []
    for name in sorted(os.listdir(sharded_path)):
        path = os.path.join(sharded_path, name)
        header, data_start = _read_safetensors_header(path)
        if fqn not in header:
            continue
        entry = header[fqn]
        offsets = _get_dcp_custom_metadata(header)[fqn]["saved_offsets"]
        start, end = entry["data_offsets"]
        found.append((offsets, entry["shape"], entry["dtype"], path, data_start + start, data_start + end))
    found.sort(key=lambda shard: shard[0][0])

    dtypes = {shard[2] for shard in found}
    next_row = 0
    for offsets, shard_shape, _, path, *_ in found:
        if offsets[0] != next_row or any(offsets[1:]) or shard_shape[1:] != shape[1:]:
            raise ValueError(
                f"{fqn}: shard {path} holds rows {offsets[0]}..{offsets[0] + shard_shape[0]} of shape "
                f"{shard_shape} where rows from {next_row} of width {shape[1:]} were expected."
            )
        next_row += shard_shape[0]
    total = sum(end - start for *_, start, end in found)
    if len(dtypes) != 1 or next_row != shape[0] or total != nbytes:
        raise ValueError(
            f"{fqn}: the shards under {sharded_path} cover {next_row} of {shape[0]} rows and {total} of "
            f"{nbytes} bytes (dtypes {sorted(dtypes)}); a shard is missing or belongs to another export."
        )
    return dtypes.pop(), [(path, start, end) for *_, path, start, end in found]


def _merge_rows(sharded_path: str, output_path: str, fqn: str, shape: List[int], nbytes: int) -> None:
    """Append ``fqn``'s row shards into a one-tensor safetensors file, a chunk at a time.

    Written under a temporary name and renamed into place: an append-only mount
    may refuse to truncate the partial file an interrupted merge left behind.
    """
    dtype, shards = _row_shards(sharded_path, fqn, shape, nbytes)
    header = json.dumps({fqn: {"dtype": dtype, "shape": shape, "data_offsets": [0, nbytes]}}).encode()
    header += b" " * (-len(header) % 8)
    partial_path = output_path + ".partial"
    for stale in (partial_path, output_path):
        if os.path.exists(stale):
            os.remove(stale)
    with open(partial_path, "wb") as out:
        out.write(struct.pack("<Q", len(header)))
        out.write(header)
        for path, start, end in shards:
            with open(path, "rb") as src:
                src.seek(start)
                remaining = end - start
                while remaining:
                    chunk = src.read(min(remaining, _COPY_CHUNK_BYTES))
                    if not chunk:
                        raise OSError(f"{path} ended {remaining} bytes early.")
                    out.write(chunk)
                    remaining -= len(chunk)
    written, data_start = _read_safetensors_header(partial_path)
    if written[fqn]["shape"] != shape or os.path.getsize(partial_path) != data_start + nbytes:
        raise OSError(f"{partial_path} does not hold the {nbytes} bytes of {fqn} it was written with.")
    os.replace(partial_path, output_path)


def merge_pending_hf_shards(save_path: str, num_threads: Optional[int] = None) -> None:
    """Write out the files an HF export deferred, then drop its marker and shards.

    Each file is written sequentially, which an append-only mount accepts, and
    memory stays at one copy chunk per file. Files are written in parallel, one
    thread each by default. Nothing is removed unless every file checks out.
    """
    from concurrent.futures import ThreadPoolExecutor

    marker = os.path.join(save_path, layout.HF_PENDING_MERGE_FILENAME)
    with open(marker) as f:
        pending = json.load(f)
    tensors = pending["tensors"]
    sharded_path = os.path.join(save_path, pending["sharded_dir"])
    index_path = os.path.join(save_path, "model.safetensors.index.json")
    weight_map = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    if any(weight_map.get(fqn) != t["file"] for fqn, t in tensors.items()):
        raise ValueError(
            f"{index_path} is missing or does not name the files {marker} lists: the save stopped "
            "before writing it. Export this checkpoint again."
        )
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_threads or len(tensors)) as pool:
        futures = [
            pool.submit(_merge_rows, sharded_path, os.path.join(save_path, t["file"]), fqn, t["shape"], t["nbytes"])
            for fqn, t in tensors.items()
        ]
        for future in futures:
            future.result()
    # Marker first: a crash after it leaves a complete export and stray shards,
    # never a marker pointing at shards that are gone.
    os.remove(marker)
    shutil.rmtree(sharded_path, ignore_errors=True)
    logger.info(f"Merged {len(tensors)} deferred tensor(s) into {save_path} in {time.time() - start_time:.2f}s")


def _save_hf_safetensor_distributed(
    model: torch.nn.Module,
    save_path: str,
    fqn_to_index_mapping: Optional[Dict[str, int]],
    model_assets: Optional[Sequence],
    parallel_state=None,
):
    """Distributed HuggingFace safetensors save using HuggingFaceStorageWriter (PyTorch >= 2.9).

    All ranks must call this function.
    """
    from torch.distributed.checkpoint import HuggingFaceStorageWriter

    # Apply DCP consolidation patch just-in-time for HDFS FUSE compatibility
    # This patches torch.distributed.checkpoint._consolidate_hf_safetensors._process_output_file
    # to use append mode instead of r+b mode, which is required for append-only file systems
    from veomni.checkpoint.dcp_consolidation import apply_dcp_consolidation_patch

    apply_dcp_consolidation_patch()

    save_state = get_model_save_state(model, fqn_to_index_mapping, parallel_state=parallel_state)

    # Filter fqn_to_index_mapping to only include keys that exist in save_state.
    # This is necessary when training excludes certain modules (e.g., MTP) but the original
    # fqn_to_index_mapping parsed from model.safetensors.index.json still contains those weights.
    # Without this filtering, HuggingFaceStorageWriter's consolidation phase would create invalid
    # metadata entries (with default values like empty shape and 0 dtype_size) for non-existent
    # weights, resulting in corrupted safetensors output.
    if fqn_to_index_mapping is not None:
        original_mapping_size = len(fqn_to_index_mapping)
        fqn_to_index_mapping = {k: v for k, v in fqn_to_index_mapping.items() if k in save_state}
        if len(fqn_to_index_mapping) < original_mapping_size:
            logger.info_rank0(
                f"Filtered fqn_to_index_mapping from {original_mapping_size} to {len(fqn_to_index_mapping)} keys "
                f"to match actual model weights"
            )
    else:
        fqn_to_index_mapping = _size_balanced_index_mapping(save_state)

    deferred = {
        fqn: {"shape": list(save_state[fqn].shape), "nbytes": save_state[fqn].numel() * save_state[fqn].element_size()}
        for fqn in _deferred_merge_fqns(model, save_state)
    }
    if deferred:
        fqn_to_index_mapping = _isolate_deferred(fqn_to_index_mapping, set(deferred))
        total_size = sum(tensor.numel() * tensor.element_size() for tensor in save_state.values())

    # Consolidation runs on every rank, one output file each, instead of in the
    # writer's finish step: there only the coordinator rewrites the whole model
    # while the others wait on a collective that times out for a large one.
    from torch.distributed.checkpoint._consolidate_hf_safetensors import (
        consolidate_safetensors_files,
        consolidate_safetensors_files_on_every_rank,
    )

    sharded_path = os.path.join(save_path, layout.HF_SHARDED_DIRNAME)
    storage_writer = HuggingFaceStorageWriter(
        path=sharded_path,
        save_distributed=True,
        fqn_to_index_mapping=fqn_to_index_mapping,
        enable_consolidation=False,
    )

    logger.info_rank0("Starting distributed HuggingFace safetensors save...")
    if dist.is_initialized():
        dist.barrier()
    start_time = time.time()
    dcp.save(
        state_dict=save_state,
        storage_writer=storage_writer,
    )
    del save_state  # Free copied tensors (e.g. fp32->bf16) to reduce peak memory
    if dist.is_initialized():
        dist.barrier()
    if deferred:
        _consolidate_all_but_deferred(sharded_path, save_path, fqn_to_index_mapping, deferred, total_size)
    elif dist.is_initialized():
        consolidate_safetensors_files_on_every_rank(
            input_dir=sharded_path,
            output_dir=save_path,
            fqn_to_index_mapping=fqn_to_index_mapping,
            process_group=_consolidation_process_group(),
        )
    else:
        consolidate_safetensors_files(
            input_dir=sharded_path, output_dir=save_path, fqn_to_index_mapping=fqn_to_index_mapping
        )
    # Rank 0 returns from the consolidation gather only after every rank has written its files.
    if not deferred and (not dist.is_initialized() or dist.get_rank() == 0):
        shutil.rmtree(sharded_path, ignore_errors=True)
    gc.collect()
    helper.empty_cache()
    elapsed_time = time.time() - start_time
    logger.info_rank0(f"Distributed HuggingFace safetensors save took {elapsed_time:.2f}s")

    # Save model assets (config, tokenizer, etc.) on rank 0
    if model_assets and (not dist.is_initialized() or dist.get_rank() == 0):
        save_model_assets(save_path, model_assets)

    logger.info_rank0(f"HuggingFace checkpoint saved at {save_path} successfully!")


def _save_hf_safetensor_legacy(
    save_checkpoint_path: str,
    save_hf_safetensor_path: str,
    model_assets: Optional[Sequence],
    ckpt_manager: str,
    output_dir: Optional[str],
    model: Optional[torch.nn.Module],
):
    """Legacy HuggingFace safetensors save via checkpoint conversion (rank-0 only)."""
    model_state_dict = ckpt_to_state_dict(
        save_checkpoint_path=save_checkpoint_path,
        ckpt_manager=ckpt_manager,
        output_dir=output_dir,
    )
    model_state_dict = _revert_model_weight_conversions_for_hf(model, model_state_dict)
    save_model_weights(save_hf_safetensor_path, model_state_dict, model_assets=model_assets)
    logger.info_rank0(f"HuggingFace checkpoint saved at {save_hf_safetensor_path} successfully!")


def save_hf_safetensor(
    save_hf_safetensor_path: Optional[str] = None,
    ckpt_manager: Optional[str] = None,
    model_assets: Optional[Sequence] = None,
    # Legacy only
    save_checkpoint_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    is_rank_0: bool = False,
    # Distributed only
    model: Optional[torch.nn.Module] = None,
    fqn_to_index_mapping: Optional[Dict[str, int]] = None,
    parallel_state=None,
):
    """Save model weights in HuggingFace safetensors format.

    This function is self-contained w.r.t. synchronization: it calls ``synchronize()`` at
    entry to flush pending GPU operations before reading tensor data, and calls
    ``dist.barrier()`` before returning to ensure all ranks complete the save. Callers
    do not need to add external synchronization around this function.

    Supports two modes:
    - Distributed mode (PyTorch >= 2.9, ckpt_manager="dcp", non-LoRA): Uses HuggingFaceStorageWriter
      for efficient distributed save directly from the live FSDP model. Must be called on all ranks.
    - Legacy mode: Loads from checkpoint and converts to safetensors on rank 0.

    Args:
        save_hf_safetensor_path: Output path for saved HuggingFace safetensors.
        ckpt_manager: Checkpoint manager type. Used for routing (distributed when "dcp")
            and passed to legacy ``ckpt_to_state_dict``.
        model_assets: Model assets (e.g., config, tokenizer) to save alongside weights.

        save_checkpoint_path: [Legacy only] Path to the distributed checkpoint for conversion.
        output_dir: [Legacy only] Output directory passed to ``ckpt_to_state_dict``.
        is_rank_0: [Legacy only] Whether the current process is global rank 0.
            Legacy save is rank-0 only; non-rank-0 processes return immediately.
            Required by non-dcp checkpoint managers (e.g., omnistore).
        model: [Distributed only] Live FSDP model for distributed save.
        fqn_to_index_mapping: [Distributed only] Maps FQNs to safetensors file indices
            for multi-file output.
    """
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

    # Disable the distributed `HuggingFaceStorageWriter` path on NPU.  Under
    # HSDP replicate ranks independently cast fp32->bf16 on-device and write
    # per-rank safetensors files.  When the casts are not bit-exact (~1 ULP on
    # Ascend 910B) the consolidation step bakes in whichever rank's data it
    # reads last, causing an intermittent mismatch against the fp32 DCP
    # checkpoint.  Route through the legacy rank-0 path until the underlying
    # fp32->bf16 determinism issue is resolved in the NPU stack.
    use_distributed = is_torch_version_greater_than("2.9") and ckpt_manager == "dcp" and not IS_NPU_AVAILABLE

    # Ensure all GPU operations are complete before reading tensor data for saving
    synchronize()

    # Wait for any pending async DCP save before HF safetensor save
    if ckpt_manager == "dcp":
        DistributedCheckpointer.wait_for_pending_save()

    if use_distributed:
        from veomni.models.checkpoint_tensor_loading import resolve_fqn_to_index_mapping_for_save

        fqn_to_index_mapping = resolve_fqn_to_index_mapping_for_save(model, fqn_to_index_mapping)
        _save_hf_safetensor_distributed(
            model, save_hf_safetensor_path, fqn_to_index_mapping, model_assets, parallel_state=parallel_state
        )
    else:
        # Legacy path is rank-0 only; non-rank-0 waits at the barrier below
        if is_rank_0:
            _save_hf_safetensor_legacy(
                save_checkpoint_path,
                save_hf_safetensor_path,
                model_assets,
                ckpt_manager,
                output_dir,
                model,
            )

    # Ensure all ranks finish saving before anyone proceeds
    if dist.is_initialized():
        dist.barrier()


@torch.no_grad()
def save_lora_adapter_with_dcp(
    model: torch.nn.Module,
    save_path: str,
    adapter_name: str = "default",
    dcp_subdir: str = ".lora_dcp_tmp",
) -> None:
    """Save LoRA adapter with DCP parallel write and rank-0 consolidation.

    All ranks must call this function. It performs:
    1. Extract LoRA-only state from the live model.
    2. Save with ``dcp.save`` in parallel to a temporary DCP directory.
    3. Consolidate on rank 0 into ``adapter_model.safetensors`` and ``adapter_config.json``.

    The consolidated adapter is written as **safetensors** (not a pickled ``.bin``)
    so a per-rank ExtraParallel-slice reader (``load_model_weights_ep_sharded`` under
    ``is_peft_model``) can stream only each rank's expert rows of an *independent*
    MoE-LoRA adapter instead of every rank reading the whole gathered ``[E, ...]``
    tensor. ``load_adapter_state_dict`` already prefers ``adapter_model.safetensors``.
    """
    from veomni.lora import is_veomni_lora_model
    from veomni.lora.state_dict import get_lora_state_dict

    if not is_veomni_lora_model(model):
        raise TypeError(
            f"save_lora_adapter_with_dcp expects a VeOmniLoraModel, got {type(model).__name__}. "
            "LoRA is served exclusively by the native veomni.lora stack."
        )

    synchronize()
    if dist.is_initialized():
        dist.barrier()

    os.makedirs(save_path, exist_ok=True)
    dcp_save_path = os.path.join(save_path, dcp_subdir)
    os.makedirs(dcp_save_path, exist_ok=True)

    lora_state = get_lora_state_dict(model, adapter_name=adapter_name, config=model.get_lora_config())
    lora_state = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in lora_state.items()}

    # Restore the EP shard dim before DCP sees the LoRA tensors. After
    # ``ParallelPlan.apply`` the EP-sharded params are stored as plain
    # local tensors (then wrapped by FSDP only along the FSDP mesh), so
    # ``get_peft_model_state_dict`` returns FSDP-only DTensors whose
    # data is the EP-local chunk. DCP would happily save these as-is
    # and ``ckpt_to_state_dict`` would consolidate only across FSDP,
    # leaving each rank's per-expert LoRA at ``[E_local, ...]`` instead
    # of the global ``[E, ...]``. Re-wrap every Shard()-placed tensor
    # as a 2-D (FSDP+EP) DTensor so DCP gathers both dimensions.
    # Mirrors what ``ModelState.get_state_dict_with_extra_parallel_dim_preprocess``
    # does for the regular checkpoint path.
    fqn2spec_info = getattr(model, "_fqn2spec_info", None)
    if fqn2spec_info is not None:
        from torch.distributed._tensor import Shard as _ShardPlacement

        from ..checkpoint.dcp_checkpointer import restore_extra_parallel_dim

        # ``get_peft_model_state_dict`` strips the adapter name from keys
        # (``...gate_proj.lora_A.default.weight`` -> ``...gate_proj.lora_A.weight``)
        # while ``_fqn2spec_info`` is built from ``model.named_parameters()``
        # and keeps the adapter name. Re-insert it (matching the save's
        # ``adapter_name``) when looking up the spec, otherwise no LoRA
        # tensor matches and the EP gather is silently skipped.
        for name in list(lora_state):
            spec = fqn2spec_info.get(name)
            if spec is None and name.endswith(".weight"):
                spec_key = f"{name[: -len('.weight')]}.{adapter_name}.weight"
                spec = fqn2spec_info.get(spec_key)
            if spec is not None and isinstance(spec.placement, _ShardPlacement) and spec.para_fsdp_mesh is not None:
                lora_state[name] = restore_extra_parallel_dim(
                    lora_state[name],
                    spec.para_fsdp_mesh,
                    spec.para_fsdp_mesh[f"{spec.para_name}_fsdp"],
                )

    # ckpt_to_state_dict's DCP conversion path only recognizes keys starting with "model.".
    # Prefix LoRA keys temporarily for DCP save so consolidation can reuse existing conversion logic.
    dcp_lora_state = {k if k.startswith("model.") else f"model.{k}": v for k, v in lora_state.items()}

    storage_writer = dcp.FileSystemWriter(
        dcp_save_path,
        thread_count=16,
        single_file_per_rank=True,
        sync_files=False,
    )
    dcp.save(
        state_dict=dcp_lora_state,
        storage_writer=storage_writer,
    )

    if dist.is_initialized():
        dist.barrier()

    is_rank_0 = not dist.is_initialized() or dist.get_rank() == 0
    if is_rank_0:
        consolidated_state = ckpt_to_state_dict(
            save_checkpoint_path=dcp_save_path,
            ckpt_manager="dcp",
        )
        # safetensors needs contiguous, non-aliased tensors; consolidation returns
        # fresh tensors but ``.contiguous()`` is a cheap no-op when already so.
        consolidated_state = {k: v.contiguous() for k, v in consolidated_state.items()}
        adapter_model_file = os.path.join(save_path, "adapter_model.safetensors")
        _save_state_dict(consolidated_state, adapter_model_file, safe_serialization=True)

        # VeOmniLoraModel writes a single PEFT-loadable adapter_config.json
        # (MoE metadata, if any, lives in its ``veomni_lora`` block -- no
        # separate sidecar).
        model.get_lora_config().save_pretrained(save_path)

        shutil.rmtree(dcp_save_path, ignore_errors=True)
        logger.info_rank0(f"LoRA adapter saved at {save_path} successfully!")

    if dist.is_initialized():
        dist.barrier()

    gc.collect()
    helper.empty_cache()
