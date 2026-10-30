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
"""Crop and restore Qwen4-Exp PLE rows around a DCP save.

Runtime parameters stay padded so ``ple`` can split them evenly. A DCP
checkpoint stores only the logical rows. Row interval exchanges preserve column
sharding without gathering a full table. Loading restores the zero tail.
"""

from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.tensor import DTensor, Shard

from veomni.models.checkpoint.convert import get_checkpoint_tensor_converter
from veomni.models.transformers.qwen4_exp.checkpoint_tensor_converter import _PLE_SHARD_PATTERN
from veomni.utils import logging


logger = logging.get_logger(__name__)


def ple_row_counts(model: nn.Module) -> Dict[str, Tuple[int, int]]:
    """Return ``weight_name -> (logical_rows, padded_rows)`` for PLE weights."""
    converter = get_checkpoint_tensor_converter(model) if model is not None else None
    return getattr(converter, "ple_rows_by_name", {})


def _is_ple_weight(name: str) -> bool:
    if _PLE_SHARD_PATTERN.match(name):
        return True
    # DCP metadata keys are ``model.`` plus the state-dict name.
    if name.startswith("model.") and _PLE_SHARD_PATTERN.match(name[len("model.") :]):
        return True
    return False


def _checkpoint_name(name: str, counts: Dict[str, Tuple[int, int]]) -> str:
    # Prefer an exact FQN: the model itself may already have a ``model`` child.
    if name in counts:
        return name
    if name.startswith("model.") and name[len("model.") :] in counts:
        return name[len("model.") :]
    raise RuntimeError(f"PLE weight {name!r} has no runtime row-count metadata.")


def _from_local(tensor: torch.Tensor, reference: DTensor, rows: int) -> DTensor:
    """Rebuild a contiguous global matrix with explicit shape and stride."""
    shape = torch.Size([rows, *reference.shape[1:]])
    stride = torch.empty(shape, device="meta").stride()
    return DTensor.from_local(
        tensor.contiguous(), reference.device_mesh, reference.placements, shape=shape, stride=stride
    )


def _row_interval(rows: int, size: int, rank: int) -> Tuple[int, int]:
    chunk = (rows + size - 1) // size
    return min(rank * chunk, rows), min((rank + 1) * chunk, rows)


def _resize_dtensor(tensor: DTensor, rows: int, *, empty: bool = False) -> DTensor:
    """Exchange overlapping row intervals within the row-sharding mesh axis.

    PLE has one row-sharding axis and optionally a separate column axis. Each
    peer retains its column slice; send/receive buffers are local-shard sized.
    """
    row_axes = [i for i, p in enumerate(tensor.placements) if isinstance(p, Shard) and p.dim == 0]
    if len(row_axes) > 1:
        raise RuntimeError("PLE checkpoint projection requires at most one row-sharding mesh axis.")
    local = tensor.to_local().detach()
    if not row_axes:
        if empty:
            resized = local.new_empty((rows, *local.shape[1:]))
        elif rows <= tensor.shape[0]:
            resized = local[:rows]
        else:
            resized = local.new_zeros((rows, *local.shape[1:]))
            resized[: local.shape[0]].copy_(local)
        return _from_local(resized, tensor, rows)

    axis = row_axes[0]
    mesh = tensor.device_mesh
    size, rank = mesh.size(axis), mesh.get_local_rank(axis)
    source_rows = int(tensor.shape[0])
    source_start, source_end = _row_interval(source_rows, size, rank)
    target_start, target_end = _row_interval(rows, size, rank)
    shape = (target_end - target_start, *local.shape[1:])
    if empty:
        return _from_local(local.new_empty(shape), tensor, rows)
    # Unchanged chunk width means only the last nonempty peer's tail changes.
    # The condition is global, so every peer takes the same collective-free path.
    if (source_rows + size - 1) // size == (rows + size - 1) // size:
        if rows < source_rows:
            return _from_local(local[: shape[0]], tensor, rows)
        output = local.new_zeros(shape)
        output[: local.shape[0]].copy_(local)
        return _from_local(output, tensor, rows)
    output = local.new_zeros(shape)
    send_sizes, recv_sizes = [], []
    for peer in range(size):
        start, end = _row_interval(rows, size, peer)
        lo, hi = max(source_start, start), min(source_end, end)
        count = max(0, hi - lo)
        send_sizes.append(count)
        start, end = _row_interval(source_rows, size, peer)
        recv_sizes.append(max(0, min(end, target_end) - max(start, target_start)))
    # Peer intersections are ordered contiguous intervals. Receive straight
    # into the valid output prefix; the remaining expansion tail stays zero.
    dist.all_to_all_single(
        output[: sum(recv_sizes)],
        local[: sum(send_sizes)].contiguous(),
        output_split_sizes=recv_sizes,
        input_split_sizes=send_sizes,
        group=mesh.get_group(axis),
    )
    return _from_local(output, tensor, rows)


def _project_one(tensor: torch.Tensor, crop, logical_rows: int, name: str, *, empty: bool) -> torch.Tensor:
    if tensor.shape[0] < logical_rows:
        raise RuntimeError(f"{name}: checkpoint rows {tensor.shape[0]} are shorter than logical rows {logical_rows}.")
    if tensor.shape[0] == logical_rows:
        return tensor
    if isinstance(tensor, DTensor):
        return _resize_dtensor(tensor, logical_rows, empty=empty)
    if empty:
        return tensor.new_empty((logical_rows, *tensor.shape[1:]))
    return crop(name, tensor.detach())


def _pad_one(tensor: torch.Tensor, padded_rows: int, name: str) -> torch.Tensor:
    rows = tensor.shape[0]
    if rows == padded_rows:
        return tensor
    if rows > padded_rows:
        raise RuntimeError(f"{name}: DCP rows {rows} are longer than padded rows {padded_rows}.")
    if isinstance(tensor, DTensor):
        return _resize_dtensor(tensor, padded_rows)
    padded = tensor.new_zeros((padded_rows, *tensor.shape[1:]))
    padded[:rows].copy_(tensor.detach())
    return padded


def project_state_dict_to_checkpoint(
    model: nn.Module, state_dict: Dict[str, torch.Tensor], *, empty: bool = False
) -> Dict[str, torch.Tensor]:
    """Store PLE weights at ``rows_per_checkpoint_shard``.

    DCP save and Hugging Face export both call this. The row cut is
    converter metadata. Dense tensors use ``crop_hf_export_tensor``; DTensors
    exchange row intersections without full-table materialization. ``empty``
    allocates load targets without copying runtime weights. Models without
    that converter are returned unchanged.
    """
    counts = ple_row_counts(model)
    if not counts:
        return state_dict
    converter = get_checkpoint_tensor_converter(model)
    crop = getattr(converter, "crop_hf_export_tensor", None)
    if crop is None:
        return state_dict
    projected = dict(state_dict)
    changed = 0
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not torch.is_tensor(tensor) or not _is_ple_weight(name):
            continue
        checkpoint_name = _checkpoint_name(name, counts)
        logical, _padded = counts[checkpoint_name]
        updated = _project_one(tensor, crop, logical, checkpoint_name, empty=empty)
        if updated is not tensor:
            changed += 1
        projected[name] = updated
    if changed:
        logger.info_rank0(f"Projected {changed} PLE tensor(s) to their logical checkpoint rows.")
    return projected if changed else state_dict


def crop_ple_state_dict_for_dcp(
    model: nn.Module, state_dict: Dict[str, torch.Tensor], *, empty: bool = False
) -> Dict[str, torch.Tensor]:
    """DCP entry point. Same projection as Hugging Face export."""
    return project_state_dict_to_checkpoint(model, state_dict, empty=empty)


def pad_ple_state_dict_from_dcp(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Pad cropped PLE weights back to the runtime ``padded_rows_per_shard``."""
    counts = ple_row_counts(model)
    if not counts:
        return state_dict
    restored = dict(state_dict)
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not torch.is_tensor(tensor) or not _is_ple_weight(name):
            continue
        _logical, padded = counts[_checkpoint_name(name, counts)]
        restored[name] = _pad_one(tensor, padded, name)
    return restored


def ple_checkpoint_is_cropped(checkpoint_dir: str, model: nn.Module) -> bool:
    """Whether ``checkpoint_dir`` stored PLE weights at the logical row count.

    Checkpoints written before this crop still contain the padded runtime shape
    and load without a second pad.
    """
    counts = ple_row_counts(model)
    if not counts:
        return False
    metadata = FileSystemReader(checkpoint_dir).read_metadata()
    formats = set()
    for key, tensor_meta in metadata.state_dict_metadata.items():
        if not hasattr(tensor_meta, "size") or not _is_ple_weight(key):
            continue
        rows = int(tensor_meta.size[0])
        logical, padded = counts[_checkpoint_name(key, counts)]
        if rows not in (logical, padded):
            raise RuntimeError(
                f"{key}: PLE checkpoint rows {rows} are neither the logical row count {logical} "
                f"nor the padded runtime row count {padded}."
            )
        # An unpadded table does not distinguish the old and new formats.
        if logical != padded:
            formats.add(rows == logical)
    if len(formats) > 1:
        raise RuntimeError("PLE checkpoint mixes logical and padded row formats.")
    return True in formats
