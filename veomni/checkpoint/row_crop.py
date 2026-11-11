# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Save logical row prefixes and restore padded runtime shards."""

from dataclasses import replace

import torch
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.nn.parallel import DistributedDataParallel

from .planner import _ExtraParallelSavePlanner, _ModelStrictLoadPlanner


def checkpoint_row_counts(model):
    """Read optional row metadata from the model-owned checkpoint converter."""
    if isinstance(model, DistributedDataParallel):
        model = model.module
    factory = getattr(model, "_create_checkpoint_tensor_converter", None)
    if not callable(factory):
        return {}
    converter = factory(model)
    getter = getattr(converter, "checkpoint_row_counts", None)
    return getter() if callable(getter) else {}


def crop_dense_state_dict(state_dict, rows_by_name):
    """Crop full tensors on the legacy rank-zero export path."""
    if not rows_by_name:
        return state_dict
    cropped = dict(state_dict)
    for name, (logical, padded) in rows_by_name.items():
        tensor = state_dict.get(name)
        if tensor is None:
            continue
        if isinstance(tensor, DTensor):
            raise TypeError(f"{name}: dense export requires a full tensor")
        if not 0 < logical <= padded or tensor.shape[0] not in (logical, padded):
            raise ValueError(f"{name}: incompatible checkpoint row count {tensor.shape[0]}")
        cropped[name] = tensor[:logical].contiguous()
    return cropped


class RowCropSavePlanner(_ExtraParallelSavePlanner):
    """Clip selected write chunks to logical rows while retaining their global offsets."""

    def __init__(self, rows_by_name, **kwargs):
        # Keep caching disabled because cropping rewrites the parent local plan.
        super().__init__(enable_plan_caching=False, **kwargs)
        self.rows_by_name = rows_by_name

    def create_local_plan(self):
        """Describe logical storage using views of the original local shards."""
        plan = super().create_local_plan()
        items = []
        for item in plan.items:
            counts = self.rows_by_name.get(item.index.fqn)
            if counts is None or item.tensor_data is None:
                items.append(item)
                continue
            logical, padded = counts
            if not 0 < logical <= padded:
                raise ValueError(f"{item.index.fqn}: invalid checkpoint/runtime rows {counts}")
            data = item.tensor_data
            if data.size[0] != padded:
                raise ValueError(f"{item.index.fqn}: unexpected runtime row count {data.size[0]}")
            chunk = data.chunk
            rows = max(0, min(chunk.sizes[0], logical - chunk.offsets[0]))
            if rows:
                clipped = replace(chunk, sizes=torch.Size((rows, *chunk.sizes[1:])))
                items.append(
                    replace(
                        item,
                        tensor_data=replace(data, chunk=clipped, size=torch.Size((logical, *data.size[1:]))),
                    )
                )
        self.plan = replace(plan, items=items)
        return self.plan

    def transform_object(self, write_item, object):
        """Return a local prefix view only for a selected row-cropped weight."""
        tensor = super().transform_object(write_item, object)
        if write_item.index.fqn in self.rows_by_name and write_item.tensor_data is not None:
            tensor = tensor.narrow(0, 0, write_item.tensor_data.chunk.sizes[0])
        return tensor


class RowCropLoadPlanner(_ModelStrictLoadPlanner):
    """Read logical row-cropped chunks directly into padded runtime shards."""

    def __init__(self, strict_model: bool, rows_by_name):
        super().__init__(strict_model=strict_model)
        self.rows_by_name = rows_by_name

    def _create_read_plan(self):
        """Allow only validated row-cropped row differences and clear the unsaved tail."""
        if not self.rows_by_name:
            return super()._create_read_plan()
        original = self.metadata
        metadata = dict(original.state_dict_metadata)
        formats = set()
        tails = []
        for name, (logical, padded) in self.rows_by_name.items():
            if not 0 < logical <= padded:
                raise ValueError(f"{name}: invalid checkpoint/runtime rows {(logical, padded)}")
            tensor = self.state_dict.get(name)
            saved = metadata.get(name)
            if tensor is None or not isinstance(saved, TensorStorageMetadata):
                continue
            if tuple(saved.size[1:]) != tuple(tensor.shape[1:]) or saved.size[0] not in (logical, padded):
                raise ValueError(f"{name}: incompatible row-cropped checkpoint shape {saved.size}")
            if tensor.shape[0] != padded:
                raise ValueError(f"{name}: incompatible runtime row-cropped shape {tensor.shape}")
            if logical != padded:
                formats.add(saved.size[0] == logical)
            if saved.size[0] == logical and logical != padded:
                # Only the planner's shape check sees padded size. Stored chunks
                # retain their real logical extents, so reads never cover the tail.
                metadata[name] = replace(saved, size=tensor.size())
                if isinstance(tensor, DTensor):
                    _, offsets = compute_local_shape_and_global_offset(
                        tensor.shape, tensor.device_mesh, tensor.placements
                    )
                    local = tensor.to_local()
                    start = max(0, min(local.shape[0], logical - offsets[0]))
                else:
                    local, start = tensor, logical
                tails.append((local, start))
        if len(formats) > 1:
            raise ValueError("Checkpoint mixes logical and padded row formats")
        self.metadata = replace(original, state_dict_metadata=metadata)
        try:
            plan = super()._create_read_plan()
        finally:
            self.metadata = original
        with torch.no_grad():
            for local, start in tails:
                local[start:].zero_()
        return plan
