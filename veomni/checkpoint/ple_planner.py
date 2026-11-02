# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""PLE-only DCP row clipping without changing the runtime tensor layout."""

from dataclasses import replace

import torch
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner, DefaultSavePlanner
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset


class PleSavePlanner(DefaultSavePlanner):
    """Clip PLE write chunks to logical rows while retaining their global offsets."""

    def __init__(self, rows_by_name, **kwargs):
        # The parent caches its unmodified plan before our PLE clipping.
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
            data = item.tensor_data
            if data.size[0] != padded:
                raise ValueError(f"{item.index.fqn}: unexpected runtime PLE row count {data.size[0]}")
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
        """Return a local prefix view only for a selected PLE weight."""
        tensor = super().transform_object(write_item, object)
        if write_item.index.fqn in self.rows_by_name and write_item.tensor_data is not None:
            tensor = tensor.narrow(0, 0, write_item.tensor_data.chunk.sizes[0])
        return tensor


class PleLoadPlanner(DefaultLoadPlanner):
    """Read logical PLE chunks directly into padded runtime shards."""

    def __init__(self, *args, ple_rows_by_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ple_rows_by_name = ple_rows_by_name or {}

    def create_local_plan(self):
        """Allow only validated PLE row differences and clear the unsaved tail."""
        if not self.ple_rows_by_name:
            return super().create_local_plan()
        original = self.metadata
        metadata = dict(original.state_dict_metadata)
        formats = set()
        tails = []
        for name, (logical, padded) in self.ple_rows_by_name.items():
            tensor = self.state_dict.get(name)
            saved = metadata.get(name)
            if tensor is None or not isinstance(saved, TensorStorageMetadata):
                continue
            if tuple(saved.size[1:]) != tuple(tensor.shape[1:]) or saved.size[0] not in (logical, padded):
                raise ValueError(f"{name}: incompatible PLE checkpoint shape {saved.size}")
            if tensor.shape[0] != padded:
                raise ValueError(f"{name}: incompatible runtime PLE shape {tensor.shape}")
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
            raise ValueError("PLE checkpoint mixes logical and padded row formats")
        self.metadata = replace(original, state_dict_metadata=metadata)
        try:
            plan = super().create_local_plan()
        finally:
            self.metadata = original
        with torch.no_grad():
            for local, start in tails:
                local[start:].zero_()
        return plan
