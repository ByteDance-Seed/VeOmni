# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""DCP planners for expert layouts and PLE row clipping."""

from dataclasses import replace

import torch
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner, DefaultSavePlanner
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import _StridedShard


_EXPERT_LAYOUTS_KEY = "__veomni_expert_layouts__"


def _expert_layout(tensor):
    """Describe the source EP/FSDP layout, independently of the load target."""
    if not isinstance(tensor, DTensor):
        return None
    mesh = tensor.device_mesh
    names = mesh.mesh_dim_names or ()
    if "ep" not in names:
        return None
    ep_axis = names.index("ep")
    ep = tensor.placements[ep_axis]
    if not isinstance(ep, Shard):
        return None
    fsdp_axis = names.index("ep_fsdp") if "ep_fsdp" in names else None
    fsdp = tensor.placements[fsdp_axis] if fsdp_axis is not None else Replicate()
    fsdp_dim = fsdp.dim if isinstance(fsdp, (Shard, _StridedShard)) else None
    fsdp_size = mesh.size(fsdp_axis) if fsdp_axis is not None else 1
    same_axis = ep.dim == fsdp_dim and mesh.size(ep_axis) > 1 and fsdp_size > 1
    canonical = not same_axis or isinstance(fsdp, _StridedShard)
    return {
        "ep_shard_dim": ep.dim,
        "fsdp_shard_dim": fsdp_dim,
        "ep_size": mesh.size(ep_axis),
        "fsdp_size": fsdp_size,
        "expert_order": "canonical" if canonical else "legacy_fsdp_first",
    }


class _ExtraParallelSavePlanner(DefaultSavePlanner):
    """Record per-tensor source layouts in DCP metadata, including optimizer state."""

    def create_local_plan(self):
        plan = super().create_local_plan()
        layouts = {key: layout for key, value in self.state_dict.items() if (layout := _expert_layout(value))}
        return replace(plan, planner_data={**(plan.planner_data or {}), _EXPERT_LAYOUTS_KEY: layouts})

    def create_global_plan(self, all_plans):
        plans, metadata = super().create_global_plan(all_plans)
        # Reuse DCP's plan exchange; sparse optimizer states may appear on only
        # some ranks. Keep the ordinary FQN -> state-dict path mappings intact.
        layouts = {}
        for plan in all_plans:
            layouts.update(plan.planner_data[_EXPERT_LAYOUTS_KEY])
        metadata.planner_data = {
            **(metadata.planner_data or {}),
            _EXPERT_LAYOUTS_KEY: {"version": 1, "tensors": layouts},
        }
        return plans, metadata


class PleSavePlanner(_ExtraParallelSavePlanner):
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


class _ModelStrictLoadPlanner(DefaultLoadPlanner):
    """Allow partial optimizer state while requiring a complete full-model DCP."""

    def __init__(self, strict_model: bool):
        super().__init__(allow_partial_load=True)
        self.strict_model = strict_model

    def create_local_plan(self):
        self._validate_expert_layouts()
        plan = self._create_read_plan()
        if not self.strict_model:
            return plan

        assert self.metadata is not None
        missing_model_keys = sorted(
            key
            for key, path in self.mappings.items()
            if path and path[0] == "model" and key not in self.metadata.state_dict_metadata
        )
        if missing_model_keys:
            preview = ", ".join(missing_model_keys[:10])
            suffix = " ..." if len(missing_model_keys) > 10 else ""
            raise RuntimeError(
                f"DCP is missing {len(missing_model_keys)} model key(s) required for a full-model resume: "
                f"{preview}{suffix}"
            )
        return plan

    def _create_read_plan(self):
        return super().create_local_plan()

    def _validate_expert_layouts(self):
        planner_data = self.metadata.planner_data or {}
        source = planner_data.get(_EXPERT_LAYOUTS_KEY)
        if source is not None:
            if source["version"] != 1:
                raise RuntimeError(f"Unsupported checkpoint expert layout version: {source['version']}")
            for key in self.state_dict.keys() & source["tensors"].keys():
                if source["tensors"][key]["expert_order"] != "canonical":
                    raise RuntimeError(
                        f"Checkpoint tensor {key!r} has noncanonical expert order: {source['tensors'][key]}. "
                        "Migrate model and optimizer state using the original training layout before loading."
                    )
            return

        # Checkpoints produced by the earlier version of this feature already
        # used canonical coordinates, but did not record individual layouts.
        if planner_data.get("__veomni_ep_first_layout__") == 1:
            return

        for key, value in self.state_dict.items():
            layout = _expert_layout(value)
            saved = self.metadata.state_dict_metadata.get(key)
            if layout is None or saved is None:
                continue
            # For legacy VeOmni EP weights, the faulty same-axis layout only
            # split the expert dimension. A source chunk that splits matrix
            # rows (old dim-1 FSDP), or spans all experts, cannot have that
            # permutation. This uses saved chunks, never the target FSDP dim.
            ep_dim = layout["ep_shard_dim"]
            if all(
                chunk.sizes[ep_dim] == saved.size[ep_dim]
                or any(size < saved.size[dim] for dim, size in enumerate(chunk.sizes) if dim != ep_dim)
                for chunk in saved.chunks
            ):
                continue
            raise RuntimeError(
                f"Cannot determine source expert order for {key!r} in this unversioned checkpoint. "
                "Expert-only chunks may come from legacy Muon dim-0 saves with permuted expert IDs. "
                "Migrate model and optimizer state using the original training layout before loading; "
                "changing the target expert_fsdp_shard_dim does not resolve the ambiguity."
            )


class PleLoadPlanner(_ModelStrictLoadPlanner):
    """Read logical PLE chunks directly into padded runtime shards."""

    def __init__(self, strict_model: bool, ple_rows_by_name):
        super().__init__(strict_model=strict_model)
        self.ple_rows_by_name = ple_rows_by_name

    def _create_read_plan(self):
        """Allow only validated PLE row differences and clear the unsaved tail."""
        if not self.ple_rows_by_name:
            return super()._create_read_plan()
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
            plan = super()._create_read_plan()
        finally:
            self.metadata = original
        with torch.no_grad():
            for local, start in tails:
                local[start:].zero_()
        return plan
