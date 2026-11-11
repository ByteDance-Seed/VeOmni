# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Common DCP planners for expert layouts and strict model restoration."""

from dataclasses import replace

from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner, DefaultSavePlanner
from torch.distributed.tensor import DTensor, Replicate, Shard
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
        """Attach source expert layouts to the local write plan."""
        plan = super().create_local_plan()
        layouts = {key: layout for key, value in self.state_dict.items() if (layout := _expert_layout(value))}
        return replace(plan, planner_data={**(plan.planner_data or {}), _EXPERT_LAYOUTS_KEY: layouts})

    def create_global_plan(self, all_plans):
        """Collect expert layouts across ranks while retaining DCP key mappings."""
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


class _ModelStrictLoadPlanner(DefaultLoadPlanner):
    """Allow partial optimizer state while requiring a complete full-model DCP."""

    def __init__(self, strict_model: bool):
        super().__init__(allow_partial_load=True)
        self.strict_model = strict_model

    def create_local_plan(self):
        """Validate expert layouts and require all model keys for strict restores."""
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
        """Build the read plan from the current tensor metadata."""
        return super().create_local_plan()

    def _validate_expert_layouts(self):
        """Reject checkpoints with noncanonical or ambiguous expert ordering."""
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
