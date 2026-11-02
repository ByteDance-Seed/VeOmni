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
import hashlib
import json
import os
import shutil
import threading
import uuid
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from typing import Any, Dict, Optional, Set, Union

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load,
)
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner, DefaultSavePlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE, Metadata
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor.placement_types import _StridedShard

from ..distributed.parallel_state import get_parallel_state
from ..models.qwen4_gdn_precision import (
    is_qwen4_exp_model,
    is_qwen4_gdn_parameter,
    qwen4_exp_model_config,
    validate_qwen4_gdn_pairs,
)
from ..utils import logging
from ..utils.device import empty_cache, synchronize
from ..utils.dist_utils import any_rank_failed, raise_if_any_rank_failed
from .bounded_writer import BoundedFileSystemWriter
from .checkpointer import CheckpointerBase
from .layout import (
    DCP_MARKER_FILENAME,
    OPTIMIZER_DIRNAME,
    WEIGHTS_DIRNAME,
    dcp_markers,
    manifest_path,
    model_dir,
    optimizer_dir,
    step_dir,
    weights_dir,
)
from .multi_optimizer_checkpoint import (
    _MULTI_GENERATION_MARKER,
    _MULTI_INVENTORY_MARKER,
    _dcp_parameter_name_map,
    _inventory_digests,
    _iter_optimizer_leaves,
    _multi_extra_snapshot,
    _preflight_multi_optimizer_layout,
    _prepare_multi_optimizer_saved_states,
    _validate_source_optimizer_states,
)
from .optimizer_resume_layout import MULTI_LAYOUT_MARKER, coordinated_multi_layout, frozen_parameter_digest


logger = logging.get_logger(__name__)

_EXTRA_STATE_KEY = "extra_state"
_EXTRA_STATE_DIRNAME = "extra_state"
_EXTRA_STATE_FORMAT = "extra_state_rank_{}.pt"
_QWEN4_GDN_FP32_LEGACY_MARKER = "__veomni_qwen4_gdn_fp32_v1__"
_QWEN4_GDN_FP32_MARKER = "__veomni_qwen4_gdn_fp32_v2__"


def read_qwen4_gdn_manifest(checkpoint_path) -> Optional[set[str]]:
    """Read the persisted GDN precision inventory, including EP namespaces."""
    reader = FileSystemReader(checkpoint_path)
    keys = []
    for key in reader.read_metadata().state_dict_metadata:
        if not key.startswith("model."):
            continue
        logical = _export_extra_parallel_layout_key(key.removeprefix("model."))
        if logical == _QWEN4_GDN_FP32_MARKER:
            keys.append(key)
    if not keys:
        return None
    if len(keys) != 1:
        raise RuntimeError("Qwen4 GDN checkpoint has multiple precision inventories")
    state = {keys[0]: ""}
    load(state, storage_reader=reader, no_dist=True)
    try:
        payload = json.loads(state[keys[0]])
        names = payload["parameter_names"]
        if payload["version"] != 1 or not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ValueError("invalid schema")
        if len(set(names)) != len(names) or any(not is_qwen4_gdn_parameter(name) for name in names):
            raise ValueError("invalid parameter inventory")
        validate_qwen4_gdn_pairs(names)
        return set(names)
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError("Invalid Qwen4 GDN FP32 checkpoint precision inventory") from exc


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
    fsdp_dim = fsdp.dim % tensor.ndim if isinstance(fsdp, (Shard, _StridedShard)) else None
    ep_dim = ep.dim % tensor.ndim
    fsdp_size = mesh.size(fsdp_axis) if fsdp_axis is not None else 1
    same_axis = ep_dim == fsdp_dim and mesh.size(ep_axis) > 1 and fsdp_size > 1
    canonical = not same_axis or isinstance(fsdp, _StridedShard) or (fsdp_axis is not None and ep_axis < fsdp_axis)
    return {
        "ep_shard_dim": ep_dim,
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


class _FileBackedSavePlanner(_ExtraParallelSavePlanner):
    """Reclaim consumed file-backed pages after a writer resolves its next item."""

    def __init__(self, *, optimizers, **kwargs):
        super().__init__(**kwargs)
        self._optimizers = optimizers
        self._thread_previous = threading.local()

    def resolve_data(self, write_item):
        previous = getattr(self._thread_previous, "value", None)
        if torch.is_tensor(previous):
            for optimizer in self._optimizers:
                optimizer.checkpoint_commit_tensor(previous)
        value = super().resolve_data(write_item)
        self._thread_previous.value = value
        return value


class _ModelStrictLoadPlanner(DefaultLoadPlanner):
    """Allow partial optimizer state while requiring a complete full-model DCP."""

    def __init__(self, strict_model: bool, expected_save_id=None, strict_gdn_fp32: bool = False):
        super().__init__(allow_partial_load=True)
        self.strict_model = strict_model
        self.expected_save_id = expected_save_id
        self.strict_gdn_fp32 = strict_gdn_fp32

    def create_local_plan(self):
        assert self.metadata is not None
        if self.strict_gdn_fp32:
            for key, path in self.mappings.items():
                if not path or path[0] != "model":
                    continue
                logical = _export_extra_parallel_layout_key(key.removeprefix("model."))
                if logical is None or not is_qwen4_gdn_parameter(logical):
                    continue
                source = self.metadata.state_dict_metadata.get(key)
                if source is not None and getattr(getattr(source, "properties", None), "dtype", None) != torch.float32:
                    raise RuntimeError(f"Qwen4 GDN parameter {logical} in DCP is not FP32")
                target = self.state_dict.get(key)
                if torch.is_tensor(target) and target.dtype != torch.float32:
                    raise RuntimeError(f"Qwen4 GDN target {logical} must be FP32")
        target = {
            key
            for key, path in self.mappings.items()
            if path and _is_extra_parallel_layout_marker(str(path[-1]).split(".")[-1])
        }
        source = {
            key for key in self.metadata.state_dict_metadata if _is_extra_parallel_layout_marker(key.split(".")[-1])
        }
        if source != target:
            raise RuntimeError("Incompatible ExtraParallel checkpoint layout; explicit layout migration is required")
        if source:
            if len(source) != 1:
                raise RuntimeError("ExtraParallel checkpoint has multiple layout markers")
            marker = next(iter(source))
            if any(
                key.startswith("model.") and key != marker and not key.startswith(marker + "__")
                for key in self.metadata.state_dict_metadata
            ):
                raise RuntimeError("ExtraParallel checkpoint mixes versioned and legacy tensor keys")
        if target:
            for key, value in self.state_dict.items():
                if not torch.is_tensor(value) or not key.startswith("model."):
                    continue
                source_value = self.metadata.state_dict_metadata.get(key)
                source_dtype = getattr(getattr(source_value, "properties", None), "dtype", None)
                if source_dtype is not None and source_dtype != value.dtype:
                    raise RuntimeError(
                        f"ExtraParallel checkpoint dtype differs for {key}; explicit conversion required"
                    )
        if self.expected_save_id is not None:
            actual = getattr(getattr(self.metadata, "storage_meta", None), "save_id", None)
            if actual != self.expected_save_id:
                raise RuntimeError("Checkpoint changed between identity preflight and tensor load")
        self._validate_expert_layouts()
        plan = super().create_local_plan()
        if not self.strict_model:
            return plan

        assert self.metadata is not None
        missing_model_keys = sorted(
            key
            for key, path in self.mappings.items()
            if path
            and path[0] == "model"
            and _export_extra_parallel_layout_key(key.removeprefix("model.")) != _QWEN4_GDN_FP32_MARKER
            and key not in self.metadata.state_dict_metadata
        )
        if missing_model_keys:
            preview = ", ".join(missing_model_keys[:10])
            suffix = " ..." if len(missing_model_keys) > 10 else ""
            raise RuntimeError(
                f"DCP is missing {len(missing_model_keys)} model key(s) required for a full-model resume: "
                f"{preview}{suffix}"
            )

        return plan

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

        # Earlier v2 checkpoints used EP-first coordinates and a versioned
        # namespace; matching source/target markers were checked above.
        marker = _extra_parallel_layout_key({})
        if any(key.split(".")[-1] == marker for key in self.metadata.state_dict_metadata):
            return

        for key, value in self.state_dict.items():
            layout = _expert_layout(value)
            saved = self.metadata.state_dict_metadata.get(key)
            if saved is None:
                continue
            # Unversioned chunks cannot distinguish a safe EP+row-FSDP layout
            # from the same-axis legacy bug combined with another parallel
            # axis. Only a source chunk spanning all experts is unambiguous.
            if layout is None:
                # Offline export uses a plain CPU tensor as its target. Old
                # stacked expert weights still require the same source check.
                if "experts" not in key.split(".") or len(saved.size) < 2:
                    continue
                ep_dim = 0
            else:
                ep_dim = layout["ep_shard_dim"]
            if all(chunk.sizes[ep_dim] == saved.size[ep_dim] for chunk in saved.chunks):
                continue
            raise RuntimeError(
                f"Cannot determine source expert order for {key!r} in this unversioned checkpoint. "
                "Expert-only chunks may come from legacy Muon dim-0 saves with permuted expert IDs. "
                "Migrate model and optimizer state using the original training layout before loading; "
                "changing the target expert_fsdp_shard_dim does not resolve the ambiguity."
            )


def _validate_extra_parallel_meshes(parallel_state) -> None:
    """Fail-fast precondition for ExtraParallel state dict preprocessing.

    At least one ExtraParallel mesh must be non-None, and at least one
    of those meshes must carry the ExtraParallel + FSDP composition:
    2D ``(ep_fsdp, ep)`` for plain FSDP or 3D
    ``(ep_replicate, ep_fsdp, ep)`` for HSDP.
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
        and parallel_state.extra_parallel_fsdp_device_mesh[para].ndim in (2, 3)
        for para in parallel_state.extra_parallel_names
    ), (
        "At least one extra_parallel fsdp_device_mesh must carry the ExtraParallel+FSDP "
        "composition: 2D (ep_fsdp, ep) for plain FSDP or 3D (ep_replicate, ep_fsdp, ep) for HSDP"
    )


_EXTRA_PARALLEL_LAYOUT_MARKER = "__veomni_ep_outer_layout_v1__"
_EXTRA_PARALLEL_FIRST_MESHES = {}


def _is_extra_parallel_layout_marker(key):
    if not isinstance(key, str):
        return False
    suffix = key.removeprefix(_EXTRA_PARALLEL_LAYOUT_MARKER)
    return (
        key.startswith(_EXTRA_PARALLEL_LAYOUT_MARKER)
        and len(suffix) == 64
        and all(c in "0123456789abcdef" for c in suffix)
    )


def _normalized_extra_parallel_dims(spec):
    shape = getattr(spec, "global_shape", None)
    if shape is None:
        raise RuntimeError("The v2 ExtraParallel checkpoint layout requires the original global_shape")
    ndim = len(shape)
    ep_dim, fsdp_dim = spec.placement.dim, spec.fsdp_shard_dim
    if not (-ndim <= ep_dim < ndim and -ndim <= fsdp_dim < ndim):
        raise RuntimeError("ExtraParallel shard dimension is outside the global tensor rank")
    return ep_dim % ndim, fsdp_dim % ndim


def _needs_extra_parallel_layout_marker(specs) -> bool:
    return any(
        getattr(spec, "checkpoint_layout_v2", False)
        and isinstance(spec.placement, Shard)
        and getattr(spec, "persistent_fsdp_shard_dim", None) is None
        for spec in (specs or {}).values()
    )


def _check_canonical_spec_coverage(model, specs, state_keys, *, trainable_only):
    from torch.distributed.checkpoint.state_dict import _get_fqns

    params = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers(remove_duplicate=False))
    for name, param in params.items():
        if trainable_only and not param.requires_grad:
            continue
        if any(canonical in state_keys for canonical in _get_fqns(model, name)) and name not in (specs or {}):
            raise RuntimeError(f"Canonical ExtraParallel plan is missing parameter: {name}")
    for name, spec in (specs or {}).items():
        if not isinstance(spec.placement, Shard) or getattr(spec, "persistent_fsdp_shard_dim", None) is not None:
            continue
        value = params.get(name, buffers.get(name))
        if value is None:
            raise RuntimeError(f"ExtraParallel checkpoint spec has no runtime tensor: {name}")
        if trainable_only and name in params and not value.requires_grad:
            continue
        if name not in state_keys:
            raise RuntimeError(f"ExtraParallel checkpoint spec is not present in the canonical state dict: {name}")


def _reject_unproven_extra_parallel_optimizer(
    state, parallel_state, *, group=None, trainable_only=False, allow_proven_optimizer=False
):
    error, signature = None, None
    try:
        specs = getattr(state.get("model"), "_fqn2spec_info", None)
        if _needs_extra_parallel_layout_marker(specs):
            _extra_parallel_layout_key(specs)
            signature = _runtime_extra_parallel_layout_signature(specs)
        if state.get("optimizer") is not None:
            ps = parallel_state if parallel_state is not None else get_parallel_state()
        else:
            ps = None
        if state.get("optimizer") is not None and getattr(ps, "dp_mode", None) == "fsdp2":
            if _needs_extra_parallel_layout_marker(specs) and not allow_proven_optimizer:
                raise RuntimeError("Canonical ExtraParallel optimizer state requires an owner-aware checkpoint codec")
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "checking canonical ExtraParallel optimizer state", group=group)
    error = None
    try:
        if signature is not None:
            model = state["model"]
            options = StateDictOptions(ignore_frozen_params=True) if trainable_only else None
            canonical_state = get_model_state_dict(model, options=options)
            _check_canonical_spec_coverage(model, specs, canonical_state.keys(), trainable_only=trainable_only)
            tensors = dict(model.named_parameters(remove_duplicate=False))
            tensors.update(model.named_buffers(remove_duplicate=False))
            for name, spec in specs.items():
                if (
                    not isinstance(spec.placement, Shard)
                    or getattr(spec, "persistent_fsdp_shard_dim", None) is not None
                ):
                    continue
                if name not in tensors:
                    raise RuntimeError(f"ExtraParallel checkpoint tensor {name} is absent from the runtime model")
                mesh = spec.para_fsdp_mesh
                fsdp_mesh = mesh[mesh.mesh_dim_names[:-1]]
                _validate_extra_parallel_tensor_v2(
                    tensors[name], mesh, fsdp_mesh, spec.placement.dim, spec.fsdp_shard_dim, spec.global_shape
                )
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "checking canonical ExtraParallel runtime tensors", group=group)


def _extra_parallel_layout_key(specs):
    """Version canonical tensors independently of their source geometry."""
    for spec in (specs or {}).values():
        if not isinstance(spec.placement, Shard):
            continue
        if getattr(spec, "persistent_fsdp_shard_dim", None) is not None:
            continue
        if not getattr(spec, "checkpoint_layout_v2", False):
            raise RuntimeError("Canonical ExtraParallel checkpoints cannot mix versioned and legacy sharded tensors")
        _normalized_extra_parallel_dims(spec)
    digest = hashlib.sha256(b"veomni-extra-parallel-canonical-v2").hexdigest()
    return f"{_EXTRA_PARALLEL_LAYOUT_MARKER}{digest}"


def _runtime_extra_parallel_layout_signature(specs):
    """Detect mutation while a single Stateful adapter is in use."""
    if _needs_extra_parallel_layout_marker(specs):
        _extra_parallel_layout_key(specs)
    return tuple(
        (
            name,
            spec.para_name,
            repr(spec.placement),
            spec.fsdp_shard_dim,
            getattr(spec, "persistent_fsdp_shard_dim", None),
            getattr(spec, "global_shape", None),
            getattr(spec, "checkpoint_layout_v2", False),
            tuple(spec.para_fsdp_mesh.shape) if spec.para_fsdp_mesh else None,
        )
        for name, spec in sorted((specs or {}).items())
    )


def _strip_extra_parallel_layout_marker(state_dict, required, marker_key):
    if not required:
        return state_dict
    state_dict = dict(state_dict)
    marker = state_dict.pop(marker_key, None)
    if marker != "ep-outer-v1":
        raise RuntimeError(
            "Unversioned same-dimension ExtraParallel checkpoint; explicit layout migration is required"
        )
    prefix = marker_key + "__"
    if any(not key.startswith(prefix) for key in state_dict):
        raise RuntimeError("ExtraParallel checkpoint contains an unexpected unversioned tensor key")
    return {key[len(prefix) :]: value for key, value in state_dict.items()}


def _namespace_extra_parallel_layout_state(state_dict, marker_key):
    """Give v2 tensors distinct keys so a legacy planner cannot read them as v1."""
    prefix = marker_key + "__"
    result = {prefix + key: value for key, value in state_dict.items()}
    result[marker_key] = "ep-outer-v1"
    return result


def _export_extra_parallel_layout_key(key):
    if not key.startswith(_EXTRA_PARALLEL_LAYOUT_MARKER):
        return key
    marker_length = len(_EXTRA_PARALLEL_LAYOUT_MARKER) + 64
    marker = key[:marker_length]
    if not _is_extra_parallel_layout_marker(marker):
        raise RuntimeError("Malformed versioned ExtraParallel checkpoint key")
    if marker != _extra_parallel_layout_key({}):
        raise RuntimeError("Unknown ExtraParallel checkpoint layout version; a verified migration is required")
    if key == marker:
        return None
    if key[marker_length : marker_length + 2] != "__":
        raise RuntimeError("Malformed versioned ExtraParallel checkpoint tensor key")
    return key[marker_length + 2 :]


def _read_extra_parallel_marker(reader, marker_key):
    scratch = {"model": {marker_key: ""}}
    dcp.load(scratch, storage_reader=reader, no_dist=True)
    if scratch["model"][marker_key] != "ep-outer-v1":
        raise RuntimeError("Invalid ExtraParallel checkpoint layout marker value")


def _extra_parallel_first_mesh(mesh: DeviceMesh) -> DeviceMesh:
    """Return a view with the ExtraParallel dim outside the FSDP dim.

    The runtime mesh is ``(para_fsdp, para)`` (or
    ``(para_replicate, para_fsdp, para)`` for HSDP). When both placements
    shard the same tensor dim, PyTorch composes mesh dims outer-to-inner.
    Keeping the runtime order would therefore describe FSDP-outer /
    ExtraParallel-inner shards, while the optimizer stores the inverse layout.
    Swapping only those two mesh dims preserves rank-local storage and makes
    DTensor/DCP publish the real EP-outer / FSDP-inner global offsets.
    """
    key = id(mesh)
    cached = _EXTRA_PARALLEL_FIRST_MESHES.get(key)
    if cached is not None and cached[0]() is mesh:
        return cached[1]
    names = mesh.mesh_dim_names
    if names is None or len(names) != mesh.ndim or mesh.ndim not in (2, 3):
        raise RuntimeError("ExtraParallel checkpoint mesh must have 2 or 3 named dimensions")
    order = tuple(range(mesh.ndim - 2)) + (mesh.ndim - 1, mesh.ndim - 2)
    reordered = DeviceMesh.from_group(
        [mesh.get_group(index) for index in order],
        device_type=mesh.device_type,
        mesh=mesh.mesh.permute(order),
        mesh_dim_names=tuple(names[index] for index in order),
    )

    def discard(reference):
        current = _EXTRA_PARALLEL_FIRST_MESHES.get(key)
        if current is not None and current[0] is reference:
            _EXTRA_PARALLEL_FIRST_MESHES.pop(key, None)

    _EXTRA_PARALLEL_FIRST_MESHES[key] = (weakref.ref(mesh, discard), reordered)
    return reordered


def _from_local_preserve_device(local_tensor, device_mesh, placements, *, shape=None, stride=None):
    """Wrap optimizer CPU state without moving it onto its accelerator mesh.

    DTensor.from_local implicitly copies CPU input to device_mesh.device_type.
    A metadata-only construction computes the same global shape/stride/spec;
    attaching the actual local shard then preserves CPU storage and the original
    mesh identity. DTensor's spec constructor is intentionally isolated here.
    No collectives or accelerator allocations are needed (run_check=False).
    """
    if local_tensor.device.type == "cpu" and device_mesh.device_type != "cpu":
        metadata = DTensor.from_local(
            local_tensor.to("meta"),
            device_mesh=device_mesh,
            placements=placements,
            run_check=False,
            shape=shape,
            stride=stride,
        )
        return DTensor(local_tensor, metadata._spec, requires_grad=local_tensor.requires_grad)
    return DTensor.from_local(
        local_tensor,
        device_mesh=device_mesh,
        placements=placements,
        run_check=False,
        shape=shape,
        stride=stride,
    )


def _chunk_extent(extent: int, num_chunks: int, index: int) -> int:
    """Extent of ``index`` under ``torch.chunk`` / DTensor ``Shard`` semantics."""
    full = -(-extent // num_chunks)
    start = min(full * index, extent)
    return min(extent, start + full) - start


def _dropped_extra_parallel_shape(loaded_tensor, ep_shard_dim, fsdp_shard_dim, extra_parallel_name=None):
    """Global shape/stride of one ExtraParallel slice on the FSDP sub-mesh."""
    shape = list(loaded_tensor.shape)
    if ep_shard_dim == fsdp_shard_dim:
        mesh = loaded_tensor.device_mesh
        mesh_dim_names = mesh.mesh_dim_names
        if mesh_dim_names is None or len(mesh_dim_names) != mesh.ndim:
            raise RuntimeError("ExtraParallel checkpoint mesh must have named dimensions")
        last_two = mesh_dim_names[-2:]
        if extra_parallel_name is not None and extra_parallel_name in last_two:
            ep_dim = len(mesh_dim_names) - 2 if last_two[0] == extra_parallel_name else len(mesh_dim_names) - 1
            fsdp_dim = len(mesh_dim_names) - 2 if ep_dim == len(mesh_dim_names) - 1 else len(mesh_dim_names) - 1
        elif last_two[-1] == f"{last_two[0]}_fsdp":
            ep_dim, fsdp_dim = len(mesh_dim_names) - 2, len(mesh_dim_names) - 1
        elif last_two[0] == f"{last_two[-1]}_fsdp":
            fsdp_dim, ep_dim = len(mesh_dim_names) - 2, len(mesh_dim_names) - 1
        else:
            raise RuntimeError(
                f"Cannot determine ExtraParallel/FSDP mesh order from mesh dimensions {mesh_dim_names}."
            )
        ep_size = mesh.size(ep_dim)
        fsdp_size = mesh.size(fsdp_dim)
        coordinate = mesh.get_coordinate()
        ep_rank, fsdp_rank = coordinate[ep_dim], coordinate[fsdp_dim]
        extent = shape[ep_shard_dim]
        if ep_dim < fsdp_dim:
            # restore_extra_parallel_dim lays same-dimension checkpoints out as
            # EP-outer / FSDP-inner nested chunks. Mirror that layout instead
            # of flat chunking when the extent is not evenly divisible.
            ep_extent = _chunk_extent(extent, ep_size, ep_rank)
            expected_local = _chunk_extent(ep_extent, fsdp_size, fsdp_rank)
        else:
            # Legacy FSDP-outer metadata has no contiguous EP slice unless the
            # extent is evenly divisible by both mesh dimensions.
            if extent % (fsdp_size * ep_size):
                raise RuntimeError("Unsupported uneven ExtraParallel/FSDP shard layout")
            ep_extent = extent // ep_size
            expected_local = ep_extent // fsdp_size
        if expected_local != loaded_tensor.to_local().shape[ep_shard_dim]:
            raise RuntimeError("Unsupported uneven ExtraParallel/FSDP shard layout")
        shape[ep_shard_dim] = ep_extent
    else:
        shape[ep_shard_dim] = loaded_tensor.to_local().shape[ep_shard_dim]
    stride = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        stride[index] = stride[index + 1] * shape[index + 1]
    return tuple(shape), tuple(stride)


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
    * ``"substring"`` (optimizer): the key starts with ``state.<FQN>.``, e.g.
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
        tensor = state_dict[name]
        if not torch.is_tensor(tensor) or tensor.ndim == 0:
            continue

        if key_match == "exact":
            if name not in extra_parallel_fqn2spec_info:
                continue
            spec_info = extra_parallel_fqn2spec_info[name]
        else:  # "substring"
            # Anchor the optimizer namespace: a nested parameter may end with
            # another parameter's FQN. Param-group fields are not tensor state.
            matches = [k for k in extra_parallel_keys if name.startswith(f"state.{k}.")]
            if not matches:
                continue
            assert len(matches) == 1, f"Ambiguous ExtraParallel spec match for state key '{name}': {matches}"
            spec_info = extra_parallel_fqn2spec_info[matches[0]]

        if not isinstance(spec_info.placement, Shard):
            continue

        # Persistent ExtraParallel parameters already use their complete
        # runtime placement (for example PLE ``[Shard(1), Shard(0)]`` on
        # ``(ple_fsdp, ple)``). They are not managed by FSDP2, so there is no
        # temporary ExtraParallel dimension to drop or restore around DCP.
        if getattr(spec_info, "persistent_fsdp_shard_dim", None) is not None:
            continue

        assert spec_info.para_fsdp_mesh is not None, f"ExtraParallel spec {name} must have an ExtraParallel FSDP mesh"

        # Drop the innermost ExtraParallel (e.g. ``ep``) dim and keep the FSDP
        # sub-mesh: 1D ``(ep_fsdp,)`` for plain FSDP, or 2D
        # ``(ep_replicate, ep_fsdp)`` for HSDP. Mirrors the mesh slicing used
        # when sharding the module in ``torch_parallelize``.
        fsdp_submesh = spec_info.para_fsdp_mesh[spec_info.para_fsdp_mesh.mesh_dim_names[:-1]]
        # The ExtraParallel (e.g. ``ep``) dim shards the tensor along
        # ``spec_info.placement.dim`` (dim 0 for experts), while FSDP shards it
        # along ``spec_info.fsdp_shard_dim`` (1 by default, 0 for Muon zero-comm).
        # These are two DIFFERENT tensor dims, so the placements must use each
        # one explicitly instead of a fixed ``[Shard(0), Shard(1)]``.
        ep_shard_dim = spec_info.placement.dim
        fsdp_shard_dim = spec_info.fsdp_shard_dim
        versioned = getattr(spec_info, "checkpoint_layout_v2", False)
        if action == "drop" and not versioned:
            tensor = drop_extra_parallel_dim(tensor, fsdp_submesh, fsdp_shard_dim)
        elif action == "restore" and not versioned:
            tensor = restore_extra_parallel_dim(
                tensor, spec_info.para_fsdp_mesh, fsdp_submesh, ep_shard_dim, fsdp_shard_dim
            )
        elif action == "drop":
            tensor = drop_extra_parallel_dim_v2(
                tensor,
                fsdp_submesh,
                fsdp_shard_dim,
                extra_parallel_name=spec_info.para_name,
            )
        else:
            tensor = restore_extra_parallel_dim_v2(
                tensor,
                spec_info.para_fsdp_mesh,
                fsdp_submesh,
                ep_shard_dim,
                fsdp_shard_dim,
                global_shape=spec_info.global_shape,
            )
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

    def __init__(self, model, trainable_only: bool = False, parallel_state=None):
        self.model = model
        self.trainable_only = trainable_only

        # Determine whether this is ExtraParallel+FSDP2 case
        # If so, we need to restore Para(e.g. EP)-dim before saving to DCP
        self.parallel_state = parallel_state if parallel_state is not None else get_parallel_state()
        self.extra_parallel_fqn2spec_info = getattr(self.model, "_fqn2spec_info", None)
        self._layout_marker_key = (
            _extra_parallel_layout_key(self.extra_parallel_fqn2spec_info)
            if _needs_extra_parallel_layout_marker(self.extra_parallel_fqn2spec_info)
            else None
        )
        self._layout_signature = _runtime_extra_parallel_layout_signature(self.extra_parallel_fqn2spec_info)
        self.should_extra_parallel_aware = (
            self.extra_parallel_fqn2spec_info is not None and self.parallel_state.dp_mode == "fsdp2"
        )

    @torch.no_grad()
    def state_dict(self):
        if self.extra_parallel_fqn2spec_info is not getattr(self.model, "_fqn2spec_info", None) or (
            self._layout_signature != _runtime_extra_parallel_layout_signature(self.extra_parallel_fqn2spec_info)
        ):
            raise RuntimeError("ExtraParallel layout changed while checkpoint adapter was active")
        options = StateDictOptions(ignore_frozen_params=True) if self.trainable_only else None
        model_state_dict = get_model_state_dict(model=self.model, options=options)
        if self.should_extra_parallel_aware and _needs_extra_parallel_layout_marker(self.extra_parallel_fqn2spec_info):
            _check_canonical_spec_coverage(
                self.model,
                self.extra_parallel_fqn2spec_info,
                model_state_dict.keys(),
                trainable_only=self.trainable_only,
            )
        if self.should_extra_parallel_aware:
            logger.info_rank0(
                "Getting model state_dict from ModelState wrapper, would restore ExtraParallel dim for ExtraParallel (e.g. Experts/Embeds) module"
            )
            # As fsdp+extra parallel and pure extra parallel have different placements, e.g. [Shard(0), Shard(1)] and [Shard(0)],
            # restoring state dict should be extra parallel aware.
            model_state_dict = self.get_state_dict_with_extra_parallel_dim_preprocess(model_state_dict, "restore")
        if is_qwen4_exp_model(self.model) and not self.trainable_only:
            validate_qwen4_gdn_pairs(model_state_dict, qwen4_exp_model_config(self.model))
            names = sorted(name for name in model_state_dict if is_qwen4_gdn_parameter(name))
            if any(model_state_dict[name].dtype != torch.float32 for name in names):
                raise RuntimeError("Qwen4 full checkpoint requires FP32 GDN parameters")
            if names:
                model_state_dict[_QWEN4_GDN_FP32_MARKER] = json.dumps({"version": 1, "parameter_names": names})
        if self.should_extra_parallel_aware and _needs_extra_parallel_layout_marker(self.extra_parallel_fqn2spec_info):
            model_state_dict = _namespace_extra_parallel_layout_state(model_state_dict, self._layout_marker_key)
        return model_state_dict

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        """
        perform the reverse operation for state_dict()
        need to drop ExtraParallel-dim when loading from DCP checkpoints
        so that ExtraParallel-FSDP would not be confused
        """
        if self.extra_parallel_fqn2spec_info is not getattr(self.model, "_fqn2spec_info", None) or (
            self._layout_signature != _runtime_extra_parallel_layout_signature(self.extra_parallel_fqn2spec_info)
        ):
            raise RuntimeError("ExtraParallel layout changed while checkpoint adapter was active")
        model_state_dict = _strip_extra_parallel_layout_marker(
            state_dict,
            self.should_extra_parallel_aware
            and _needs_extra_parallel_layout_marker(self.extra_parallel_fqn2spec_info),
            self._layout_marker_key,
        )
        model_state_dict.pop(_QWEN4_GDN_FP32_MARKER, None)
        model_state_dict.pop(_QWEN4_GDN_FP32_LEGACY_MARKER, None)
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


def _requires_rank_local_optimizer_checkpoint(optimizer):
    # A registered adapter declares an exact, rank-local state contract even
    # when the underlying optimizer does not expose a marker attribute.
    from ..optim.checkpoint_adapter import rank_local_adamw_adapter

    return (
        getattr(optimizer, "requires_exact_state_dtypes", False) is True
        or rank_local_adamw_adapter(optimizer) is not None
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
    cannot be scoped to optimizer-only). ``_ModelStrictLoadPlanner`` therefore
    validates model-key completeness from checkpoint metadata before loading a
    non-LoRA full-model DCP.

    Args:
        model: the model whose parameters the optimizer owns.
        optimizer: the optimizer to save or load.
        parallel_state: optional parallel state; otherwise ``get_parallel_state()`` is used.
        load: when ``True`` (the normal resume path), ``state_dict()`` returns a
            dense state dict so DCP knows where to place restored optimizer
            entries.  When ``False`` (the save path), the returned dict is
            sparse and skips sub-optimizers that have never been stepped,
            avoiding synthetic zero state for unused parameters (e.g. unused
            MoE experts) in the checkpoint.
    """

    def __init__(self, model, optimizer, parallel_state=None, *, load: bool = False):
        self.model = model
        self.optimizer = optimizer
        self.parallel_state = parallel_state if parallel_state is not None else get_parallel_state()
        self.extra_parallel_fqn2spec_info = getattr(self.model, "_fqn2spec_info", None)
        self._layout_marker_key = (
            _extra_parallel_layout_key(self.extra_parallel_fqn2spec_info)
            if _needs_extra_parallel_layout_marker(self.extra_parallel_fqn2spec_info)
            else None
        )
        self._layout_signature = _runtime_extra_parallel_layout_signature(self.extra_parallel_fqn2spec_info)
        self.should_extra_parallel_aware = (
            self.extra_parallel_fqn2spec_info is not None and self.parallel_state.dp_mode == "fsdp2"
        )
        self._load = load

    def state_dict(self):
        if _requires_rank_local_optimizer_checkpoint(self.optimizer):
            raise RuntimeError("Exact-precision optimizers require MultiOptimizer for sparse DCP recovery")
        if self.extra_parallel_fqn2spec_info is not getattr(self.model, "_fqn2spec_info", None) or (
            self._layout_signature != _runtime_extra_parallel_layout_signature(self.extra_parallel_fqn2spec_info)
        ):
            raise RuntimeError("ExtraParallel layout changed while checkpoint adapter was active")
        if self.should_extra_parallel_aware and self._layout_marker_key is not None:
            raise RuntimeError("Canonical ExtraParallel optimizer state requires an owner-aware checkpoint codec")
        if getattr(self.optimizer, "_is_multi_optimizer", False):
            raise RuntimeError(
                "Raw DCP MultiOptimizer does not prove parameter groups or scheduler identity; "
                "use DistributedCheckpointer with the live scheduler."
            )
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
            if _needs_extra_parallel_layout_marker(self.extra_parallel_fqn2spec_info):
                optim_sd_with_extra_parallel_dim = _namespace_extra_parallel_layout_state(
                    optim_sd_with_extra_parallel_dim, self._layout_marker_key
                )
            return optim_sd_with_extra_parallel_dim

        if getattr(self.optimizer, "_is_multi_optimizer", False):
            if self._load:
                return self.optimizer.state_dict()
            return self.optimizer._sparse_state_dict()

        return get_optimizer_state_dict(model=self.model, optimizers=self.optimizer)

    def load_state_dict(self, state_dict):
        """Restore optimizer state from the loaded DCP state dict."""
        if _requires_rank_local_optimizer_checkpoint(self.optimizer):
            raise RuntimeError("Exact-precision optimizers require MultiOptimizer for sparse DCP recovery")
        if self.should_extra_parallel_aware and self._layout_marker_key is not None:
            raise RuntimeError("Canonical ExtraParallel optimizer state requires an owner-aware checkpoint codec")
        if self.extra_parallel_fqn2spec_info is not getattr(self.model, "_fqn2spec_info", None) or (
            self._layout_signature != _runtime_extra_parallel_layout_signature(self.extra_parallel_fqn2spec_info)
        ):
            raise RuntimeError("ExtraParallel layout changed while checkpoint adapter was active")
        optim_state_from_dcp_load = _strip_extra_parallel_layout_marker(
            state_dict,
            self.should_extra_parallel_aware
            and _needs_extra_parallel_layout_marker(self.extra_parallel_fqn2spec_info),
            self._layout_marker_key,
        )
        if self.should_extra_parallel_aware:
            # we need to drop ExtraParallel dim before loading them into optimizers
            optim_state_without_extra_parallel_dim = self.get_state_dict_with_extra_parallel_dim_preprocess(
                optim_state_from_dcp_load, "drop"
            )
            # Delegate to MultiOptimizer (it will split/filter correctly)
            self.optimizer.load_state_dict(optim_state_without_extra_parallel_dim)
            return

        if getattr(self.optimizer, "_is_multi_optimizer", False):
            self.optimizer.load_state_dict(optim_state_from_dcp_load)
            return

        # Single torch optimizer.
        # ``strict=False`` matches the DCP planner's allow_partial_load intent:
        # params that never received a gradient (and thus have no saved Adam
        # state) keep the default-initialized state that
        # ``set_optimizer_state_dict`` / ``_init_optim_state`` already created.
        # Torch 2.11+ raises under the default strict=True when any
        # requires_grad param is missing from the checkpoint (DeepSeek-V4
        # indexer ``position_bias`` is one such case on short toy runs).
        set_optimizer_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            optim_state_dict=optim_state_from_dcp_load,
            options=StateDictOptions(strict=False),
        )

    def get_state_dict_with_extra_parallel_dim_preprocess(self, state_dict, action):
        return _apply_extra_parallel_dim(
            state_dict,
            self.extra_parallel_fqn2spec_info,
            self.parallel_state,
            action,
            key_match="substring",
        )


class _ProvenModelState(ModelState):
    def __init__(self, *args, optimizer, group, writing=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.group = group

    def state_dict(self):
        result, error = None, None
        try:
            result = super().state_dict()
            logical_keys = {key for raw in result if (key := _export_extra_parallel_layout_key(raw)) is not None}
            if self.trainable_only:
                names = _dcp_parameter_name_map(self.model)
                for leaf in _iter_optimizer_leaves(self.optimizer):
                    for param, saved in leaf.state.items():
                        aliases = [name for name, owned in names.items() if owned is param]
                        if saved and not any(name in logical_keys for name in aliases):
                            raise RuntimeError("trainable-only save would omit a parameter with optimizer history")
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "constructing proven model state", group=self.group)
        return result

    def load_state_dict(self, state_dict):
        error = None
        try:
            super().load_state_dict(state_dict)
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "applying proven model state", group=self.group)


class _ProvenMultiOptimizerState(OptimizerState):
    """High-level DCP adapter; direct optimizer APIs retain their own contract."""

    def __init__(
        self,
        *args,
        generation,
        scheduler_snapshot,
        scheduler_object,
        extra_digest,
        expected_layout,
        group,
        frozen_digest=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.generation = generation
        self.scheduler_snapshot = scheduler_snapshot
        self.scheduler_object = scheduler_object
        self.extra_digest = extra_digest
        self.expected_layout = expected_layout
        self.group = group
        self.frozen_digest = frozen_digest

    def state_dict(self):
        result, error = None, None
        try:
            result = self.optimizer._sparse_state_dict(coordinate=True, group=self.group)
            if self.should_extra_parallel_aware:
                from .rank_local_optimizer import (
                    current_rank,
                    local_optimizer_state,
                    rank_inventory,
                    rank_inventory_key,
                )

                inventory = rank_inventory(result, current_rank())
                result = local_optimizer_state(result, current_rank())
                result[rank_inventory_key(current_rank())] = "" if self._load else json.dumps(inventory)
            else:
                from torch.distributed.checkpoint._nested_dict import flatten_state_dict

                flattened, _ = flatten_state_dict({"optimizer": result})
                inventory = sorted(flattened)
                result[_MULTI_INVENTORY_MARKER] = "" if self._load else json.dumps(inventory)
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "constructing proven MultiOptimizer state", group=self.group)
        if self._load:
            result[MULTI_LAYOUT_MARKER] = ""
            return result
        initial = coordinated_multi_layout(
            self.model,
            self.optimizer,
            self.scheduler_snapshot,
            self.scheduler_object,
            writing=True,
            group=self.group,
            frozen_digest=self.frozen_digest,
        )
        if initial != self.expected_layout:
            raise RuntimeError("MultiOptimizer layout changed while saving checkpoint")
        current = coordinated_multi_layout(
            self.model,
            self.optimizer,
            self.scheduler_snapshot,
            self.scheduler_object,
            self.extra_digest,
            writing=True,
            group=self.group,
            frozen_digest=self.frozen_digest,
        )
        result[MULTI_LAYOUT_MARKER] = json.dumps(
            {
                "generation": self.generation,
                "layout": current,
                "inventory_digests": _inventory_digests(inventory, group=self.group),
            },
            sort_keys=True,
        )
        return result

    def load_state_dict(self, state_dict):
        error = None
        try:
            state_dict = dict(state_dict)
            marker = json.loads(state_dict.pop(MULTI_LAYOUT_MARKER))
            if (
                not isinstance(marker, dict)
                or set(marker) != {"generation", "layout", "inventory_digests"}
                or marker["generation"] != self.generation
                or marker["layout"] != self.expected_layout
            ):
                raise RuntimeError("MultiOptimizer identity changed during tensor loading")
            if self.should_extra_parallel_aware:
                from .rank_local_optimizer import (
                    current_rank,
                    rank_inventory,
                    rank_inventory_key,
                    restore_optimizer_state,
                )

                inventory = json.loads(state_dict.pop(rank_inventory_key(current_rank())))
                template = self.optimizer._sparse_state_dict()
                if inventory != rank_inventory(template, current_rank()):
                    raise RuntimeError("rank-local optimizer checkpoint inventory changed during loading")
                state_dict = restore_optimizer_state(state_dict, template, current_rank())
            else:
                from torch.distributed.checkpoint._nested_dict import flatten_state_dict

                inventory = json.loads(state_dict.pop(_MULTI_INVENTORY_MARKER))
                flattened, _ = flatten_state_dict({"optimizer": state_dict})
                if inventory != sorted(flattened):
                    raise RuntimeError("MultiOptimizer checkpoint inventory changed during loading")
            # The DCP reader populated mmap-backed PLE moments. Flush and
            # evict those pages before applying the optimizer state, so the
            # restore does not retain two full passes of resident pages.
            for leaf in _iter_optimizer_leaves(self.optimizer):
                if getattr(leaf, "_ple_file_mode_enabled", False):
                    leaf.checkpoint_evict_file_backed_state()
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "validating loaded MultiOptimizer state", group=self.group)
        if marker["inventory_digests"] != _inventory_digests(inventory, group=self.group):
            raise RuntimeError("MultiOptimizer inventory identity changed during tensor loading")
        current = coordinated_multi_layout(
            self.model,
            self.optimizer,
            self.scheduler_snapshot,
            self.scheduler_object,
            self.extra_digest,
            group=self.group,
            frozen_digest=self.frozen_digest,
        )
        if current != self.expected_layout:
            raise RuntimeError("MultiOptimizer layout changed while loading optimizer state")
        error = None
        try:
            self.optimizer._dcp_load_state_dict(state_dict, group=self.group)
            # Applying the state may fault the same mmap pages back in. Keep
            # the explicit post-load release previously provided by hashing.
            for leaf in _iter_optimizer_leaves(self.optimizer):
                if getattr(leaf, "_ple_file_mode_enabled", False):
                    leaf.checkpoint_evict_file_backed_state()
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "applying proven MultiOptimizer state", group=self.group)


def drop_extra_parallel_dim(loaded_tensor: torch.Tensor, device_mesh: DeviceMesh, fsdp_shard_dim: int = 1):
    """
    Drop ExtraParallel dims after loading from DCP so that ExtraParallel-FSDP would not be confused.

    ``device_mesh`` is the FSDP sub-mesh (ExtraParallel dim already excluded):
    1D ``(ep_fsdp,)`` for plain FSDP, or 2D ``(ep_replicate, ep_fsdp)`` for HSDP.
    ``fsdp_shard_dim`` is the tensor dim FSDP shards along (1 by default, 0 for
    the Muon zero-comm layout). The number of placements on the loaded DTensor
    reflects the full saved mesh:

    * 1 placement: pure ExtraParallel, no FSDP -> return the plain local tensor.
    * 2 placements: ``(ep_fsdp, ep)`` -> keep FSDP ``Shard(fsdp_shard_dim)`` on
      the 1D sub-mesh.
    * 3 placements: ``(ep_replicate, ep_fsdp, ep)`` (HSDP) -> keep
      ``[Replicate(), Shard(fsdp_shard_dim)]`` on the 2D sub-mesh.
    """

    num_placements = len(loaded_tensor.placements)
    if num_placements == 1:
        tensor_to_put = loaded_tensor.to_local()
    elif num_placements == 2:
        tensor_to_put = DTensor.from_local(
            loaded_tensor._local_tensor, device_mesh=device_mesh, placements=[Shard(fsdp_shard_dim)]
        )
    elif num_placements == 3:
        tensor_to_put = DTensor.from_local(
            loaded_tensor._local_tensor, device_mesh=device_mesh, placements=[Replicate(), Shard(fsdp_shard_dim)]
        )
    else:
        raise RuntimeError(
            "Expect ExtraParallel parameters from checkpoints to be DTensor with 1-dim (no FSDP), "
            f"2-dim (ExtraParallel+FSDP) or 3-dim (ExtraParallel+HSDP), got {loaded_tensor}"
        )

    return tensor_to_put


def restore_extra_parallel_dim(
    orgin_tensor: torch.Tensor,
    fsdp_mesh: DeviceMesh,
    extra_parallel_fsdp_mesh: DeviceMesh,
    ep_shard_dim: int = 0,
    fsdp_shard_dim: int = 1,
):
    """
    Restore ExtraParallel dim so that DCP can be aware about ExtraParallel ranks

    The ExtraParallel (e.g. ``ep``) dim and the FSDP dim shard the tensor along
    their configured axes: ``ep_shard_dim`` (the EP plan's ``Shard.dim``, dim 0 for
    experts) and ``fsdp_shard_dim`` (1 by default, 0 for the Muon zero-comm
    layout). They must be mapped to the matching mesh dims explicitly.

    args:
        orgin_tensor (torch.Tensor): The orgin tensor (FSDP-local shard).
        fsdp_mesh (DeviceMesh): The full ExtraParallel mesh, i.e. 2D
            ``(ep_fsdp, ep)`` for plain FSDP or 3D
            ``(ep_replicate, ep_fsdp, ep)`` for HSDP.
        extra_parallel_fsdp_mesh (DeviceMesh): The FSDP sub-mesh (ExtraParallel
            dim excluded), used for the pure-ExtraParallel (no FSDP) path.
        ep_shard_dim (int): Tensor dim the ExtraParallel dim shards along.
        fsdp_shard_dim (int): Tensor dim FSDP shards along.

    Note:
        When both splits use the same tensor dimension, physical sharding is
        EP-first/FSDP-second despite the mesh order ``(ep_fsdp, ep)``. Describe
        that with ``_StridedShard`` on the FSDP mesh axis, just as native FSDP2
        does for same-dimension TP. This preserves global expert identities in
        DCP metadata without communication or copying the local tensor.
    """
    assert fsdp_mesh.ndim in (2, 3), f"global_mesh.ndim must be 2 or 3, got {fsdp_mesh.ndim}"
    if not torch.is_tensor(orgin_tensor):
        raise RuntimeError(f"origin_tensor - {orgin_tensor} is not a tensor!")
    ndim = orgin_tensor.ndim
    if not -ndim <= ep_shard_dim < ndim:
        raise RuntimeError("ExtraParallel shard dimension is outside the tensor rank")
    ep_shard_dim %= ndim

    if isinstance(orgin_tensor, DTensor):
        if not -ndim <= fsdp_shard_dim < ndim:
            raise RuntimeError("FSDP shard dimension is outside the tensor rank")
        fsdp_shard_dim %= ndim
        # ExtraParallel+FSDP2. mesh order is (ep_fsdp, ep) or, for HSDP,
        # (ep_replicate, ep_fsdp, ep): ep_fsdp -> Shard(fsdp_shard_dim),
        # ep -> Shard(ep_shard_dim), ep_replicate -> Replicate().
        fsdp_placement = (
            _StridedShard(fsdp_shard_dim, split_factor=fsdp_mesh.size(-1))
            if ep_shard_dim == fsdp_shard_dim and fsdp_mesh.size(-1) > 1
            else Shard(fsdp_shard_dim)
        )
        if fsdp_mesh.ndim == 3:
            placements = [Replicate(), fsdp_placement, Shard(ep_shard_dim)]
        else:
            placements = [fsdp_placement, Shard(ep_shard_dim)]
        dtensor = DTensor.from_local(orgin_tensor._local_tensor, device_mesh=fsdp_mesh, placements=placements)
    elif torch.is_tensor(orgin_tensor):
        # No FSDP: each rank owns an EP slice. The supplied submesh is the
        # FSDP mesh; use the final (EP) dimension of the full mesh instead.
        if fsdp_mesh.mesh_dim_names is None:
            raise RuntimeError("ExtraParallel mesh must have named dimensions")
        ep_mesh = fsdp_mesh[fsdp_mesh.mesh_dim_names[-1]]
        dtensor = DTensor.from_local(orgin_tensor, device_mesh=ep_mesh, placements=[Shard(ep_shard_dim)])
    else:
        raise RuntimeError(f"origin_tensor - {orgin_tensor} is not a tensor!")

    return dtensor


def drop_extra_parallel_dim_v2(
    loaded_tensor: torch.Tensor,
    device_mesh: DeviceMesh,
    fsdp_shard_dim: int = 1,
    *,
    extra_parallel_name: Optional[str] = None,
):
    """
    Drop ExtraParallel dims after loading from DCP so that ExtraParallel-FSDP would not be confused.

    ``device_mesh`` is the FSDP sub-mesh (ExtraParallel dim already excluded):
    1D ``(ep_fsdp,)`` for plain FSDP, or 2D ``(ep_replicate, ep_fsdp)`` for HSDP.
    ``fsdp_shard_dim`` is the tensor dim FSDP shards along (1 by default, 0 for
    the Muon zero-comm layout). The number of placements on the loaded DTensor
    reflects the full saved mesh:

    * 1 placement: pure ExtraParallel, no FSDP -> return the plain local tensor.
    * 2 placements: ``(ep_fsdp, ep)`` -> keep FSDP ``Shard(fsdp_shard_dim)`` on
      the 1D sub-mesh.
    * 3 placements: ``(ep_replicate, ep_fsdp, ep)`` (HSDP) -> keep
      ``[Replicate(), Shard(fsdp_shard_dim)]`` on the 2D sub-mesh.
    """

    if not -loaded_tensor.ndim <= fsdp_shard_dim < loaded_tensor.ndim:
        raise RuntimeError("FSDP shard dimension is outside the tensor rank")
    fsdp_shard_dim %= loaded_tensor.ndim
    num_placements = len(loaded_tensor.placements)
    if num_placements == 1:
        tensor_to_put = loaded_tensor.to_local()
    elif num_placements == 2:
        ep_placement = loaded_tensor.placements[-1]
        if not isinstance(ep_placement, Shard):
            raise RuntimeError(f"ExtraParallel placement must be Shard, got {ep_placement}")
        ep_shard_dim = ep_placement.dim
        shape, stride = _dropped_extra_parallel_shape(loaded_tensor, ep_shard_dim, fsdp_shard_dim, extra_parallel_name)
        tensor_to_put = _from_local_preserve_device(
            loaded_tensor._local_tensor,
            device_mesh=device_mesh,
            placements=[Shard(fsdp_shard_dim)],
            shape=shape,
            stride=stride,
        )
    elif num_placements == 3:
        ep_placement = loaded_tensor.placements[-1]
        if not isinstance(ep_placement, Shard):
            raise RuntimeError(f"ExtraParallel placement must be Shard, got {ep_placement}")
        ep_shard_dim = ep_placement.dim
        shape, stride = _dropped_extra_parallel_shape(loaded_tensor, ep_shard_dim, fsdp_shard_dim, extra_parallel_name)
        tensor_to_put = _from_local_preserve_device(
            loaded_tensor._local_tensor,
            device_mesh=device_mesh,
            placements=[Replicate(), Shard(fsdp_shard_dim)],
            shape=shape,
            stride=stride,
        )
    else:
        raise RuntimeError(
            "Expect ExtraParallel parameters from checkpoints to be DTensor with 1-dim (no FSDP), "
            f"2-dim (ExtraParallel+FSDP) or 3-dim (ExtraParallel+HSDP), got {loaded_tensor}"
        )

    return tensor_to_put


def _validate_extra_parallel_tensor_v2(tensor, mesh, fsdp_mesh, ep_dim, fsdp_dim, global_shape):
    if not isinstance(tensor, DTensor):
        raise RuntimeError("The v2 ExtraParallel checkpoint layout requires an FSDP DTensor")
    ndim = tensor.ndim
    if not (-ndim <= ep_dim < ndim and -ndim <= fsdp_dim < ndim):
        raise RuntimeError("ExtraParallel shard dimension is outside the tensor rank")
    ep_dim, fsdp_dim = ep_dim % ndim, fsdp_dim % ndim
    if tensor.device_mesh != fsdp_mesh:
        raise RuntimeError("ExtraParallel source tensor does not use the declared FSDP mesh")
    if tuple(tensor.placements) != (Replicate(),) * (fsdp_mesh.ndim - 1) + (Shard(fsdp_dim),):
        raise RuntimeError("ExtraParallel source tensor does not use the declared FSDP placement")
    if global_shape is None or len(global_shape) != ndim:
        raise RuntimeError("FSDP-only ExtraParallel tensors require the original global_shape")

    expected = list(global_shape)
    ep_size, ep_rank = mesh.size(-1), mesh.get_local_rank(-1)
    ep_chunk = (expected[ep_dim] + ep_size - 1) // ep_size
    expected[ep_dim] = max(min(expected[ep_dim] - ep_rank * ep_chunk, ep_chunk), 0)
    if tuple(tensor.shape) != tuple(expected):
        raise RuntimeError("ExtraParallel source tensor shape does not match the declared global shape")
    local_size, local_rank = fsdp_mesh.size(-1), fsdp_mesh.get_local_rank(-1)
    local_chunk = (expected[fsdp_dim] + local_size - 1) // local_size
    expected[fsdp_dim] = max(min(expected[fsdp_dim] - local_rank * local_chunk, local_chunk), 0)
    if tuple(tensor.to_local().shape) != tuple(expected):
        raise RuntimeError("ExtraParallel local tensor shape does not match its declared FSDP placement")


def restore_extra_parallel_dim_v2(
    orgin_tensor: torch.Tensor,
    fsdp_mesh: DeviceMesh,
    extra_parallel_fsdp_mesh: DeviceMesh,
    ep_shard_dim: int = 0,
    fsdp_shard_dim: int = 1,
    global_shape: Optional[tuple[int, ...]] = None,
):
    """
    Restore ExtraParallel dim so that DCP can be aware about ExtraParallel ranks

    The ExtraParallel (e.g. ``ep``) dim and the FSDP dim shard the tensor along
    DIFFERENT axes: ``ep_shard_dim`` (the EP plan's ``Shard.dim``, dim 0 for
    experts) and ``fsdp_shard_dim`` (1 by default, 0 for the Muon zero-comm
    layout). They must be mapped to the matching mesh dims explicitly.

    args:
        orgin_tensor (torch.Tensor): The orgin tensor (FSDP-local shard).
        fsdp_mesh (DeviceMesh): The full ExtraParallel mesh, i.e. 2D
            ``(ep_fsdp, ep)`` for plain FSDP or 3D
            ``(ep_replicate, ep_fsdp, ep)`` for HSDP.
        extra_parallel_fsdp_mesh (DeviceMesh): The FSDP sub-mesh (ExtraParallel
            dim excluded), used for the pure-ExtraParallel (no FSDP) path.
        ep_shard_dim (int): Tensor dim the ExtraParallel dim shards along.
        fsdp_shard_dim (int): Tensor dim FSDP shards along.
        global_shape (tuple[int, ...], optional): The model parameter's shape
            before ExtraParallel and FSDP sharding. It is required for an
            FSDP-only DTensor: a local slice alone cannot prove global shape.

    Note:
        When ``ep_shard_dim == fsdp_shard_dim`` (e.g. the Muon zero-comm layout
        shards experts on dim 0 just like EP), both mesh dims split the SAME
        tensor dim. The temporary checkpoint DTensor uses an EP-outer /
        FSDP-inner mesh view so DCP metadata matches the optimizer's physical
        local shards and global offsets instead of block-transposing the dim.
    """
    assert fsdp_mesh.ndim in (2, 3), f"global_mesh.ndim must be 2 or 3, got {fsdp_mesh.ndim}"
    ndim = orgin_tensor.ndim
    if not (-ndim <= ep_shard_dim < ndim and -ndim <= fsdp_shard_dim < ndim):
        raise RuntimeError("ExtraParallel shard dimension is outside the tensor rank")
    ep_shard_dim %= ndim
    fsdp_shard_dim %= ndim

    if isinstance(orgin_tensor, DTensor):
        _validate_extra_parallel_tensor_v2(
            orgin_tensor, fsdp_mesh, extra_parallel_fsdp_mesh, ep_shard_dim, fsdp_shard_dim, global_shape
        )
        # ExtraParallel+FSDP2. mesh order is (ep_fsdp, ep) or, for HSDP,
        # (ep_replicate, ep_fsdp, ep): ep_fsdp -> Shard(fsdp_shard_dim),
        # ep -> Shard(ep_shard_dim), ep_replicate -> Replicate().
        if ep_shard_dim == fsdp_shard_dim:
            checkpoint_mesh = _extra_parallel_first_mesh(fsdp_mesh)
            if checkpoint_mesh.ndim == 3:
                placements = [Replicate(), Shard(ep_shard_dim), Shard(fsdp_shard_dim)]
            else:
                placements = [Shard(ep_shard_dim), Shard(fsdp_shard_dim)]
        else:
            checkpoint_mesh = fsdp_mesh
            if fsdp_mesh.ndim == 3:
                placements = [Replicate(), Shard(fsdp_shard_dim), Shard(ep_shard_dim)]
            else:
                placements = [Shard(fsdp_shard_dim), Shard(ep_shard_dim)]
        shape = tuple(global_shape)
        stride = [1] * len(shape)
        for index in range(len(shape) - 2, -1, -1):
            stride[index] = stride[index + 1] * shape[index + 1]
        dtensor = _from_local_preserve_device(
            orgin_tensor._local_tensor,
            device_mesh=checkpoint_mesh,
            placements=placements,
            shape=shape,
            stride=tuple(stride),
        )
    elif torch.is_tensor(orgin_tensor):
        raise RuntimeError("The v2 ExtraParallel checkpoint layout requires an FSDP DTensor")
    else:
        raise RuntimeError(f"origin_tensor - {orgin_tensor} is not a tensor!")

    return dtensor


def _local_rank() -> int:
    """This process's rank within its node, as set by the elastic launcher.

    Elects one rank per node for the node-local staging work; defaults to 0 so
    single-process runs still take the leader path.
    """
    value = os.environ.get("LOCAL_RANK", "0")
    return int(value) if value.isdigit() else 0


def _stage_key(path: str) -> str:
    """Directory name that isolates one run's staged files from another's.

    Substituting separators would collide -- it maps ``/tmp/a_b/c`` and
    ``/tmp/a/b_c`` onto one name -- so the name is a digest of the absolute path,
    prefixed with the basename to stay identifiable on disk.
    """
    absolute = os.path.abspath(path)
    digest = hashlib.sha256(absolute.encode("utf-8")).hexdigest()[:16]
    return f"{os.path.basename(absolute) or 'ckpt'}-{digest}"


class _Promotion:
    """Failure state shared by the phases of one promotion.

    ``error`` is what this rank saw, ``failed`` what the whole group saw: the work
    is split across ranks -- one leader per node copies that node's files -- so a
    failure starts out visible to a single rank.
    """

    def __init__(self) -> None:
        self.error: Optional[BaseException] = None
        self.failed = False


def _promotion_phase(state: _Promotion, work, *, participates: bool, group: Optional[Any] = None) -> None:
    """Run one phase on the ranks that take part, then let every rank agree on the result.

    The closing reduction is the phase's only collective and every rank reaches it
    on every path, including the failing one. Collectives are untagged, so a rank
    that returned early would leave the others pairing up with the wrong one from
    then on and the save would hang instead of failing; one collective per phase
    keeps the counts equal by construction rather than by inspection.

    ``BaseException`` because ``work`` is arbitrary and the guarantee above is
    structural: anything that escapes this catch skips the reduction, and the
    ranks that did reach it wait for a peer that has already left.
    """
    if participates and not state.failed:
        try:
            work()
        except BaseException as e:  # noqa: BLE001 - raised once every phase is done
            if state.error is None:
                state.error = e
    try:
        state.failed = any_rank_failed(state.error is not None, group=group) or state.failed
    except BaseException as group_error:
        # The group itself failed, e.g. peers timed out first. Keep this rank's own
        # error as the cause, or the log only shows the connection closing.
        if state.error is not None:
            raise group_error from state.error
        raise


_STAGE_ROOT = "veomni_ckpt_stage"


def _gloo_timeout(timeout_seconds: Optional[int]) -> Optional[timedelta]:
    """``save_timeout_seconds`` as a process-group timeout; None keeps gloo's default."""
    return timedelta(seconds=timeout_seconds) if timeout_seconds is not None else None


def _prepare_stage_dir(stage_dir: str, path: str, group: Optional[Any] = None) -> str:
    """Create the empty staging directory for the run writing to ``path``.

    One directory per run, shared by every checkpoint it writes and emptied
    first. That is also how a save killed part-way is cleaned up: its copy -- the
    size of the model plus its optimizer state -- is left exactly where the next
    save clears it, instead of stranded under a key naming its own step.

    Keyed on ``path`` because ``stage_dir`` is often something generic like /tmp
    and this directory gets swept: two runs sharing a node must not land in the
    same place, or one would delete the other's staged data.

    Only the node leader touches the filesystem; peers would race the sweep and
    have no need to, since the reduction below is a collective. That reduction
    also keeps a per-node failure -- a full or read-only scratch disk -- from
    being seen by one rank alone, which would leave the rest waiting in
    ``dcp.save`` on a collective that never arrives.
    """
    stage_path = os.path.join(stage_dir, _STAGE_ROOT, _stage_key(path))
    error: Optional[BaseException] = None
    if _local_rank() == 0:
        try:
            if os.path.lexists(stage_path):
                shutil.rmtree(stage_path)
            # Failing rather than reusing an uncleared directory prevents a
            # previous checkpoint's sidecars from joining this transaction.
            os.makedirs(stage_path, exist_ok=False)
        except BaseException as e:  # noqa: BLE001 - raised once every rank has agreed
            # ``BaseException`` for the reason ``_promotion_phase`` documents:
            # anything that escapes here skips the reduction below, and the ranks
            # that did reach it wait for a peer that has already left.
            error = e
    if any_rank_failed(error is not None, group=group):
        raise error or RuntimeError(f"another rank could not prepare a staging directory under {stage_dir}")
    return stage_path


def _promote_staged_checkpoint(
    stage_path: str, final_path: str, step_root: Optional[str] = None, group: Optional[Any] = None
) -> None:
    """Copy a staged checkpoint to its destination, then drop the staged copy.

    The staging directory is node-local, so one rank per node copies all of it
    rather than each rank working out which files it wrote; that keeps this
    independent of DCP's file naming.

    Three phases, each ending in a single collective on ``group`` (see
    ``_promotion_phase``). A phase's error is re-raised on every rank once all
    three have run; a failure of the group itself -- a timeout, a dead peer --
    raises out of the phase it hit, where NCCL would have aborted the process.
    Either way a node leader frees its staged copy in a ``finally``; one killed
    mid-copy leaves it for the next save's ``_prepare_stage_dir`` to sweep.

    ``.metadata`` is what DCP reads as "this DCP directory is complete", and the
    staged tree holds one per directory -- ``ckpt/`` and ``optimizer/``. The
    destination is emptied first, before anything is overwritten, and the new
    markers go last and only if every rank's data landed -- so a reader sees
    either the previous complete checkpoint or none, never a completion marker
    over data that is only partly there, and never this save's files mixed with
    a previous one's.

    This is the *only* place a staged save invalidates its destination. Doing it
    when the save starts, as the unstaged path does, would throw away the
    previous checkpoint before the new one exists anywhere -- and keeping that
    checkpoint readable until the last possible moment is what staging is for.

    ``step_root`` is the step directory, given when the destination may hold a
    pre-split checkpoint whose marker sits there rather than inside
    ``final_path``. Nested files are copied in the data phase, before any
    ``.metadata`` is. The per-rank ``extra_state/`` files are among those files.
    """
    metadata_name = DCP_MARKER_FILENAME
    is_node_leader = _local_rank() == 0
    is_coordinator = (not dist.is_initialized()) or dist.get_rank() == 0
    state = _Promotion()

    def staged_markers() -> list[str]:
        """Relative path of every ``.metadata`` in the staged tree.

        DCP writes the marker from its coordinator rank, so only that rank has
        them staged -- which is also the only rank that deletes and copies them.
        """
        rels: list[str] = []
        for dirpath, _dirnames, filenames in os.walk(stage_path):
            for filename in filenames:
                if filename == metadata_name:
                    rels.append(os.path.relpath(os.path.join(dirpath, filename), stage_path))
        return sorted(rels)

    # Listed once, up front. Retracting a marker must not depend on the staged
    # tree still being there: the cleanup below removes it, and a promotion that
    # failed is exactly when both run.
    markers = staged_markers()

    def clear_destination() -> None:
        """Empty the destination, so the staged tree replaces it rather than merges.

        Deleting only the files about to be overwritten would leave whatever the
        previous save wrote and this one does not: a weights-only save over a
        step that has an optimizer keeps that optimizer's shards *and* its
        ``.metadata``, and the step then reads as a complete checkpoint pairing
        this run's weights with an earlier run's optimizer.

        Nothing is lost that the destination could still have used. A marker has
        to go before its shards are overwritten either way, and without one DCP
        will not read the directory at all.

        One module's directory. A sibling module of the same job lives beside it,
        is not being written, and keeps everything it has.
        """
        # The step may predate this layout, in which case DCP left its marker at
        # the step root rather than in here. Delete this import (and
        # veomni/checkpoint/legacy_v0_1_12.py) to drop that layout.
        from .legacy_v0_1_12 import drop_marker

        shutil.rmtree(final_path, ignore_errors=True)
        os.makedirs(final_path, exist_ok=True)
        if step_root is not None:
            drop_marker(step_root)

    def copy_this_nodes_files() -> None:
        """Copy every staged file on this node, except the completion markers.

        Walks the tree so nested directories are copied, not skipped.
        """
        os.makedirs(final_path, exist_ok=True)

        names: list[str] = []
        for dirpath, _dirnames, filenames in os.walk(stage_path):
            for filename in filenames:
                if filename == metadata_name:
                    continue
                names.append(os.path.relpath(os.path.join(dirpath, filename), stage_path))
        names.sort()

        def _copy(rel: str) -> None:
            """Copy one staged file to the destination, preserving its relative path."""
            src = os.path.join(stage_path, rel)
            dst = os.path.join(final_path, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)

        if names:
            with ThreadPoolExecutor(max_workers=min(16, len(names))) as pool:
                list(pool.map(_copy, names))

    def drop_destination_markers() -> None:
        """Remove every marker this promotion could have copied.

        Earlier markers go with the one that failed: half a model is not
        resumable, and leaving one valid directory behind would misreport which
        part survived.
        """
        for rel in markers:
            dst = os.path.join(final_path, rel)
            try:
                if os.path.exists(dst):
                    os.remove(dst)
            except OSError:
                logger.error(f"could not remove the completion marker {dst}", exc_info=True)

    def copy_markers() -> None:
        """Copy the completion markers last, or leave none behind if that fails."""
        try:
            for rel in markers:
                dst = os.path.join(final_path, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(os.path.join(stage_path, rel), dst)
        except BaseException:
            # copyfile creates the destination before writing it, so a failure
            # can leave a truncated marker -- worse than none, since DCP would
            # read it as a complete directory.
            drop_destination_markers()
            raise

    promoted = False
    try:
        _promotion_phase(state, clear_destination, participates=is_coordinator, group=group)
        _promotion_phase(state, copy_this_nodes_files, participates=is_node_leader, group=group)
        _promotion_phase(state, copy_markers, participates=is_coordinator, group=group)
        promoted = not state.failed
    finally:
        # The markers are copied before the phase that agrees on them, so a
        # failure of the agreement itself -- a peer gone, the group timed out --
        # would leave them standing over a save that did not finish. The step
        # keeps the previous run's manifest until the cursor files are rewritten,
        # and the two together read as a complete checkpoint pairing this run's
        # model state with that run's cursor.
        #
        # This covers the failures the coordinator itself observes, which is
        # every agreed one. A reduction that fails on a peer alone, or a
        # coordinator that dies between copying the markers and agreeing on
        # them, still leaves them behind; the next save of that step clears the
        # destination before writing, so they are stale for one step at most.
        if is_coordinator and not promoted:
            drop_destination_markers()
        # The staged copy is as large as the model plus its optimizer state, and
        # nothing is lost by dropping it: without a marker the destination reads
        # as incomplete, which it is.
        if is_node_leader:
            shutil.rmtree(stage_path, ignore_errors=True)

    if state.error is not None:
        raise state.error
    if state.failed:
        raise RuntimeError("checkpoint promotion failed on another rank; no completion marker was written")


class DistributedCheckpointer(CheckpointerBase):
    """
    Distributed checkpointer for torch.distributed.checkpoint
    """

    # One in-flight async save per slot. A step writes the weights and the
    # optimizer into two directories, so there are two saves to track rather than
    # one.
    #
    # Each slot gets its own Gloo group. ``dcp.async_save`` hands the group it is
    # given to a background thread that runs the *whole* save, collectives
    # included -- it asserts a CPU backend for exactly that reason. Two saves
    # sharing one group would have two threads issuing collectives on it at once,
    # and ranks that interleave them differently deadlock. Separate groups let
    # the two writes overlap; sharing one would force the second to wait out the
    # first, which is what draining before each save used to do.
    _save_futures: Dict[str, Any] = {}
    _async_process_groups: Dict[str, Any] = {}
    # The gloo group a staged save's own collectives run on, created on first use.
    _stage_process_group: Optional[Any] = None
    # Both group caches live as long as the process: a run initialises distributed
    # once and tears it down on the way out, so there is no point at which a stale
    # group could be read back. Reusing a process across runs would have to clear
    # them here, together with ``_save_futures``.

    @classmethod
    def save(
        cls,
        path: str,
        state: Dict[str, Any],
        save_async: bool = False,
        global_steps: int = None,
        module: str = "",
        trainable_only: bool = False,
        save_to_lowest_rank: bool = False,
        parallel_state=None,
        stage_dir: Optional[str] = None,
        save_timeout_seconds: Optional[int] = None,
    ) -> None:
        """
        save training state to distributed checkpoint

        Writes three things under ``model/`` (see ``veomni.checkpoint.layout``):
        ``ckpt/`` for the weights, ``optimizer/`` for the optimizer state, and a
        per-rank ``extra_state/`` for model-bound extra state (lr scheduler and
        condition-model RNG). Weights and optimizer are separate DCP
        directories so the weights can be shipped or converted on their own; a
        single directory interleaves both into the same ``.distcp`` files.

        args:
            path: path to save checkpoint
            state: state to save
            save_async: whether to save asynchronously
            global_steps: step this checkpoint belongs to. Given, the checkpoint goes
                into a per-step subdirectory of ``path`` and ``path`` identifies the run,
                which is what ``stage_dir`` keys its staging directory on. Callers that
                fold the step into ``path`` themselves get a staging directory per step.
            module: name of the model this checkpoint belongs to, for a job that trains
                several (SeedOmni V2). Nests every artifact under ``model/<module>/``.
                Empty for a single-model job, which is the only difference between the
                two cases.
            trainable_only: when True, only persist parameters with ``requires_grad=True``
                (LoRA / PEFT path). Frozen base weights are skipped on save and must be
                re-materialised from ``model.model_path`` at resume time. The optimizer
                state is already trainable-only by construction (the optimizer is built
                from ``filter(lambda p: p.requires_grad, ...)``), so this flag only
                affects the model state dump.
            save_to_lowest_rank: forwarded to the DCP ``DefaultSavePlanner``. When True, each
                replicated shard is written by the lowest global rank that holds it, instead of
                being load-balanced across all replica holders. On a non-shared filesystem this
                concentrates the (already deduplicated) copy onto the lowest-ranked replica group
                instead of scattering it across replicas; in the standard HSDP layout (shard within
                a node, replicate across nodes) that group is one node, which then holds a complete
                checkpoint. Note this only consolidates *replicated* data: unique shards from
                expert/tensor/pipeline parallelism are never deduplicated and remain distributed.
                See ``CheckpointConfig.dcp_save_to_lowest_rank``.
            stage_dir: write the checkpoint here and copy it to ``path`` afterwards,
                instead of writing straight to ``path``. Intended for a destination far
                slower than local disk. The whole ``model/`` subtree — both DCP
                directories and the scheduler sidecar — lands under the staging
                directory and is copied together, with every ``.metadata`` copied
                last. Staging is keyed per module as well as per run, so two modules
                of one job do not sweep each other's staged files. The caller
                owns the choice: this does not probe
                for a usable directory or check free space, and an unusable ``stage_dir``
                fails the save rather than silently writing elsewhere. See
                ``CheckpointConfig.stage_dir``.
            save_timeout_seconds: collective timeout for the gloo groups this save runs
                its own collectives on -- staging's, and each ``save_async`` slot's. It
                has to outlast the work, since the ranks not writing wait on it for the
                whole duration. Unset keeps gloo's default.
        return:
            None
        """
        if not isinstance(state, dict) or "model" not in state:
            raise ValueError("Model must be provided to save a distributed checkpoint.")
        optimizer = state.get("optimizer")
        multi = getattr(optimizer, "_is_multi_optimizer", False) is True
        file_optimizers = (
            [leaf for leaf in _iter_optimizer_leaves(optimizer) if getattr(leaf, "_ple_file_mode_enabled", False)]
            if multi
            else []
        )
        file_mode = bool(file_optimizers)
        # An async write may outlive a staged copy, so the modes cannot combine.
        if stage_dir and save_async:
            raise ValueError("stage_dir cannot be combined with save_async")
        single_exact = not multi and _requires_rank_local_optimizer_checkpoint(optimizer)
        if single_exact:
            raise RuntimeError("Exact-precision optimizers require MultiOptimizer for sparse DCP recovery")
        _reject_unproven_extra_parallel_optimizer(
            state, parallel_state, trainable_only=trainable_only, allow_proven_optimizer=multi
        )
        scheduler_object = state.get("_veomni_scheduler_object") if multi else None
        source_model = state.get("model")
        extra = state.get("extra_state") if multi else None
        snapshot = extra.get("lr_scheduler") if isinstance(extra, dict) else None
        initial_layout, generation = None, None
        if multi:
            error = None
            try:
                parallel_state = parallel_state if parallel_state is not None else get_parallel_state()
                if "_veomni_scheduler_object" not in state:
                    raise RuntimeError(
                        "MultiOptimizer checkpoint requires an explicit scheduler declaration; "
                        "set _veomni_scheduler_object to the scheduler or None"
                    )
                if save_async:
                    raise RuntimeError("Proven MultiOptimizer checkpoints currently require synchronous DCP save")
                _validate_source_optimizer_states(optimizer, source_model)
                if "extra_state" in state and not isinstance(extra, dict):
                    raise RuntimeError("MultiOptimizer extra_state must be a mapping")
            except BaseException as exc:
                error = exc
            raise_if_any_rank_failed(error, "validating MultiOptimizer checkpoint policy")
            frozen_digest = frozen_parameter_digest(state["model"], enabled=trainable_only)
            initial_layout = coordinated_multi_layout(
                state["model"], optimizer, snapshot, scheduler_object, writing=True, frozen_digest=frozen_digest
            )
            error, generation_value = None, None
            if not dist.is_initialized() or dist.get_rank() == 0:
                try:
                    generation_value = str(uuid.uuid4())
                except BaseException as exc:
                    error = exc
            raise_if_any_rank_failed(error, "generating MultiOptimizer checkpoint identity")
            values = [generation_value]
            if dist.is_initialized():
                dist.broadcast_object_list(values, src=0)
            generation = values[0]

        # ``is not None`` rather than truthiness: step 0 is a step like any other, and
        # folding it onto ``path`` would write it over the run's own directory.
        checkpoint_dir = step_dir(path, global_steps) if global_steps is not None else path
        model_root = model_dir(checkpoint_dir, module)
        if multi:
            error = (
                RuntimeError("Cannot overwrite a completed checkpoint transaction with MultiOptimizer state")
                if os.path.exists(manifest_path(checkpoint_dir))
                else None
            )
            raise_if_any_rank_failed(error, "checking MultiOptimizer checkpoint destination")
        if multi and extra is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
            extra_path = os.path.join(model_root, _EXTRA_STATE_DIRNAME, _EXTRA_STATE_FORMAT.format(rank))
            error = (
                RuntimeError("Existing checkpoint sidecar cannot be reused when extra_state is omitted")
                if os.path.exists(extra_path)
                else None
            )
            raise_if_any_rank_failed(error, "checking for stale MultiOptimizer extra state")

        # Keyed on the module's own run-level path, not just ``path``: a
        # multi-module job calls this once per module, and a single key would have
        # each module clear the previous one's staged files.
        stage_key_path = os.path.join(path, module) if module else path
        # Staging's collectives run on a gloo group rather than the training backend:
        # the copy to a slow destination outlasts NCCL's watchdog, which aborts the
        # process where gloo raises. Created up front, before any rank can fail.
        stage_group = cls._get_stage_process_group(save_timeout_seconds) if stage_dir else None
        stage_path = _prepare_stage_dir(stage_dir, stage_key_path, group=stage_group) if stage_dir else None
        write_root = stage_path or model_root

        error = None
        try:
            if stage_path is None:
                # Invalidate stale metadata before writing any new shards.
                cls._remove_dcp_markers(checkpoint_dir, module)
            cls._create_checkpoint_dir(write_root)
            if multi:
                os.makedirs(os.path.join(write_root, OPTIMIZER_DIRNAME), exist_ok=True)
            cls._save_extra_state(checkpoint_dir=write_root, state=state)
        except BaseException as exc:
            error = exc
        if multi:
            raise_if_any_rank_failed(error, "preparing MultiOptimizer checkpoint")
        elif error is not None:
            raise error
        extra_digest = None
        if multi:
            error = None
            try:
                extra_digest, _ = _multi_extra_snapshot(write_root, read_state=False)
                if (extra is None) != (extra_digest == "absent"):
                    raise RuntimeError("MultiOptimizer checkpoint sidecar presence does not match this save")
            except BaseException as exc:
                error = exc
            raise_if_any_rank_failed(error, "hashing MultiOptimizer extra state")

        def prepare_writer_state(factory, what):
            result, error = None, None
            try:
                result = factory()
            except BaseException as exc:
                error = exc
            if multi:
                raise_if_any_rank_failed(error, what)
            elif error is not None:
                raise error
            return result

        def storage_writer(path):
            writer = cls._create_storage_writer(path)
            if file_mode:
                if type(writer) is not FileSystemWriter:
                    raise TypeError("File-backed checkpoint requires the standard filesystem writer")
                writer = BoundedFileSystemWriter(
                    writer.path,
                    sync_files=writer.sync_files,
                    overwrite=writer.overwrite,
                    release_page_cache=True,
                )
                writer._veomni_file_optimizers = file_optimizers
            return writer

        try:
            model_state, model_writer = prepare_writer_state(
                lambda: (
                    _ProvenModelState(
                        state["model"],
                        trainable_only=trainable_only,
                        parallel_state=parallel_state,
                        optimizer=optimizer,
                        group=None,
                        writing=True,
                    )
                    if multi
                    else ModelState(state["model"], trainable_only=trainable_only, parallel_state=parallel_state),
                    storage_writer(os.path.join(write_root, WEIGHTS_DIRNAME)),
                ),
                "preparing model DCP writer",
            )
            model_payload = {"model": model_state}
            if multi:
                model_payload[_MULTI_GENERATION_MARKER] = generation
            cls.execute_save(
                save_state=model_payload,
                storage_writer=model_writer,
                save_async=save_async,
                save_to_lowest_rank=save_to_lowest_rank,
                slot=WEIGHTS_DIRNAME,
                timeout_seconds=save_timeout_seconds,
            )

            if multi:
                error = None
                if state.get("model") is not source_model or state.get("optimizer") is not optimizer:
                    error = RuntimeError("Model or optimizer object changed while saving model state")
                elif (
                    state.get("extra_state") is not extra
                    or state.get("_veomni_scheduler_object") is not scheduler_object
                ):
                    error = RuntimeError("Scheduler or extra-state object changed while saving model state")
                raise_if_any_rank_failed(error, "rechecking MultiOptimizer checkpoint ownership")

            if "optimizer" in state and state["optimizer"] is not None:
                optim_state, optim_writer = prepare_writer_state(
                    lambda: (
                        _ProvenMultiOptimizerState(
                            model=state["model"],
                            optimizer=optimizer,
                            parallel_state=parallel_state,
                            generation=generation,
                            scheduler_snapshot=snapshot,
                            scheduler_object=scheduler_object,
                            extra_digest=extra_digest,
                            expected_layout=initial_layout,
                            group=None,
                            frozen_digest=frozen_digest,
                        )
                        if multi
                        else OptimizerState(model=state["model"], optimizer=optimizer, parallel_state=parallel_state),
                        storage_writer(os.path.join(write_root, OPTIMIZER_DIRNAME)),
                    ),
                    "preparing optimizer DCP writer",
                )
                cls.execute_save(
                    save_state={"optimizer": optim_state},
                    storage_writer=optim_writer,
                    save_async=save_async,
                    save_to_lowest_rank=save_to_lowest_rank,
                    slot=OPTIMIZER_DIRNAME,
                    timeout_seconds=save_timeout_seconds,
                )
        except BaseException:
            if stage_path is not None and _local_rank() == 0:
                shutil.rmtree(stage_path, ignore_errors=True)
            raise

        if stage_path is not None:
            _promote_staged_checkpoint(stage_path, model_root, step_root=checkpoint_dir, group=stage_group)

        logger.info_rank0(f"Saved checkpoint to {model_root}")
        return generation if multi else None

    @classmethod
    def _remove_dcp_markers(cls, checkpoint_dir: str, module: str) -> None:
        """Delete the ``.metadata`` files vouching for what this save overwrites.

        The unstaged path's invalidation, run before the first byte lands. A
        staged save has ``_promote_staged_checkpoint`` do the same job at
        promotion time, because until then it has overwritten nothing.

        Two places to look. This module's ``ckpt/`` and ``optimizer/``, and the
        step root, where an older VeOmni's fused save left the same file. Both
        are DCP's own marker, which is why both are dropped here rather than by
        the callback: this class is the one that puts them back.

        Only this module's. A sibling module is not being rewritten and its
        markers still hold. The step's ``checkpoint_manifest.json`` is not ours
        either -- ``GlobalStateCallback`` writes it and clears it.

        One rank owns these files, so only that rank deletes them, and the
        reduction turns a failure there into one every rank sees.
        """
        # The step being overwritten may predate this layout, in which case DCP
        # left its marker at the step root. Delete this import (and
        # veomni/checkpoint/legacy_v0_1_12.py) to drop that layout.
        from .legacy_v0_1_12 import drop_marker

        is_coordinator = (not dist.is_initialized()) or dist.get_rank() == 0
        error: Optional[Exception] = None
        if is_coordinator:
            try:
                drop_marker(checkpoint_dir)
                for marker in dcp_markers(checkpoint_dir, [module]):
                    if os.path.exists(marker):
                        os.remove(marker)
            except Exception as e:  # noqa: BLE001 - raised once every rank has agreed
                error = e
        raise_if_any_rank_failed(error, f"removing the old DCP markers under {checkpoint_dir}")

    @classmethod
    def load(
        cls,
        path: str,
        state: Dict[str, Any],
        process_group=None,
        module: str = "",
        trainable_only: bool = False,
        parallel_state=None,
    ) -> Dict[str, Any]:
        """Load a checkpoint, invalidating file-backed state after any failure."""
        try:
            return cls._load_impl(path, state, process_group, module, trainable_only, parallel_state)
        except BaseException:
            try:
                optimizer = state.get("optimizer") if isinstance(state, dict) else None
                if getattr(optimizer, "_is_multi_optimizer", False):
                    for leaf in _iter_optimizer_leaves(optimizer):
                        if getattr(leaf, "_ple_file_mode_enabled", False):
                            leaf.checkpoint_abort_file_backed_state()
            except Exception:
                # Preserve the original load failure, not a cleanup exception.
                pass
            raise

    @classmethod
    def _load_impl(
        cls,
        path: str,
        state: Dict[str, Any],
        process_group=None,
        module: str = "",
        trainable_only: bool = False,
        parallel_state=None,
    ) -> Dict[str, Any]:
        """
        load training state from distributed checkpoint

        Mirrors :meth:`save`: weights from ``model/<module>/ckpt``, optimizer from
        ``model/<module>/optimizer``, model-bound extra state from the per-rank
        ``extra_state/`` beside them. A checkpoint written before the split has
        no ``model/`` at all and keeps both in one directory; that shape is
        detected and read as-is.

        args:
            path: step directory to load from
            state: state to load, "model" is required; "optimizer" and "extra_state" are optional
            process_group: process group for loading checkpoint
            module: name of the model to load, for a job that trains several.
                Empty for a single-model job. See :meth:`save`.
            trainable_only: when True, ``set_model_state_dict`` runs in non-strict
                mode (``StateDictOptions(strict=False)``). Use this for LoRA / PEFT
                resumes where the DCP only contains trainable adapter weights and the
                frozen base must come from ``model.model_path``. Safe to enable when
                the DCP is full (extra strictness is just dropped).

        return:
            state: state loaded
        """
        if isinstance(state, dict):
            state.pop("_veomni_optimizer_identity_verified", None)
            state.pop("_veomni_optimizer_checkpoint_generation", None)
        if not isinstance(state, dict) or "model" not in state:
            if state is None:
                raise ValueError("State dict must be provided to load a distributed checkpoint.")
            raise ValueError("Model must be provided to load a distributed checkpoint.")
        optimizer = state.get("optimizer")
        multi = getattr(optimizer, "_is_multi_optimizer", False) is True
        single_exact = not multi and _requires_rank_local_optimizer_checkpoint(optimizer)
        if single_exact:
            raise RuntimeError("Exact-precision optimizers require MultiOptimizer for sparse DCP recovery")
        _reject_unproven_extra_parallel_optimizer(
            state, parallel_state, group=process_group, trainable_only=trainable_only, allow_proven_optimizer=multi
        )

        def _model_state() -> "ModelState":
            return ModelState(state["model"], trainable_only=trainable_only, parallel_state=parallel_state)

        def _optimizer_state() -> "OptimizerState":
            return OptimizerState(
                model=state["model"],
                optimizer=state["optimizer"],
                parallel_state=parallel_state,
                load=True,
            )

        wants_optimizer = state.get("optimizer") is not None

        fused_dir, error = None, None
        try:
            fused_dir = cls._legacy_fused_dir(path, module)
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "inspecting checkpoint layout", group=process_group)
        specs, needs_marker, error = None, False, None
        try:
            specs = getattr(state["model"], "_fqn2spec_info", None)
            needs_marker = _needs_extra_parallel_layout_marker(specs)
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "checking ExtraParallel marker policy", group=process_group)
        if needs_marker:
            marker_error = None
            try:
                source_path = fused_dir if fused_dir is not None else weights_dir(path, module)
                _read_extra_parallel_marker(cls._create_storage_reader(source_path), _extra_parallel_layout_key(specs))
            except BaseException as exc:
                marker_error = exc
            raise_if_any_rank_failed(marker_error, "validating ExtraParallel checkpoint marker", group=process_group)
        if multi:
            error = None
            try:
                parallel_state = parallel_state if parallel_state is not None else get_parallel_state()
                if "_veomni_scheduler_object" not in state:
                    raise RuntimeError(
                        "MultiOptimizer checkpoint requires an explicit scheduler declaration; "
                        "set _veomni_scheduler_object to the scheduler or None"
                    )
            except BaseException as exc:
                error = exc
            raise_if_any_rank_failed(error, "validating MultiOptimizer checkpoint policy", group=process_group)
            error = (
                RuntimeError("Fused MultiOptimizer checkpoints require an explicitly verified migration")
                if fused_dir is not None
                else None
            )
            raise_if_any_rank_failed(error, "checking MultiOptimizer checkpoint layout", group=process_group)
            scheduler_object = state.get("_veomni_scheduler_object")
            optimizer_path = optimizer_dir(path, module)
            model_path = weights_dir(path, module)
            root = model_dir(path, module)
            optimizer_id, extra, model_id, generation, expected_layout, extra_digest, frozen_digest = (
                _preflight_multi_optimizer_layout(
                    state["model"],
                    optimizer,
                    cls._create_storage_reader,
                    optimizer_path,
                    model_path,
                    root,
                    "extra_state" in state,
                    scheduler_object,
                    parallel_state=parallel_state,
                    trainable_only=trainable_only,
                    group=process_group,
                )
            )
            _prepare_multi_optimizer_saved_states(
                state["model"],
                optimizer,
                cls._create_storage_reader,
                optimizer_path,
                optimizer_id,
                group=process_group,
                parallel_state=parallel_state,
                validate_only=True,
            )
            model_state, reader, planner, error = None, None, None, None
            try:
                model_state = _ProvenModelState(
                    state["model"],
                    trainable_only=trainable_only,
                    parallel_state=parallel_state,
                    optimizer=optimizer,
                    group=process_group,
                )
                reader = cls._create_storage_reader(model_path)
                planner = _ModelStrictLoadPlanner(
                    strict_model=not trainable_only,
                    expected_save_id=model_id,
                    strict_gdn_fp32=is_qwen4_exp_model(state["model"]),
                )
            except BaseException as exc:
                error = exc
            raise_if_any_rank_failed(error, "preparing proven model checkpoint state", group=process_group)
            dcp.load(
                state_dict={"model": model_state},
                storage_reader=reader,
                process_group=process_group,
                planner=planner,
            )
            error = (
                RuntimeError("Optimizer object changed while loading model state")
                if state.get("optimizer") is not optimizer
                else None
            )
            raise_if_any_rank_failed(error, "rechecking MultiOptimizer identity", group=process_group)
            snapshot = extra.get("lr_scheduler") if extra is not None else None
            current = coordinated_multi_layout(
                state["model"],
                optimizer,
                snapshot,
                scheduler_object,
                extra_digest,
                group=process_group,
                frozen_digest=frozen_digest,
            )
            error = (
                RuntimeError("MultiOptimizer layout changed while loading model state")
                if current != expected_layout
                else None
            )
            raise_if_any_rank_failed(error, "rechecking MultiOptimizer layout", group=process_group)
            _prepare_multi_optimizer_saved_states(
                state["model"],
                optimizer,
                cls._create_storage_reader,
                optimizer_path,
                optimizer_id,
                group=process_group,
                parallel_state=parallel_state,
            )
            optimizer_state, reader, planner, error = None, None, None, None
            try:
                optimizer_state = _ProvenMultiOptimizerState(
                    model=state["model"],
                    optimizer=optimizer,
                    parallel_state=parallel_state,
                    load=True,
                    generation=generation,
                    scheduler_snapshot=snapshot,
                    scheduler_object=scheduler_object,
                    extra_digest=extra_digest,
                    expected_layout=expected_layout,
                    group=process_group,
                    frozen_digest=frozen_digest,
                )
                reader = cls._create_storage_reader(optimizer_path)
                planner = _ModelStrictLoadPlanner(strict_model=False, expected_save_id=optimizer_id)
            except BaseException as exc:
                error = exc
            raise_if_any_rank_failed(error, "preparing proven optimizer checkpoint state", group=process_group)
            dcp.load(
                state_dict={"optimizer": optimizer_state},
                storage_reader=reader,
                process_group=process_group,
                planner=planner,
            )
            if extra is not None:
                state["extra_state"].update(extra)
            state["_veomni_optimizer_identity_verified"] = True
            state["_veomni_optimizer_checkpoint_generation"] = generation
            return state
        if fused_dir is not None:
            load_state: Dict[str, Any] = {"model": _model_state()}
            if wants_optimizer:
                load_state["optimizer"] = _optimizer_state()
            dcp.load(
                state_dict=load_state,
                storage_reader=cls._create_storage_reader(fused_dir),
                process_group=process_group,
                planner=_ModelStrictLoadPlanner(
                    strict_model=not trainable_only, strict_gdn_fp32=is_qwen4_exp_model(state["model"])
                ),
            )
            cls._load_extra_state(checkpoint_dir=fused_dir, state=state)
            logger.info_rank0(f"Loaded pre-split checkpoint from {fused_dir}")
            return state

        model_root = model_dir(path, module)
        dcp.load(
            state_dict={"model": _model_state()},
            storage_reader=cls._create_storage_reader(weights_dir(path, module)),
            process_group=process_group,
            planner=_ModelStrictLoadPlanner(
                strict_model=not trainable_only, strict_gdn_fp32=is_qwen4_exp_model(state["model"])
            ),
        )
        if wants_optimizer:
            dcp.load(
                state_dict={"optimizer": _optimizer_state()},
                storage_reader=cls._create_storage_reader(optimizer_dir(path, module)),
                process_group=process_group,
                # The strict check looks for missing ``model`` keys; there are none
                # in an optimizer-only load, so strictness has nothing to say here.
                planner=_ModelStrictLoadPlanner(strict_model=False),
            )

        cls._load_extra_state(checkpoint_dir=model_root, state=state)

        logger.info_rank0(f"Loaded checkpoint from {model_root}")

        return state

    @classmethod
    def _legacy_fused_dir(cls, path: str, module: str) -> Optional[str]:
        """Directory holding a pre-split checkpoint, or ``None`` for a current one.

        Before weights and optimizer were separated there was one DCP directory
        per model, marked by its own ``.metadata``: at the step root for a
        single-model job, and at ``<step>/<module>/`` for SeedOmni V2. Neither has
        a ``model/`` subtree, which is what tells the two apart.
        """
        if os.path.exists(os.path.join(weights_dir(path, module), DCP_MARKER_FILENAME)):
            return None
        legacy_dir = os.path.join(path, module) if module else path
        if os.path.exists(os.path.join(legacy_dir, DCP_MARKER_FILENAME)):
            return legacy_dir
        return None

    @classmethod
    def wait_for_pending_save(cls) -> None:
        """Block until every pending async DCP save completes.

        Safe to call when no save is pending (no-op).  Every rank ends up
        raising if any rank's save failed, and the reductions that decide
        that are also the synchronization callers rely on before starting a
        new collective. There is one per drained slot, on that slot's own
        gloo group: the write it covers ran there too, so the ranks that
        finished first wait out the others off the training backend.

        This is the single entrypoint for all async-save coordination —
        prefer calling this over poking ``_save_futures`` directly.

        Every slot is drained even when one raises, so a failed weights write
        cannot leave the optimizer write running into the next step's
        collectives. That is also why the catch below is ``BaseException``:
        DCP reports a failed save as ``CheckpointException``, which derives
        from ``BaseException`` rather than ``Exception``, so catching the
        latter would let the very failures this drains for escape the loop --
        leaving the remaining slots running and their futures unreachable.
        """
        if not cls._save_futures:
            return
        rank = dist.get_rank() if dist.is_initialized() else 0
        futures = cls._save_futures
        cls._save_futures = {}
        errors: Dict[str, BaseException] = {}
        for slot, future in futures.items():
            try:
                logger.info(f"[RANK {rank}] waiting for pending DCP save ({slot}) to end...")
                future.result()
            except BaseException as e:  # noqa: BLE001 - re-raised once every rank has agreed
                logger.error(f"[RANK {rank}] pending async DCP save ({slot}) raised; propagating", exc_info=True)
                errors[slot] = e
        # Agreeing after the loop rather than inside it is what drains every slot
        # even when one raised. The slots are the same on every rank, and so is
        # their order, so the reductions below line up. Reducing on the slot's
        # group is safe here because every rank's future has resolved: the save
        # that owned that group has stopped issuing collectives on it. The lookup
        # misses only when the save ran without a process group at all, in a
        # single process, where the reduction is a no-op anyway.
        for slot in futures:
            raise_if_any_rank_failed(
                errors.get(slot), f"a pending async DCP save ({slot})", group=cls._async_process_groups.get(slot)
            )

    @classmethod
    def _drain_slot(cls, slot: str) -> None:
        """Finish this slot's previous save before its process group is reused.

        DCP surfaces an async failure through the future of the rank that hit
        it, and the save's own process group does not reduce that across the
        group. Raising here on the failing rank alone would leave its peers in
        the reduction below with nobody to meet -- which is also why the catch
        is ``BaseException``: DCP's ``CheckpointException`` does not derive
        from ``Exception``, and letting it past the reduction is exactly the
        hang this method exists to prevent.

        The agreement runs on the slot's own gloo group, like the write it
        follows: ranks whose write finished first wait out the rest there rather
        than in the training backend, whose watchdog aborts the process.
        """
        future = cls._save_futures.pop(slot, None)
        if future is None:
            return
        rank = dist.get_rank() if dist.is_initialized() else 0
        error: Optional[BaseException] = None
        try:
            logger.info(f"[RANK {rank}] waiting for previous DCP save ({slot}) to end...")
            future.result()
        except BaseException as e:  # noqa: BLE001 - re-raised once every rank has agreed
            logger.error(f"[RANK {rank}] previous async DCP save ({slot}) raised; propagating", exc_info=True)
            error = e
        raise_if_any_rank_failed(
            error, f"the previous async DCP save ({slot})", group=cls._async_process_groups.get(slot)
        )

    @classmethod
    def _get_stage_process_group(cls, timeout_seconds: Optional[int]) -> Optional[Any]:
        """The gloo group a staged save runs its collectives on, created on first use.

        Creating a group is itself a collective, so every rank must reach this in the
        same order. Cached for the life of the process, like the async slot groups:
        ``timeout_seconds`` takes effect on the first staged save, and a group broken
        by a timeout is not rebuilt -- the save that broke it has already failed the run.
        """
        if cls._stage_process_group is None and dist.is_initialized():
            cls._stage_process_group = dist.new_group(backend="gloo", timeout=_gloo_timeout(timeout_seconds))
        return cls._stage_process_group

    @classmethod
    def execute_save(
        cls,
        save_state: Dict[str, Any],
        storage_writer: FileSystemWriter,
        save_async: bool,
        save_to_lowest_rank: bool = False,
        slot: str = WEIGHTS_DIRNAME,
        timeout_seconds: Optional[int] = None,
    ) -> None:
        """Execute DCP save with optional async support.

        ``save_to_lowest_rank`` is forwarded to ``DefaultSavePlanner``; the default
        (False) preserves DCP's load-balanced write assignment across replica holders.

        ``slot`` names the concurrent async save this call belongs to — one per
        directory a step writes. Only the *same* slot's previous save is drained,
        so the weights and the optimizer overlap within a step while neither can
        outlive its own next write. ``timeout_seconds`` is that slot's group
        timeout, applied when the group is created.
        """
        file_optimizers = getattr(storage_writer, "_veomni_file_optimizers", None)
        if file_optimizers is not None and save_async:
            raise RuntimeError("File-backed optimizer state requires synchronous DCP save")
        planner = (
            _FileBackedSavePlanner(optimizers=file_optimizers, dedup_save_to_lowest_rank=save_to_lowest_rank)
            if file_optimizers is not None
            else _ExtraParallelSavePlanner(dedup_save_to_lowest_rank=save_to_lowest_rank)
        )
        if save_async:
            # Lazily create this slot's dedicated Gloo process group. Creating a
            # group is itself collective, and every rank runs the same save
            # sequence, so every rank creates the same groups in the same order.
            if slot not in cls._async_process_groups:
                cls._async_process_groups[slot] = dist.new_group(
                    backend="gloo", timeout=_gloo_timeout(timeout_seconds)
                )

            cls._drain_slot(slot)

            cls._save_futures[slot] = dcp.async_save(
                state_dict=save_state,
                storage_writer=storage_writer,
                process_group=cls._async_process_groups[slot],
                planner=planner,
            )
        else:
            try:
                dcp.save(
                    state_dict=save_state,
                    storage_writer=storage_writer,
                    planner=planner,
                )
            except BaseException:
                for optimizer in file_optimizers or ():
                    optimizer.checkpoint_abort_file_backed_state()
                raise
            if file_optimizers is not None:
                error = None
                try:
                    for optimizer in file_optimizers:
                        optimizer.checkpoint_evict_file_backed_state()
                except BaseException as exc:
                    error = exc
                try:
                    raise_if_any_rank_failed(error, "evicting file-backed optimizer checkpoint state")
                except BaseException:
                    for optimizer in file_optimizers:
                        optimizer.checkpoint_abort_file_backed_state()
                    raise
            if dist.is_initialized():
                dist.barrier()
            error = None
            try:
                gc.collect()
                empty_cache()
                synchronize()
            except BaseException as exc:
                error = exc
            raise_if_any_rank_failed(error, "finalizing DCP save")

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
        """Pickle this rank's ``extra_state`` dict into ``extra_state/``.

        The condition-model RNG is rank-local, so every rank writes its own file.
        """
        if _EXTRA_STATE_KEY not in state:
            logger.warning_rank0("extra_state not found in state, skipping extra_state save")
            return
        extra_state_dir = os.path.join(checkpoint_dir, _EXTRA_STATE_DIRNAME)
        os.makedirs(extra_state_dir, exist_ok=True)
        rank = dist.get_rank() if dist.is_initialized() else 0
        extra_state_path = os.path.join(extra_state_dir, _EXTRA_STATE_FORMAT.format(rank))
        torch.save(state[_EXTRA_STATE_KEY], extra_state_path)

    @classmethod
    def _load_extra_state(cls, checkpoint_dir: str, state: Dict[str, Any]) -> None:
        """Load this rank's ``extra_state`` dict from ``extra_state/``."""
        if _EXTRA_STATE_KEY not in state:
            logger.warning_rank0("extra_state not found in state, skipping extra_state load")
            return
        rank = dist.get_rank() if dist.is_initialized() else 0
        extra_state_path = os.path.join(checkpoint_dir, _EXTRA_STATE_DIRNAME, _EXTRA_STATE_FORMAT.format(rank))
        if os.path.exists(extra_state_path):
            state[_EXTRA_STATE_KEY] = torch.load(extra_state_path, weights_only=False)


def get_dtype_size(dtype: torch.dtype) -> int:
    """Return size in bytes for a given dtype."""
    return torch.empty((), dtype=dtype).element_size()


def _normalize_key(key: str) -> Optional[str]:
    """
    Convert DCP key to HuggingFace format. Returns None for non-model weights.

    Conversion rules:
    - "model.model.*" -> "model.*" (remove first "model." prefix)
    - "model.lm_head.weight" -> "lm_head.weight" (special case)
    - "model.base_model.*" -> "base_model.*" (PEFT LoRA adapter case;
      ``save_lora_adapter_with_dcp`` re-prefixes already-PEFT-prefixed keys
      with ``model.`` so DCP keeps them, and we strip that here on read)
    - Other "model.*" keys -> log warning and strip "model." prefix
    """
    if not key.startswith("model."):
        return None
    normalized = _export_extra_parallel_layout_key(key[6:])
    if normalized is None or normalized in (_QWEN4_GDN_FP32_MARKER, _QWEN4_GDN_FP32_LEGACY_MARKER):
        return None
    key = "model." + normalized

    if key.startswith("model.model."):
        # Standard case: model.model.* -> model.*
        return key[6:]  # Remove first "model." prefix
    elif key == "model.lm_head.weight":
        # Special case: model.lm_head.weight -> lm_head.weight
        return "lm_head.weight"
    elif key.startswith("model.base_model."):
        # PEFT LoRA adapter save: ``save_lora_adapter_with_dcp`` writes keys
        # of the form ``model.base_model.model.<...>.lora_A.weight`` so the
        # DCP-side ``model.`` filter keeps them. The HF-side adapter file is
        # the standard PEFT layout ``base_model.model.<...>.lora_A.weight``,
        # which is exactly ``key[6:]``. This is a known, expected pattern
        # — silent strip, no warning.
        return key[6:]
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
    drop_hf_keys: Optional[Set[str]] = None,
    preserve_fp32_keys: Optional[Set[str]] = None,
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

    model_keys = {key for key in metadata.state_dict_metadata if key.startswith("model.")}
    version_prefix = "model." + _EXTRA_PARALLEL_LAYOUT_MARKER
    versioned_keys = {key for key in model_keys if key.startswith(version_prefix)}
    if versioned_keys:
        marker = "model." + _extra_parallel_layout_key({})
        if marker not in versioned_keys:
            raise RuntimeError("Versioned ExtraParallel checkpoint marker is missing or unknown")
        if any(key != marker and not key.startswith(marker + "__") for key in model_keys):
            raise RuntimeError("ExtraParallel checkpoint mixes versioned and legacy tensor keys")
        _read_extra_parallel_marker(FileSystemReader(checkpoint_path), marker.removeprefix("model."))

    # Collect model tensors and calculate sizes
    tensor_infos = []
    all_dcp_keys = []
    seen_hf_keys = {}
    dropped = 0

    for key, tensor_meta in metadata.state_dict_metadata.items():
        hf_key = _normalize_key(key)
        if hf_key:
            if drop_hf_keys and hf_key in drop_hf_keys:
                dropped += 1
                continue
            if hf_key in seen_hf_keys:
                raise ValueError(f"Multiple DCP tensors map to HF key {hf_key}: {seen_hf_keys[hf_key]} and {key}")
            seen_hf_keys[hf_key] = key

            # Determine dtype for size calculation
            if not hasattr(tensor_meta.properties, "dtype"):
                raise ValueError(
                    f"Cannot determine dtype for tensor '{key}': metadata does not contain dtype information"
                )
            source_dtype = tensor_meta.properties.dtype
            if (
                save_dtype
                and source_dtype.is_floating_point
                and not (preserve_fp32_keys and hf_key in preserve_fp32_keys)
            ):
                dtype = getattr(torch, save_dtype) if isinstance(save_dtype, str) else save_dtype
            else:
                dtype = source_dtype

            # Calculate tensor size in bytes
            numel = 1
            for dim in tensor_meta.size:
                numel *= dim

            byte_size = numel * get_dtype_size(dtype)

            tensor_infos.append({"dcp_key": key, "hf_key": hf_key, "size": byte_size, "metadata": tensor_meta})
            all_dcp_keys.append(key)

    if dropped:
        logger.info(f"Excluded {dropped} tensor(s) named by `drop_hf_keys` from the plan")

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
    preserve_fp32_keys: Optional[Set[str]] = None,
) -> str:
    reader = FileSystemReader(checkpoint_path)
    metadata = reader.read_metadata()

    state_dict = OrderedDict()
    dcp_keys_to_load = list(shard_keys.values())

    # A partial offline shard must carry the checked layout marker too, so
    # the same planner can validate the source namespace before copying data.
    marker = "model." + _extra_parallel_layout_key({})
    if marker in metadata.state_dict_metadata:
        _read_extra_parallel_marker(reader, marker.removeprefix("model."))
        state_dict[marker] = ""

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
        planner=_ModelStrictLoadPlanner(strict_model=True),
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

        if target_dtype and tensor.is_floating_point() and not (preserve_fp32_keys and hf_key in preserve_fp32_keys):
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
