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
import re
import shutil
import uuid
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta
from typing import Any, Dict, Optional, Set, Union

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed._tensor import DeviceMesh, DTensor, Partial, Replicate, Shard
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load,
)
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner, DefaultSavePlanner
from torch.distributed.checkpoint.metadata import (
    STATE_DICT_TYPE,
    BytesStorageMetadata,
    Metadata,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor.placement_types import _StridedShard

from ..distributed.parallel_state import ParallelState, get_parallel_state, is_parallel_state_initialized
from ..utils import logging
from ..utils.device import empty_cache, get_device_type, synchronize
from ..utils.dist_utils import any_rank_failed, raise_if_any_rank_failed
from ..utils.dtensor_utils import rewrap_dtensor_local
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
    _clear_swap_checkpoint_staging,
    _dcp_parameter_name_map,
    _inventory_digests,
    _iter_optimizer_leaves,
    _multi_extra_snapshot,
    _preflight_multi_optimizer_layout,
    _prepare_multi_optimizer_saved_states,
    _validate_fresh_swap_optimizer,
    _validate_source_optimizer_states,
)
from .optimizer_resume_layout import (
    MULTI_LAYOUT_MARKER,
    capture_multi_layout,
    finish_multi_layout,
    frozen_parameter_digest,
)


logger = logging.get_logger(__name__)

_EXTRA_STATE_KEY = "extra_state"
_EXTRA_STATE_DIRNAME = "extra_state"
_EXTRA_STATE_FORMAT = "extra_state_rank_{}.pt"


class _PreparedLoadState:
    """Expose a rank-coordinated Stateful snapshot to DCP exactly once."""

    def __init__(self, owner, snapshot, group):
        self.owner = owner
        self.snapshot = snapshot
        self.group = group

    def state_dict(self):
        return self.snapshot

    def load_state_dict(self, state_dict):
        error = None
        try:
            self.owner.load_state_dict(state_dict)
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "applying checkpoint state", group=self.group)


_EXPERT_LAYOUTS_KEY = "__veomni_expert_layouts__"
_UNKNOWN_EXPERT_LAYOUTS_KEY = "__veomni_unknown_expert_layouts__"
_LEGACY_VEOMNI_TP1_EP0_FSDP1 = "veomni_tp1_ep0_fsdp1"
_LEGACY_VEOMNI_TP1_NO_EP = "veomni_tp1_no_ep"


def _validate_legacy_expert_source_layout(value):
    if value is not None and value not in (_LEGACY_VEOMNI_TP1_EP0_FSDP1, _LEGACY_VEOMNI_TP1_NO_EP):
        raise ValueError(f"Unsupported legacy expert source layout: {value!r}")


def _expert_layout(tensor, extra_parallel_names=()):
    """Describe a source ExtraParallel/FSDP layout independently of the target."""
    if not isinstance(tensor, DTensor):
        return None
    mesh = tensor.device_mesh
    names = mesh.mesh_dim_names or ()
    if any(isinstance(p, Partial) and mesh.size(axis) > 1 for axis, p in enumerate(tensor.placements)):
        return {"expert_order": "unproven"}
    extra_names = [name for name in names if name == "ep" or name in extra_parallel_names or f"{name}_fsdp" in names]
    if not extra_names:
        return None
    if len(extra_names) != 1:
        raise RuntimeError(f"Cannot determine the ExtraParallel axis from mesh names {names}")
    extra_name = extra_names[0]
    ep_axis = names.index(extra_name)
    ep = tensor.placements[ep_axis]
    if isinstance(ep, _StridedShard):
        return {"expert_order": "unproven"}
    if not isinstance(ep, Shard):
        return None
    fsdp_name = f"{extra_name}_fsdp"
    fsdp_axis = names.index(fsdp_name) if fsdp_name in names else None
    fsdp = tensor.placements[fsdp_axis] if fsdp_axis is not None else Replicate()
    fsdp_dim = fsdp.dim % tensor.ndim if isinstance(fsdp, (Shard, _StridedShard)) else None
    ep_dim = ep.dim % tensor.ndim
    fsdp_size = mesh.size(fsdp_axis) if fsdp_axis is not None else 1
    same_axis = ep_dim == fsdp_dim and mesh.size(ep_axis) > 1 and fsdp_size > 1
    shard_axes = [
        (axis, placement.dim % tensor.ndim)
        for axis, placement in enumerate(tensor.placements)
        if isinstance(placement, (Shard, _StridedShard)) and mesh.size(axis) > 1
    ]
    unknown_overlap = any(
        axis not in (ep_axis, fsdp_axis)
        and any(other_axis != axis and other_dim == dim for other_axis, other_dim in shard_axes)
        for axis, dim in shard_axes
    )
    canonical = not same_axis or isinstance(fsdp, _StridedShard) or (fsdp_axis is not None and ep_axis < fsdp_axis)
    order = "canonical" if canonical else "legacy_fsdp_first"
    unknown_strided = any(
        axis not in (ep_axis, fsdp_axis) and isinstance(placement, _StridedShard) and mesh.size(axis) > 1
        for axis, placement in enumerate(tensor.placements)
    )
    if (
        unknown_overlap
        or unknown_strided
        or (isinstance(fsdp, _StridedShard) and (fsdp_dim != ep_dim or fsdp.split_factor != mesh.size(ep_axis)))
    ):
        order = "unproven"
    return {
        "ep_shard_dim": ep_dim,
        "fsdp_shard_dim": fsdp_dim,
        "ep_size": mesh.size(ep_axis),
        "fsdp_size": fsdp_size,
        "expert_order": order,
    }


def _ordinary_expert_fqns(model, parallel_state):
    """Prove which parameters are outside every active ExtraParallel plan."""
    if not isinstance(model, torch.nn.Module) or not isinstance(getattr(model, "_fqn2spec_info", None), dict):
        return set()
    from torch.distributed.checkpoint.state_dict import _get_fqns

    from ..distributed.parallel_plan import ParallelPlan, SpecInfo, get_runtime_parallel_plan

    plan = get_runtime_parallel_plan(model)
    if not isinstance(plan, ParallelPlan):
        return set()
    patterns = [
        pattern
        for name, entries in plan.extra_parallel_plan.items()
        if parallel_state.extra_parallel_sizes.get(name, 1) > 1
        for pattern in entries
    ]
    exact_patterns = {pattern for pattern in patterns if "*" not in pattern}
    wildcard_patterns = [
        re.compile("^" + re.escape(pattern).replace(r"\*", ".*") + "$") for pattern in patterns if "*" in pattern
    ]
    aliases = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(param), []).append(name)
    result, owners, ambiguous = set(), {}, set()
    for owner, raw_names in aliases.items():
        if not any("experts" in name.split(".") for name in raw_names):
            continue
        names = [(name, _get_fqns(model, name)) for name in raw_names]
        specs = [model._fqn2spec_info.get(name) for name in raw_names]
        if any(
            not isinstance(spec, SpecInfo)
            or not isinstance(spec.placement, Replicate)
            or parallel_state.extra_parallel_sizes.get(spec.para_name, 1) <= 1
            or spec.para_fsdp_mesh != parallel_state.extra_parallel_fsdp_device_mesh.get(spec.para_name)
            for spec in specs
        ):
            continue
        if any(
            name in exact_patterns or any(pattern.match(name) for pattern in wildcard_patterns) for name in raw_names
        ):
            continue
        if any(len(canonical) != 1 for _, canonical in names):
            continue
        for _, canonical in names:
            fqn = next(iter(canonical))
            if fqn in owners and owners[fqn] != owner:
                ambiguous.add(fqn)
            owners[fqn] = owner
            result.add(fqn)
    return result - ambiguous


def _logical_extra_parallel_key(key, state_dict):
    # The save planner's metadata keeps the serialized key, but ownership is
    # expressed in logical FQNs. Canonical v2 checkpoints namespace both model
    # and optimizer keys, so strip only the verified layout marker for lookup.
    root, sep, rest = key.partition(".")
    if not sep or root not in ("model", "optimizer"):
        return key
    marker = _extra_parallel_layout_key({})
    marker_value = state_dict.get(f"{root}.{marker}")
    if isinstance(marker_value, str) and marker_value == "ep-outer-v1":
        if rest == marker:
            return None
        prefix = marker + "__"
        if not rest.startswith(prefix):
            raise RuntimeError("ExtraParallel checkpoint mixes versioned and legacy tensor keys")
        return f"{root}.{rest[len(prefix) :]}"
    return key


def _key_belongs_to_fqns(key, fqns):
    if key.startswith("model.") and (key in fqns or key.removeprefix("model.") in fqns):
        return True
    if key.startswith("optimizer.state."):
        rest = key.removeprefix("optimizer.state.")
        while "." in rest:
            rest = rest.rsplit(".", 1)[0]
            if rest in fqns:
                return True
    return False


def _shared_lora_fqns(model, ordinary_fqns):
    """Identify the shared LoRA parameters whose EP gradients are synchronized."""
    from torch.distributed.checkpoint.state_dict import _get_fqns

    from ..lora.moe_layers import LoraSharedExperts, _is_lora_param_name

    shared_ids = {
        id(param)
        for module in model.modules()
        if isinstance(module, LoraSharedExperts)
        for name, param in module.named_parameters(remove_duplicate=False)
        if _is_lora_param_name(name) and param.requires_grad
    }
    return {
        fqn
        for name, param in model.named_parameters(remove_duplicate=False)
        if id(param) in shared_ids
        for fqn in _get_fqns(model, name)
        if fqn in ordinary_fqns
    }


def _no_ep_expert_layout(key, tensor, parallel_state, ordinary_fqns=(), shared_lora_fqns=()):
    """Prove ordinary FSDP from the actual runtime and, if needed, model plan."""
    if "experts" not in key.split(".") or not torch.is_tensor(tensor) or not tensor.ndim:
        return None
    if not isinstance(parallel_state, ParallelState):
        return None
    if parallel_state.tp_size != 1 or not parallel_state.extra_parallel_sizes:
        return None
    single_process = dist.get_world_size() == 1 if dist.is_initialized() else os.environ.get("WORLD_SIZE", "1") == "1"
    if (
        parallel_state.device_mesh is None
        and single_process
        and all(size == 1 for size in parallel_state.extra_parallel_sizes.values())
        and (not isinstance(tensor, DTensor) or tensor.device_mesh.size() == 1)
    ):
        return {
            "ep_shard_dim": None,
            "fsdp_shard_dim": None,
            "ep_size": 1,
            "fsdp_size": 1,
            "expert_order": "canonical",
            "source_kind": "veomni_tp1_no_ep",
        }
    ordinary = _key_belongs_to_fqns(key, ordinary_fqns)
    if any(size != 1 for size in parallel_state.extra_parallel_sizes.values()) and not ordinary:
        return None
    if parallel_state.device_mesh is None and (
        not is_parallel_state_initialized()
        or parallel_state is not get_parallel_state()
        or (dist.is_initialized() and dist.get_world_size() != 1)
    ):
        return None
    if not isinstance(tensor, DTensor):
        if any(size != 1 for size in parallel_state.extra_parallel_sizes.values()):
            return None
        return {
            "ep_shard_dim": None,
            "fsdp_shard_dim": None,
            "ep_size": 1,
            "fsdp_size": 1,
            "expert_order": "canonical",
            "source_kind": "veomni_tp1_no_ep",
        }
    if parallel_state.device_mesh is None:
        return None
    mesh = tensor.device_mesh
    names = mesh.mesh_dim_names or ()
    normal_fsdp = (
        names
        in (
            ("dp_shard",),
            ("dp_shard_sp",),
            ("dp_replicate", "dp_shard"),
            ("dp_replicate", "dp_shard_sp"),
        )
        and mesh == parallel_state.fsdp_mesh
    )
    # Only the shared-LoRA wrapper installs the cross-EP gradient hooks needed
    # to keep replicas on complementary FSDP meshes consistent.
    shared_lora = _key_belongs_to_fqns(key, shared_lora_fqns)
    ep_mesh = parallel_state.extra_parallel_fsdp_device_mesh.get("ep")
    replicated_extra_fsdp = (
        ordinary and shared_lora and ep_mesh is not None and mesh == ep_mesh[ep_mesh.mesh_dim_names[:-1]]
    )
    if not (normal_fsdp or replicated_extra_fsdp) or len(tensor.placements) != len(names):
        return None
    if len(names) == 2 and not isinstance(tensor.placements[0], Replicate):
        return None
    shard = tensor.placements[-1]
    if type(shard) not in (Shard, Replicate):
        return None
    return {
        "ep_shard_dim": None,
        "fsdp_shard_dim": shard.dim % tensor.ndim if isinstance(shard, Shard) else None,
        "ep_size": 1,
        "fsdp_size": mesh.size(-1),
        "expert_order": "canonical",
        "source_kind": "veomni_tp1_no_ep",
    }


def _unidentified_same_axis_shards(tensor):
    """Do not certify a multi-axis layout whose mesh names hide its origin."""
    if not isinstance(tensor, DTensor) or not tensor.ndim:
        return False
    if any(
        isinstance(placement, (_StridedShard, Partial)) and tensor.device_mesh.size(axis) > 1
        for axis, placement in enumerate(tensor.placements)
    ):
        return True
    dims = [
        placement.dim % tensor.ndim
        for axis, placement in enumerate(tensor.placements)
        if isinstance(placement, (Shard, _StridedShard)) and tensor.device_mesh.size(axis) > 1
    ]
    return len(dims) != len(set(dims))


def _verified_rank_local_keys(adapter, *, loading):
    """Only the proven optimizer adapter can supply rank-local inventory keys."""
    if (
        type(adapter) is not _ProvenMultiOptimizerState
        or adapter._load is not loading
        or not adapter.should_extra_parallel_aware
    ):
        return frozenset()
    return adapter._verified_rank_local_keys


def _tp_only_source_parameters(model, parallel_state):
    """Identify unambiguous model owners on the runtime's TP/FSDP mesh.

    This does not certify custom sharding or any active ExtraParallel plan.
    Like the rest of the checkpointer, it assumes the supplied runtime model
    was built from a complete logical model rather than pre-sliced parameters.
    """
    if not isinstance(model, torch.nn.Module) or not isinstance(parallel_state, ParallelState):
        return {}, None
    if (
        parallel_state.device_mesh is None
        or not dist.is_initialized()
        or parallel_state.device_mesh._get_root_mesh().size() != dist.get_world_size()
        or parallel_state.tp_size <= 1
        or not parallel_state.extra_parallel_sizes
        or any(size != 1 for size in parallel_state.extra_parallel_sizes.values())
    ):
        return {}, None
    from ..distributed.parallel_state import _PARALLEL_STATE_CACHE

    # A multi-model trainer may save a registered non-ambient state. Require a
    # state created by the runtime's topology cache, not just matching names.
    if not any(parallel_state is state for state in _PARALLEL_STATE_CACHE.values()):
        return {}, None

    from torch.distributed.checkpoint.state_dict import _get_fqns

    try:
        mesh = DeviceMesh._concatenate([parallel_state.fsdp_mesh, parallel_state.tp_mesh])
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return {}, None
    if mesh._get_root_mesh() is not parallel_state.device_mesh._get_root_mesh():
        return {}, None

    aliases = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(param), []).append((name, param))
    candidates, ambiguous = {}, set()
    for names in aliases.values():
        if len(names) != 1:
            continue
        name, param = names[0]
        fqns = _get_fqns(model, name)
        if len(fqns) != 1:
            continue
        fqn = next(iter(fqns))
        if fqn in candidates:
            ambiguous.add(fqn)
        candidates[fqn] = param
    for fqn in ambiguous:
        candidates.pop(fqn, None)
    return candidates, mesh


def _tp_only_source_layout(key, tensor, parameters, mesh, *, allow_staged_copies=False):
    """Certify only a model parameter or same-layout state of its unique owner."""
    if mesh is None or not isinstance(tensor, DTensor) or not tensor.ndim:
        return None
    if key.startswith("model."):
        fqn = key.removeprefix("model.")
    elif key.startswith("optimizer.state."):
        fqn = key.removeprefix("optimizer.state.")
        while fqn not in parameters and "." in fqn:
            fqn = fqn.rsplit(".", 1)[0]
    else:
        return None
    param = parameters.get(fqn)
    if not isinstance(param, DTensor) or tuple(tensor.shape) != tuple(param.shape):
        return None
    if param.device_mesh._get_root_mesh() is not mesh._get_root_mesh():
        return None
    if tensor.device_mesh != mesh or param.device_mesh != mesh or tensor.placements != param.placements:
        return None
    if tensor.device_mesh._get_root_mesh() is not mesh._get_root_mesh():
        if not allow_staged_copies:
            return None
        # Async CPU staging deep-copies DTensor meshes. Admit a staged copy
        # only for the production async writer and only if every axis still
        # resolves to the runtime's identical process group.
        try:
            if any(tensor.device_mesh.get_group(i) is not mesh.get_group(i) for i in range(mesh.ndim)):
                return None
        except (AssertionError, AttributeError, IndexError, KeyError, RuntimeError, ValueError):
            return None
    placements = tensor.placements
    if len(placements) not in (2, 3) or any(isinstance(placement, Partial) for placement in placements):
        return None
    if len(placements) == 3 and not isinstance(placements[0], Replicate):
        return None
    fsdp, tp = placements[-2:]
    if type(tp) not in (Shard, Replicate) or type(fsdp) not in (Shard, _StridedShard):
        return None
    same_axis = isinstance(tp, Shard) and fsdp.dim % tensor.ndim == tp.dim % tensor.ndim
    if same_axis:
        if type(fsdp) is not _StridedShard or fsdp.split_factor != mesh.size(-1):
            return None
    elif type(fsdp) is not Shard:
        return None
    return {
        "ep_shard_dim": None,
        "fsdp_shard_dim": fsdp.dim % tensor.ndim,
        "ep_size": 1,
        "fsdp_size": mesh.size(-2),
        "expert_order": "canonical",
        "source_kind": "veomni_tp_fsdp_no_extra",
    }


class _ExtraParallelSavePlanner(DefaultSavePlanner):
    """Record per-tensor source layouts in DCP metadata, including optimizer state."""

    def __init__(
        self,
        *args,
        parallel_state=None,
        source_model=None,
        adapter_name=None,
        rank_local_adapter=None,
        allow_staged_copies=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.parallel_state = parallel_state
        self.rank_local_adapter = rank_local_adapter
        self.allow_staged_copies = allow_staged_copies
        self.tp_only_parameters, self.tp_only_mesh = _tp_only_source_parameters(source_model, parallel_state)
        self.ordinary_fqns = (
            _ordinary_expert_fqns(source_model, parallel_state)
            if isinstance(parallel_state, ParallelState)
            and any(size != 1 for size in parallel_state.extra_parallel_sizes.values())
            else set()
        )
        self.shared_lora_fqns = _shared_lora_fqns(source_model, self.ordinary_fqns) if self.ordinary_fqns else set()
        if adapter_name is not None and self.ordinary_fqns:
            from torch.distributed.checkpoint.state_dict import _get_fqns

            from ..lora.state_dict import get_lora_state_dict, strip_adapter_name

            exported = get_lora_state_dict(
                source_model, adapter_name=adapter_name, config=source_model.get_lora_config()
            )
            owners, candidates, shared_candidates = {}, set(), set()
            for name, param in source_model.named_parameters(remove_duplicate=False):
                exported_name = strip_adapter_name(name, adapter_name)
                if exported_name not in exported:
                    continue
                owners.setdefault(exported_name, set()).add(id(param))
                if name in self.ordinary_fqns or _get_fqns(source_model, name) & self.ordinary_fqns:
                    candidates.add(exported_name)
                if name in self.shared_lora_fqns or _get_fqns(source_model, name) & self.shared_lora_fqns:
                    shared_candidates.add(exported_name)
            self.ordinary_fqns = {name for name in candidates if len(owners[name]) == 1}
            self.shared_lora_fqns = {name for name in shared_candidates if len(owners[name]) == 1}

    def create_local_plan(self):
        plan = super().create_local_plan()
        layouts = {}
        rank_local_keys = _verified_rank_local_keys(self.rank_local_adapter, loading=False)
        for key, value in self.state_dict.items():
            if key in rank_local_keys:
                continue
            logical_key = _logical_extra_parallel_key(key, self.state_dict)
            if logical_key is None:
                continue
            extra_names = getattr(self.parallel_state, "extra_parallel_names", ())
            layout = (
                _expert_layout(value, extra_names)
                or _no_ep_expert_layout(
                    logical_key, value, self.parallel_state, self.ordinary_fqns, self.shared_lora_fqns
                )
                or _tp_only_source_layout(
                    logical_key,
                    value,
                    self.tp_only_parameters,
                    self.tp_only_mesh,
                    allow_staged_copies=self.allow_staged_copies,
                )
            )
            if layout is not None:
                layouts[key] = layout
            elif _unidentified_same_axis_shards(value) or (
                "experts" in logical_key.split(".") and torch.is_tensor(value) and value.ndim > 0
            ):
                layouts[key] = {"expert_order": "unproven"}
        return replace(plan, planner_data={**(plan.planner_data or {}), _EXPERT_LAYOUTS_KEY: layouts})

    def create_global_plan(self, all_plans):
        plans, metadata = super().create_global_plan(all_plans)
        # Reuse DCP's plan exchange; sparse optimizer states may appear on only
        # some ranks. Keep the ordinary FQN -> state-dict path mappings intact.
        layouts = {}
        for plan in all_plans:
            for key, value in plan.planner_data[_EXPERT_LAYOUTS_KEY].items():
                if key in layouts and layouts[key] != value:
                    raise RuntimeError(f"Checkpoint tensor {key!r} has inconsistent source layouts across ranks")
                layouts[key] = value
        has_v2_model = any(
            key.startswith("model.") and _is_extra_parallel_layout_marker(key.split(".")[-1])
            for key in metadata.state_dict_metadata
        )
        if has_v2_model:
            unproven = sorted(
                key
                for key, value in layouts.items()
                if key.startswith("model.") and value.get("expert_order") != "canonical"
            )
            if unproven:
                raise RuntimeError(
                    f"Cannot save canonical ExtraParallel checkpoint with unproven expert layout: {unproven[0]}"
                )
        metadata.planner_data = {
            **(metadata.planner_data or {}),
            _EXPERT_LAYOUTS_KEY: {"version": 1, "tensors": layouts},
        }
        return plans, metadata


class _ModelStrictLoadPlanner(DefaultLoadPlanner):
    """Allow partial optimizer state while requiring a complete full-model DCP."""

    def __init__(
        self, strict_model: bool, expected_save_id=None, legacy_expert_source_layout=None, *, rank_local_adapter=None
    ):
        super().__init__(allow_partial_load=True)
        _validate_legacy_expert_source_layout(legacy_expert_source_layout)
        self.strict_model = strict_model
        self.expected_save_id = expected_save_id
        self.legacy_expert_source_layout = legacy_expert_source_layout
        self.rank_local_adapter = rank_local_adapter

    def create_local_plan(self):
        assert self.metadata is not None
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

    def _validate_expert_layouts(self):
        planner_data = self.metadata.planner_data or {}
        source = planner_data.get(_EXPERT_LAYOUTS_KEY)
        rank_local_keys = _verified_rank_local_keys(self.rank_local_adapter, loading=True)
        source_tensors = {}
        if source is not None:
            if source["version"] != 1:
                raise RuntimeError(f"Unsupported checkpoint expert layout version: {source['version']}")
            source_tensors = source["tensors"]
            for key in self.state_dict.keys() & source_tensors.keys():
                if key in rank_local_keys:
                    continue
                if source_tensors[key]["expert_order"] != "canonical":
                    raise RuntimeError(
                        f"Checkpoint tensor {key!r} has noncanonical expert order: {source_tensors[key]}. "
                        "Migrate model and optimizer state using the original training layout before loading."
                    )

        # The v2 namespace proves how sharded EP tensors were serialized, but
        # not whether an ordinary/shared expert was a synchronized replica.
        # Each saved expert needs its own layout proof even in a v2 checkpoint.
        marker = _extra_parallel_layout_key({})
        has_v2_marker = any(key.split(".")[-1] == marker for key in self.metadata.state_dict_metadata)
        # Preserve the earlier unversioned EP-first contract. A v2 marker must
        # not bypass the per-tensor proof even if both markers are present.
        if not has_v2_marker and planner_data.get("__veomni_ep_first_layout__") == 1:
            return

        for key, value in self.state_dict.items():
            if key in source_tensors or key in rank_local_keys:
                continue
            layout = _expert_layout(value)
            saved = self.metadata.state_dict_metadata.get(key)
            if saved is None or not torch.is_tensor(value) or not hasattr(saved, "size"):
                continue
            root = key.partition(".")[0]
            root_marker = f"{root}.{marker}"
            source_marker = {root_marker: "ep-outer-v1"} if root_marker in self.metadata.state_dict_metadata else {}
            logical_key = _logical_extra_parallel_key(key, source_marker)
            is_expert = logical_key is not None and "experts" in logical_key.split(".")
            if (
                is_expert
                and bool(saved.size)
                and (has_v2_marker or (source is not None and self.legacy_expert_source_layout is None))
            ):
                raise RuntimeError(
                    f"Cannot determine source expert order for {key!r}: checkpoint metadata omits its layout. "
                    "Verify and migrate the original training layout before loading."
                )
            if has_v2_marker:
                continue
            # Unversioned chunks cannot identify the source EP axis. The
            # target axis is not source evidence: a source sharded along one
            # axis can appear whole along a different target expert axis.
            if layout is None:
                # Offline export uses a plain CPU tensor as its target. Old
                # stacked expert weights still require the same source check.
                if not is_expert:
                    continue
            if all(tuple(chunk.sizes) == tuple(saved.size) for chunk in saved.chunks):
                continue
            # This declaration is a trusted statement about the source, not
            # an inference from target topology or ambiguous chunk geometry.
            if self.legacy_expert_source_layout is not None:
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


def _has_live_expert_checkpoint_tensors(model):
    """Conservatively include aliases, frozen parameters, and persistent buffers."""
    if not isinstance(model, torch.nn.Module):
        return False

    specs = getattr(model, "_fqn2spec_info", None) or {}

    def is_expert(name, tensor):
        spec = specs.get(name)
        return tensor.ndim and (
            "experts" in name.split(".")
            or (getattr(spec, "para_name", None) == "ep" and isinstance(getattr(spec, "placement", None), Shard))
        )

    if any(is_expert(name, param) for name, param in model.named_parameters(remove_duplicate=False)):
        return True
    for prefix, module in model.named_modules(remove_duplicate=False):
        for name, buffer in module._buffers.items():
            fqn = f"{prefix}.{name}" if prefix else name
            if name not in module._non_persistent_buffers_set and buffer is not None and is_expert(fqn, buffer):
                return True
    return False


def _reject_unproven_extra_parallel_optimizer(
    state, parallel_state, *, group=None, trainable_only=False, allow_proven_optimizer=False, saving=False
):
    error, has_v2, logical_signature = None, False, None
    try:
        model = state.get("model")
        ps = (
            (parallel_state if parallel_state is not None else get_parallel_state())
            if saving or state.get("optimizer") is not None
            else None
        )
        if saving and getattr(ps, "tp_size", 1) > 1 and _has_live_expert_checkpoint_tensors(model):
            raise RuntimeError(
                "Cannot save a tensor-parallel model containing expert tensors: "
                "the expert checkpoint source layout is not proven for TP > 1"
            )
        specs = getattr(model, "_fqn2spec_info", None)
        has_v2 = _needs_extra_parallel_layout_marker(specs)
        if has_v2:
            _extra_parallel_layout_key(specs)
        if state.get("optimizer") is not None and getattr(ps, "dp_mode", None) == "fsdp2":
            if has_v2 and not allow_proven_optimizer:
                raise RuntimeError("Canonical ExtraParallel optimizer state requires an owner-aware checkpoint codec")
        if has_v2:
            named_params = list(state["model"].named_parameters(remove_duplicate=False))
            if ps is None:
                ps = parallel_state if parallel_state is not None else get_parallel_state()
            aliases = {}
            for name, param in named_params:
                aliases.setdefault(id(param), []).append(name)
            expert_aliases = {
                name
                for names in aliases.values()
                if any("experts" in alias.split(".") for alias in names)
                for name in names
            }
            ordinary_experts = (
                tuple(sorted(_ordinary_expert_fqns(state["model"], ps)))
                if isinstance(ps, ParallelState) and any(size != 1 for size in ps.extra_parallel_sizes.values())
                else ()
            )
            logical_signature = _checkpoint_extra_parallel_signature(
                specs,
                optimizer_present=state.get("optimizer") is not None,
                trainable_only=trainable_only,
                trainability=(
                    tuple(sorted((name, param.requires_grad) for name, param in named_params))
                    if trainable_only
                    else ()
                ),
                runtime_ndims={
                    name: tensor.ndim
                    for name, tensor in (named_params + list(state["model"].named_buffers(remove_duplicate=False)))
                },
                expert_aliases=expert_aliases,
                ordinary_experts=ordinary_experts,
                shared_lora=tuple(sorted(_shared_lora_fqns(state["model"], ordinary_experts)))
                if ordinary_experts
                else (),
            )
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "checking canonical ExtraParallel optimizer state", group=group)
    _check_extra_parallel_signature_across_ranks(has_v2, logical_signature, group=group)
    error = None
    try:
        if has_v2:
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


def _checkpoint_extra_parallel_signature(
    specs,
    *,
    optimizer_present,
    trainable_only,
    trainability=(),
    runtime_ndims=None,
    expert_aliases=(),
    ordinary_experts=(),
    shared_lora=(),
):
    """Describe logical layout, excluding rank-local coordinates and tensor contents."""
    entries = []
    for name, spec in sorted((specs or {}).items()):
        mesh = getattr(spec, "para_fsdp_mesh", None)
        shape = getattr(spec, "global_shape", None)
        placement = getattr(spec, "placement", None)
        if isinstance(placement, Replicate) and ("experts" in name.split(".") or name in expert_aliases):
            mesh_names = getattr(mesh, "mesh_dim_names", None)
            entries.append(
                (
                    name,
                    spec.para_name,
                    "Replicate",
                    tuple(mesh_names) if mesh_names is not None else None,
                    tuple(mesh.shape) if mesh is not None else None,
                )
            )
            continue
        if not isinstance(placement, Shard):
            continue
        axis = placement.dim
        fsdp_axis = getattr(spec, "fsdp_shard_dim", None)
        persistent_axis = getattr(spec, "persistent_fsdp_shard_dim", None)
        layout_v2 = bool(getattr(spec, "checkpoint_layout_v2", False))
        if persistent_axis is not None:
            # Persistent shards do not use the ordinary FSDP axis or v2 marker.
            fsdp_axis, layout_v2 = None, None
        ndim = len(shape) if shape is not None else (runtime_ndims or {}).get(name)
        if ndim:
            for shard_axis in (axis, fsdp_axis, persistent_axis):
                if shard_axis is not None and not -ndim <= shard_axis < ndim:
                    raise RuntimeError(f"ExtraParallel checkpoint axis is outside the tensor rank: {name}")
            axis = axis % ndim if axis is not None else None
            fsdp_axis = fsdp_axis % ndim if fsdp_axis is not None else None
            persistent_axis = persistent_axis % ndim if persistent_axis is not None else None
        mesh_names = getattr(mesh, "mesh_dim_names", None)
        entries.append(
            (
                name,
                spec.para_name,
                type(placement).__name__,
                axis,
                fsdp_axis,
                persistent_axis,
                tuple(shape) if shape is not None else None,
                layout_v2,
                tuple(mesh_names) if mesh_names is not None else None,
                tuple(mesh.shape) if mesh is not None else None,
            )
        )
    return (
        bool(optimizer_present),
        bool(trainable_only),
        tuple(trainability),
        tuple(ordinary_experts),
        tuple(shared_lora),
        tuple(entries),
    )


def _check_extra_parallel_signature_across_ranks(has_v2, signature, *, group=None):
    """Reject rank-inconsistent checkpoint policy before collectives or file mutation."""
    if not dist.is_initialized():
        return
    device = "cpu" if dist.get_backend(group) == "gloo" else get_device_type()
    presence = torch.tensor([int(has_v2), int(not has_v2)], dtype=torch.int32, device=device)
    dist.all_reduce(presence, op=dist.ReduceOp.MAX, group=group)
    if bool(presence[0]) and bool(presence[1]):
        raise RuntimeError("checkpoint ExtraParallel layout version differs across ranks")
    if not bool(presence[0]):
        return
    digest = hashlib.sha256(repr(signature).encode("utf-8")).digest()
    low = torch.tensor(list(digest), dtype=torch.int32, device=device)
    high = low.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(high, op=dist.ReduceOp.MAX, group=group)
    if not torch.equal(low, high):
        raise RuntimeError("checkpoint ExtraParallel layout or optimizer policy differs across ranks")


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


def _assert_extra_parallel_layout_unchanged(owner):
    specs = owner.extra_parallel_fqn2spec_info
    if specs is not getattr(owner.model, "_fqn2spec_info", None) or (
        owner._layout_signature != _runtime_extra_parallel_layout_signature(specs)
    ):
        raise RuntimeError("ExtraParallel layout changed while checkpoint adapter was active")


def _coordinate_extra_parallel_layout_check(owner, group):
    error = None
    try:
        _assert_extra_parallel_layout_unchanged(owner)
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "checking checkpoint adapter layout", group=group)


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
    """Wrap CPU optimizer shards without moving or changing their strides."""
    if local_tensor.device.type != "cpu" or device_mesh.device_type == "cpu":
        return DTensor.from_local(
            local_tensor, device_mesh=device_mesh, placements=placements, run_check=False, shape=shape, stride=stride
        )
    wrapped = rewrap_dtensor_local(
        local_tensor,
        mesh=device_mesh,
        placements=placements,
        shape=shape,
        stride=stride,
    )
    if wrapped.requires_grad != local_tensor.requires_grad:
        wrapped = DTensor(local_tensor, wrapped._spec, requires_grad=local_tensor.requires_grad)
    return wrapped


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
        if ep_dim < fsdp_dim or ep_size == 1 or isinstance(loaded_tensor.placements[fsdp_dim], _StridedShard):
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
        When both dimensions shard the same tensor axis, the FSDP placement
        uses ``_StridedShard`` on the FSDP-first runtime mesh. If the global
        axis is uneven across EP, use an EP-first mesh view instead: current
        Torch strided metadata can report incorrect offsets for that case.
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
        same_axis = ep_shard_dim == fsdp_shard_dim and fsdp_mesh.size(-1) > 1
        if same_axis and global_shape[ep_shard_dim] % fsdp_mesh.size(-1):
            checkpoint_mesh = _extra_parallel_first_mesh(fsdp_mesh)
            placements = [Shard(ep_shard_dim), Shard(fsdp_shard_dim)]
            if checkpoint_mesh.ndim == 3:
                placements.insert(0, Replicate())
        else:
            checkpoint_mesh = fsdp_mesh
            fsdp_placement = (
                _StridedShard(fsdp_shard_dim, split_factor=fsdp_mesh.size(-1)) if same_axis else Shard(fsdp_shard_dim)
            )
            placements = [fsdp_placement, Shard(ep_shard_dim)]
            if checkpoint_mesh.ndim == 3:
                placements.insert(0, Replicate())
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
        if (
            action == "restore"
            and key_match == "substring"
            and not isinstance(tensor, DTensor)
            and fsdp_submesh.size() > 1
        ):
            raise RuntimeError(
                "Plain optimizer state on multiple ExtraParallel FSDP ranks has no proven replica layout; "
                "use a rank-local optimizer checkpoint codec."
            )
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
        _assert_extra_parallel_layout_unchanged(self)
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
        _assert_extra_parallel_layout_unchanged(self)
        model_state_dict = _strip_extra_parallel_layout_marker(
            state_dict,
            self.should_extra_parallel_aware
            and _needs_extra_parallel_layout_marker(self.extra_parallel_fqn2spec_info),
            self._layout_marker_key,
        )
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


def _checkpoint_optimizer_policy(state, *, loading, group=None):
    """Agree on the optimizer path before entering any checkpoint collective."""
    optimizer, multi, swap, error = None, False, False, None
    try:
        from ..optim.swap_adamw import SwapAdamW

        if not isinstance(state, dict) or "model" not in state:
            if loading and state is None:
                raise ValueError("State dict must be provided to load a distributed checkpoint.")
            raise ValueError(f"Model must be provided to {'load' if loading else 'save'} a distributed checkpoint.")
        optimizer = state.get("optimizer")
        multi = getattr(optimizer, "_is_multi_optimizer", False) is True
        swap = type(optimizer) is SwapAdamW
        if not multi and _requires_rank_local_optimizer_checkpoint(optimizer):
            raise RuntimeError("Exact-precision optimizers require MultiOptimizer for sparse DCP recovery")
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "checking checkpoint optimizer policy", group=group)

    if any_rank_failed(multi, group=group) and any_rank_failed(not multi, group=group):
        raise RuntimeError("Checkpoint optimizer types differ across ranks")
    if any_rank_failed(optimizer is None, group=group) and any_rank_failed(optimizer is not None, group=group):
        raise RuntimeError("Checkpoint optimizer presence differs across ranks")
    if any_rank_failed(swap, group=group) and any_rank_failed(not swap, group=group):
        raise RuntimeError("Checkpoint optimizer offload types differ across ranks")
    return optimizer, multi


class OptimizerState(Stateful):
    """A wrapper around an optimizer to make it stateful.

    On save, only optimizer state that actually exists is persisted — params
    that never received a gradient (e.g. unused MoE experts, frozen LoRA
    base weights) are simply absent from the checkpoint.  ``SwapAdamW`` is the
    exception for the parameters it owns as trainable: it allocates their state
    at construction, so those entries are dense and never-stepped ones carry
    zero moments and ``step`` 0 — fresh state, as a newly built optimizer has
    it.  A parameter it was built with frozen (or added while frozen) has no
    state until its first gradient, like a stock optimizer's, and is absent
    from the checkpoint until then.  A stock optimizer resumed through this
    class instead ends an absent param at ``step`` 1 (the checkpoint
    machinery's eager zero-grad initialization); a direct ``load_state_dict``
    that omits the param leaves it stateless until its next ``step()``.

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

    def __init__(self, model, optimizer, parallel_state=None, *, load: bool = False, process_group=None):
        self.model = model
        self.optimizer = optimizer
        self.process_group = process_group
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
        if self.should_extra_parallel_aware:
            # Reject unsupported raw DCP on every rank before any optimizer
            # can enter its own collectives, even if a local check fails.
            error = None
            try:
                _assert_extra_parallel_layout_unchanged(self)
                if _requires_rank_local_optimizer_checkpoint(self.optimizer):
                    raise RuntimeError("Exact-precision optimizers require MultiOptimizer for sparse DCP recovery")
                if self._layout_marker_key is not None:
                    raise RuntimeError(
                        "Canonical ExtraParallel optimizer state requires an owner-aware checkpoint codec"
                    )
                if not getattr(self.optimizer, "_is_multi_optimizer", False):
                    raise RuntimeError("ExtraParallel is enabled but optimizer is not a MultiOptimizer instance")
                raise RuntimeError(
                    "Raw DCP MultiOptimizer does not prove parameter groups or scheduler identity; "
                    "use DistributedCheckpointer with the live scheduler."
                )
            except BaseException as exc:
                error = exc
            raise_if_any_rank_failed(error, "validating ExtraParallel optimizer checkpoint", group=self.process_group)
            raise RuntimeError("ExtraParallel optimizer checkpoint preflight unexpectedly succeeded")

        _assert_extra_parallel_layout_unchanged(self)
        if _requires_rank_local_optimizer_checkpoint(self.optimizer):
            raise RuntimeError("Exact-precision optimizers require MultiOptimizer for sparse DCP recovery")
        if getattr(self.optimizer, "_is_multi_optimizer", False):
            raise RuntimeError(
                "Raw DCP MultiOptimizer does not prove parameter groups or scheduler identity; "
                "use DistributedCheckpointer with the live scheduler."
            )
        return get_optimizer_state_dict(model=self.model, optimizers=self.optimizer)

    def load_state_dict(self, state_dict):
        """Restore optimizer state from the loaded DCP state dict."""
        _assert_extra_parallel_layout_unchanged(self)
        if _requires_rank_local_optimizer_checkpoint(self.optimizer):
            raise RuntimeError("Exact-precision optimizers require MultiOptimizer for sparse DCP recovery")
        if self.should_extra_parallel_aware and self._layout_marker_key is not None:
            raise RuntimeError("Canonical ExtraParallel optimizer state requires an owner-aware checkpoint codec")
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
        _coordinate_extra_parallel_layout_check(self, self.group)
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
        _coordinate_extra_parallel_layout_check(self, self.group)
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
        expected_layout,
        group,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.generation = generation
        self.expected_layout = expected_layout
        self.group = group
        self._verified_rank_local_keys = frozenset()

    def state_dict(self):
        _coordinate_extra_parallel_layout_check(self, self.group)
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
                self._verified_rank_local_keys = frozenset(inventory)
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
        result[MULTI_LAYOUT_MARKER] = json.dumps(
            {
                "generation": self.generation,
                "layout": self.expected_layout,
                "inventory_digests": _inventory_digests(inventory, group=self.group),
            },
            sort_keys=True,
        )
        return result

    def load_state_dict(self, state_dict):
        _coordinate_extra_parallel_layout_check(self, self.group)
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
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "validating loaded MultiOptimizer state", group=self.group)
        digests = _inventory_digests(inventory, group=self.group)
        error = (
            RuntimeError("MultiOptimizer inventory identity changed during tensor loading")
            if marker["inventory_digests"] != digests
            else None
        )
        raise_if_any_rank_failed(error, "validating MultiOptimizer inventory identity", group=self.group)
        error = None
        try:
            self.optimizer._dcp_load_state_dict(state_dict, group=self.group)
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

    * A pure ExtraParallel shard, with only replicated FSDP axes -> return the
      plain local tensor.
    * 2 placements: ``(ep_fsdp, ep)`` -> keep FSDP ``Shard(fsdp_shard_dim)`` on
      the 1D sub-mesh.
    * 3 placements: ``(ep_replicate, ep_fsdp, ep)`` (HSDP) -> keep
      ``[Replicate(), Shard(fsdp_shard_dim)]`` on the 2D sub-mesh.
    """

    num_placements = len(loaded_tensor.placements)
    has_fsdp_shard = any(not isinstance(p, Replicate) for p in loaded_tensor.placements[:-1])
    if num_placements in (2, 3) and has_fsdp_shard:
        ndim = loaded_tensor.ndim
        if not -ndim <= fsdp_shard_dim < ndim:
            raise RuntimeError("FSDP shard dimension is outside the tensor rank")
        fsdp_shard_dim %= ndim
    if num_placements == 1 or (num_placements in (2, 3) and not has_fsdp_shard):
        tensor_to_put = loaded_tensor.to_local()
    elif num_placements == 2:
        tensor_to_put = _from_local_preserve_device(
            loaded_tensor._local_tensor, device_mesh=device_mesh, placements=[Shard(fsdp_shard_dim)]
        )
    elif num_placements == 3:
        tensor_to_put = _from_local_preserve_device(
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

    if isinstance(orgin_tensor, DTensor):
        ndim = orgin_tensor.ndim
        if not (-ndim <= ep_shard_dim < ndim and -ndim <= fsdp_shard_dim < ndim):
            raise RuntimeError("ExtraParallel shard dimension is outside the tensor rank")
        ep_shard_dim %= ndim
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
        dtensor = _from_local_preserve_device(orgin_tensor._local_tensor, device_mesh=fsdp_mesh, placements=placements)
    elif torch.is_tensor(orgin_tensor):
        # Plain model tensors must be replicated across FSDP coordinates for
        # the same ExtraParallel rank. Optimizer shards without that proof are
        # rejected by the caller.
        ndim = orgin_tensor.ndim
        if not -ndim <= ep_shard_dim < ndim:
            raise RuntimeError("ExtraParallel shard dimension is outside the tensor rank")
        if fsdp_mesh.mesh_dim_names is None:
            raise RuntimeError("A plain ExtraParallel tensor requires named mesh dimensions")
        placements = [Replicate()] * (fsdp_mesh.ndim - 1) + [Shard(ep_shard_dim % ndim)]
        dtensor = _from_local_preserve_device(orgin_tensor, device_mesh=fsdp_mesh, placements=placements)
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


def _promote_staged_checkpoint(stage_path: str, final_path: str, group: Optional[Any] = None) -> None:
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

    Nested files are copied in the data phase, before any ``.metadata`` is. The
    per-rank ``extra_state/`` files are among those files.
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
        shutil.rmtree(final_path, ignore_errors=True)
        os.makedirs(final_path, exist_ok=True)

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


@contextmanager
def _single_swap_frozen_resume(model, optimizer, reader, *, group=None, trainable_only=False):
    """Include saved frozen SwapAdamW owners in the legacy DCP template."""
    from ..optim.swap_adamw import SwapAdamW

    if type(optimizer) is not SwapAdamW:
        yield ()
        return
    names, owned, error = None, None, None
    try:
        names = _dcp_parameter_name_map(model)
        owned = {id(p): p for g in optimizer.param_groups for p in g["params"]}
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "inspecting SwapAdamW parameter ownership", group=group)
    needs_inspection = any(not p.requires_grad or id(p) not in owned for p in names.values())
    needs_inspection = any_rank_failed(needs_inspection, group=group)
    metadata, group_indices, error = None, None, None
    try:
        if needs_inspection:
            _validate_fresh_swap_optimizer(optimizer)
        metadata = reader.read_metadata().state_dict_metadata
        group_indices = sorted(
            int(match.group(1))
            for key in metadata
            if (match := re.fullmatch(r"optimizer\.param_groups\.(\d+)\.params", key))
        )
        if not group_indices or group_indices != list(range(len(group_indices))):
            raise RuntimeError("SwapAdamW checkpoint parameter groups cannot be proven")
        if any(
            not isinstance(metadata[f"optimizer.param_groups.{index}.params"], BytesStorageMetadata)
            for index in group_indices
        ):
            raise RuntimeError("SwapAdamW checkpoint parameter groups have an unsupported format")
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "inspecting SwapAdamW checkpoint groups", group=group)
    if dist.is_initialized():
        layouts = [None] * dist.get_world_size(group)
        dist.all_gather_object(layouts, group_indices, group=group)
        if any(layout != layouts[0] for layout in layouts[1:]):
            raise RuntimeError("SwapAdamW checkpoint group metadata differs across ranks")
    saved_groups = {"optimizer": {"param_groups": [{"params": None} for _ in group_indices]}}
    dcp.load(state_dict=saved_groups, storage_reader=reader, process_group=group)
    error = None
    try:
        names_by_id = {id(param): name for name, param in names.items()}
        current_groups = [[names_by_id[id(param)] for param in entry["params"]] for entry in optimizer.param_groups]
        source_groups = [entry["params"] for entry in saved_groups["optimizer"]["param_groups"]]
        if source_groups != current_groups:
            raise RuntimeError(
                "SwapAdamW checkpoint parameter groups differ; restore the original groups or use a verified migration"
            )
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "validating SwapAdamW checkpoint groups", group=group)
    if not needs_inspection:
        try:
            yield ()
        except BaseException:
            optimizer._veomni_dcp_restore_failed = True
            raise
        return
    changed, allocated, error = [], [], None
    try:
        for name, param in names.items():
            if id(param) in owned and param.requires_grad:
                continue
            fields = {
                field: metadata.get(f"optimizer.state.{name}.{field}") for field in ("step", *optimizer._state_keys)
            }
            present = {key for key, value in fields.items() if value is not None}
            if not present:
                continue
            if present != set(fields):
                raise RuntimeError(f"SwapAdamW frozen state for {name} is incomplete")
            if id(param) not in owned:
                raise RuntimeError(
                    f"SwapAdamW checkpoint has state for unowned parameter {name}; "
                    "restore the original optimizer groups or use a verified migration"
                )
            if trainable_only:
                raise RuntimeError("Frozen SwapAdamW history requires a full-model checkpoint")
            if optimizer._optimizer_load_state_dict_pre_hooks or optimizer._optimizer_load_state_dict_post_hooks:
                raise RuntimeError("Frozen SwapAdamW history does not support optimizer load-state hooks")
            for field, value in fields.items():
                expected_shape = torch.Size([]) if field == "step" else param.shape
                expected_dtype = torch.float32 if field == "step" else param.dtype
                if (
                    not isinstance(value, TensorStorageMetadata)
                    or value.size != expected_shape
                    or value.properties.dtype != expected_dtype
                ):
                    raise RuntimeError(f"SwapAdamW frozen state {name}.{field} has incompatible shape or dtype")
            if param not in optimizer._host_states:
                allocated.append(param)
                optimizer._ensure_param_state(param, bool(optimizer._group_by_param[param]["amsgrad"]))
            param.requires_grad_(True)
            changed.append(param)
    except BaseException as exc:
        error = exc
    try:
        raise_if_any_rank_failed(error, "preparing frozen SwapAdamW checkpoint state", group=group)
    except BaseException:
        for param in changed:
            param.requires_grad_(False)
        for param in allocated:
            optimizer._host_states.pop(param, None)
            optimizer._device_states.pop(param, None)
            optimizer.state.pop(param, None)
        raise
    try:
        yield changed
    except BaseException:
        optimizer._veomni_dcp_restore_failed = True
        raise
    finally:
        for param in changed:
            param.requires_grad_(False)


@contextmanager
def _single_swap_frozen_model_view(changed):
    for param in changed:
        param.requires_grad_(False)
    try:
        yield
        if any(param.requires_grad for param in changed):
            raise RuntimeError("Model load-state hooks changed frozen SwapAdamW parameter trainability")
    finally:
        for param in changed:
            param.requires_grad_(True)


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
                directories and the per-rank ``extra_state/`` pickles — lands under
                the staging directory and is copied together, with every
                ``.metadata`` copied
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
        optimizer, multi = _checkpoint_optimizer_policy(state, loading=False)
        # An async write may outlive a staged copy, so the modes cannot combine.
        if stage_dir and save_async:
            raise ValueError("stage_dir cannot be combined with save_async")
        _reject_unproven_extra_parallel_optimizer(
            state, parallel_state, trainable_only=trainable_only, allow_proven_optimizer=multi, saving=True
        )
        from ..optim.swap_adamw import SwapAdamW

        single_swap_snapshot = None
        if type(optimizer) is SwapAdamW:
            if save_async:
                cls._drain_slot(OPTIMIZER_DIRNAME)
            error = None
            try:
                single_swap_snapshot = OptimizerState(
                    model=state["model"], optimizer=optimizer, parallel_state=parallel_state
                ).state_dict()
            except BaseException as exc:
                error = exc
            raise_if_any_rank_failed(error, "preparing SwapAdamW checkpoint state")
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
            captured_layout = capture_multi_layout(
                state["model"], optimizer, snapshot, scheduler_object, frozen_digest=frozen_digest
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
            error = None
            try:
                os.stat(manifest_path(checkpoint_dir))
            except FileNotFoundError:
                pass
            except BaseException as exc:
                error = exc
            else:
                error = RuntimeError("Cannot overwrite a completed checkpoint transaction with MultiOptimizer state")
            raise_if_any_rank_failed(error, "checking MultiOptimizer checkpoint destination")
        if multi and extra is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
            extra_path = os.path.join(model_root, _EXTRA_STATE_DIRNAME, _EXTRA_STATE_FORMAT.format(rank))
            error = None
            try:
                os.stat(extra_path)
            except FileNotFoundError:
                pass
            except BaseException as exc:
                error = exc
            else:
                error = RuntimeError("Existing checkpoint sidecar cannot be reused when extra_state is omitted")
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
        raise_if_any_rank_failed(error, "preparing checkpoint files")
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
            try:
                initial_layout = finish_multi_layout(captured_layout, extra_digest)
            except BaseException:
                if stage_path is not None and _local_rank() == 0:
                    shutil.rmtree(stage_path, ignore_errors=True)
                raise

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
                    cls._create_storage_writer(os.path.join(write_root, WEIGHTS_DIRNAME)),
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
                parallel_state=parallel_state if parallel_state is not None else get_parallel_state(),
                source_model=state["model"],
            )

            if "optimizer" in state and state["optimizer"] is not None:
                optim_state, optim_writer = prepare_writer_state(
                    lambda: (
                        _ProvenMultiOptimizerState(
                            model=state["model"],
                            optimizer=optimizer,
                            parallel_state=parallel_state,
                            generation=generation,
                            expected_layout=initial_layout,
                            group=None,
                        )
                        if multi
                        else single_swap_snapshot
                        if single_swap_snapshot is not None
                        else OptimizerState(model=state["model"], optimizer=optimizer, parallel_state=parallel_state),
                        cls._create_storage_writer(os.path.join(write_root, OPTIMIZER_DIRNAME)),
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
                    parallel_state=parallel_state if parallel_state is not None else get_parallel_state(),
                    source_model=state["model"],
                )
        except BaseException:
            if stage_path is not None and _local_rank() == 0:
                shutil.rmtree(stage_path, ignore_errors=True)
            raise

        if stage_path is not None:
            _promote_staged_checkpoint(stage_path, model_root, group=stage_group)

        logger.info_rank0(f"Saved checkpoint to {model_root}")
        return generation if multi else None

    @classmethod
    def _remove_dcp_markers(cls, checkpoint_dir: str, module: str) -> None:
        """Delete the ``.metadata`` files vouching for what this save overwrites.

        The unstaged path's invalidation, run before the first byte lands. A
        staged save has ``_promote_staged_checkpoint`` do the same job at
        promotion time, because until then it has overwritten nothing.

        This module's ``ckpt/`` and ``optimizer/``: DCP's own marker, which is
        why the checkpointer drops them rather than the callback -- this class
        is the one that puts them back.

        Only this module's. A sibling module is not being rewritten and its
        markers still hold. The step's ``checkpoint_manifest.json`` is not ours
        either -- ``GlobalStateCallback`` writes it and clears it.

        One rank owns these files, so only that rank deletes them, and the
        reduction turns a failure there into one every rank sees.
        """
        is_coordinator = (not dist.is_initialized()) or dist.get_rank() == 0
        error: Optional[Exception] = None
        if is_coordinator:
            try:
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
        legacy_expert_source_layout: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        load training state from distributed checkpoint

        Mirrors :meth:`save`: weights from ``model/<module>/ckpt``, optimizer from
        ``model/<module>/optimizer``, model-bound extra state from the per-rank
        ``extra_state/`` beside them.

        args:
            path: step directory to load from
            state: state to load, "model" is required; "optimizer" and "extra_state" are optional
            process_group: process group for loading checkpoint
            module: name of the model to load, for a job that trains several.
                Empty for a single-model job. See :meth:`save`.
            legacy_expert_source_layout: Optional, trusted declaration of an
                unversioned model-only checkpoint's source layout. It must be
                identical on all ranks and cannot be used for optimizer restore.
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
        optimizer, multi = _checkpoint_optimizer_policy(state, loading=True, group=process_group)
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
                process_group=process_group,
            )

        wants_optimizer = state.get("optimizer") is not None

        error = None
        try:
            _validate_legacy_expert_source_layout(legacy_expert_source_layout)
            if legacy_expert_source_layout is not None and optimizer is not None:
                raise ValueError("Legacy expert source declarations are only supported for model-only loads")
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "inspecting checkpoint layout", group=process_group)

        model_root = model_dir(path, module)
        # Every directory this load will read has to be complete, or ``dcp.load``
        # fails with a bare "metadata is None" that names neither the path nor
        # the reason. Two ways to get here: a step written by a layout this
        # version no longer produces (see ``docs/usage/checkpoint.md``), or a
        # save that was interrupted — on disk the two are the same shape.
        #
        # Reduced before any rank enters ``dcp.load``: the markers are read from
        # the filesystem, and a job writing to non-shared storage
        # (``dcp_save_to_lowest_rank``) can see a marker on one rank and not
        # another. A rank that raised here alone would leave its peers waiting in
        # the load's collectives until the backend timed out.
        #
        # Requiring the global ``.metadata`` is safe here because that is the
        # only one VeOmni writes: ``FileSystemWriter`` is constructed with its
        # defaults, so each save emits a single step-level ``.metadata`` and
        # never the ``__{rank}.metadata`` a DCP load can fall back to. A
        # checkpoint carrying only rank-local metadata is therefore not
        # something this framework produces.
        error: Optional[Exception] = None
        for directory, wanted in (
            (weights_dir(path, module), True),
            (optimizer_dir(path, module), wants_optimizer),
        ):
            if not wanted:
                continue
            marker = os.path.join(directory, DCP_MARKER_FILENAME)
            try:
                # Opened, not stat-ed. A marker that exists but cannot be read
                # passes ``os.stat`` and then fails inside ``dcp.load`` as
                # "metadata is None" -- the message this guard exists to replace.
                with open(marker, "rb"):
                    pass
            except FileNotFoundError:
                error = FileNotFoundError(
                    f"no complete DCP checkpoint under {directory}: {DCP_MARKER_FILENAME} is "
                    "missing. Checkpoints written before the step directory was split by state "
                    "owner are not resumable, and a save that did not finish leaves the same "
                    "shape. See docs/usage/checkpoint.md."
                )
            except OSError as read_error:
                # Unreadable is not missing. Reporting a permissions or I/O
                # failure as "not resumable" would send the user to the layout
                # doc for a problem that has nothing to do with the layout.
                error = read_error
            if error is not None:
                break
        raise_if_any_rank_failed(error, "validating the checkpoint to load", group=process_group)
        if error is not None:
            raise error
        specs = getattr(state["model"], "_fqn2spec_info", None)
        if _needs_extra_parallel_layout_marker(specs):
            marker_error = None
            try:
                _read_extra_parallel_marker(
                    cls._create_storage_reader(weights_dir(path, module)), _extra_parallel_layout_key(specs)
                )
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
                planner = _ModelStrictLoadPlanner(strict_model=not trainable_only, expected_save_id=model_id)
            except BaseException as exc:
                error = exc
            raise_if_any_rank_failed(error, "preparing proven model checkpoint state", group=process_group)
            dcp.load(
                state_dict={"model": model_state},
                storage_reader=reader,
                process_group=process_group,
                planner=planner,
            )
            try:
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
                        expected_layout=expected_layout,
                        group=process_group,
                    )
                    reader = cls._create_storage_reader(optimizer_path)
                    planner = _ModelStrictLoadPlanner(
                        strict_model=False,
                        expected_save_id=optimizer_id,
                        rank_local_adapter=optimizer_state if optimizer_state.should_extra_parallel_aware else None,
                    )
                except BaseException as exc:
                    error = exc
                raise_if_any_rank_failed(error, "preparing proven optimizer checkpoint state", group=process_group)
                dcp.load(
                    state_dict={"optimizer": optimizer_state},
                    storage_reader=reader,
                    process_group=process_group,
                    planner=planner,
                )
            except BaseException:
                from ..optim.swap_adamw import SwapAdamW

                optimizer._veomni_dcp_restore_failed = True
                for leaf in _iter_optimizer_leaves(optimizer):
                    if type(leaf) is SwapAdamW:
                        leaf._veomni_dcp_restore_failed = True
                raise
            finally:
                _clear_swap_checkpoint_staging(optimizer)
            if extra is not None:
                state["extra_state"].update(extra)
            state["_veomni_optimizer_identity_verified"] = True
            state["_veomni_optimizer_checkpoint_generation"] = generation
            return state
        model_state, optimizer_state, model_reader, reader, error = None, None, None, None, None
        try:
            model_state = _model_state()
            optimizer_state = _optimizer_state() if wants_optimizer else None
            model_reader = cls._create_storage_reader(weights_dir(path, module))
            if wants_optimizer:
                reader = cls._create_storage_reader(optimizer_dir(path, module))
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "preparing distributed checkpoint load", group=process_group)
        if dist.is_initialized():
            decisions = [None] * dist.get_world_size(group=process_group)
            local = (wants_optimizer, multi)
            dist.all_gather_object(decisions, local, group=process_group)
            if any(decision != local for decision in decisions):
                raise RuntimeError("Checkpoint load format differs across ranks")

        def prepared(owner, description):
            snapshot, error = None, None
            try:
                snapshot = owner.state_dict()
            except BaseException as exc:
                error = exc
            raise_if_any_rank_failed(error, description, group=process_group)
            return _PreparedLoadState(owner, snapshot, process_group)

        def load_extra_coordinated(checkpoint_dir):
            error = None
            try:
                cls._load_extra_state(checkpoint_dir=checkpoint_dir, state=state)
            except BaseException as exc:
                error = exc
            raise_if_any_rank_failed(error, "loading checkpoint extra state", group=process_group)

        with _single_swap_frozen_resume(
            state["model"], optimizer, reader, group=process_group, trainable_only=trainable_only
        ) as frozen_swap_owners:
            has_frozen = any_rank_failed(bool(frozen_swap_owners), group=process_group)
            error = None
            if has_frozen and (
                optimizer._optimizer_load_state_dict_pre_hooks or optimizer._optimizer_load_state_dict_post_hooks
            ):
                error = RuntimeError("Frozen SwapAdamW history does not support optimizer load-state hooks")
            raise_if_any_rank_failed(error, "checking frozen optimizer recovery hooks", group=process_group)
            model_root = model_dir(path, module)
            error = None
            try:
                with _single_swap_frozen_model_view(frozen_swap_owners):
                    model_state = prepared(model_state, "preparing model checkpoint state")
                    dcp.load(
                        state_dict={"model": model_state},
                        storage_reader=model_reader,
                        process_group=process_group,
                        planner=_ModelStrictLoadPlanner(
                            strict_model=not trainable_only, legacy_expert_source_layout=legacy_expert_source_layout
                        ),
                    )
            except BaseException as exc:
                error = exc
            raise_if_any_rank_failed(error, "loading model before optimizer recovery", group=process_group)
            if wants_optimizer:
                optimizer_state = prepared(optimizer_state, "preparing optimizer checkpoint state")
                dcp.load(
                    state_dict={"optimizer": optimizer_state},
                    storage_reader=reader,
                    process_group=process_group,
                    planner=_ModelStrictLoadPlanner(strict_model=False),
                )
            load_extra_coordinated(model_root)
            logger.info_rank0(f"Loaded checkpoint from {model_root}")
            return state

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
        parallel_state=None,
        source_model=None,
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
        planner, error = None, None
        try:
            optimizer_adapter = save_state.get("optimizer")
            planner = _ExtraParallelSavePlanner(
                dedup_save_to_lowest_rank=save_to_lowest_rank,
                parallel_state=parallel_state,
                source_model=source_model,
                rank_local_adapter=optimizer_adapter
                if type(optimizer_adapter) is _ProvenMultiOptimizerState
                else None,
                allow_staged_copies=save_async,
            )
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "preparing the checkpoint save planner")
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
            dcp.save(
                state_dict=save_state,
                storage_writer=storage_writer,
                planner=planner,
            )
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
    if normalized is None:
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


def _validate_expert_export_provenance(metadata: Metadata, model_keys, legacy_expert_source_layout=None) -> None:
    """Reject expert layouts that a plain CPU export cannot prove safe."""
    planner_data = metadata.planner_data or {}
    recorded = planner_data.get(_EXPERT_LAYOUTS_KEY)
    layouts = {}
    if recorded is not None:
        if recorded["version"] != 1:
            raise RuntimeError(f"Unsupported checkpoint expert layout version: {recorded['version']}")
        layouts = recorded["tensors"]
        for key in model_keys & layouts.keys():
            if layouts[key]["expert_order"] != "canonical":
                raise RuntimeError(
                    f"Checkpoint tensor {key!r} has noncanonical expert order; migrate it before export"
                )

    marker = _extra_parallel_layout_key({})
    marker_key = "model." + marker
    has_v2_marker = marker_key in metadata.state_dict_metadata
    if not has_v2_marker and planner_data.get("__veomni_ep_first_layout__") == 1:
        return

    # An old expert-only chunk may use either EP-first or FSDP-first order.
    # Plain CPU export targets do not carry the EP mesh needed to distinguish it.
    for key in model_keys:
        tensor_metadata = metadata.state_dict_metadata[key]
        shape = getattr(tensor_metadata, "size", ())
        logical_key = key
        if has_v2_marker:
            fqn = _export_extra_parallel_layout_key(key.removeprefix("model."))
            logical_key = f"model.{fqn}" if fqn is not None else ""
        if "experts" not in logical_key.split(".") or not shape or key in layouts:
            continue
        if has_v2_marker or (recorded is not None and legacy_expert_source_layout is None):
            raise RuntimeError(
                f"Cannot determine source expert order for {key!r}: checkpoint metadata omits its layout. "
                "Verify and migrate the original training layout before exporting."
            )
        if legacy_expert_source_layout is None and not all(
            tuple(chunk.sizes) == tuple(shape) for chunk in tensor_metadata.chunks
        ):
            raise RuntimeError(
                f"Cannot determine source expert order for {key!r} in this unversioned checkpoint. "
                "Verify and migrate the original training layout before exporting."
            )


def _get_sharding_plan(
    checkpoint_path: Union[str, os.PathLike],
    shard_size: int = None,
    save_dtype: Optional[Union[str, torch.dtype]] = None,
    drop_hf_keys: Optional[Set[str]] = None,
    *,
    legacy_expert_source_layout=None,
    include_hf_key=None,
):
    """
    Create sharding plan from checkpoint metadata without loading weights.

    Returns:
        shards: List of {hf_key: dcp_key} dicts per shard
        total_size: Total size in bytes
        all_dcp_keys: All valid DCP model keys
    """
    _validate_legacy_expert_source_layout(legacy_expert_source_layout)
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
            if include_hf_key is not None and not include_hf_key(hf_key):
                continue
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
            if save_dtype and source_dtype.is_floating_point:
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

    _validate_expert_export_provenance(metadata, set(all_dcp_keys), legacy_expert_source_layout)

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
        if not shards and include_hf_key is not None:
            return {}, total_size, all_dcp_keys
        assert len(shards) == 1, "Shard size None should result in a single shard"
        shards = shards[0]
    return shards, total_size, all_dcp_keys


def _process_shard(
    shard_keys: Dict[str, str],
    checkpoint_path: str,
    save_dtype: Optional[Union[str, torch.dtype]] = None,
    *,
    legacy_expert_source_layout: Optional[str] = None,
) -> str:
    _validate_legacy_expert_source_layout(legacy_expert_source_layout)
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
        planner=_ModelStrictLoadPlanner(strict_model=True, legacy_expert_source_layout=legacy_expert_source_layout),
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

        if target_dtype and tensor.is_floating_point():
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


def dcp_to_torch_state_dict(
    save_checkpoint_path: Union[str, os.PathLike], *, legacy_expert_source_layout: Optional[str] = None
) -> STATE_DICT_TYPE:
    """
    Given a directory containing a DCP checkpoint, this function will convert it into a
    Torch state_dict.

    Args:
        save_checkpoint_path: Directory containing the DCP checkpoint.
        legacy_expert_source_layout: Trusted declaration of the source's legacy
            expert layout. See ``docs/usage/checkpoint.md`` for its contract.

    .. warning::
        To avoid OOM, it's recommended to only run this function on a single rank.
    """
    shard, _, _ = _get_sharding_plan(save_checkpoint_path, legacy_expert_source_layout=legacy_expert_source_layout)

    processed_dict = _process_shard(
        shard, save_checkpoint_path, legacy_expert_source_layout=legacy_expert_source_layout
    )

    return processed_dict
