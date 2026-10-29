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
import math
import os
import shutil
import threading
import weakref
from bisect import bisect_left, bisect_right, insort
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, nullcontext
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
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE, Metadata, TensorStorageMetadata
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    _get_fqns,
    get_model_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from ..distributed.parallel_state import get_parallel_state
from ..models.qwen4_gdn_precision import (
    is_qwen4_exp_model,
    is_qwen4_gdn_parameter,
    qwen4_exp_model_config,
    validate_qwen4_gdn_pairs,
)
from ..optim.optimizer import optimizer_state_dict_for_checkpoint
from ..utils import logging
from ..utils.device import empty_cache, synchronize
from ..utils.dist_utils import any_rank_failed, raise_if_any_rank_failed, require_same_rank_value
from .bounded_writer import BoundedFileSystemWriter
from .checkpointer import CheckpointerBase
from .layout import (
    DCP_MARKER_FILENAME,
    OPTIMIZER_DIRNAME,
    WEIGHTS_DIRNAME,
    dcp_markers,
    model_dir,
    optimizer_dir,
    step_dir,
    weights_dir,
)


logger = logging.get_logger(__name__)

_EXTRA_STATE_KEY = "extra_state"
_EXTRA_STATE_DIRNAME = "extra_state"
_EXTRA_STATE_FORMAT = "extra_state_rank_{}.pt"
_EXTRA_PARALLEL_LAYOUT_MARKER = "__veomni_ep_outer_layout_v1__"
_QWEN4_GDN_FP32_LEGACY_MARKER = "__veomni_qwen4_gdn_fp32_v1__"
_QWEN4_GDN_FP32_MARKER = "__veomni_qwen4_gdn_fp32_v2__"
_ZERO_OPTIMIZER_STATE_MARKER = "__veomni_zero_optimizer_state__"
_ABSENT_OPTIMIZER_STATE_ATTR = "_veomni_absent_optimizer_state_bases"
_EXTRA_PARALLEL_FIRST_MESHES = weakref.WeakKeyDictionary()
_EXTRA_PARALLEL_FIRST_MESH_LOCK = threading.Lock()


def read_qwen4_gdn_manifest(checkpoint_path) -> Optional[set[str]]:
    """Read the small persisted precision inventory, if present, from model DCP."""
    reader = FileSystemReader(checkpoint_path)
    key = f"model.{_QWEN4_GDN_FP32_MARKER}"
    if key not in reader.read_metadata().state_dict_metadata:
        return None
    state = {key: ""}
    load(state, storage_reader=reader, no_dist=True)
    try:
        payload = json.loads(state[key])
        names = payload["parameter_names"]
        if payload["version"] != 1 or not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ValueError("invalid schema")
        if len(set(names)) != len(names) or any(not is_qwen4_gdn_parameter(name) for name in names):
            raise ValueError("invalid parameter inventory")
        validate_qwen4_gdn_pairs(names)
        return set(names)
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError("Invalid Qwen4 GDN FP32 checkpoint precision inventory") from exc


def _needs_extra_parallel_layout_marker(specs) -> bool:
    return any(
        isinstance(spec.placement, Shard)
        and spec.placement.dim == spec.fsdp_shard_dim
        and getattr(spec, "persistent_fsdp_shard_dim", None) is None
        for spec in (specs or {}).values()
    )


def _strip_extra_parallel_layout_marker(state_dict, required):
    if not required:
        return state_dict
    state_dict = dict(state_dict)
    if state_dict.pop(_EXTRA_PARALLEL_LAYOUT_MARKER, None) is not True:
        raise RuntimeError(
            "Unversioned same-dimension ExtraParallel checkpoint; explicit layout migration is required"
        )
    return state_dict


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
    with _EXTRA_PARALLEL_FIRST_MESH_LOCK:
        cached = _EXTRA_PARALLEL_FIRST_MESHES.get(mesh)
        if cached is not None:
            return cached

        names = mesh.mesh_dim_names
        if names is None or len(names) != mesh.ndim or mesh.ndim not in (2, 3):
            raise RuntimeError("ExtraParallel checkpoint mesh must have 2 or 3 named dimensions")
        order = tuple(range(mesh.ndim - 2)) + (mesh.ndim - 1, mesh.ndim - 2)
        reordered = DeviceMesh(
            mesh.device_type,
            mesh.mesh.permute(order),
            mesh_dim_names=tuple(names[index] for index in order),
        )
        _EXTRA_PARALLEL_FIRST_MESHES[mesh] = reordered
        return reordered


def _mark_zero_optimizer_state_resume(optimizer) -> None:
    if optimizer is None:
        raise RuntimeError("A zero-state optimizer checkpoint requires the target optimizer")
    targets = optimizer.values() if isinstance(optimizer, dict) else (optimizer,)
    for target in targets:
        target._veomni_resume_zero_optimizer_state = True


def _reset_zero_optimizer_state_resume(optimizer) -> None:
    """Discard an unconsumed resume marker without touching optimizer state."""
    if optimizer is None:
        return
    targets = optimizer.values() if isinstance(optimizer, dict) else (optimizer,)
    for target in targets:
        target._veomni_resume_zero_optimizer_state = False


def _clear_optimizer_state(optimizer) -> None:
    if isinstance(optimizer, dict):
        for child in optimizer.values():
            _clear_optimizer_state(child)
        return
    children = getattr(optimizer, "optimizers_dict", None)
    if children is not None:
        for child in children.values():
            _clear_optimizer_state(child)
        return
    optimizer.state.clear()


def _finish_zero_optimizer_state_resume(optimizer) -> None:
    if optimizer is None:
        return
    targets = optimizer.values() if isinstance(optimizer, dict) else (optimizer,)
    if not any(getattr(target, "_veomni_resume_zero_optimizer_state", False) for target in targets):
        return
    try:
        _clear_optimizer_state(optimizer)
    finally:
        _reset_zero_optimizer_state_resume(optimizer)


def _reset_absent_optimizer_state(optimizer) -> None:
    """Clear a planner-to-loader handoff for wholly absent optimizer state."""
    if optimizer is None:
        return
    targets = optimizer.values() if isinstance(optimizer, dict) else (optimizer,)
    for target in targets:
        setattr(target, _ABSENT_OPTIMIZER_STATE_ATTR, frozenset())


def _record_absent_optimizer_state(optimizer, bases) -> None:
    if optimizer is None:
        return
    targets = optimizer.values() if isinstance(optimizer, dict) else (optimizer,)
    frozen = frozenset(bases)
    for target in targets:
        setattr(target, _ABSENT_OPTIMIZER_STATE_ATTR, frozen)


def _iter_optimizer_leaves(optimizer):
    if isinstance(optimizer, dict):
        for child in optimizer.values():
            yield from _iter_optimizer_leaves(child)
        return
    children = getattr(optimizer, "optimizers_dict", None)
    if children is not None:
        for child in children.values():
            yield from _iter_optimizer_leaves(child)
        return
    yield optimizer


def _file_backed_optimizers(optimizer):
    if optimizer is None:
        return ()
    return tuple(
        leaf for leaf in _iter_optimizer_leaves(optimizer) if getattr(leaf, "_ple_moment_storage", None) is not None
    )


def _file_backed_mode(optimizer):
    return optimizer is not None and any(
        getattr(leaf, "_ple_file_mode_enabled", False) is True for leaf in _iter_optimizer_leaves(optimizer)
    )


def _agree_file_backed_mode(optimizer, group=None):
    """Check optimizer types and scratch-mode flags before branching collectively."""
    signature, error = None, None
    try:
        leaves = () if optimizer is None else tuple(_iter_optimizer_leaves(optimizer))
        signature = tuple(
            (
                type(leaf).__module__,
                type(leaf).__qualname__,
                getattr(leaf, "_ple_world_profile", False),
                getattr(leaf, "_ple_file_mode_enabled", False),
                getattr(leaf, "_ple_scratch_root", None),
            )
            for leaf in leaves
        )
    except Exception as exc:
        error = exc
    raise_if_any_rank_failed(error, "reading checkpoint optimizer scratch mode", group=group)
    require_same_rank_value(signature, "checkpoint optimizer scratch mode", group=group)
    return any(row[3] is True for row in signature)


def _file_mode_for_checkpoint(optimizer, group=None, receipt=None):
    """Keep ordinary DCP unchanged; authenticated file profiles agree together."""
    file_mode, error = False, None
    try:
        file_mode = _file_backed_mode(optimizer)
    except Exception as exc:
        error = exc
    if not dist.is_initialized():
        if error is not None:
            raise error
        return file_mode
    if group is not None and group == dist.GroupMember.NON_GROUP_MEMBER:
        raise ValueError("A non-member cannot participate in this checkpoint process group")
    # Every participant must join this check, including ranks with no receipt or
    # no local optimizer. Otherwise one rank can raise while a peer waits in
    # the following collective. Ordinary DCP performs one small reduction.
    requested = error is not None or receipt is not None or file_mode
    if not any_rank_failed(requested, group=group):
        return False
    try:
        if error is not None:
            raise error
        if receipt is None:
            raise RuntimeError("Distributed file-backed DCP requires its all-rank WORLD profile receipt")
        if group is not None and group is not dist.group.WORLD:
            raise RuntimeError("File-backed DCP currently requires the WORLD process group")
        from ..optim.ple_world_profile import _validate_receipt

        _validate_receipt(receipt)
    except Exception as exc:
        error = exc
    raise_if_any_rank_failed(error, "validating file-backed checkpoint receipt", group=group)
    agreed = _agree_file_backed_mode(optimizer, group=group)
    if not agreed:
        raise RuntimeError("Authenticated file-backed checkpoint is missing its optimizer")
    error = None
    try:
        for leaf in _iter_optimizer_leaves(optimizer):
            if getattr(leaf, "_ple_file_mode_enabled", False) is True:
                if getattr(leaf, "_ple_world_receipt", None) is not receipt:
                    raise RuntimeError("File-backed optimizer does not belong to the checkpoint profile")
    except Exception as exc:
        error = exc
    raise_if_any_rank_failed(error, "validating file-backed checkpoint profile", group=group)
    return agreed


def _abort_file_backed_optimizers(optimizer):
    if optimizer is None:
        return
    for leaf in _iter_optimizer_leaves(optimizer):
        abort = getattr(leaf, "checkpoint_abort_file_backed_state", None)
        if abort is not None:
            abort()


def _persistent_optimizer_rows(model, optimizer):
    """Describe persistent extra-parallel optimizer steps before serialization.

    A scalar Adam step is replicated in DCP, even when its parameter is
    owner-sharded.  Do not silently save distinct owner-local steps as one.
    """
    specs = getattr(model, "_fqn2spec_info", None) or {}
    persistent = {name for name, spec in specs.items() if getattr(spec, "persistent_fsdp_shard_dim", None) is not None}
    names = {id(param): name for name, param in model.named_parameters() if name in persistent}
    rows = {}
    for leaf in _iter_optimizer_leaves(optimizer):
        for group in leaf.param_groups:
            for param in group["params"]:
                name = names.get(id(param))
                if name is None:
                    continue
                if name in rows:
                    raise RuntimeError(f"Duplicate optimizer parameter for persistent shard {name}")
                if not param.requires_grad:
                    raise RuntimeError(
                        f"Cannot checkpoint frozen persistent optimizer parameter {name}; rebuild the optimizer"
                    )
                layout = json.loads(group.get("qwen38_ple_step_layout", "null"))
                owner_grid = (
                    group.get("qwen38_ple_lookup") is True
                    and getattr(leaf, "requires_exact_recipe", False)
                    and isinstance(param, DTensor)
                    and layout == ["owner-grid-v1", param.device_mesh.mesh.tolist()]
                )
                if any(
                    getattr(leaf, key, None)
                    for key in ("_optimizer_state_dict_pre_hooks", "_optimizer_state_dict_post_hooks")
                ):
                    raise RuntimeError("Persistent-shard optimizer checkpoints do not support state-dict hooks")
                if owner_grid:
                    rows[name] = ("owner-grid", True, None)
                    continue
                state = leaf.state.get(param, {})
                local = param.to_local() if isinstance(param, DTensor) else param
                if not state and (local.numel() or not leaf.state):
                    raise RuntimeError(
                        f"Saving an uninitialized persistent optimizer state for {name} is unsupported; "
                        "save after all owners have state or use a lazy-state-aware optimizer"
                    )
                signature = None
                if state:
                    step = state.get("step")
                    if torch.is_tensor(step) and step.ndim == 0:
                        if isinstance(step, DTensor):
                            raise RuntimeError(f"Persistent-shard optimizer scalar step is distributed for {name}")
                        value, dtype = float(step.item()), str(step.dtype)
                    elif type(step) in (int, float):
                        value, dtype = float(step), type(step).__qualname__
                    else:
                        raise RuntimeError(f"Persistent-shard optimizer step is not a scalar for {name}")
                    if not math.isfinite(value) or value < 0 or not value.is_integer():
                        raise RuntimeError(f"Persistent-shard optimizer step is invalid for {name}")
                    fields = tuple(
                        (key, str(item.dtype) if torch.is_tensor(item) else type(item).__qualname__)
                        for key, item in sorted(state.items())
                    )
                    signature = (fields, dtype, value)
                rows[name] = ("scalar", bool(local.numel()), signature)
    return rows


def _checkpoint_group_has_complete_replica(mesh, placements, ranks):
    """Whether the group covers every distinct shard, across any replicas."""
    grid = mesh.mesh
    if len(placements) != grid.ndim:
        raise RuntimeError("Checkpoint mesh placements do not match mesh dimensionality")
    flat_ranks = grid.reshape(-1).tolist()
    if set(flat_ranks).issubset(ranks):
        return True
    shard_axes = [axis for axis, placement in enumerate(placements) if not isinstance(placement, Replicate)]
    strides = [math.prod(grid.shape[axis + 1 :]) for axis in range(grid.ndim)]
    covered = {
        tuple((index // strides[axis]) % grid.shape[axis] for axis in shard_axes)
        for index, rank in enumerate(flat_ranks)
        if rank in ranks
    }
    return len(covered) == math.prod(grid.shape[axis] for axis in shard_axes)


def _validate_persistent_optimizer_ownership(model, optimizer, group=None, *, coordinated=False, check_coverage=True):
    """Check frozen targets and ensure the DCP group covers each persistent mesh."""
    error = None
    try:
        specs = getattr(model, "_fqn2spec_info", None) or {}
        persistent = {
            name for name, spec in specs.items() if getattr(spec, "persistent_fsdp_shard_dim", None) is not None
        }
        names = {id(param): name for name, param in model.named_parameters() if name in specs}
        params = [
            param
            for leaf in _iter_optimizer_leaves(optimizer)
            for param_group in leaf.param_groups
            for param in param_group["params"]
            if id(param) in names
        ]
        if dist.is_initialized() and group is not None and not coordinated:
            raise RuntimeError("Optimizer checkpoints with an explicit group require coordinated DCP validation")
        if (
            dist.is_initialized()
            and dist.get_world_size() > 1
            and not coordinated
            and any(names[id(param)] in persistent for param in params)
        ):
            raise RuntimeError("Persistent optimizer checkpoints require coordinated DCP validation")
        ranks = None
        if dist.is_initialized():
            ranks = set(range(dist.get_world_size())) if group is None else set(dist.get_process_group_ranks(group))
        coverage = {}
        for param in params:
            name = names[id(param)]
            if name in persistent and not param.requires_grad:
                raise RuntimeError(
                    f"Cannot checkpoint frozen persistent optimizer parameter {name}; rebuild the optimizer"
                )
            if ranks is not None and check_coverage:
                spec = specs[name]
                mesh = (
                    getattr(spec, "para_fsdp_mesh", None)
                    if isinstance(getattr(spec, "placement", None), Shard)
                    else None
                )
                if mesh is not None:
                    # ParallelPlan uses (... replica, FSDP, ExtraParallel).
                    placements = [Replicate()] * max(0, mesh.ndim - 2) + [Shard(0)] * min(2, mesh.ndim)
                elif isinstance(param, DTensor):
                    mesh, placements = param.device_mesh, param.placements
                else:
                    continue
                key = (id(mesh), tuple(isinstance(placement, Replicate) for placement in placements))
                if key not in coverage:
                    coverage[key] = _checkpoint_group_has_complete_replica(mesh, placements, ranks)
                if not coverage[key]:
                    raise RuntimeError(f"DCP group does not cover extra-parallel parameter owners for {name}")
    except BaseException as exc:
        error = exc
    if coordinated:
        raise_if_any_rank_failed(error, "validating persistent optimizer ownership", group=group)
    elif error is not None:
        raise error


def _validate_persistent_optimizer_steps(model, optimizer, group=None, *, coordinated=False):
    """Guard scalar owner steps without introducing implicit process groups."""
    rows, error = {}, None
    try:
        rows = _persistent_optimizer_rows(model, optimizer)
        if not coordinated and rows and dist.is_initialized() and dist.get_world_size() > 1:
            raise RuntimeError(
                "Saving persistent-shard optimizer steps requires coordinated DCP validation; "
                "use DistributedCheckpointer or explicitly supply the DCP process group"
            )
    except BaseException as exc:
        error = exc
    if not dist.is_initialized() or dist.get_world_size(group) <= 1 or not coordinated:
        if error is not None:
            raise error
        return rows
    raise_if_any_rank_failed(error, "inspecting persistent optimizer checkpoint steps", group=group)
    gathered = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, rows, group=group)
    if any(set(entry) != set(rows) for entry in gathered):
        raise RuntimeError("Persistent optimizer checkpoint parameter inventory differs across ranks")
    for name in rows:
        entries = [entry[name] for entry in gathered]
        if any(entry[0] != entries[0][0] for entry in entries):
            raise RuntimeError(f"Persistent optimizer step layouts differ across ranks for {name}")
        if entries[0][0] == "owner-grid":
            continue
        contributors = [entry[2] for entry in entries if entry[1]]
        writers = [entry[2] for entry in entries if entry[2] is not None]
        if (
            (contributors and any(value != contributors[0] for value in contributors[1:]))
            or (writers and any(value != writers[0] for value in writers[1:]))
            or (writers and contributors and contributors[0] is None)
        ):
            raise RuntimeError(
                f"DCP cannot preserve different persistent owner optimizer states/steps for {name}; "
                "a shard-aware step representation is required"
            )
    return rows


def _verify_serialized_persistent_steps(model, rows, state_dict):
    """Check the actual serialized representation, not just its recipe marker."""
    if not rows:
        return
    params = dict(model.named_parameters())
    nested_state = state_dict.get("state", {})
    for name, (kind, _active, signature) in rows.items():
        step = state_dict.get(f"state.{name}.step")
        if step is None and name in nested_state:
            step = nested_state[name].get("step")
        if kind == "owner-grid":
            param = params[name]
            shape = tuple(param.device_mesh.mesh.shape)
            if not (
                isinstance(step, DTensor)
                and tuple(step.shape) == shape
                and tuple(step.placements) == tuple(Shard(axis) for axis in range(len(shape)))
                and torch.equal(step.device_mesh.mesh, param.device_mesh.mesh)
            ):
                raise RuntimeError(f"Persistent optimizer step for {name} is not saved with its owner grid")
        elif signature is None:
            if step is not None:
                raise RuntimeError(f"Saving created synthetic persistent optimizer state for {name}")
        else:
            _fields, dtype, value = signature
            if torch.is_tensor(step) and not isinstance(step, DTensor) and step.ndim == 0:
                actual = (str(step.dtype), float(step.item()))
            elif type(step) in (int, float):
                actual = (type(step).__qualname__, float(step))
            else:
                raise RuntimeError(f"Saved persistent optimizer step is not a scalar for {name}")
            if actual != (dtype, value):
                raise RuntimeError(f"Persistent optimizer step changed during checkpoint serialization for {name}")


def _validate_exact_optimizer_ownership(optimizer, group=None, *, coordinated=False, check_coverage=True):
    """Reject frozen restore targets and incomplete distributed owner groups."""
    error = None
    try:
        leaves = [leaf for leaf in _iter_optimizer_leaves(optimizer) if getattr(leaf, "requires_exact_recipe", False)]
        if dist.is_initialized() and dist.get_world_size() > 1 and not coordinated and leaves:
            raise RuntimeError("Distributed Muon checkpoints require coordinated DCP validation")
        ranks = None
        if dist.is_initialized():
            ranks = set(range(dist.get_world_size())) if group is None else set(dist.get_process_group_ranks(group))
        coverage = {}
        for leaf in leaves:
            for param_group in leaf.param_groups:
                for param in param_group["params"]:
                    if not param.requires_grad:
                        raise RuntimeError("Cannot checkpoint frozen Muon optimizer parameters; rebuild the optimizer")
                    if ranks is not None and check_coverage and isinstance(param, DTensor):
                        key = (id(param.device_mesh), tuple(isinstance(p, Replicate) for p in param.placements))
                        if key not in coverage:
                            coverage[key] = _checkpoint_group_has_complete_replica(
                                param.device_mesh, param.placements, ranks
                            )
                        if not coverage[key]:
                            raise RuntimeError("DCP group does not cover Muon optimizer parameter owners")
    except BaseException as exc:
        error = exc
    if coordinated:
        raise_if_any_rank_failed(error, "validating Muon optimizer ownership", group=group)
    elif error is not None:
        raise error


def _validate_trainable_optimizer_owners(optimizer, group=None, *, coordinated=False):
    if dist.is_initialized() and dist.get_world_size() > 1 and not coordinated:
        return
    error = None
    try:
        if any(
            not param.requires_grad
            for leaf in _iter_optimizer_leaves(optimizer)
            for param_group in leaf.param_groups
            for param in param_group["params"]
        ):
            raise RuntimeError("Cannot checkpoint frozen optimizer parameters; rebuild the optimizer")
    except BaseException as exc:
        error = exc
    if coordinated:
        raise_if_any_rank_failed(error, "validating trainable optimizer ownership", group=group)
    elif error is not None:
        raise error


def _validate_nonple_save_state(model, optimizer, group=None, *, require_step=False):
    """Fail closed if non-PLE shards cannot share a faithful DCP state."""
    if not dist.is_initialized():
        return
    local_rows, error = {}, None
    try:
        validate_groups = getattr(optimizer, "_validate_checkpoint_param_groups", None)
        if validate_groups is not None:
            validate_groups()
        names = {id(param): name for name, param in model.named_parameters()}
        for leaf in _iter_optimizer_leaves(optimizer):
            for param_group in leaf.param_groups:
                if not param_group["params"]:
                    raise RuntimeError("DCP optimizer checkpoint cannot represent an empty parameter group")
                if param_group.get("qwen38_ple_lookup"):
                    if any(not param.requires_grad for param in param_group["params"]):
                        raise RuntimeError("Cannot checkpoint frozen PLE optimizer parameters; rebuild the optimizer")
                    layout = json.loads(param_group.get("qwen38_ple_step_layout", "null"))
                    if not getattr(leaf, "requires_exact_recipe", False) or not (
                        isinstance(layout, list) and len(layout) == 2 and layout[0] == "owner-grid-v1"
                    ):
                        raise RuntimeError("Unrecognized PLE checkpoint owner-step layout")
                    if all(
                        isinstance(param, DTensor) and layout[1] == param.device_mesh.mesh.tolist()
                        for param in param_group["params"]
                    ):
                        continue
                for param in param_group["params"]:
                    name = names.get(id(param))
                    if name is None or name in local_rows:
                        raise RuntimeError("Cannot identify a unique non-PLE checkpoint optimizer parameter")
                    if not param.requires_grad:
                        raise RuntimeError(
                            f"Cannot checkpoint frozen optimizer parameter {name}; rebuild the optimizer"
                        )
                    local_param = param.to_local() if isinstance(param, DTensor) else param
                    value = leaf.state.get(param, {})
                    if value:
                        fields = tuple(
                            (str(name), str(item.dtype) if torch.is_tensor(item) else type(item).__qualname__)
                            for name, item in sorted(value.items())
                        )
                        step = value.get("step")
                        step = step.to_local() if isinstance(step, DTensor) else step
                        if step is None and not require_step:
                            signature = (fields, None, None)
                        else:
                            if torch.is_tensor(step) and step.ndim == 0:
                                step_value, step_type = float(step.item()), str(step.dtype)
                            elif not require_step and type(step) in (int, float):
                                step_value, step_type = float(step), type(step).__qualname__
                            else:
                                raise RuntimeError("Non-PLE optimizer checkpoint step must be a scalar")
                            if not math.isfinite(step_value) or step_value < 0 or not step_value.is_integer():
                                raise RuntimeError(
                                    "Non-PLE optimizer checkpoint step must be a finite nonnegative integer"
                                )
                            signature = (fields, step_type, step_value)
                    else:
                        signature = None
                    local_rows[name] = (bool(local_param.numel()), signature)
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "inspecting non-PLE optimizer checkpoint state", group=group)
    gathered = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, local_rows, group=group)
    names = set(local_rows)
    if any(set(rows) != names for rows in gathered):
        raise RuntimeError("Non-PLE optimizer checkpoint parameter inventory differs across ranks")
    for name in sorted(names):
        contributing = [rows[name][1] for rows in gathered if rows[name][0]]
        scalar_writers = [rows[name][1] for rows in gathered if rows[name][1] is not None]
        if (
            (contributing and any(value != contributing[0] for value in contributing[1:]))
            or (scalar_writers and any(value != scalar_writers[0] for value in scalar_writers[1:]))
            or (scalar_writers and contributing and contributing[0] is None)
        ):
            raise RuntimeError(
                f"Current checkpoint format cannot preserve differing non-PLE owner state/step for {name}; "
                "a shard-aware checkpoint representation is required"
            )


class _SourceCacheEvictor:
    """Advise away consumed DCP source ranges without changing checkpoint bytes."""

    def __init__(self, reader):
        if not hasattr(os, "posix_fadvise"):
            raise RuntimeError("File-backed DCP resume requires POSIX source-cache advice")
        self.reader = reader
        self._fds = OrderedDict()
        self._lock = threading.Lock()
        self._closed = False
        self.requests = 0
        self.hinted_bytes = 0

    def evict(self, item):
        info = self.reader.storage_data[item.storage_index]
        if info.length <= 0:
            return
        path = self.reader.fs.concat_path(self.reader.path, info.relative_path)
        with self._lock:
            if self._closed:
                raise RuntimeError("DCP source-cache adviser is already closed")
            fd = self._fds.get(path)
            if fd is None:
                if len(self._fds) >= 8:
                    _, old_fd = self._fds.popitem(last=False)
                    os.close(old_fd)
                fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
                self._fds[path] = fd
            else:
                self._fds.move_to_end(path)
            page = os.sysconf("SC_PAGE_SIZE")
            start = info.offset // page * page
            end = ((info.offset + info.length + page - 1) // page) * page
            os.posix_fadvise(fd, start, end - start, os.POSIX_FADV_DONTNEED)
            self.requests += 1
            self.hinted_bytes += info.length

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            first_error = None
            for fd in self._fds.values():
                try:
                    os.close(fd)
                except OSError as exc:
                    if first_error is None:
                        first_error = exc
            self._fds.clear()
            if first_error is not None:
                raise first_error


@contextmanager
def _source_cache_evictor(reader, enabled, group=None):
    if not enabled:
        yield None
        return
    evictor, error = None, None
    try:
        evictor = _SourceCacheEvictor(reader)
    except BaseException as exc:
        error = exc
    try:
        raise_if_any_rank_failed(error, "preparing DCP source-cache advice", group=group)
    except BaseException:
        if evictor is not None:
            evictor.close()
        raise
    error = None
    try:
        yield evictor
        logger.info_rank0(
            "DCP source cache advice: %d consumed ranges, %d bytes", evictor.requests, evictor.hinted_bytes
        )
    except BaseException as exc:
        error = exc
    try:
        evictor.close()
    except BaseException as exc:
        if error is None:
            error = exc
        else:
            error.add_note(f"DCP source-cache close also failed: {exc!r}")
    raise_if_any_rank_failed(error, "loading and closing DCP source-cache advice", group=group)


@contextmanager
def _claim_file_checkpoint(checkpoint_dir, module):
    """Exclusively claim a module destination for opt-in synchronous writers.

    A process death may leave the small claim file behind; failing closed then
    requires an operator to inspect the incomplete checkpoint before retrying.
    This protects cooperating writers; external writers must not share the path.
    """
    coordinator = not dist.is_initialized() or dist.get_rank() == 0
    module_id = hashlib.sha256((module or "").encode()).hexdigest()[:16]
    lock_path = os.path.join(checkpoint_dir, f".veomni-file-state-save-{module_id}.lock")
    fd, error = None, None
    if coordinator:
        try:
            os.makedirs(os.path.dirname(lock_path), exist_ok=True)
            fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except BaseException as exc:
            error = exc
    raise_if_any_rank_failed(error, "claiming file-backed optimizer checkpoint destination")
    try:
        yield
    finally:
        error = None
        if coordinator and fd is not None:
            try:
                os.close(fd)
                os.unlink(lock_path)
            except BaseException as exc:
                error = exc
        raise_if_any_rank_failed(error, "releasing file-backed optimizer checkpoint destination")


def _validate_file_checkpoint_layout(model, parallel_state, group):
    from torch.distributed.fsdp import FullyShardedDataParallel

    error = None
    try:
        state = parallel_state if parallel_state is not None else get_parallel_state()
        if state.dp_mode not in ("fsdp2", "ddp") or any(
            isinstance(module, FullyShardedDataParallel) for module in model.modules()
        ):
            raise ValueError("File-backed optimizer checkpointing requires FSDP2 or DDP")
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "validating file-backed optimizer checkpoint layout", group=group)


class _FileBackedSavePlanner(DefaultSavePlanner):
    """Reclaim previous file-backed pages after each writer resolves its next item."""

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


def _dcp_parameter_name_map(model) -> dict[str, torch.nn.Parameter]:
    """Map DCP-canonical parameter FQNs back to their runtime parameters.

    ``get_optimizer_state_dict`` strips wrapper prefixes such as DDP's
    ``module.`` when it canonicalizes state keys. The absent-state handoff uses
    those canonical keys, so resolving them through ``named_parameters()``
    directly breaks every wrapped model. Reuse DCP's own FQN conversion so both
    sides of the handoff name the same parameter.
    """
    names = {}
    for runtime_name, param in model.named_parameters():
        for fqn in _get_fqns(model, runtime_name):
            existing = names.get(fqn)
            if existing is not None and existing is not param:
                raise RuntimeError(f"DCP parameter FQN collision for {fqn}")
            names[fqn] = param
    return names


def _drop_absent_optimizer_state(model, optimizer) -> None:
    """Remove DCP-created placeholders for parameters with no saved state.

    ``_init_optim_state`` calls ``step()`` with zero gradients, which advances a
    placeholder ``step`` to one and copies a master weight before the model may
    be restored. Leaving that state behind makes the first real update use step
    two. Drop the whole state for a parameter that was entirely absent so the
    optimizer recreates it lazily from the restored parameter.
    """
    bases = getattr(optimizer, _ABSENT_OPTIMIZER_STATE_ATTR, ())
    if not bases:
        return
    names = _dcp_parameter_name_map(model)
    fqns = {
        base.removeprefix("optimizer.state.")
        for base in bases
        if isinstance(base, str) and base.startswith("optimizer.state.")
    }
    leaves = tuple(_iter_optimizer_leaves(optimizer))
    dropped = 0
    for fqn in sorted(fqns):
        param = names.get(fqn)
        if param is None:
            raise RuntimeError(f"Cannot resolve parameter for absent optimizer state: {fqn}")
        for leaf in leaves:
            if leaf.state.pop(param, None) is not None:
                dropped += 1
    if dropped:
        logger.info_rank0("DCP resume dropped %d wholly absent optimizer state entr(ies)", dropped)


def _validate_dcp_optimizer_state_keys(optimizer):
    """DCP's optimizer FQN mapper supports only parameter-owned state."""
    for leaf in _iter_optimizer_leaves(optimizer):
        parameter_ids = {id(param) for group in leaf.param_groups for param in group["params"]}
        has_state_hooks = getattr(leaf, "_optimizer_state_dict_pre_hooks", None) or getattr(
            leaf, "_optimizer_state_dict_post_hooks", None
        )
        if not has_state_hooks and any(id(key) not in parameter_ids for key in leaf.state):
            raise RuntimeError("DCP cannot serialize non-parameter optimizer state; remove or save it separately")


@contextmanager
def _filter_synthetic_optimizer_state_on_load(model, optimizer):
    """Hide DCP-created placeholder entries before user load hooks run.

    PyTorch reconstructs the state dict using the target optimizer's entries,
    including temporary entries for parameters absent from the checkpoint.
    Filtering the incoming dict, rather than deleting state after loading,
    preserves both the hook contract and changes made by user post-load hooks.
    """
    targets = optimizer.values() if isinstance(optimizer, dict) else (optimizer,)
    handles, expected, invoked = [], set(), set()
    names = None
    try:
        for target in targets:
            zero = getattr(target, "_veomni_resume_zero_optimizer_state", False)
            bases = getattr(target, _ABSENT_OPTIMIZER_STATE_ATTR, ())
            leaves = tuple(_iter_optimizer_leaves(target))
            for leaf in leaves:
                if isinstance(leaf, torch.optim.AdamW):
                    expected.add(id(leaf))

                    def validate_adamw(current, incoming):
                        if any(group.get("decoupled_weight_decay") is not True for group in incoming["param_groups"]):
                            raise RuntimeError("AdamW checkpoint requires decoupled weight decay; migrate explicitly")
                        invoked.add(id(current))

                    # Validate the saved value and the result of any user hook.
                    handles.append(leaf.register_load_state_dict_pre_hook(validate_adamw, prepend=True))
                    handles.append(leaf.register_load_state_dict_pre_hook(validate_adamw))
            if not zero and not bases:
                continue
            if names is None:
                names = _dcp_parameter_name_map(model)
            missing = set()
            for base in bases:
                if not isinstance(base, str) or not base.startswith("optimizer.state."):
                    raise RuntimeError(f"Invalid absent optimizer state name: {base!r}")
                fqn = base.removeprefix("optimizer.state.")
                if fqn not in names:
                    raise RuntimeError(f"Cannot resolve parameter for absent optimizer state: {fqn}")
                missing.add(id(names[fqn]))

            for leaf in leaves:
                expected.add(id(leaf))

                def filter_state(current, incoming, *, missing=missing, zero=zero):
                    groups = incoming["param_groups"]
                    if len(groups) != len(current.param_groups) or any(
                        len(saved["params"]) != len(group["params"])
                        for saved, group in zip(groups, current.param_groups, strict=True)
                    ):
                        raise RuntimeError("Cannot filter synthetic state with mismatched optimizer groups")
                    result = incoming.copy()
                    result["state"] = incoming["state"].copy()
                    for saved, group in zip(groups, current.param_groups, strict=True):
                        for key, param in zip(saved["params"], group["params"], strict=True):
                            if zero or id(param) in missing:
                                result["state"].pop(key, None)
                    invoked.add(id(current))
                    return result

                filter_state._veomni_internal_dcp_filter = True
                handles.append(leaf.register_load_state_dict_pre_hook(filter_state, prepend=True))
        yield
        if invoked != expected:
            raise RuntimeError("Optimizer load bypassed hooks needed to remove synthetic checkpoint state")
    finally:
        for handle in handles:
            handle.remove()


def _requires_exact_optimizer_state_dtypes(optimizer):
    """Opt in only precision-sensitive optimizers, including MultiOptimizer."""
    optimizer = getattr(optimizer, "optimizer", optimizer)
    if getattr(optimizer, "requires_exact_state_dtypes", False):
        return True
    children = optimizer if isinstance(optimizer, dict) else getattr(optimizer, "optimizers_dict", {})
    return any(_requires_exact_optimizer_state_dtypes(child) for child in children.values())


def _requires_exact_optimizer_recipe(optimizer):
    optimizer = getattr(optimizer, "optimizer", optimizer)
    if getattr(optimizer, "requires_exact_recipe", False):
        return True
    children = optimizer if isinstance(optimizer, dict) else getattr(optimizer, "optimizers_dict", {})
    return any(_requires_exact_optimizer_recipe(child) for child in children.values())


def _validate_unique_optimizer_parameters(optimizer):
    """Flattened DCP optimizer groups cannot encode repeated parameters."""
    seen = set()
    for leaf in _iter_optimizer_leaves(optimizer):
        for group in leaf.param_groups:
            for param in group["params"]:
                if id(param) in seen:
                    raise RuntimeError(
                        "DCP optimizer groups contain duplicate parameters; explicit migration required"
                    )
                seen.add(id(param))


def _validate_exact_optimizer_state_targets(optimizer):
    """Validate parameter inventory even when PyTorch skips partial-state initialization."""
    optimizer = getattr(optimizer, "optimizer", optimizer)
    children = optimizer if isinstance(optimizer, dict) else getattr(optimizer, "optimizers_dict", {})
    if children:
        for child in children.values():
            _validate_exact_optimizer_state_targets(child)
        return
    if not getattr(optimizer, "requires_exact_state_dtypes", False):
        return
    for group in optimizer.param_groups:
        for param in group["params"]:
            if not param.requires_grad:
                continue
            fields = optimizer.state.get(param, {})
            inventory = getattr(optimizer, "checkpoint_state_fields", None)
            required = (
                inventory(param) if inventory is not None else ("master_weight", "exp_avg", "exp_avg_sq", "step")
            )
            missing = set(required).difference(fields)
            if missing:
                raise RuntimeError(
                    "Incomplete precision-sensitive optimizer target state: "
                    f"missing {sorted(missing)} for parameter shape {tuple(param.shape)}; "
                    "use a fresh optimizer for resume or explicitly migrate its state"
                )


def _optimizer_chunks_are_complete(tensor_metadata: TensorStorageMetadata) -> bool:
    """Reject missing, overlapping or out-of-bounds optimizer state shards."""
    shape, chunks = tensor_metadata.size, tensor_metadata.chunks
    if not shape:
        return len(chunks) == 1 and not chunks[0].offsets and not chunks[0].sizes
    ndim = len(shape)
    volume = 0
    for chunk in chunks:
        if len(chunk.offsets) != ndim or len(chunk.sizes) != ndim:
            return False
        if any(
            start < 0 or size < 0 or start + size > bound
            for start, size, bound in zip(chunk.offsets, chunk.sizes, shape, strict=True)
        ):
            return False
        volume += math.prod(chunk.sizes)
    if volume != math.prod(shape):
        return False

    # Sweep the axis with the lowest average interval coverage. The widest
    # tensor dimension is often not the sharded dimension and can leave all
    # chunks active, producing quadratic planner cost for FSDP matrices.
    axis = min(range(ndim), key=lambda dim: sum(chunk.sizes[dim] for chunk in chunks) / (shape[dim] or 1))
    ordered = sorted((chunk for chunk in chunks if math.prod(chunk.sizes)), key=lambda chunk: chunk.offsets[axis])
    active = []
    for index, chunk in enumerate(ordered):
        start, end = chunk.offsets[axis], chunk.offsets[axis] + chunk.sizes[axis]
        del active[: bisect_right(active, (start, len(ordered)))]
        for _, other_index in active:
            other = ordered[other_index]
            if all(
                max(a, b) < min(a + size_a, b + size_b)
                for a, size_a, b, size_b in zip(chunk.offsets, chunk.sizes, other.offsets, other.sizes, strict=True)
            ):
                return False
        insort(active, (end, index))
    return True


class _ModelStrictLoadPlanner(DefaultLoadPlanner):
    """Allow partial optimizer state while requiring a complete full-model DCP."""

    def __init__(
        self,
        strict_model: bool,
        strict_optimizer_dtype: bool = False,
        strict_optimizer_recipe: bool = False,
        strict_gdn_fp32: bool = False,
        optimizer=None,
        model=None,
        source_cache_evictor=None,
    ):
        super().__init__(allow_partial_load=True)
        self.strict_model = strict_model
        self.strict_optimizer_dtype = strict_optimizer_dtype
        self.strict_optimizer_recipe = strict_optimizer_recipe
        self.strict_gdn_fp32 = strict_gdn_fp32
        self.optimizer = optimizer
        self.model = model
        self.source_cache_evictor = source_cache_evictor

    def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
        super().set_up_planner(state_dict, metadata, is_coordinator)
        if metadata is None:
            return
        # Older runs may have saved the PyTorch-added differentiable flag even
        # when the fresh custom optimizer has no such default. Preserve an
        # existing source value instead of silently dropping the group field.
        for key in metadata.state_dict_metadata:
            if not key.startswith("optimizer.param_groups.") or not key.endswith(".differentiable"):
                continue
            reference = key.removesuffix("differentiable") + "lr"
            if key in self.mappings or reference not in self.mappings:
                continue
            path = self.mappings[reference]
            self.mappings[key] = (*path[:-1], path[-1].removesuffix("lr") + "differentiable")
            self.state_dict[key] = False

    def load_bytes(self, read_item, value):
        super().load_bytes(read_item, value)
        if self.source_cache_evictor is not None:
            self.source_cache_evictor.evict(read_item)

    def commit_tensor(self, read_item, tensor):
        super().commit_tensor(read_item, tensor)
        if self.optimizer is not None:
            for optimizer in _iter_optimizer_leaves(self.optimizer):
                callback = getattr(optimizer, "checkpoint_commit_tensor", None)
                if callback is not None:
                    callback(tensor)
        if self.source_cache_evictor is not None:
            self.source_cache_evictor.evict(read_item)

    def create_local_plan(self):
        # The marker is an ephemeral handoff from planning to load_state_dict.
        # Never let a failed plan leave it behind for a later checkpoint.
        _reset_zero_optimizer_state_resume(self.optimizer)
        if self.optimizer is not None:
            _reset_absent_optimizer_state(self.optimizer)
        metadata = self.metadata.state_dict_metadata if self.metadata is not None else {}
        optimizer_tensors = {
            key: value
            for key, value in metadata.items()
            if key.startswith("optimizer.state.") and key in self.mappings and isinstance(value, TensorStorageMetadata)
        }
        if any(not _optimizer_chunks_are_complete(value) for value in optimizer_tensors.values()):
            raise RuntimeError("DCP optimizer tensor chunks do not completely and uniquely cover their global shape")
        if self.strict_gdn_fp32:
            for key, path in self.mappings.items():
                if not path or path[0] != "model" or not is_qwen4_gdn_parameter(key):
                    continue
                source = metadata.get(key)
                if source is not None and getattr(getattr(source, "properties", None), "dtype", None) != torch.float32:
                    raise RuntimeError(
                        f"DCP Qwen4 GDN parameter {key} is not FP32; explicit checkpoint migration is required"
                    )
                target = self.state_dict.get(key)
                if source is not None and (not torch.is_tensor(target) or target.dtype != torch.float32):
                    raise RuntimeError(f"DCP Qwen4 GDN target {key} must be FP32 before restore")
        for key, path in self.mappings.items():
            if path and path[-1] == _EXTRA_PARALLEL_LAYOUT_MARKER and key not in metadata:
                raise RuntimeError(
                    "Unversioned same-dimension ExtraParallel checkpoint; explicit layout migration is required"
                )
        has_optimizer_state = any(key.startswith("optimizer.state.") for key in metadata)
        has_zero_state_marker = f"optimizer.{_ZERO_OPTIMIZER_STATE_MARKER}" in metadata
        if has_zero_state_marker and has_optimizer_state:
            raise RuntimeError(
                "DCP zero-state optimizer marker conflicts with saved tensor state; "
                "rank-local lazy state cannot be reconstructed without explicit checkpoint migration"
            )
        resume_zero_optimizer_state = has_zero_state_marker and not has_optimizer_state

        if self.strict_optimizer_dtype:
            # DCP otherwise casts saved tensors to the target placeholders before
            # optimizer.load_state_dict sees them. Validate original metadata
            # before any read can erase evidence of a precision-policy change.
            assert self.metadata is not None
            if any(path and path[0] == "optimizer" for path in self.mappings.values()):
                _validate_exact_optimizer_state_targets(self.optimizer)
        # DCP silently ignores saved keys without target placeholders. Check
        # the optimizer-owned names from its parameter groups, so partially
        # initialized targets or a changed optimizer recipe cannot drop a
        # saved moment while still allowing unrelated frozen parameters.
        owned_names = set()
        group_prefix = "optimizer.param_groups."
        for key in self.mappings:
            if not key.startswith(group_prefix):
                continue
            if key.endswith(".params"):
                params = self.state_dict.get(key)
                if isinstance(params, (tuple, list)):
                    owned_names.update(name for name in params if isinstance(name, str))
            elif key.endswith(".lr") and getattr(self.optimizer, "_is_multi_optimizer", False):
                name = key[len(group_prefix) : -len(".lr")]
                if name:
                    owned_names.add(name)
        saved_keys = sorted(metadata)
        # A tied parameter may acquire a different canonical FQN if model
        # registration order changes. Its old moment must not be mistaken for
        # an unrelated frozen parameter while the new name appears lazy.
        alias_model = self.model if self.model is not None else getattr(self.optimizer, "model", None)
        if alias_model is not None and owned_names:
            aliases_by_id = {}
            for runtime_name, parameter in alias_model.named_parameters(remove_duplicate=False):
                aliases_by_id.setdefault(id(parameter), set()).update(_get_fqns(alias_model, runtime_name))
            for aliases in aliases_by_id.values():
                if not aliases.intersection(owned_names):
                    continue
                for alias in aliases.difference(owned_names):
                    base = f"optimizer.state.{alias}"
                    index = bisect_left(saved_keys, base + ".")
                    if base in metadata or (index < len(saved_keys) and saved_keys[index].startswith(base + ".")):
                        raise RuntimeError("DCP optimizer tied parameter name changed; explicit migration is required")
        name_map = None
        for name in owned_names:
            base = f"optimizer.state.{name}"
            if base in metadata and base not in self.mappings:
                raise RuntimeError(f"DCP optimizer state has an incompatible target for {base}")
            prefix = base + "."
            index = bisect_left(saved_keys, prefix)
            while index < len(saved_keys) and saved_keys[index].startswith(prefix):
                key = saved_keys[index]
                if key not in self.mappings:
                    # Older compact Muon expert checkpoints can contain an
                    # unused FP32 second moment. Validate and ignore only that
                    # explicitly supported legacy field on a real compact expert.
                    if key == prefix + "exp_avg_sq":
                        model = self.model if self.model is not None else getattr(self.optimizer, "model", None)
                        if model is not None:
                            if name_map is None:
                                name_map = _dcp_parameter_name_map(model)
                            param = name_map.get(name)
                            master = self.state_dict.get(prefix + "master_weight")
                            source = metadata[key]
                            compact = param is not None and any(
                                id(param) in getattr(leaf, "_compact_expert_ids", ())
                                for leaf in _iter_optimizer_leaves(self.optimizer)
                            )
                            if (
                                compact
                                and torch.is_tensor(master)
                                and isinstance(source, TensorStorageMetadata)
                                and source.size == master.shape
                                and source.properties.dtype == torch.float32
                                and _optimizer_chunks_are_complete(source)
                            ):
                                index += 1
                                continue
                    raise RuntimeError(
                        f"DCP saved optimizer state has no target placeholder: {key}; "
                        "use a fresh optimizer with the same recipe or explicitly migrate its state"
                    )
                index += 1
        # A parameter's tensor state may be lazy, but a missing group option
        # cannot safely be replaced by the new run's value for any optimizer.
        for key, path in self.mappings.items():
            if (
                path
                and path[0] == "optimizer"
                and len(path) > 1
                and (path[1] == "param_groups" or str(path[1]).startswith("param_groups."))
                and key not in metadata
            ):
                raise RuntimeError(
                    f"DCP optimizer recipe/state is incomplete: {key}; "
                    "missing optimizer group fields require explicit checkpoint migration"
                )
        group_prefixes = {
            key.removesuffix(".lr")
            for key in self.mappings
            if key.startswith("optimizer.param_groups.") and key.endswith(".lr")
        }
        group_prefixes.update(
            f"optimizer.param_groups.{path[2]}"
            for path in self.mappings.values()
            if len(path) > 2 and path[:2] == ("optimizer", "param_groups") and isinstance(path[2], int)
        )
        model = self.model if self.model is not None else getattr(self.optimizer, "model", None)
        if model is not None:
            if name_map is None:
                name_map = _dcp_parameter_name_map(model)
            group_prefixes.update(f"optimizer.param_groups.{name}" for name in name_map)
        for key in metadata:
            if not key.startswith("optimizer.param_groups.") or key in self.mappings:
                continue
            if any(key.startswith(prefix + ".") for prefix in group_prefixes):
                raise RuntimeError(
                    f"DCP optimizer parameter group has an unsupported saved field: {key}; "
                    "explicit checkpoint migration is required"
                )
        target_group_indices = {
            path[2]
            for path in self.mappings.values()
            if len(path) > 2 and path[:2] == ("optimizer", "param_groups") and isinstance(path[2], int)
        }
        if target_group_indices:
            source_group_indices = {
                path[2]
                for path in (getattr(self.metadata, "planner_data", None) or {}).values()
                if len(path) > 2 and path[:2] == ("optimizer", "param_groups") and isinstance(path[2], int)
            }
            if source_group_indices != target_group_indices:
                raise RuntimeError("DCP optimizer parameter group count changed; explicit migration is required")
        missing_state_fields = []
        for key, path in self.mappings.items():
            if not path or path[0] != "optimizer" or not key.startswith("optimizer.state."):
                continue
            # Optimizer state may have arbitrary, nested fields (e.g. Kahan
            # compensation); match the longest owned FQN rather than using a
            # fixed list of Adam field names.
            parts = key.removeprefix("optimizer.state.").split(".")
            name = next(
                (".".join(parts[:i]) for i in range(len(parts) - 1, 0, -1) if ".".join(parts[:i]) in owned_names), None
            )
            base = f"optimizer.state.{name}" if name is not None else key.rpartition(".")[0]
            source = metadata.get(key)
            target = self.state_dict.get(key)
            if source is None:
                missing_state_fields.append(base)
                continue
            if not self.strict_optimizer_dtype or not torch.is_tensor(target):
                continue
            source_shape = getattr(source, "size", None)
            if source_shape != target.shape:
                raise RuntimeError(
                    f"DCP optimizer shape mismatch for {key}: saved={source_shape}, configured={target.shape}; "
                    "explicit shape migration is required"
                )
            source_dtype = getattr(getattr(source, "properties", None), "dtype", None)
            if source_dtype != target.dtype:
                raise RuntimeError(
                    f"DCP optimizer dtype mismatch for {key}: saved={source_dtype}, configured={target.dtype}; "
                    "explicit precision migration is required"
                )
        if missing_state_fields:
            lazy_bases = set()
            for base in missing_state_fields:
                prefix = base + "."
                saved_index = bisect_left(saved_keys, prefix)
                if base in metadata or (saved_index < len(saved_keys) and saved_keys[saved_index].startswith(prefix)):
                    kind = "precision-sensitive optimizer state" if self.strict_optimizer_dtype else "optimizer state"
                    raise RuntimeError(
                        f"DCP is missing part of {kind} for {base}; explicit checkpoint migration is required"
                    )
                lazy_bases.add(base)
            if self.strict_optimizer_dtype and not has_optimizer_state and not has_zero_state_marker:
                raise RuntimeError(
                    "DCP is missing every precision-sensitive optimizer state entry; "
                    "explicit checkpoint migration is required"
                )
            if lazy_bases and not resume_zero_optimizer_state:
                if self.optimizer is not None:
                    _record_absent_optimizer_state(self.optimizer, lazy_bases)
                logger.info_rank0(
                    "DCP resume preserves lazy optimizer state for %d parameter(s) absent from the checkpoint",
                    len(lazy_bases),
                )
        plan = super().create_local_plan()
        if self.strict_model:
            assert self.metadata is not None
            missing_model_keys = sorted(
                key
                for key, path in self.mappings.items()
                if path
                and path[0] == "model"
                and path[-1] != _QWEN4_GDN_FP32_MARKER
                and key not in self.metadata.state_dict_metadata
            )
            if missing_model_keys:
                preview = ", ".join(missing_model_keys[:10])
                suffix = " ..." if len(missing_model_keys) > 10 else ""
                raise RuntimeError(
                    f"DCP is missing {len(missing_model_keys)} model key(s) required for a full-model resume: "
                    f"{preview}{suffix}"
                )

        if resume_zero_optimizer_state:
            _mark_zero_optimizer_state_resume(self.optimizer)
            logger.info_rank0("DCP resume restored an explicitly zero-step optimizer checkpoint")

        return plan


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

        # Persistent ExtraParallel parameters already use their complete
        # runtime placement (for example PLE ``[Shard(1), Shard(0)]`` on
        # ``(ple_fsdp, ple)``). They are not managed by FSDP2, so there is no
        # temporary ExtraParallel dimension to drop or restore around DCP.
        if getattr(spec_info, "persistent_fsdp_shard_dim", None) is not None:
            continue

        tensor = state_dict[name]
        if not torch.is_tensor(tensor):
            continue
        if tensor.ndim == 0:
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
        if action == "drop":
            tensor = drop_extra_parallel_dim(
                tensor,
                fsdp_submesh,
                fsdp_shard_dim,
                extra_parallel_name=spec_info.para_name,
            )
        else:
            tensor = restore_extra_parallel_dim(
                tensor,
                spec_info.para_fsdp_mesh,
                fsdp_submesh,
                ep_shard_dim,
                fsdp_shard_dim,
                global_shape=spec_info.global_shape,
            )
        state_dict[name] = tensor

    return state_dict


def _coordinated_prime_checkpoint_meshes(specs, enabled, group):
    """Agree on the group-creation plan before creating any checkpoint mesh."""
    plan, meshes, error = [], [], None
    try:
        seen = set()
        if enabled:
            for name in sorted(specs):
                info = specs[name]
                if (
                    isinstance(info.placement, Shard)
                    and getattr(info, "persistent_fsdp_shard_dim", None) is None
                    and info.para_fsdp_mesh is not None
                    and info.placement.dim == info.fsdp_shard_dim
                ):
                    mesh = info.para_fsdp_mesh
                    if mesh in seen:
                        continue
                    seen.add(mesh)
                    if mesh.ndim not in (2, 3) or mesh.mesh_dim_names is None:
                        raise ValueError("File-backed checkpoint mesh must have 2 or 3 named dimensions")
                    with _EXTRA_PARALLEL_FIRST_MESH_LOCK:
                        cached = mesh in _EXTRA_PARALLEL_FIRST_MESHES
                    plan.append((name, mesh.device_type, tuple(mesh.mesh_dim_names), mesh.mesh.tolist(), cached))
                    meshes.append(mesh)
        if dist.is_initialized() and group is not None and dist.get_world_size(group) != dist.get_world_size():
            if any(not item[-1] for item in plan):
                raise ValueError("A subgroup checkpoint requires the global ExtraParallel meshes to be precreated")
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "preparing checkpoint mesh plan", group=group)
    if dist.is_initialized():
        plans = [None] * dist.get_world_size(group=group)
        dist.all_gather_object(plans, plan, group=group)
        if any(other != plan for other in plans):
            raise RuntimeError("File-backed checkpoint mesh plans differ across ranks")
    for mesh in meshes:
        _extra_parallel_first_mesh(mesh)


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

    def __init__(
        self, model, trainable_only: bool = False, parallel_state=None, *, coordinated=False, coordinated_group=None
    ):
        self.model = model
        self.trainable_only = trainable_only

        # Determine whether this is ExtraParallel+FSDP2 case
        # If so, we need to restore Para(e.g. EP)-dim before saving to DCP
        self.parallel_state = parallel_state if parallel_state is not None else get_parallel_state()
        self.extra_parallel_fqn2spec_info = getattr(self.model, "_fqn2spec_info", None)
        self.should_extra_parallel_aware = (
            self.extra_parallel_fqn2spec_info is not None and self.parallel_state.dp_mode == "fsdp2"
        )
        self._coordinated = coordinated
        self._coordinated_group = coordinated_group

    @torch.no_grad()
    def state_dict(self):
        if not self._coordinated:
            return self._state_dict_impl()
        _coordinated_prime_checkpoint_meshes(
            self.extra_parallel_fqn2spec_info, self.should_extra_parallel_aware, self._coordinated_group
        )
        result, error = None, None
        try:
            result = self._state_dict_impl()
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "constructing model checkpoint state", group=self._coordinated_group)
        return result

    def _state_dict_impl(self):
        options = StateDictOptions(ignore_frozen_params=True) if self.trainable_only else None
        model_state_dict = get_model_state_dict(model=self.model, options=options)
        if self.should_extra_parallel_aware:
            logger.info_rank0(
                "Getting model state_dict from ModelState wrapper, would restore ExtraParallel dim for ExtraParallel (e.g. Experts/Embeds) module"
            )
            # As fsdp+extra parallel and pure extra parallel have different placements, e.g. [Shard(0), Shard(1)] and [Shard(0)],
            # restoring state dict should be extra parallel aware.
            model_state_dict = self.get_state_dict_with_extra_parallel_dim_preprocess(model_state_dict, "restore")

        if self.should_extra_parallel_aware and _needs_extra_parallel_layout_marker(self.extra_parallel_fqn2spec_info):
            model_state_dict[_EXTRA_PARALLEL_LAYOUT_MARKER] = True
        if is_qwen4_exp_model(self.model) and not self.trainable_only:
            validate_qwen4_gdn_pairs(model_state_dict, qwen4_exp_model_config(self.model))
            gdn_names = sorted(name for name in model_state_dict if is_qwen4_gdn_parameter(name))
            if any(model_state_dict[name].dtype != torch.float32 for name in gdn_names):
                raise RuntimeError("Qwen4 full checkpoint requires FP32 GDN parameters")
            if gdn_names:
                model_state_dict[_QWEN4_GDN_FP32_MARKER] = json.dumps({"version": 1, "parameter_names": gdn_names})
        return model_state_dict

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        if not self._coordinated:
            return self._load_state_dict_impl(state_dict)
        error = None
        try:
            self._load_state_dict_impl(state_dict)
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "restoring model checkpoint state", group=self._coordinated_group)

    def _load_state_dict_impl(self, state_dict):
        """
        perform the reverse operation for state_dict()
        need to drop ExtraParallel-dim when loading from DCP checkpoints
        so that ExtraParallel-FSDP would not be confused
        """
        state_dict = dict(state_dict)
        state_dict.pop(_QWEN4_GDN_FP32_MARKER, None)
        state_dict.pop(_QWEN4_GDN_FP32_LEGACY_MARKER, None)
        model_state_dict = _strip_extra_parallel_layout_marker(
            state_dict,
            self.should_extra_parallel_aware
            and _needs_extra_parallel_layout_marker(self.extra_parallel_fqn2spec_info),
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


def _has_optimizer_checkpoint_state(state_dict):
    """Ignore empty per-parameter mappings, which DCP does not serialize."""
    if any(key.startswith("state.") for key in state_dict):
        return True
    nested = state_dict.get("state", {})
    return any(not isinstance(value, dict) or bool(value) for value in nested.values())


def _validate_single_optimizer_groups(model, optimizer, saved_state):
    groups = saved_state.get("param_groups", ())
    if len(groups) != len(optimizer.param_groups):
        raise RuntimeError("DCP optimizer parameter group count changed; explicit migration is required")
    names_to_ids = {}
    for runtime_name, parameter in model.named_parameters(remove_duplicate=False):
        for name in _get_fqns(model, runtime_name):
            names_to_ids.setdefault(name, set()).add(id(parameter))
    for saved, current in zip(groups, optimizer.param_groups, strict=True):
        saved_ids = set()
        for name in saved.get("params", ()):
            identities = names_to_ids.get(name, set())
            if len(identities) != 1:
                raise RuntimeError("DCP optimizer parameter group has an unknown or ambiguous parameter")
            saved_ids.update(identities)
        if saved_ids != {id(parameter) for parameter in current["params"]}:
            raise RuntimeError("DCP optimizer parameter group membership changed; explicit migration is required")


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
    Placeholders for entirely absent parameter state are removed from the
    incoming state before load hooks, preserving lazy initialization and the
    user hook contract for the parameter's first actual update.

    Persistent scalar-owner saves reject uninitialized owners or differing
    owner states. Coordinated saves and loads reject frozen optimizer-owned
    parameters; persistent and exact-recipe restore targets must be trainable.

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
            sparse, preserving parameter groups while avoiding synthetic
            optimizer state for unused parameters (e.g. unused MoE experts).
        coordinated_group_validation: every rank in the DCP group must call the
            wrapper collectively. Required for distributed optimizer saves and
            loads, unless ``coordinated`` provides equivalent group handling.
        coordinated_group: the same group passed to ``dcp.save`` or ``dcp.load``;
            ``None`` denotes the default world group.

    Precision-sensitive optimizers opt out of partial state restoration of an
    existing but incomplete state. A parameter with no saved state at all keeps
    the fresh optimizer's default lazy state; unused parameters can be absent.
    A zero-step sparse MultiOptimizer checkpoint records an explicit marker, so
    an entirely empty optimizer state restores lazy instead of being rejected.
    """

    def __init__(
        self,
        model,
        optimizer,
        parallel_state=None,
        *,
        load: bool = False,
        coordinated: bool = False,
        coordinated_group=None,
        coordinated_group_validation: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.parallel_state = parallel_state if parallel_state is not None else get_parallel_state()
        self.extra_parallel_fqn2spec_info = getattr(self.model, "_fqn2spec_info", None)
        self.should_extra_parallel_aware = (
            self.extra_parallel_fqn2spec_info is not None and self.parallel_state.dp_mode == "fsdp2"
        )
        self._load = load
        self._coordinated = coordinated
        self._coordinated_group = coordinated_group
        self._coordinated_group_validation = coordinated_group_validation
        self._coordinated_file_mode = False
        self._coordinate_state_dict = False

    def state_dict(self):
        coordinated = self._coordinated or self._coordinated_group_validation
        if dist.is_initialized() and dist.get_world_size() > 1 and not coordinated:
            raise RuntimeError("Distributed optimizer checkpoints require coordinated DCP validation")
        _validate_persistent_optimizer_ownership(
            self.model, self.optimizer, self._coordinated_group, coordinated=coordinated, check_coverage=not self._load
        )
        _validate_exact_optimizer_ownership(
            self.optimizer, self._coordinated_group, coordinated=coordinated, check_coverage=not self._load
        )
        _validate_trainable_optimizer_owners(self.optimizer, self._coordinated_group, coordinated=coordinated)
        if not self._coordinated:
            if not self._coordinated_group_validation:
                rows = (
                    _validate_persistent_optimizer_steps(self.model, self.optimizer, self._coordinated_group)
                    if not self._load
                    else {}
                )
                result = self._state_dict_impl()
                if not self._load:
                    _verify_serialized_persistent_steps(self.model, rows, result)
                return result
            from torch.distributed.fsdp import FullyShardedDataParallel

            if any(isinstance(module, FullyShardedDataParallel) for module in self.model.modules()):
                # FSDP1's state-dict construction contains its own collectives;
                # do not interleave them with the FSDP2/DDP error agreement.
                return self._state_dict_impl()
            # DCP calls Stateful.state_dict before its own planning collectives.
            # Prepare custom optimizer buffers locally and agree on failures
            # before EP mesh construction or DCP planning on any rank.
            with ExitStack() as prepared:
                error = None
                try:
                    if self._load:
                        for optimizer in _iter_optimizer_leaves(self.optimizer):
                            prepare = getattr(optimizer, "checkpoint_state", None)
                            if prepare is not None:
                                prepared.enter_context(prepare(load=True))
                except BaseException as exc:
                    error = exc
                raise_if_any_rank_failed(error, "preparing optimizer checkpoint state", group=self._coordinated_group)
                self._agree_parameter_uniqueness()
                self._agree_external_optimizer_hooks()
                persistent_rows = {}
                if not self._load:
                    _validate_nonple_save_state(self.model, self.optimizer, group=self._coordinated_group)
                    persistent_rows = _validate_persistent_optimizer_steps(
                        self.model, self.optimizer, self._coordinated_group, coordinated=True
                    )
                self._prime_checkpoint_meshes()
                result, error = None, None
                try:
                    self._coordinate_state_dict = True
                    result = self._state_dict_impl()
                    if not self._load:
                        _verify_serialized_persistent_steps(self.model, persistent_rows, result)
                except BaseException as exc:
                    error = exc
                finally:
                    self._coordinate_state_dict = False
                raise_if_any_rank_failed(
                    error, "constructing optimizer checkpoint state", group=self._coordinated_group
                )
                self._agree_save_state_coverage(result)
                return result
        self._coordinated_file_mode = _agree_file_backed_mode(self.optimizer, group=self._coordinated_group)
        if not self._coordinated_file_mode:
            self._coordinated = False
            self._coordinated_group_validation = True
            try:
                return self.state_dict()
            finally:
                self._coordinated = True
        from torch.distributed.fsdp import FullyShardedDataParallel

        unsupported = self.parallel_state.dp_mode not in ("fsdp2", "ddp") or any(
            isinstance(module, FullyShardedDataParallel) for module in self.model.modules()
        )
        raise_if_any_rank_failed(
            ValueError("File-backed optimizer checkpointing requires FSDP2 or DDP") if unsupported else None,
            "validating file-backed optimizer checkpoint layout",
            group=self._coordinated_group,
        )
        if not self._load:
            _validate_nonple_save_state(self.model, self.optimizer, group=self._coordinated_group, require_step=True)
        # Allocate local file-backed targets before entering state-dict code
        # that may contain FSDP/ExtraParallel collectives. A local allocation
        # failure is coordinated while every peer is still at this preflight.
        with ExitStack() as prepared:
            error = None
            try:
                for optimizer in _file_backed_optimizers(self.optimizer):
                    prepared.enter_context(optimizer.checkpoint_state(load=self._load))
            except BaseException as exc:
                error = exc
            raise_if_any_rank_failed(error, "preparing optimizer checkpoint state", group=self._coordinated_group)
            self._agree_parameter_uniqueness()
            self._agree_external_optimizer_hooks()
            persistent_rows = {}
            if not self._load:
                persistent_rows = _validate_persistent_optimizer_steps(
                    self.model, self.optimizer, self._coordinated_group, coordinated=True
                )
            self._prime_checkpoint_meshes()
            result, error = None, None
            try:
                self._coordinate_state_dict = True
                result = self._state_dict_impl()
                if not self._load:
                    _verify_serialized_persistent_steps(self.model, persistent_rows, result)
            except BaseException as exc:
                error = exc
            finally:
                self._coordinate_state_dict = False
            try:
                raise_if_any_rank_failed(
                    error, "constructing optimizer checkpoint state", group=self._coordinated_group
                )
            except BaseException:
                _abort_file_backed_optimizers(self.optimizer)
                raise
            self._agree_save_state_coverage(result)
            return result

    def _prime_checkpoint_meshes(self):
        _coordinated_prime_checkpoint_meshes(
            self.extra_parallel_fqn2spec_info, self.should_extra_parallel_aware, self._coordinated_group
        )

    def _agree_parameter_uniqueness(self):
        error = None
        try:
            _validate_unique_optimizer_parameters(self.optimizer)
            _validate_dcp_optimizer_state_keys(self.optimizer)
            for leaf in _iter_optimizer_leaves(self.optimizer):
                if any(not group["params"] for group in leaf.param_groups):
                    raise RuntimeError("DCP optimizer checkpoint cannot represent an empty parameter group")
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "validating optimizer checkpoint parameters", group=self._coordinated_group)

    def _agree_save_state_coverage(self, state_dict):
        if self._load:
            return
        # The per-parameter preflight rejects unrepresentable lazy states.
        # A rank with only zero-sized shards may legitimately have no local
        # state; do not publish its local marker when another rank has tensors.
        local_has_state = _has_optimizer_checkpoint_state(state_dict)
        any_has_state = any_rank_failed(local_has_state, group=self._coordinated_group)
        if any_has_state:
            state_dict.pop(_ZERO_OPTIMIZER_STATE_MARKER, None)
        else:
            state_dict[_ZERO_OPTIMIZER_STATE_MARKER] = True

    def _agree_external_optimizer_hooks(self):
        if not dist.is_initialized() or dist.get_world_size(self._coordinated_group) <= 1:
            return
        error = None
        try:
            names = ["_optimizer_state_dict_pre_hooks", "_optimizer_state_dict_post_hooks"]
            if self._load:
                names.extend(["_optimizer_load_state_dict_pre_hooks", "_optimizer_load_state_dict_post_hooks"])
            if any(getattr(leaf, name, None) for leaf in _iter_optimizer_leaves(self.optimizer) for name in names):
                raise RuntimeError(
                    "Distributed DCP does not support external optimizer state-dict hooks; "
                    "remove the hooks or materialize their state before checkpointing"
                )
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(
            error, "validating distributed optimizer checkpoint hooks", group=self._coordinated_group
        )

    def _state_dict_impl(self):
        if not self._coordinate_state_dict:
            _validate_unique_optimizer_parameters(self.optimizer)
            _validate_dcp_optimizer_state_keys(self.optimizer)
        if self.should_extra_parallel_aware:
            logger.info_rank0(
                "Getting optimizer state_dict from OptimizerState wrapper, would restore ExtraParallel dim for Experts module"
            )
            assert self.optimizer._is_multi_optimizer, (
                "ExtraParallel is enabled but optimizer is not a MultiOptimizer instance"
            )
            vanilla_optim_sd = self.optimizer.state_dict(for_load=self._load)
            optim_sd_with_extra_parallel_dim = self.get_state_dict_with_extra_parallel_dim_preprocess(
                vanilla_optim_sd, "restore"
            )
            if (
                not self._load
                and not self._coordinated_file_mode
                and not _has_optimizer_checkpoint_state(optim_sd_with_extra_parallel_dim)
            ):
                optim_sd_with_extra_parallel_dim[_ZERO_OPTIMIZER_STATE_MARKER] = True
            if _needs_extra_parallel_layout_marker(self.extra_parallel_fqn2spec_info):
                optim_sd_with_extra_parallel_dim[_EXTRA_PARALLEL_LAYOUT_MARKER] = True
            return optim_sd_with_extra_parallel_dim

        if getattr(self.optimizer, "_is_multi_optimizer", False):
            if self._load:
                return self.optimizer.state_dict(for_load=True)
            sparse = self.optimizer._sparse_state_dict()
            has_state = any(key.startswith("state.") for key in sparse)
            if not self._coordinated_file_mode and not has_state:
                sparse[_ZERO_OPTIMIZER_STATE_MARKER] = True
            return sparse

        state_dict = optimizer_state_dict_for_checkpoint(self.model, self.optimizer, for_load=self._load)
        if not self._load:
            has_state = _has_optimizer_checkpoint_state(state_dict)
            if not self._coordinated_file_mode and not has_state:
                state_dict[_ZERO_OPTIMIZER_STATE_MARKER] = True
        return state_dict

    def load_state_dict(self, state_dict):
        if not self._coordinated_file_mode and not self._coordinated_group_validation:
            return self._load_state_dict_impl(state_dict)
        error = None
        try:
            self._load_state_dict_impl(state_dict)
        except BaseException as exc:
            error = exc
        try:
            raise_if_any_rank_failed(error, "restoring optimizer checkpoint state", group=self._coordinated_group)
        except BaseException:
            _abort_file_backed_optimizers(self.optimizer)
            raise

    def _load_state_dict_impl(self, state_dict):
        succeeded = False
        try:
            optim_state_from_dcp_load, error = None, None
            try:
                optim_state_from_dcp_load = _strip_extra_parallel_layout_marker(
                    state_dict,
                    self.should_extra_parallel_aware
                    and _needs_extra_parallel_layout_marker(self.extra_parallel_fqn2spec_info),
                )
            except BaseException as exc:
                error = exc
            if self._coordinated_group_validation or self._coordinated_file_mode:
                raise_if_any_rank_failed(error, "reading optimizer checkpoint marker", group=self._coordinated_group)
                self._agree_parameter_uniqueness()
                self._agree_external_optimizer_hooks()
            elif error is not None:
                raise error
            else:
                _validate_unique_optimizer_parameters(self.optimizer)
            with ExitStack() as hooks:
                hook_error = None
                try:
                    hooks.enter_context(_filter_synthetic_optimizer_state_on_load(self.model, self.optimizer))
                except BaseException as exc:
                    hook_error = exc
                if self._coordinated_group_validation or self._coordinated_file_mode:
                    raise_if_any_rank_failed(
                        hook_error, "preparing optimizer checkpoint load hooks", group=self._coordinated_group
                    )
                elif hook_error is not None:
                    raise hook_error
                if self.should_extra_parallel_aware:
                    # we need to drop ExtraParallel dim before loading them into optimizers
                    optim_state_without_extra_parallel_dim, error = None, None
                    try:
                        optim_state_without_extra_parallel_dim = (
                            self.get_state_dict_with_extra_parallel_dim_preprocess(optim_state_from_dcp_load, "drop")
                        )
                    except BaseException as exc:
                        error = exc
                    if self._coordinated_group_validation or self._coordinated_file_mode:
                        raise_if_any_rank_failed(
                            error, "preparing extra-parallel optimizer restore", group=self._coordinated_group
                        )
                    elif error is not None:
                        raise error
                    self._validate_multi_optimizer_groups(optim_state_without_extra_parallel_dim)
                    # Delegate to MultiOptimizer (it will split/filter correctly)
                    self.optimizer.load_state_dict(optim_state_without_extra_parallel_dim)
                elif getattr(self.optimizer, "_is_multi_optimizer", False):
                    self._validate_multi_optimizer_groups(optim_state_from_dcp_load)
                    self.optimizer.load_state_dict(optim_state_from_dcp_load)
                else:
                    # Single torch optimizer.
                    # ``strict=False`` matches the DCP planner's allow_partial_load intent:
                    # params that never received a gradient (and thus have no saved Adam
                    # state) keep the default-initialized state that
                    # ``set_optimizer_state_dict`` / ``_init_optim_state`` already created.
                    # Torch 2.11+ raises under the default strict=True when any
                    # requires_grad param is missing from the checkpoint (DeepSeek-V4
                    # indexer ``position_bias`` is one such case on short toy runs).
                    self._validate_single_optimizer_groups(optim_state_from_dcp_load)
                    set_optimizer_state_dict(
                        model=self.model,
                        optimizers=self.optimizer,
                        optim_state_dict=optim_state_from_dcp_load,
                        options=StateDictOptions(strict=False),
                    )
            succeeded = True
        finally:
            if succeeded:
                _reset_zero_optimizer_state_resume(self.optimizer)
            else:
                _drop_absent_optimizer_state(self.model, self.optimizer)
                _finish_zero_optimizer_state_resume(self.optimizer)
            _reset_absent_optimizer_state(self.optimizer)

    def _validate_multi_optimizer_groups(self, state_dict):
        if not self._coordinated_group_validation:
            return
        error = None
        try:
            self.optimizer.validate_checkpoint_groups(state_dict)
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "validating optimizer checkpoint groups", group=self._coordinated_group)

    def _validate_single_optimizer_groups(self, state_dict):
        error = None
        try:
            _validate_single_optimizer_groups(self.model, self.optimizer, state_dict)
        except BaseException as exc:
            error = exc
        if self._coordinated_group_validation or self._coordinated_file_mode:
            raise_if_any_rank_failed(error, "validating optimizer checkpoint groups", group=self._coordinated_group)
        elif error is not None:
            raise error

    def get_state_dict_with_extra_parallel_dim_preprocess(self, state_dict, action):
        return _apply_extra_parallel_dim(
            state_dict,
            self.extra_parallel_fqn2spec_info,
            self.parallel_state,
            action,
            key_match="substring",
        )


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


def drop_extra_parallel_dim(
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


def restore_extra_parallel_dim(
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
            before ExtraParallel and FSDP sharding. This is required for an
            FSDP-only DTensor whose local storage is already one ExtraParallel
            slice and the ExtraParallel shard is uneven; without it the helper
            can only infer an evenly divisible extent and otherwise fails.

    Note:
        When ``ep_shard_dim == fsdp_shard_dim`` (e.g. the Muon zero-comm layout
        shards experts on dim 0 just like EP), both mesh dims split the SAME
        tensor dim. The temporary checkpoint DTensor uses an EP-outer /
        FSDP-inner mesh view so DCP metadata matches the optimizer's physical
        local shards and global offsets instead of block-transposing the dim.
    """
    assert fsdp_mesh.ndim in (2, 3), f"global_mesh.ndim must be 2 or 3, got {fsdp_mesh.ndim}"

    if isinstance(orgin_tensor, DTensor):
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
        if global_shape is None:
            shape = list(orgin_tensor.shape)
            if len(orgin_tensor.placements) < fsdp_mesh.ndim:
                ep_size = fsdp_mesh.size(fsdp_mesh.ndim - 1)
                if shape[ep_shard_dim] % ep_size:
                    raise RuntimeError(
                        "Cannot infer the full ExtraParallel tensor shape from an FSDP-only DTensor with an "
                        f"uneven {ep_shard_dim=} extent {shape[ep_shard_dim]} over {ep_size} ranks; "
                        "pass the original global_shape."
                    )
                shape[ep_shard_dim] *= ep_size
            global_shape = tuple(shape)
        elif len(global_shape) != orgin_tensor.ndim:
            raise RuntimeError(
                f"ExtraParallel global shape {global_shape} does not match tensor rank {orgin_tensor.ndim}."
            )
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
        # If there is no FSDP but only ExtraParallel
        dtensor = _from_local_preserve_device(
            orgin_tensor, device_mesh=extra_parallel_fsdp_mesh, placements=[Shard(0)]
        )
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
            shutil.rmtree(stage_path, ignore_errors=True)
            os.makedirs(stage_path, exist_ok=True)
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
        file_profile_receipt=None,
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
        if "model" not in state:
            raise ValueError("Model must be provided to save a distributed checkpoint.")

        optimizer = state.get("optimizer")
        file_mode = _file_mode_for_checkpoint(optimizer, receipt=file_profile_receipt)
        if save_async and file_mode:
            raise ValueError("File-backed optimizer state requires synchronous DCP save")
        if file_mode:
            _validate_file_checkpoint_layout(state["model"], parallel_state, group=None)

        # Rejected up front, before anything reaches disk: an async write is still
        # running when save() returns and drops the staged copy, so it would write
        # straight to the slow destination staging was meant to avoid.
        if stage_dir and save_async:
            raise ValueError("stage_dir cannot be combined with save_async")

        # ``is not None`` rather than truthiness: step 0 is a step like any other, and
        # folding it onto ``path`` would write it over the run's own directory.
        checkpoint_dir = step_dir(path, global_steps) if global_steps is not None else path
        claim = _claim_file_checkpoint(checkpoint_dir, module) if file_mode else nullcontext()
        with claim:
            model_root = model_dir(checkpoint_dir, module)
            if file_mode:
                error = None
                try:
                    module_id = hashlib.sha256((module or "").encode()).hexdigest()[:16]
                    own_claim = f".veomni-file-state-save-{module_id}.lock"
                    # A failed write may have shard/sidecar files even when no
                    # DCP metadata or trainer manifest was ever committed.
                    # Preserve all such artifacts and require a fresh root.
                    if any(name != own_claim for name in os.listdir(checkpoint_dir)):
                        raise FileExistsError("File-backed optimizer save refuses to overwrite an existing checkpoint")
                except BaseException as exc:
                    error = exc
                raise_if_any_rank_failed(error, "checking checkpoint destination", group=None)

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

            if stage_path is None:
                # Nothing stands between this save and the previous checkpoint's
                # shards, so its markers have to go before the first byte lands --
                # otherwise a save that dies part-way leaves one describing shards
                # that are half this step and half the last. A staged save invalidates
                # its destination in ``_promote_staged_checkpoint`` instead, at the
                # point where it is about to overwrite it for real.
                #
                # The manifest is not ours either way; ``GlobalStateCallback`` clears
                # the one it writes.
                cls._remove_dcp_markers(checkpoint_dir, module)

            error = None
            try:
                cls._create_checkpoint_dir(write_root)
                # The manifest is written after all modules have completed.
                cls._save_extra_state(checkpoint_dir=write_root, state=state)
            except BaseException as exc:
                error = exc
            if file_mode:
                raise_if_any_rank_failed(error, "preparing file-backed checkpoint directory and sidecar")
            elif error is not None:
                raise error

            def prepare_slot(slot):
                writer, slot_state, slot_error = None, None, None
                try:
                    writer = cls._create_storage_writer(os.path.join(write_root, slot))
                    if file_mode:
                        # A single PyTorch 2.10 file retains all of its CPU
                        # copies until close. Keep serial failure propagation,
                        # but bound each file to a limited tensor-byte bucket.
                        if type(writer) is not FileSystemWriter:
                            raise TypeError("File-backed checkpoint requires the standard filesystem writer")
                        writer = BoundedFileSystemWriter(
                            writer.path,
                            sync_files=writer.sync_files,
                            overwrite=writer.overwrite,
                            release_page_cache=True,
                        )
                    if slot == WEIGHTS_DIRNAME:
                        slot_state = {
                            "model": ModelState(
                                state["model"],
                                trainable_only=trainable_only,
                                parallel_state=parallel_state,
                                coordinated=True,
                            )
                        }
                    else:
                        slot_state = {
                            "optimizer": OptimizerState(
                                model=state["model"],
                                optimizer=state["optimizer"],
                                parallel_state=parallel_state,
                                load=False,
                                coordinated=file_mode,
                                coordinated_group_validation=True,
                            )
                        }
                except BaseException as exc:
                    slot_error = exc
                if file_mode:
                    raise_if_any_rank_failed(slot_error, f"preparing {slot} checkpoint writer")
                elif slot_error is not None:
                    raise slot_error
                return slot_state, writer

            try:
                model_state, model_writer = prepare_slot(WEIGHTS_DIRNAME)
                cls.execute_save(
                    save_state=model_state,
                    storage_writer=model_writer,
                    save_async=save_async,
                    save_to_lowest_rank=save_to_lowest_rank,
                    slot=WEIGHTS_DIRNAME,
                    timeout_seconds=save_timeout_seconds,
                    file_mode_override=file_mode,
                )

                if "optimizer" in state and state["optimizer"] is not None:
                    optimizer_state, optimizer_writer = prepare_slot(OPTIMIZER_DIRNAME)
                    cls.execute_save(
                        save_state=optimizer_state,
                        storage_writer=optimizer_writer,
                        save_async=save_async,
                        save_to_lowest_rank=save_to_lowest_rank,
                        slot=OPTIMIZER_DIRNAME,
                        timeout_seconds=save_timeout_seconds,
                        file_mode_override=file_mode,
                    )
            except BaseException as original_error:
                if file_mode:
                    _abort_file_backed_optimizers(state.get("optimizer"))
                if stage_path is not None and _local_rank() == 0:
                    shutil.rmtree(stage_path, ignore_errors=True)
                if stage_path is None and file_mode and not save_async:
                    try:
                        cls._remove_dcp_markers(checkpoint_dir, module)
                    except BaseException as marker_error:
                        raise marker_error from original_error
                raise

            if stage_path is not None:
                _promote_staged_checkpoint(stage_path, model_root, step_root=checkpoint_dir, group=stage_group)

            logger.info_rank0(f"Saved checkpoint to {model_root}")

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
        file_profile_receipt=None,
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
        if state is None:
            raise ValueError("State dict must be provided to load a distributed checkpoint.")

        if "model" not in state:
            raise ValueError("Model must be provided to load a distributed checkpoint.")

        optimizer = state.get("optimizer")
        file_mode = _file_mode_for_checkpoint(optimizer, group=process_group, receipt=file_profile_receipt)
        if file_mode:
            _validate_file_checkpoint_layout(state["model"], parallel_state, group=process_group)

        def _model_state() -> "ModelState":
            return ModelState(
                state["model"],
                trainable_only=trainable_only,
                parallel_state=parallel_state,
                # Model loading can construct ExtraParallel meshes before the
                # optimizer phase. Preflight the same group plan even for the
                # ordinary path so a cold subgroup cannot create a global mesh.
                coordinated=True,
                coordinated_group=process_group,
            )

        def _optimizer_state() -> "OptimizerState":
            return OptimizerState(
                model=state["model"],
                optimizer=state["optimizer"],
                parallel_state=parallel_state,
                load=True,
                coordinated=file_mode,
                coordinated_group=process_group,
                coordinated_group_validation=True,
            )

        wants_optimizer = state.get("optimizer") is not None

        fused_dir = cls._legacy_fused_dir(path, module)
        if fused_dir is not None:
            load_state: Dict[str, Any] = {"model": _model_state()}
            if wants_optimizer:
                load_state["optimizer"] = _optimizer_state()
            try:
                reader = cls._create_storage_reader(fused_dir)
                with _source_cache_evictor(reader, file_mode, group=process_group) as cache_evictor:
                    dcp.load(
                        state_dict=load_state,
                        storage_reader=reader,
                        process_group=process_group,
                        planner=_ModelStrictLoadPlanner(
                            strict_model=not trainable_only,
                            strict_gdn_fp32=is_qwen4_exp_model(state["model"]),
                            strict_optimizer_dtype=_requires_exact_optimizer_state_dtypes(state.get("optimizer")),
                            strict_optimizer_recipe=_requires_exact_optimizer_recipe(state.get("optimizer")),
                            optimizer=state.get("optimizer"),
                            model=state["model"],
                            source_cache_evictor=cache_evictor,
                        ),
                    )
            except BaseException:
                _abort_file_backed_optimizers(state.get("optimizer"))
                raise
            finally:
                _reset_zero_optimizer_state_resume(state.get("optimizer"))
            cls._load_extra_state(checkpoint_dir=fused_dir, state=state, required=file_mode, group=process_group)
            logger.info_rank0(f"Loaded pre-split checkpoint from {fused_dir}")
            return state

        model_root = model_dir(path, module)
        try:
            reader = cls._create_storage_reader(weights_dir(path, module))
            with _source_cache_evictor(reader, file_mode, group=process_group) as cache_evictor:
                dcp.load(
                    state_dict={"model": _model_state()},
                    storage_reader=reader,
                    process_group=process_group,
                    planner=_ModelStrictLoadPlanner(
                        strict_model=not trainable_only,
                        strict_gdn_fp32=is_qwen4_exp_model(state["model"]),
                        strict_optimizer_dtype=_requires_exact_optimizer_state_dtypes(state.get("optimizer")),
                        strict_optimizer_recipe=_requires_exact_optimizer_recipe(state.get("optimizer")),
                        optimizer=state.get("optimizer"),
                        model=state["model"],
                        source_cache_evictor=cache_evictor,
                    ),
                )
        except BaseException:
            _abort_file_backed_optimizers(state.get("optimizer"))
            raise
        finally:
            _reset_zero_optimizer_state_resume(state.get("optimizer"))
        if wants_optimizer:
            try:
                reader = cls._create_storage_reader(optimizer_dir(path, module))
                with _source_cache_evictor(reader, file_mode, group=process_group) as cache_evictor:
                    dcp.load(
                        state_dict={"optimizer": _optimizer_state()},
                        storage_reader=reader,
                        process_group=process_group,
                        # The optimizer-only state contains no model keys.
                        planner=_ModelStrictLoadPlanner(
                            strict_model=False,
                            strict_optimizer_dtype=_requires_exact_optimizer_state_dtypes(state.get("optimizer")),
                            strict_optimizer_recipe=_requires_exact_optimizer_recipe(state.get("optimizer")),
                            optimizer=state.get("optimizer"),
                            model=state["model"],
                            source_cache_evictor=cache_evictor,
                        ),
                    )
            except BaseException:
                _abort_file_backed_optimizers(state.get("optimizer"))
                raise
            finally:
                _reset_zero_optimizer_state_resume(state.get("optimizer"))

        cls._load_extra_state(checkpoint_dir=model_root, state=state, required=file_mode, group=process_group)

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
        file_mode_override: Optional[bool] = None,
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
        optimizer_state = save_state.get("optimizer")
        root_optimizer = getattr(optimizer_state, "optimizer", None)
        file_backed = _file_backed_optimizers(root_optimizer)
        file_mode = file_mode_override
        if file_mode is None:
            file_mode = _file_mode_for_checkpoint(root_optimizer)
        if save_async and file_mode:
            raise ValueError("File-backed optimizer state requires synchronous DCP save")
        if file_mode:
            # PyTorch's multithreaded filesystem writer does not reliably
            # propagate worker-thread exceptions. Keep this opt-in route
            # serial so a failed flush/write fails the collective checkpoint.
            storage_writer.thread_count = 1
            storage_writer.per_thread_copy_ahead = 0
        planner = (
            _FileBackedSavePlanner(optimizers=file_backed, dedup_save_to_lowest_rank=save_to_lowest_rank)
            if file_backed
            else DefaultSavePlanner(dedup_save_to_lowest_rank=save_to_lowest_rank)
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

            # async_save resolves Stateful objects synchronously before it
            # starts the background writer. Use the same slot's long-timeout
            # Gloo group for that resolution, not the training process group.
            for value in save_state.values():
                if isinstance(value, (ModelState, OptimizerState)):
                    value._coordinated_group = cls._async_process_groups[slot]

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
                _abort_file_backed_optimizers(root_optimizer)
                raise
            if file_mode:
                error = None
                try:
                    for leaf in file_backed:
                        leaf.checkpoint_evict_file_backed_state()
                except Exception as exc:
                    error = exc
                # Every rank participates, even when it owns no PLE data.
                try:
                    raise_if_any_rank_failed(error, "evicting file-backed optimizer checkpoint state")
                except BaseException:
                    _abort_file_backed_optimizers(root_optimizer)
                    raise
            elif dist.is_initialized():
                dist.barrier()
            cleanup_error = None
            try:
                gc.collect()
                empty_cache()
                synchronize()
            except BaseException as exc:
                cleanup_error = exc
            if file_mode:
                try:
                    raise_if_any_rank_failed(cleanup_error, "finishing file-backed checkpoint slot")
                except BaseException:
                    _abort_file_backed_optimizers(root_optimizer)
                    raise
            elif cleanup_error is not None:
                raise cleanup_error

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
    def _load_extra_state(cls, checkpoint_dir: str, state: Dict[str, Any], *, required=False, group=None) -> None:
        """Load this rank's ``extra_state`` dict from ``extra_state/``."""
        if _EXTRA_STATE_KEY not in state:
            logger.warning_rank0("extra_state not found in state, skipping extra_state load")
            return
        rank = dist.get_rank() if dist.is_initialized() else 0
        extra_state_path = os.path.join(checkpoint_dir, _EXTRA_STATE_DIRNAME, _EXTRA_STATE_FORMAT.format(rank))
        error = None
        try:
            if os.path.exists(extra_state_path):
                state[_EXTRA_STATE_KEY] = torch.load(extra_state_path, weights_only=False)
            elif required:
                raise FileNotFoundError("File-backed full resume requires model extra state: " + extra_state_path)
        except BaseException as exc:
            error = exc
        if required:
            raise_if_any_rank_failed(error, "reading file-backed model checkpoint extra state", group=group)
        elif error is not None:
            raise error


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
    if key in (
        f"model.{_EXTRA_PARALLEL_LAYOUT_MARKER}",
        f"model.{_QWEN4_GDN_FP32_MARKER}",
        f"model.{_QWEN4_GDN_FP32_LEGACY_MARKER}",
    ):
        return None
    if not key.startswith("model."):
        return None

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

    # Collect model tensors and calculate sizes
    tensor_infos = []
    all_dcp_keys = []
    dropped = 0

    for key, tensor_meta in metadata.state_dict_metadata.items():
        hf_key = _normalize_key(key)
        if hf_key:
            if drop_hf_keys and hf_key in drop_hf_keys:
                dropped += 1
                continue

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
