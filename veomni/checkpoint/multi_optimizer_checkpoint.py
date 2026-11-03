# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""Versioned identity and lazy-state restoration for MultiOptimizer checkpoints."""

import hashlib
import json
import os
import tempfile
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint import DefaultLoadPlanner
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.checkpoint.metadata import BytesStorageMetadata, TensorStorageMetadata
from torch.distributed.checkpoint.state_dict import _get_fqns

from ..utils.dist_utils import raise_if_any_rank_failed
from .optimizer_resume_layout import (
    MULTI_LAYOUT_MARKER,
    coordinated_multi_layout,
    frozen_parameter_digest,
)


_EXTRA_STATE_DIRNAME = "extra_state"
_EXTRA_STATE_FORMAT = "extra_state_rank_{}.pt"
_MULTI_GENERATION_MARKER = "__veomni_multi_generation_v1__"
_MULTI_INVENTORY_MARKER = "__veomni_multi_inventory_v1__"


def _inventory_digests(inventory, *, group=None):
    from ..utils.dist_utils import raise_if_any_rank_failed

    local, error = None, None
    try:
        local = hashlib.sha256(json.dumps(inventory, separators=(",", ":")).encode()).hexdigest()
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "hashing optimizer checkpoint inventory", group=group)
    if not dist.is_initialized():
        return [local]
    digests = [None] * dist.get_world_size(group)
    dist.all_gather_object(digests, local, group=group)
    return digests


def _coordinated_storage_reader(reader_factory, path, *, group=None):
    from ..utils.dist_utils import raise_if_any_rank_failed

    reader, error = None, None
    try:
        reader = reader_factory(path)
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "constructing optimizer inventory reader", group=group)
    return reader


def _iter_optimizer_leaves(optimizer):
    yield from optimizer.optimizers_dict.values()


def _expected_optimizer_fields(leaf, param_group):
    from ..optim.checkpoint_adapter import rank_local_adamw_adapter
    from ..optim.muon import DistributedMuon
    from ..optim.optimizer import AnyPrecisionAdamW

    if isinstance(leaf, DistributedMuon):
        return {"momentum_buffer"}
    if rank_local_adamw_adapter(leaf) is not None:
        return {"step", "master_weight", "exp_avg", "exp_avg_sq"}
    if type(leaf) is torch.optim.AdamW:
        fields = {"step", "exp_avg", "exp_avg_sq"}
        if param_group.get("amsgrad"):
            fields.add("max_exp_avg_sq")
        return fields
    if isinstance(leaf, AnyPrecisionAdamW):
        fields = {"step", "exp_avg", "exp_avg_sq"}
        if param_group.get("use_kahan_summation"):
            fields.add("compensation")
        return fields
    raise RuntimeError("Unsupported MultiOptimizer leaf")


def _validate_source_optimizer_states(optimizer, model):
    from ..optim.checkpoint_adapter import rank_local_adamw_adapter
    from ..optim.optimizer import AnyPrecisionAdamW

    for leaf in _iter_optimizer_leaves(optimizer):
        for group in leaf.param_groups:
            expected = _expected_optimizer_fields(leaf, group)
            for param in group["params"]:
                state = leaf.state.get(param, {})
                if not state:
                    continue
                if set(state) != expected:
                    raise RuntimeError("MultiOptimizer has incomplete or unsupported parameter state")
                for field, value in state.items():
                    if not torch.is_tensor(value):
                        raise RuntimeError(f"MultiOptimizer state {field} must be a tensor")
                    if field == "step":
                        if tuple(value.shape) != ():
                            raise RuntimeError("MultiOptimizer step must be a scalar")
                        if isinstance(leaf, AnyPrecisionAdamW) and isinstance(value, DTensor):
                            raise RuntimeError("AnyPrecisionAdamW step must be an ordinary scalar tensor")
                        adapter = rank_local_adamw_adapter(leaf)
                        if adapter is not None and value.dtype != adapter.state_dtype(leaf, param, field):
                            raise RuntimeError("MultiOptimizer step dtype differs from its registered policy")
                    else:
                        if tuple(value.shape) != tuple(param.shape):
                            raise RuntimeError(f"MultiOptimizer state shape differs for {field}")
                        if isinstance(leaf, AnyPrecisionAdamW) and (
                            isinstance(param, DTensor) != isinstance(value, DTensor)
                            or (
                                isinstance(param, DTensor)
                                and (
                                    value.device_mesh != param.device_mesh
                                    or value.placements != param.placements
                                    or value.to_local().shape != param.to_local().shape
                                )
                            )
                        ):
                            raise RuntimeError(f"AnyPrecisionAdamW state layout differs for {field}")
                        adapter = rank_local_adamw_adapter(leaf)
                        expected_dtype = (
                            adapter.state_dtype(leaf, param, field)
                            if adapter is not None
                            else group[
                                {
                                    "exp_avg": "momentum_dtype",
                                    "exp_avg_sq": "variance_dtype",
                                    "compensation": "compensation_buffer_dtype",
                                }[field]
                            ]
                            if isinstance(leaf, AnyPrecisionAdamW)
                            else param.dtype
                        )
                        if value.dtype != expected_dtype:
                            raise RuntimeError(f"MultiOptimizer state dtype cannot be safely restored for {field}")


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


def _multi_extra_snapshot(extra_root, *, read_state):
    """Hash the exact sidecar bytes; read through a bounded-memory snapshot."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    extra_path = os.path.join(extra_root, _EXTRA_STATE_DIRNAME, _EXTRA_STATE_FORMAT.format(rank))
    if not os.path.exists(extra_path):
        return "absent", None
    digest = hashlib.sha256()
    with open(extra_path, "rb") as source:
        if not read_state:
            while chunk := source.read(4 * 1024 * 1024):
                digest.update(chunk)
            return digest.hexdigest(), None
        with tempfile.TemporaryFile() as snapshot:
            while chunk := source.read(4 * 1024 * 1024):
                digest.update(chunk)
                snapshot.write(chunk)
            snapshot.seek(0)
            value = torch.load(snapshot, weights_only=False)
    if not isinstance(value, dict):
        raise RuntimeError("MultiOptimizer checkpoint extra_state is not a mapping")
    return digest.hexdigest(), value


def _preflight_multi_optimizer_layout(
    model,
    optimizer,
    reader_factory,
    checkpoint_path,
    model_checkpoint_path,
    extra_root,
    load_extra_state,
    scheduler_object,
    parallel_state=None,
    trainable_only=False,
    group=None,
):
    """Reject an unproven or changed MultiOptimizer group/scheduler layout."""
    multi = getattr(optimizer, "_is_multi_optimizer", False) is True
    if not multi:
        return None, None, None, None, None, None, None
    snapshot, extra, extra_digest, source_metadata, error = None, None, None, None, None
    try:
        source_metadata = reader_factory(checkpoint_path).read_metadata()
        parameter_names = _dcp_parameter_name_map(model)
        from .rank_local_optimizer import current_rank, load_rank_scalars, rank_prefix

        rank_local = (
            getattr(model, "_fqn2spec_info", None) is not None and getattr(parallel_state, "dp_mode", None) == "fsdp2"
        )
        prefix = f"optimizer.{rank_prefix(current_rank())}" if rank_local else "optimizer."
        source_bases = {
            "optimizer." + key.removeprefix(prefix).rpartition(".")[0]
            for key in source_metadata.state_dict_metadata
            if key.startswith(prefix + "state.")
        }
        if trainable_only:
            for name, param in parameter_names.items():
                if not param.requires_grad and f"optimizer.state.{name}" in source_bases:
                    raise RuntimeError(
                        "trainable-only resume cannot prove the value of a frozen parameter with optimizer history; "
                        "use a full-model checkpoint or a verified migration"
                    )
        for leaf in _iter_optimizer_leaves(optimizer):
            if leaf.state:
                raise RuntimeError("MultiOptimizer checkpoint resume requires a fresh optimizer without state")
            if any(param.grad is not None for param_group in leaf.param_groups for param in param_group["params"]):
                raise RuntimeError("MultiOptimizer checkpoint resume requires cleared gradients")
        extra_digest, extra = _multi_extra_snapshot(extra_root, read_state=True)
        if extra is not None:
            snapshot = extra.get("lr_scheduler")
        if extra is not None and not load_extra_state:
            raise RuntimeError("MultiOptimizer model extra state cannot be discarded during optimizer resume")
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "reading MultiOptimizer scheduler checkpoint", group=group)
    if rank_local:
        from .rank_local_optimizer import current_rank, rank_inventory_key

        rank = current_rank()
        inventory_name = rank_inventory_key(rank)
        inventory_path = "optimizer." + inventory_name
        inventory_error = (
            RuntimeError("rank-local optimizer checkpoint inventory is missing")
            if not isinstance(source_metadata.state_dict_metadata.get(inventory_path), BytesStorageMetadata)
            else None
        )
        raise_if_any_rank_failed(inventory_error, "inspecting rank-local optimizer inventory", group=group)
        inventory_state = {"optimizer": {inventory_name: ""}}
        reader = _coordinated_storage_reader(reader_factory, checkpoint_path, group=group)
        dcp.load(inventory_state, storage_reader=reader, process_group=group)
        inventory_error = None
        try:
            inventory = json.loads(inventory_state["optimizer"][inventory_name])
            expected = sorted(
                key for key in source_metadata.state_dict_metadata if key.startswith(prefix) and key != inventory_path
            )
            if inventory != expected:
                raise RuntimeError("rank-local optimizer checkpoint inventory differs from DCP metadata")
            load_rank_scalars(reader_factory, checkpoint_path, rank)
        except BaseException as exc:
            inventory_error = exc
        raise_if_any_rank_failed(inventory_error, "validating rank-local optimizer inventory", group=group)
    else:
        inventory_path = f"optimizer.{_MULTI_INVENTORY_MARKER}"
        inventory_error = (
            RuntimeError(
                "MultiOptimizer checkpoint group layout is unproven or inventory is missing; "
                "original grouping and scheduler must be verified and migrated explicitly"
            )
            if not isinstance(source_metadata.state_dict_metadata.get(inventory_path), BytesStorageMetadata)
            else None
        )
        raise_if_any_rank_failed(inventory_error, "inspecting MultiOptimizer inventory", group=group)
        inventory_state = {"optimizer": {_MULTI_INVENTORY_MARKER: ""}}
        reader = _coordinated_storage_reader(reader_factory, checkpoint_path, group=group)
        dcp.load(inventory_state, storage_reader=reader, process_group=group)
        inventory_error = None
        try:
            inventory = json.loads(inventory_state["optimizer"][_MULTI_INVENTORY_MARKER])
            expected = sorted(
                key
                for key in source_metadata.state_dict_metadata
                if key.startswith("optimizer.") and key not in (inventory_path, f"optimizer.{MULTI_LAYOUT_MARKER}")
            )
            if inventory != expected:
                raise RuntimeError("MultiOptimizer checkpoint inventory differs from DCP metadata")
        except BaseException as exc:
            inventory_error = exc
        raise_if_any_rank_failed(inventory_error, "validating MultiOptimizer inventory", group=group)
    inventory_digests = _inventory_digests(inventory, group=group)
    frozen_digest = frozen_parameter_digest(model, enabled=trainable_only, group=group)
    current = coordinated_multi_layout(
        model, optimizer, snapshot, scheduler_object, extra_digest, group=group, frozen_digest=frozen_digest
    )
    metadata, reader, error = None, None, None
    key = f"optimizer.{MULTI_LAYOUT_MARKER}"
    try:
        reader = reader_factory(checkpoint_path)
        metadata = reader.read_metadata()
        first_id = getattr(getattr(source_metadata, "storage_meta", None), "save_id", None)
        second_id = getattr(getattr(metadata, "storage_meta", None), "save_id", None)
        if first_id and first_id != second_id:
            raise RuntimeError("MultiOptimizer checkpoint changed during preflight")
        if not isinstance(metadata.state_dict_metadata.get(key), BytesStorageMetadata):
            raise RuntimeError(
                "MultiOptimizer checkpoint group layout is unproven; original grouping and scheduler "
                "must be verified and migrated explicitly"
            )
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "inspecting MultiOptimizer checkpoint layout", group=group)
    model_key = _MULTI_GENERATION_MARKER
    model_reader, model_error = None, None
    try:
        model_reader = reader_factory(model_checkpoint_path)
        model_metadata = model_reader.read_metadata()
        if not isinstance(model_metadata.state_dict_metadata.get(model_key), BytesStorageMetadata):
            raise RuntimeError("MultiOptimizer model checkpoint generation is missing")
        from .dcp_checkpointer import ModelState, _export_extra_parallel_layout_key

        # Save always uses ModelState. Its extra precision inventories and
        # ExtraParallel namespaces must also be present during preflight.
        current_model_state = ModelState(
            model, trainable_only=trainable_only, parallel_state=parallel_state
        ).state_dict()

        flat_model, _ = flatten_state_dict({"model": current_model_state})
        expected_keys = set(flat_model)
        source_keys = {key for key in model_metadata.state_dict_metadata if key.startswith("model.")}
        if source_keys != expected_keys:
            raise RuntimeError("MultiOptimizer checkpoint model state inventory differs")
        for name, tensor in flat_model.items():
            source_entry = model_metadata.state_dict_metadata[name]
            if not torch.is_tensor(tensor):
                if not isinstance(source_entry, BytesStorageMetadata):
                    raise RuntimeError(f"MultiOptimizer model checkpoint value type differs for {name}")
                continue
            if not isinstance(source_entry, TensorStorageMetadata):
                raise RuntimeError(f"MultiOptimizer model checkpoint value type differs for {name}")
            if source_entry.properties.dtype != tensor.dtype or tuple(source_entry.size) != tuple(tensor.shape):
                raise RuntimeError(f"MultiOptimizer model checkpoint tensor dtype or shape differs for {name}")
        if trainable_only:
            logical_source_keys = {
                logical_key
                for raw in source_keys
                if (logical_key := _export_extra_parallel_layout_key(raw.removeprefix("model."))) is not None
            }
            for name in parameter_names:
                if f"optimizer.state.{name}" in source_bases and name not in logical_source_keys:
                    raise RuntimeError(
                        "trainable-only resume cannot prove the model value for a parameter with optimizer history; "
                        "use a full-model checkpoint or a verified migration"
                    )
    except BaseException as exc:
        model_error = exc
    raise_if_any_rank_failed(model_error, "inspecting MultiOptimizer model generation", group=group)
    model_scratch = {model_key: ""}
    dcp.load(
        state_dict=model_scratch,
        storage_reader=model_reader,
        process_group=group,
        planner=DefaultLoadPlanner(flatten_state_dict=False, flatten_sharded_tensors=False, allow_partial_load=False),
    )
    scratch = {key: ""}
    dcp.load(
        state_dict=scratch,
        storage_reader=reader,
        process_group=group,
        planner=DefaultLoadPlanner(flatten_state_dict=False, flatten_sharded_tensors=False, allow_partial_load=False),
    )
    mismatch = None
    try:
        saved = json.loads(scratch[key])
        if (
            not isinstance(saved, dict)
            or set(saved) != {"generation", "layout", "inventory_digests"}
            or not isinstance(saved["generation"], str)
            or not saved["generation"]
            or saved["generation"] != model_scratch[model_key]
            or saved["layout"] != current
            or saved["inventory_digests"] != inventory_digests
        ):
            raise RuntimeError("MultiOptimizer checkpoint generation, group or scheduler layout differs")
    except (ValueError, TypeError, KeyError) as exc:
        mismatch = RuntimeError("MultiOptimizer checkpoint layout marker is malformed")
        mismatch.__cause__ = exc
    except RuntimeError as exc:
        mismatch = exc
    raise_if_any_rank_failed(mismatch, "validating MultiOptimizer checkpoint layout", group=group)
    optimizer_id = getattr(getattr(metadata, "storage_meta", None), "save_id", None)
    model_id = getattr(getattr(model_metadata, "storage_meta", None), "save_id", None)
    id_error = (
        RuntimeError("MultiOptimizer checkpoint cannot prove model/optimizer save generations")
        if not optimizer_id or not model_id
        else None
    )
    raise_if_any_rank_failed(id_error, "validating MultiOptimizer DCP generation IDs", group=group)
    return optimizer_id, extra, model_id, saved["generation"], current, extra_digest, frozen_digest


def _prepare_multi_optimizer_saved_states(
    model,
    optimizer,
    reader_factory,
    checkpoint_path,
    expected_save_id,
    group=None,
    *,
    parallel_state=None,
    validate_only=False,
):
    """Validate or allocate saved leaf state, without a synthetic optimizer step."""
    multi = getattr(optimizer, "_is_multi_optimizer", False) is True
    if not multi:
        return
    error = None
    try:
        from ..optim.checkpoint_adapter import rank_local_adamw_adapter
        from ..optim.optimizer import AnyPrecisionAdamW

        metadata = reader_factory(checkpoint_path).read_metadata()
        save_id = getattr(getattr(metadata, "storage_meta", None), "save_id", None)
        if expected_save_id is not None and save_id != expected_save_id:
            raise RuntimeError("MultiOptimizer checkpoint changed before preparing its saved state")
        names = _dcp_parameter_name_map(model)
        names_by_id = defaultdict(list)
        for name, param in names.items():
            names_by_id[id(param)].append(name)
        saved = defaultdict(dict)
        from .rank_local_optimizer import current_rank, load_rank_scalars, rank_prefix

        rank_local = (
            getattr(model, "_fqn2spec_info", None) is not None and getattr(parallel_state, "dp_mode", None) == "fsdp2"
        )
        checkpoint_prefix = f"optimizer.{rank_prefix(current_rank())}" if rank_local else "optimizer."
        unknown_state_names = set()
        for key, value in metadata.state_dict_metadata.items():
            if not key.startswith(checkpoint_prefix):
                continue
            key = "optimizer." + key.removeprefix(checkpoint_prefix)
            prefix, separator, field = key.rpartition(".")
            name = prefix.removeprefix("optimizer.state.")
            if separator and prefix.startswith("optimizer.state."):
                if name not in names:
                    unknown_state_names.add(name)
                else:
                    saved[name][field] = value
        if unknown_state_names:
            raise RuntimeError("optimizer checkpoint contains state for unknown model parameters")
        expected_groups = {key for key in optimizer._sparse_state_dict() if key.startswith("param_groups.")}
        if rank_local:
            scalars = load_rank_scalars(reader_factory, checkpoint_path, current_rank())
            actual_groups = {key for key in scalars if key.startswith("param_groups.")}
            if expected_groups != actual_groups:
                raise RuntimeError("rank-local optimizer parameter group inventory differs")
            owned_names = {
                name
                for leaf in _iter_optimizer_leaves(optimizer)
                for param_group in leaf.param_groups
                for param in param_group["params"]
                for name in names_by_id[id(param)]
            }
            unknown_steps = {
                key
                for key in scalars
                if key.startswith("state.") and key.removeprefix("state.").removesuffix(".step") not in owned_names
            }
            if unknown_steps:
                raise RuntimeError("rank-local optimizer checkpoint has unowned parameter steps")
            if set(saved) - owned_names:
                raise RuntimeError("rank-local optimizer checkpoint has unowned parameter state")
            for name in names:
                step_key = f"state.{name}.step"
                if step_key in scalars:
                    saved[name]["step"] = scalars[step_key]
        else:
            actual_groups = {
                key.removeprefix("optimizer.")
                for key in metadata.state_dict_metadata
                if key.startswith("optimizer.param_groups.")
            }
            if expected_groups != actual_groups:
                raise RuntimeError("MultiOptimizer checkpoint parameter group inventory differs")

        for leaf in _iter_optimizer_leaves(optimizer):
            if any(not param.requires_grad for group in leaf.param_groups for param in group["params"]):
                if getattr(leaf, "_optimizer_load_state_dict_pre_hooks", None) or getattr(
                    leaf, "_optimizer_load_state_dict_post_hooks", None
                ):
                    raise RuntimeError(
                        "Cannot safely restore frozen optimizer history with custom optimizer load hooks"
                    )
            for param_group in leaf.param_groups:
                for param in param_group["params"]:
                    candidates = [name for name in names_by_id[id(param)] if name in saved]
                    if len(candidates) > 1:
                        raise RuntimeError("Ambiguous shared parameter optimizer history in MultiOptimizer checkpoint")
                    if not candidates:
                        continue
                    fields = saved[candidates[0]]
                    expected = _expected_optimizer_fields(leaf, param_group)
                    if set(fields) != expected:
                        raise RuntimeError("MultiOptimizer checkpoint has incomplete or unsupported parameter state")
                    state = {}
                    for field, entry in fields.items():
                        if field == "step":
                            if rank_local:
                                if not torch.is_tensor(entry):
                                    raise RuntimeError("MultiOptimizer checkpoint step is not a tensor")
                                step_shape, step_dtype = tuple(entry.shape), entry.dtype
                            else:
                                if not isinstance(entry, TensorStorageMetadata):
                                    raise RuntimeError("MultiOptimizer checkpoint state is not a supported tensor")
                                step_shape, step_dtype = tuple(entry.size), entry.properties.dtype
                            if step_shape != ():
                                raise RuntimeError("Non-scalar optimizer steps require an explicit owner-layout codec")
                            adapter = rank_local_adamw_adapter(leaf)
                            if adapter is not None and step_dtype != adapter.state_dtype(leaf, param, field):
                                raise RuntimeError(
                                    "MultiOptimizer checkpoint step dtype differs from its registered policy"
                                )
                            device = (
                                param.device if param_group.get("capturable") or param_group.get("fused") else "cpu"
                            )
                            if not validate_only:
                                state[field] = torch.zeros((), dtype=step_dtype, device=device)
                            continue
                        if not isinstance(entry, TensorStorageMetadata):
                            raise RuntimeError("MultiOptimizer checkpoint state is not a supported tensor")
                        adapter = rank_local_adamw_adapter(leaf)
                        if adapter is not None:
                            dtype = adapter.state_dtype(leaf, param, field)
                        elif isinstance(leaf, AnyPrecisionAdamW):
                            dtype = param_group[
                                {
                                    "exp_avg": "momentum_dtype",
                                    "exp_avg_sq": "variance_dtype",
                                    "compensation": "compensation_buffer_dtype",
                                }[field]
                            ]
                        else:
                            dtype = param.dtype
                        if entry.properties.dtype != dtype:
                            raise RuntimeError(f"MultiOptimizer checkpoint dtype mismatch for {candidates[0]}.{field}")
                        local_param = param.to_local() if rank_local and hasattr(param, "to_local") else param
                        if tuple(entry.size) != tuple(local_param.shape):
                            raise RuntimeError(f"MultiOptimizer checkpoint shape mismatch for {candidates[0]}.{field}")
                        if not validate_only:
                            allocated = (
                                adapter.allocate_state(leaf, param, field, dtype)
                                if adapter is not None and adapter.allocate_state is not None
                                else None
                            )
                            if allocated is not None:
                                local = allocated.to_local() if isinstance(allocated, DTensor) else allocated
                                parameter_local = param.to_local() if isinstance(param, DTensor) else param
                                expected_device = (
                                    "cpu" if adapter.state_is_offloaded(leaf, param) else param.device.type
                                )
                                if (
                                    not torch.is_tensor(allocated)
                                    or isinstance(allocated, DTensor) != isinstance(param, DTensor)
                                    or local.shape != parameter_local.shape
                                    or allocated.dtype != dtype
                                    or local.device.type != expected_device
                                    or (
                                        isinstance(allocated, DTensor)
                                        and (
                                            allocated.shape != param.shape
                                            or allocated.placements != param.placements
                                            or allocated.device_mesh != param.device_mesh
                                        )
                                    )
                                ):
                                    raise RuntimeError("Registered optimizer allocated an invalid checkpoint state")
                                state[field] = allocated
                            elif adapter is not None and adapter.state_is_offloaded(leaf, param):
                                from .rank_local_optimizer import zeros_like_parameter_on_cpu

                                state[field] = zeros_like_parameter_on_cpu(param, dtype=dtype)
                            else:
                                state[field] = torch.zeros_like(param, dtype=dtype, device=param.device)
                    if not validate_only:
                        leaf.state[param] = state
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "preparing MultiOptimizer checkpoint state", group=group)
