# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""Identity of a MultiOptimizer's rank-local parameter and scheduler layout."""

import dis
import hashlib
import json
import marshal
import math
import types

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import _get_fqns

from ..utils.dist_utils import raise_if_any_rank_failed


MULTI_LAYOUT_MARKER = "__veomni_multi_optimizer_layout_v1__"

_SCHEDULER_SAFE_BUILTINS = frozenset(
    ("abs", "bool", "float", "int", "len", "max", "min", "pow", "range", "round", "sum")
)
_SCHEDULER_MATH_MEMBERS = frozenset(
    name
    for name, value in vars(math).items()
    if not name.startswith("_") and type(value) in (float, int, types.BuiltinFunctionType)
)


def _immutable_value(value):
    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value) is tuple:
        return [_immutable_value(item) for item in value]
    raise RuntimeError("MultiOptimizer scheduler captures unsupported mutable state")


def _scheduler_snapshot_value(value):
    if value is None or type(value) in (bool, int, float, str):
        return value
    if torch.is_tensor(value) and value.numel() == 1:
        return {"dtype": str(value.dtype), "value": value.item()}
    if isinstance(value, (list, tuple)):
        return [_scheduler_snapshot_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _scheduler_snapshot_value(item) for key, item in sorted(value.items())}
    raise RuntimeError("MultiOptimizer scheduler checkpoint contains an unsupported value")


def _group_option_value(value):
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_group_option_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _group_option_value(item) for key, item in sorted(value.items())}
    if torch.is_tensor(value) and value.numel() == 1:
        return {"dtype": str(value.dtype), "value": value.item()}
    return _immutable_value(value)


def _lr_layout(group, key, aliases):
    if key not in group:
        return None
    value = group[key]
    if torch.is_tensor(value):
        if value.device.type == "meta" or value.numel() != 1:
            raise RuntimeError(f"Cannot prove optimizer {key} tensor storage")
        storage = (str(value.device), value.untyped_storage().data_ptr())
        alias = aliases.setdefault(storage, len(aliases))
        return {"type": "tensor", "dtype": str(value.dtype), "shape": list(value.shape), "storage_alias": alias}
    if type(value) not in (int, float):
        raise RuntimeError(f"Unsupported optimizer {key} type")
    return {"type": type(value).__name__}


def _scheduler_functions(scheduler):
    """Describe LambdaLR callables using the existing checkpoint identity format."""

    def has_unordered_constant(value):
        if isinstance(value, frozenset):
            return True
        return isinstance(value, tuple) and any(has_unordered_constant(item) for item in value)

    def normalize(code, globals_dict):
        # Nested code objects share this function's global namespace. A
        # referenced global can change the next LR without changing code.
        # LOAD_ATTR names are not globals; inspect actual global lookups instead.
        instructions = list(dis.get_instructions(code))
        for index, instruction in enumerate(instructions):
            opcode = instruction.opname
            if opcode in ("STORE_DEREF", "DELETE_DEREF") and instruction.argval in code.co_freevars:
                raise RuntimeError("MultiOptimizer scheduler uses unsupported dynamic state")
            if opcode in (
                "LOAD_NAME",
                "STORE_NAME",
                "DELETE_NAME",
                "STORE_GLOBAL",
                "DELETE_GLOBAL",
                "STORE_ATTR",
                "DELETE_ATTR",
                "STORE_SUBSCR",
                "DELETE_SUBSCR",
                "IMPORT_NAME",
                "IMPORT_FROM",
                "IMPORT_STAR",
                "LOAD_BUILD_CLASS",
                # Hash-randomized set iteration can change floating-point LR
                # results across processes without changing the recipe hash.
                "BUILD_SET",
                "SET_ADD",
                "SET_UPDATE",
            ):
                raise RuntimeError("MultiOptimizer scheduler uses unsupported dynamic state")
            if opcode in ("LOAD_ATTR", "LOAD_METHOD"):
                previous = instructions[index - 1] if index else None
                if previous is None or previous.opname != "LOAD_GLOBAL" or previous.argval != "math":
                    raise RuntimeError("MultiOptimizer scheduler uses unsupported attribute access")
                if instruction.argval not in _SCHEDULER_MATH_MEMBERS:
                    raise RuntimeError("MultiOptimizer scheduler uses unsupported math state")
                continue
            if opcode != "LOAD_GLOBAL":
                continue
            name = instruction.argval
            if name in globals_dict:
                if name != "math" or globals_dict[name] is not math:
                    raise RuntimeError("MultiOptimizer scheduler refers to unsupported global state")
            elif name not in _SCHEDULER_SAFE_BUILTINS:
                raise RuntimeError("MultiOptimizer scheduler uses an unsupported builtin")
        if any(has_unordered_constant(item) for item in code.co_consts):
            raise RuntimeError("MultiOptimizer scheduler uses unsupported unordered state")
        constants = tuple(
            normalize(item, globals_dict) if isinstance(item, types.CodeType) else item for item in code.co_consts
        )
        return code.replace(co_filename="", co_firstlineno=0, co_consts=constants)

    functions = []
    for function in scheduler.lr_lambdas:
        if not isinstance(function, types.FunctionType):
            raise RuntimeError("MultiOptimizer scheduler uses an unsupported callable")

        functions.append(
            {
                "code": hashlib.sha256(marshal.dumps(normalize(function.__code__, function.__globals__))).hexdigest(),
                "defaults": _immutable_value(function.__defaults__),
                "kwdefaults": _immutable_value(function.__kwdefaults__),
                "closure": [_immutable_value(cell.cell_contents) for cell in function.__closure__ or ()],
            }
        )
    return functions


def _scheduler_layout(leaf):
    from ..optim.lr_scheduler import checkpoint_scheduler_binding

    scheduler = checkpoint_scheduler_binding(leaf)
    if scheduler is None:
        return None
    if type(scheduler) is not torch.optim.lr_scheduler.LambdaLR:
        raise RuntimeError("MultiOptimizer requires a supported scheduler")
    if scheduler.optimizer is not leaf:
        raise RuntimeError("MultiOptimizer scheduler binding changed")
    return {
        "type": f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
        "functions": _scheduler_functions(scheduler),
    }


def _local_layout(
    model, optimizer, scheduler_snapshot, scheduler_object, extra_digest, *, writing, frozen_digest=None, capture=False
):
    from ..optim.checkpoint_adapter import rank_local_adamw_adapter
    from ..optim.muon import DistributedMuon
    from ..optim.optimizer import AnyPrecisionAdamW, MultiOptimizer
    from ..optim.swap_adamw import SwapAdamW

    if type(optimizer) is not MultiOptimizer:
        raise RuntimeError("MultiOptimizer checkpoint requires the standard serialization implementation")
    seen_mutable_options = {}
    seen_tensor_options = {}

    def check_aliases(value, path):
        if torch.is_tensor(value):
            if value.device.type == "meta":
                raise RuntimeError(f"Cannot prove optimizer option tensor storage: {path}")
            storage = (str(value.device), value.untyped_storage().data_ptr())
            previous = seen_tensor_options.setdefault(storage, path)
            if previous != path:
                raise RuntimeError(
                    f"Shared mutable optimizer tensor option cannot be restored safely: {previous} and {path}"
                )
        elif type(value) in (dict, list):
            previous = seen_mutable_options.setdefault(id(value), path)
            if previous != path:
                raise RuntimeError(f"Shared mutable optimizer option cannot be restored safely: {previous} and {path}")
            values = value.values() if type(value) is dict else value
            for index, item in enumerate(values):
                check_aliases(item, f"{path}.{index}")
        elif type(value) is tuple:
            for index, item in enumerate(value):
                check_aliases(item, f"{path}.{index}")

    for leaf_name, leaf in optimizer.optimizers_dict.items():
        for key, value in leaf.defaults.items():
            check_aliases(value, f"{leaf_name}.defaults.{key}")
        for index, group in enumerate(leaf.param_groups):
            for key, value in group.items():
                if key != "params":
                    check_aliases(value, f"{leaf_name}.group{index}.{key}")
    keys = optimizer.key_names
    if len(keys) != len(set(keys)) or set(keys) != set(optimizer.optimizers_dict):
        raise RuntimeError("MultiOptimizer keys do not uniquely describe its optimizer leaves")
    names, model_trainability = {}, {}
    for runtime_name, param in model.named_parameters(remove_duplicate=False):
        # Even parameters outside the optimizer can affect the next global
        # gradient clipping operation. DCP does not persist pending gradients.
        if param.grad is not None:
            raise RuntimeError("MultiOptimizer checkpoint cannot omit pending gradients")
        fqns = _get_fqns(model, runtime_name)
        names.setdefault(id(param), set()).update(fqns)
        model_trainability.update((name, bool(param.requires_grad)) for name in fqns)
    optimizer_names = {}
    for runtime_name, param in optimizer.model.named_parameters(remove_duplicate=False):
        optimizer_names.setdefault(id(param), set()).update(_get_fqns(optimizer.model, runtime_name))
    if names != optimizer_names:
        raise RuntimeError("MultiOptimizer and checkpoint model have different canonical parameter identities")
    from .dynamic_rope import _is_native_qwen3_dynamic_reset_alias, dynamic_rope_checkpoint_contract

    model_aliases = {}
    for kind, items in (
        ("parameter", model.named_parameters(remove_duplicate=False)),
        ("buffer", model.named_buffers(remove_duplicate=False)),
    ):
        for runtime_name, value in items:
            model_aliases.setdefault(id(value), set()).update(
                f"{kind}:{name}" for name in _get_fqns(model, runtime_name)
            )
    safe_rope_pairs = set()
    for module_name, module in model.named_modules(remove_duplicate=False):
        if _is_native_qwen3_dynamic_reset_alias(module):
            rope_names = {
                f"buffer:{fqn}"
                for buffer_name in ("inv_freq", "original_inv_freq")
                for fqn in _get_fqns(model, f"{module_name}.{buffer_name}" if module_name else buffer_name)
            }
            if len(rope_names) == 2:
                safe_rope_pairs.add(frozenset(rope_names))
    alias_graph = sorted(
        sorted(aliases)
        for aliases in model_aliases.values()
        if len(aliases) > 1 and frozenset(aliases) not in safe_rope_pairs
    )

    leaves = []
    seen_param_ids, seen_fqns, lr_aliases, lr_owners = set(), set(), {}, {}
    mesh_digests = {}

    def mesh_identity(mesh):
        if mesh is None:
            return None
        mesh_id = id(mesh)
        if mesh_id not in mesh_digests:
            identity = {
                "ranks": mesh.mesh.tolist(),
                "device_type": mesh.device_type,
                "dimension_names": mesh.mesh_dim_names,
            }
            mesh_digests[mesh_id] = hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        return mesh_digests[mesh_id]

    specs = getattr(model, "_fqn2spec_info", None)
    if specs is not None and not isinstance(specs, dict):
        raise RuntimeError("MultiOptimizer checkpoint cannot prove ExtraParallel plan identity")
    extra_parallel_specs = []
    for name, spec in sorted((specs or {}).items()):
        extra_parallel_specs.append(
            {
                "name": name,
                "parallel_name": spec.para_name,
                "placement": repr(spec.placement),
                "fsdp_shard_dim": spec.fsdp_shard_dim,
                "persistent_fsdp_shard_dim": getattr(spec, "persistent_fsdp_shard_dim", None),
                "global_shape": getattr(spec, "global_shape", None),
                "checkpoint_layout_v2": getattr(spec, "checkpoint_layout_v2", False),
                "mesh": mesh_identity(spec.para_fsdp_mesh),
            }
        )
    for name in keys:
        leaf = optimizer.optimizers_dict[name]
        if rank_local_adamw_adapter(leaf) is None and type(leaf) not in (
            torch.optim.AdamW,
            AnyPrecisionAdamW,
            DistributedMuon,
            SwapAdamW,
        ):
            raise RuntimeError("MultiOptimizer checkpoint requires an explicitly supported optimizer leaf")
        groups = []
        for group_index, group in enumerate(leaf.param_groups):
            if not group["params"]:
                raise RuntimeError("MultiOptimizer cannot checkpoint an empty parameter group")
            params = []
            for param in group["params"]:
                fqns = names.get(id(param))
                if not fqns:
                    raise RuntimeError("MultiOptimizer owns a parameter absent from the checkpoint model")
                if id(param) in seen_param_ids or seen_fqns.intersection(fqns):
                    raise RuntimeError("MultiOptimizer parameter ownership is duplicated across groups or leaves")
                seen_param_ids.add(id(param))
                seen_fqns.update(fqns)
                mesh = getattr(param, "device_mesh", None)
                mesh_digest = mesh_identity(mesh)
                adapter = rank_local_adamw_adapter(leaf)
                params.append(
                    {
                        "names": sorted(fqns),
                        "shape": list(param.shape),
                        "dtype": str(param.dtype),
                        "device_type": param.device.type,
                        "mesh": mesh_digest,
                        "placements": [repr(place) for place in param.placements] if mesh is not None else None,
                        "cpu_offload": bool(adapter.state_is_offloaded(leaf, param)) if adapter is not None else False,
                        "state_dtypes": {
                            key: str(adapter.state_dtype(leaf, param, key))
                            for key in ("step", "master_weight", "exp_avg", "exp_avg_sq")
                        }
                        if adapter is not None
                        else None,
                    }
                )
            lr_layout = _lr_layout(group, "lr", lr_aliases)
            initial_lr_layout = _lr_layout(group, "initial_lr", lr_aliases)
            for lr_spec in (lr_layout, initial_lr_layout):
                if lr_spec is not None and lr_spec["type"] == "tensor":
                    alias = lr_spec["storage_alias"]
                    owner = lr_owners.setdefault(alias, (name, group_index))
                    if owner != (name, group_index):
                        raise RuntimeError(
                            "MultiOptimizer cannot safely restore LR tensor storage shared across groups"
                        )
            groups.append(
                {
                    "params": params,
                    "lr_layout": lr_layout,
                    "initial_lr_layout": initial_lr_layout,
                    # LR is restored from the checkpoint and may have advanced;
                    # structural and algorithm options must already agree.
                    "options": {
                        key: _group_option_value(value)
                        for key, value in sorted(group.items())
                        if key not in ("params", "lr", "initial_lr")
                    },
                }
            )
        leaves.append(
            {
                "name": name,
                "type": f"{type(leaf).__module__}.{type(leaf).__qualname__}",
                "algorithm_identity": (
                    _immutable_value(rank_local_adamw_adapter(leaf).algorithm_identity(leaf))
                    if rank_local_adamw_adapter(leaf) is not None
                    else None
                ),
                "groups": groups,
                "scheduler": _scheduler_layout(leaf),
            }
        )
    has_scheduler = any(leaf["scheduler"] is not None for leaf in leaves)
    if has_scheduler:
        if not isinstance(scheduler_snapshot, dict) or set(scheduler_snapshot) != set(keys):
            raise RuntimeError("MultiOptimizer scheduler checkpoint is missing or has different leaves")
        from ..optim.lr_scheduler import MultiLRScheduler, checkpoint_scheduler_binding

        if type(scheduler_object) is not MultiLRScheduler or set(scheduler_object) != set(keys):
            raise RuntimeError("MultiOptimizer scheduler checkpoint requires its live scheduler object")
        for name in keys:
            current_scheduler = scheduler_object[name]
            if current_scheduler is not checkpoint_scheduler_binding(optimizer.optimizers_dict[name]):
                raise RuntimeError("MultiOptimizer scheduler object differs from its optimizer binding")
            if writing and _scheduler_snapshot_value(scheduler_snapshot[name]) != _scheduler_snapshot_value(
                current_scheduler.state_dict()
            ):
                raise RuntimeError("MultiOptimizer scheduler snapshot differs from its live scheduler")
    elif scheduler_snapshot is not None:
        raise RuntimeError("MultiOptimizer scheduler checkpoint has no matching scheduler binding")
    elif scheduler_object is not None:
        raise RuntimeError("MultiOptimizer scheduler object has no matching scheduler binding")
    layout = {
        "leaves": leaves,
        "model_aliases": alias_graph,
        "model_trainability": sorted(model_trainability.items()),
        "extra_parallel_specs": extra_parallel_specs,
        "scheduler_snapshot": _scheduler_snapshot_value(scheduler_snapshot),
        "extra_digest": extra_digest,
        "frozen_digest": frozen_digest,
    }
    rope_contract = dynamic_rope_checkpoint_contract(model)
    if rope_contract:
        # Earlier checkpoints did not prove that transient cache history was
        # reconstructible by a fresh process.
        layout["dynamic_rope_cache_v2"] = rope_contract
    return layout if capture else _layout_digest(layout)


def _layout_digest(layout):
    return hashlib.sha256(
        json.dumps(layout, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def frozen_parameter_digest(model, *, enabled, group=None, chunk_bytes=4 << 20):
    """Bind omitted frozen model values to a trainable-only checkpoint."""
    if not enabled:
        return None
    digest, error = hashlib.sha256(), None
    try:
        for name, param in sorted(model.named_parameters(remove_duplicate=False)):
            if param.requires_grad:
                continue
            local = param.to_local() if hasattr(param, "to_local") else param
            descriptor = (name, str(param.dtype), tuple(param.shape), tuple(local.shape), str(param.device.type))
            digest.update(json.dumps(descriptor, separators=(",", ":")).encode())
            from .tensor_bytes import tensor_byte_chunks

            for chunk in tensor_byte_chunks(local, chunk_bytes=chunk_bytes):
                digest.update(chunk)
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "hashing frozen checkpoint parameters", group=group)
    return digest.hexdigest()


def capture_multi_layout(model, optimizer, scheduler_snapshot, scheduler_object, *, group=None, frozen_digest=None):
    """Validate the live save layout once, before creating checkpoint files."""
    encoded, error = None, None
    try:
        payload = _local_layout(
            model,
            optimizer,
            scheduler_snapshot,
            scheduler_object,
            None,
            writing=True,
            frozen_digest=frozen_digest,
            capture=True,
        )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "capturing MultiOptimizer checkpoint layout", group=group)
    return encoded


def finish_multi_layout(encoded, extra_digest, *, group=None):
    """Bind saved sidecar bytes without inspecting live model or optimizer state again."""
    local, error = None, None
    try:
        payload = json.loads(encoded)
        payload["extra_digest"] = extra_digest
        local = _layout_digest(payload)
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "finalizing MultiOptimizer checkpoint layout", group=group)
    if dist.is_initialized():
        digests = [None] * dist.get_world_size(group)
        dist.all_gather_object(digests, local, group=group)
        ranks = list(range(len(digests))) if group is None else dist.get_process_group_ranks(group)
    else:
        ranks, digests = [0], [local]
    return json.dumps({"version": 1, "ranks": ranks, "layouts": digests}, sort_keys=True, separators=(",", ":"))


def coordinated_multi_layout(
    model,
    optimizer,
    scheduler_snapshot=None,
    scheduler_object=None,
    extra_digest=None,
    *,
    writing=False,
    group=None,
    frozen_digest=None,
):
    """Return the same rank-indexed layout on every participant or fail together."""
    local = None
    error = None
    try:
        local = _local_layout(
            model,
            optimizer,
            scheduler_snapshot,
            scheduler_object,
            extra_digest,
            writing=writing,
            frozen_digest=frozen_digest,
        )
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "inspecting MultiOptimizer layout", group=group)
    if dist.is_initialized():
        world = dist.get_world_size(group)
        digests = [None] * world
        dist.all_gather_object(digests, local, group=group)
        ranks = list(range(world)) if group is None else dist.get_process_group_ranks(group)
    else:
        ranks, digests = [0], [local]
    return json.dumps({"version": 1, "ranks": ranks, "layouts": digests}, sort_keys=True, separators=(",", ":"))
