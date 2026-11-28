# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Install MXFP8 transport on the plain weight shards used by dense MXFP8 rows."""

from types import MethodType

import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule

from ...ops import VeomniOp
from ...ops.torchao import MXFP8_BLOCK_SIZE, _require_mxfp8_runtime
from .mxfp8_weight import MXFP8AllGather


def _mxfp8_handles(model):
    for path, module in model.named_modules(remove_duplicate=False):
        for attr, op in vars(module).items():
            if (
                isinstance(op, VeomniOp)
                and op.op in ("linear", "gated_mlp")
                and op.variant == "standard"
                and op.impl == "torchao"
                and op.kwargs.get("dtype") == "mxfp8"
            ):
                yield path, module, attr, op


def has_mxfp8_all_gather_targets(model: nn.Module) -> bool:
    """Whether this model bound an MXFP8 dense projection or gated MLP."""
    return next(_mxfp8_handles(model), None) is not None


def _selected_weights(model):
    selected = {}
    shared_aliases = {}
    selected_names = set()
    for path, owner, attr, op in _mxfp8_handles(model):
        # Standard linear sites call self.veomni_<proj>(self.<proj>, x).
        # The standard gated MLP row consumes all three projections together.
        names = (attr.removeprefix("veomni_"),) if op.op == "linear" else ("gate_proj", "up_proj", "down_proj")
        emulated = op.kwargs.get("emulated")
        if emulated is None:
            emulated = False
        elif type(emulated) is not bool:
            raise ValueError(f"MXFP8 emulated must be a bool or None: {path}.{attr}.")
        for name in names:
            layer = getattr(owner, name, None)
            fqn = ".".join(filter(None, (path, name, "weight")))
            if type(layer) is not nn.Linear:
                raise ValueError(f"MXFP8 all-gather requires a plain nn.Linear at {fqn}.")
            param_id = id(layer.weight)
            first_name, first_mode = selected.setdefault(param_id, (fqn, emulated))
            if first_mode != emulated:
                raise ValueError(f"Shared MXFP8 weight has conflicting emulated modes: {first_name}, {fqn}.")
            if first_name != fqn:
                shared_aliases[fqn] = first_name
            selected_names.add(fqn)
    # FSDP exposes the same gathered weight at every alias. An eager consumer
    # (for example a tied embedding) cannot read the two MXFP8 representations.
    if selected:
        for fqn, weight in model.named_parameters(remove_duplicate=False):
            if id(weight) in selected and fqn not in selected_names:
                raise ValueError(f"Selected MXFP8 weight {selected[id(weight)][0]} is also used by {fqn}.")
    return selected, shared_aliases


def shared_mxfp8_weight_aliases(model: nn.Module) -> dict[str, str]:
    """Validate the original consumers and record selected ties before FSDP replaces parameters."""
    _, aliases = _selected_weights(model)
    return aliases


def register_mxfp8_all_gather(model: nn.Module) -> int:
    """Bind native extensions after materialization and before the first unshard."""
    selected, _ = _selected_weights(model)
    if not selected:
        return 0

    pending = []
    for module in model.modules():
        if not isinstance(module, FSDPModule):
            continue
        group = module._get_fsdp_state()._fsdp_param_group
        if group is None:
            continue
        for param in group.fsdp_params:
            if id(param.sharded_param) not in selected:
                continue
            fqn, emulated = selected[id(param.sharded_param)]
            if group._reset_sharded_params or hasattr(param, "_unsharded_param"):
                raise ValueError(f"Install MXFP8 all-gather before the first FSDP unshard of {fqn}.")
            if param.offload_to_cpu or (
                param.post_forward_mesh_info is not None and param.post_forward_mesh_info is not param.mesh_info
            ):
                raise ValueError(f"MXFP8 all-gather requires accelerator shards and full or no resharding: {fqn}.")
            if param.mp_policy.param_dtype != torch.bfloat16:
                raise ValueError(f"MXFP8 all-gather requires BF16 mixed precision: {fqn}.")
            shape = param._orig_size
            local = param._sharded_local_tensor
            world = param.shard_mesh_from_root.size()
            if (
                param.fsdp_placement.dim != 0
                or param.is_dtensor
                or len(shape) != 2
                or shape[0] % world
                or tuple(local.shape) != (shape[0] // world, shape[1])
                or any(n == 0 or n % MXFP8_BLOCK_SIZE for n in local.shape)
                or not local.is_contiguous()
                or local.is_meta
                or local.dtype not in (torch.float32, torch.bfloat16)
            ):
                raise ValueError(f"MXFP8 all-gather needs an even dim-0 shard of complete 32-element blocks: {fqn}.")
            if hasattr(local, "fsdp_pre_all_gather") or hasattr(local, "fsdp_post_all_gather"):
                raise ValueError(f"An FSDP all-gather hook is already installed on {fqn}.")
            _require_mxfp8_runtime(None if emulated else local.device)
            pending.append((local, MXFP8AllGather(emulated)))

    if len(pending) != len(selected):
        raise ValueError("Every selected MXFP8 weight must be managed by native FSDP2.")
    for local, hooks in pending:
        local.fsdp_pre_all_gather = MethodType(hooks.pre, local)
        local.fsdp_post_all_gather = MethodType(hooks.post, local)
    return len(pending)
