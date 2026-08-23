import os
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor

from .fsdp2 import clip_grad_norm as fsdp2_clip_grad_norm
from .fsdp2.clip_grad_norm import _fsdp_grad_norm_reduce_groups
from .parallel_state import get_parallel_state


_GRAD_PARITY_TRACE_GROUPS = (
    "non_extra_all",
    "gdn_all",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "conv1d",
    "A_log",
    "dt_bias",
    "norm",
    "out_proj",
    "gdn_other",
    "self_attn",
    "router",
    "other_norms",
    "embeddings",
    "dense_other",
)


def _grad_parity_trace_steps() -> int:
    raw = os.environ.get("VEOMNI_GRAD_PARITY_TRACE_STEPS", "0")
    if not raw.isdecimal():
        raise ValueError("VEOMNI_GRAD_PARITY_TRACE_STEPS must be a non-negative base-10 integer")
    return int(raw)


def _gdn_grad_trace_group(name: str) -> str | None:
    if name.startswith("linear_attn."):
        suffix = name.removeprefix("linear_attn.")
    elif ".linear_attn." in name:
        suffix = name.split(".linear_attn.", 1)[1]
    else:
        return None
    for group in (
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_a",
        "in_proj_b",
        "conv1d",
        "A_log",
        "dt_bias",
        "norm",
        "out_proj",
    ):
        if suffix == group or suffix.startswith(f"{group}."):
            return group
    return "gdn_other"


def _dense_grad_trace_category(name: str) -> str:
    if _gdn_grad_trace_group(name) is not None:
        return "gdn_all"
    if ".self_attn." in name or name.startswith("self_attn."):
        return "self_attn"
    if ".mlp.gate." in name or ".router." in name:
        return "router"
    if "norm" in name:
        return "other_norms"
    if "embed_tokens" in name or "lm_head" in name:
        return "embeddings"
    return "dense_other"


@torch.no_grad()
def _collect_gdn_grad_parity_trace(model: Any) -> dict[str, Any]:
    """Collect scalar-only GDN gradient fingerprints before clipping.

    This diagnostic deliberately avoids cloning or casting complete gradients.
    GDN parameters are dense/non-EP parameters, so their local shard statistics
    are reduced over the same FSDP shard groups used by gradient clipping.
    """
    grouped_params = getattr(model, "_extra_parallel_param_groups", None)
    non_extra_ids: set[int] | None = None
    extra_ids: set[int] = set()
    if grouped_params is not None:
        non_extra_ids = {id(param) for param in grouped_params.get("non_extra_parallel", [])}
        for group_name, params in grouped_params.items():
            if group_name != "non_extra_parallel":
                extra_ids.update(id(param) for param in params)

    named: list[tuple[str, torch.Tensor]] = []
    live_gdn = 0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        gdn_group = _gdn_grad_trace_group(name)
        is_non_extra = non_extra_ids is None or id(param) in non_extra_ids
        if gdn_group is not None and (id(param) in extra_ids or not is_non_extra):
            raise RuntimeError(f"GDN gradient trace expected a non-extra FSDP parameter, got {name!r}")
        if not is_non_extra:
            continue
        grad = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
        if grad.numel() == 0:
            continue
        named.append((name, grad.detach()))
        live_gdn += int(gdn_group is not None)

    if live_gdn == 0:
        raise RuntimeError("GDN gradient trace found no live '.linear_attn.' gradients")

    reference = named[0][1]
    device = reference.device
    sum_stats = torch.zeros((len(_GRAD_PARITY_TRACE_GROUPS), 3), device=device, dtype=torch.float32)
    max_abs = torch.zeros(len(_GRAD_PARITY_TRACE_GROUPS), device=device, dtype=torch.float32)
    group_indices = {group: index for index, group in enumerate(_GRAD_PARITY_TRACE_GROUPS)}
    for name, grad in named:
        norm_sq = torch.linalg.vector_norm(grad, ord=2, dtype=torch.float32).square()
        signed_sum = grad.sum(dtype=torch.float32)
        local_abs_max = torch.maximum(grad.amin().float().abs(), grad.amax().float().abs())
        targets = ["non_extra_all", _dense_grad_trace_category(name)]
        gdn_group = _gdn_grad_trace_group(name)
        if gdn_group is not None:
            targets.append(gdn_group)
        for group in targets:
            index = group_indices[group]
            sum_stats[index, 0].add_(norm_sq)
            sum_stats[index, 1].add_(signed_sum)
            sum_stats[index, 2].add_(float(grad.numel()))
            max_abs[index] = torch.maximum(max_abs[index], local_abs_max)

    for _, group in _fsdp_grad_norm_reduce_groups(get_parallel_state()):
        if group is not None:
            dist.all_reduce(sum_stats, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(max_abs, op=dist.ReduceOp.MAX, group=group)

    sum_stats_cpu = sum_stats.cpu()
    max_abs_cpu = max_abs.cpu()
    metrics: dict[str, dict[str, float | int]] = {}
    for index, group in enumerate(_GRAD_PARITY_TRACE_GROUPS):
        count = int(sum_stats_cpu[index, 2].item())
        if count == 0:
            continue
        metrics[group] = {
            "l2": float(sum_stats_cpu[index, 0].sqrt().item()),
            "signed_sum": float(sum_stats_cpu[index, 1].item()),
            "max_abs": float(max_abs_cpu[index].item()),
            "numel": count,
        }
    return {"groups": metrics}


def veomni_clip_grad_norm(
    model, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None
):
    trace_steps = _grad_parity_trace_steps()
    trace_call = int(getattr(model, "_veomni_grad_parity_trace_calls", 0)) + 1
    if trace_steps:
        model._veomni_grad_parity_trace_calls = trace_call
    trace = _collect_gdn_grad_parity_trace(model) if 0 < trace_call <= trace_steps else None

    parallel_state = get_parallel_state()
    dp_mode = parallel_state.dp_mode
    if dp_mode == "fsdp2":
        grad_norm = fsdp2_clip_grad_norm(model, max_norm, norm_type, error_if_nonfinite, foreach)
    elif dp_mode == "ddp":
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, foreach=foreach)
    else:
        raise RuntimeError(f"Unknown dp mode {dp_mode}")

    grad_norm = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
    if trace is not None:
        trace.update({"call": trace_call, "norm_type": float(norm_type), "pre_clip_total_grad_norm": grad_norm})
        non_extra_l2 = trace["groups"]["non_extra_all"]["l2"]
        if float(norm_type) == 2.0:
            # The production clipper computes the global L2 norm as the sum of
            # non-extra and extra-parallel squared norms.  Derive the missing
            # bucket without another EP collective or touching full gradients.
            trace["extra_parallel_inferred_l2"] = max(grad_norm**2 - non_extra_l2**2, 0.0) ** 0.5
        model._veomni_grad_parity_trace = trace
    return grad_norm
