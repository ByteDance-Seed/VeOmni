import os
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor

from ..utils.device import get_device_type
from .fsdp2 import clip_grad_norm as fsdp2_clip_grad_norm
from .fsdp2.clip_grad_norm import _fsdp_grad_norm_reduce_groups
from .parallel_state import ParallelState, get_parallel_state


@torch.no_grad()
def _allreduce_ddp_sp_grads(model: torch.nn.Module, parallel_state: ParallelState) -> None:
    """Average a DDP module's gradients over ``fsdp_group`` (``dp_sp``).

    DDP wraps with ``process_group=dp_group``. Turning on Ulysses shrinks ``dp``
    (``dp_size = world / sp_size``), so at ``dp_size == 1`` DDP's all-reduce is a
    no-op while each rank still holds gradients from only its ``1 / sp_size``
    slice of the sequence — the optimizer would then step on a partial gradient,
    with no shape error to reveal it.

    ``dp_sp`` is exactly FSDP2's effective sync surface (``dp_replicate``
    included, so HSDP is covered too), which is why FSDP2 does not have this bug
    and why averaging over it makes the two dp modes numerically equivalent.
    Averaging over ``dp_sp`` after DDP already averaged over ``dp`` is not a
    double count: every rank of a ``dp`` group holds the same value going in, so
    the second average reproduces the plain mean over all ``dp_sp`` ranks.
    """
    group = parallel_state.fsdp_group
    if group is None:
        return

    group_size = dist.get_world_size(group)
    if group_size <= 1:
        return

    # One collective per gradient, so the set of gradients has to be the same on
    # every rank of the group. Keying it on ``requires_grad`` rather than on
    # ``grad is not None`` keeps that true even when a parameter goes unused on
    # some ranks only: a zero grad is what an unused parameter contributed, and
    # skipping it instead would desynchronize the collectives and hang.
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        # Defensive: DDP replicates plain tensors, so a DTensor grad would only
        # show up if TP were applied under it, and TP is inert today (the wrap
        # passes no ``parallelize_plan``, which makes it a no-op). ``dp_sp``
        # excludes ``tp``, so every rank of the group holds the same TP shard and
        # reducing the local shards is the right thing if that changes.
        grad = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
        # SUM + divide rather than ReduceOp.AVG, which the NPU backend rejects.
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=group)
        grad.div_(group_size)


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
    "mlp_nonrouter",
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
    if ".mlp." in name or name.startswith("mlp."):
        return "mlp_nonrouter"
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
    misplaced_gdn = 0
    for name, param in model.named_parameters():
        gdn_group = _gdn_grad_trace_group(name)
        is_non_extra = non_extra_ids is None or id(param) in non_extra_ids
        if gdn_group is not None and (id(param) in extra_ids or not is_non_extra):
            misplaced_gdn += 1
        if param.grad is None or not is_non_extra:
            continue
        grad = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
        if grad.numel() == 0:
            continue
        named.append((name, grad.detach()))
        live_gdn += int(gdn_group is not None)

    device = named[0][1].device if named else torch.device(get_device_type())
    sum_stats = torch.zeros((len(_GRAD_PARITY_TRACE_GROUPS), 2), device=device, dtype=torch.float32)
    counts = torch.zeros(len(_GRAD_PARITY_TRACE_GROUPS), device=device, dtype=torch.int64)
    max_abs = torch.zeros(len(_GRAD_PARITY_TRACE_GROUPS), device=device, dtype=torch.float32)
    status = torch.tensor((misplaced_gdn, live_gdn), device=device, dtype=torch.int64)
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
            counts[index].add_(grad.numel())
            max_abs[index] = torch.maximum(max_abs[index], local_abs_max)

    for _, group in _fsdp_grad_norm_reduce_groups(get_parallel_state()):
        if group is not None:
            dist.all_reduce(sum_stats, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(max_abs, op=dist.ReduceOp.MAX, group=group)
            dist.all_reduce(status, op=dist.ReduceOp.SUM, group=group)

    # Fail only after every rank has executed the same collectives.  A
    # rank-local exception here would strand peers in the first all-reduce.
    if int(status[0].item()) != 0:
        raise RuntimeError("GDN gradient trace expected all GDN parameters in the non-extra FSDP bucket")
    if int(status[1].item()) == 0:
        raise RuntimeError("GDN gradient trace found no live '.linear_attn.' gradients")

    sum_stats_cpu = sum_stats.cpu()
    counts_cpu = counts.cpu()
    max_abs_cpu = max_abs.cpu()
    metrics: dict[str, dict[str, float | int]] = {}
    for index, group in enumerate(_GRAD_PARITY_TRACE_GROUPS):
        count = int(counts_cpu[index].item())
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
    parallel_state = get_parallel_state()
    dp_mode = parallel_state.dp_mode
    if dp_mode == "ddp" and parallel_state.sp_size > 1:
        # Match FSDP2 semantics before collecting optional diagnostics: the
        # trace must observe the globally synchronized gradient, not one
        # sequence-parallel rank's partial contribution.
        _allreduce_ddp_sp_grads(model, parallel_state)

    trace_steps = _grad_parity_trace_steps()
    trace_call = int(getattr(model, "_veomni_grad_parity_trace_calls", 0)) + 1
    if trace_steps:
        model._veomni_grad_parity_trace_calls = trace_call
    trace = _collect_gdn_grad_parity_trace(model) if 0 < trace_call <= trace_steps else None

    if dp_mode == "fsdp2":
        grad_norm = fsdp2_clip_grad_norm(model, max_norm, norm_type, error_if_nonfinite, foreach)
    elif dp_mode == "ddp":
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, foreach=foreach)
        if isinstance(grad_norm, DTensor):
            # TP grads would make the norm a DTensor too, and ``.item()`` below
            # reads a DTensor's local shard rather than the global value. The
            # clipping itself is already global -- ``clip_grad_norm_`` applies it
            # internally on DTensors -- so this corrects only what gets reported.
            # Same conversion the fsdp2 path makes before returning.
            grad_norm = grad_norm.full_tensor()
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
