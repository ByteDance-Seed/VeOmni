import math
from typing import List

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.utils._foreach_utils import _device_has_foreach_support, _has_foreach_support

from ...utils.device import get_device_type
from ...utils.logging import get_logger
from ..parallel_state import get_parallel_state


logger = get_logger(__name__)
_LOCAL_NORM_CHUNK_SIZE = 128


def clip_grad_norm(
    model, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None
) -> torch.Tensor:
    if hasattr(model, "_extra_parallel_param_groups"):
        return extra_parallel_fsdp2_clip_grad_norm(
            model,
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )

    if getattr(model, "_fsdp_cpu_offload_enabled", False):
        return _cpu_offload_fsdp2_clip_grad_norm(
            model,
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm,
        norm_type=norm_type,
        error_if_nonfinite=error_if_nonfinite,
        foreach=foreach,
    )
    if isinstance(grad_norm, DTensor):
        grad_norm = grad_norm.full_tensor()
    return grad_norm


def _fsdp_grad_norm_reduce_groups(ps) -> List[tuple[str, dist.ProcessGroup | None]]:
    """Return process groups that own distinct FSDP gradient shards."""
    if ps.dp_mode != "fsdp2":
        return []
    if not ps.dp_replicate_enabled:
        return [("fsdp", ps.fsdp_group)]
    # In HSDP, replicas already hold equivalent gradients after backward; reduce
    # only across the shard side of the FSDP mesh.
    if ps.dp_shard_sp_enabled:
        # When SP participates in FSDP sharding, the shard group is dp_shard_sp.
        return [("fsdp_shard_sp", ps.dp_shard_sp_group)]
    if ps.dp_shard_size > 1:
        return [("fsdp_shard", ps.dp_shard_group)]
    return []


@torch.no_grad()
def _cpu_offload_fsdp2_clip_grad_norm(
    model, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None
) -> torch.Tensor:
    ps = get_parallel_state()
    params = [p for p in model.parameters() if p.grad is not None]
    total_norm_or_pth_sum = _fsdp2_reduce_group(
        params=params,
        norm_type=norm_type,
        reduce_groups=_fsdp_grad_norm_reduce_groups(ps),
    )
    total_norm = _finalize_total_norm(total_norm_or_pth_sum, norm_type)
    _raise_if_nonfinite(total_norm, norm_type, error_if_nonfinite)

    torch.nn.utils.clip_grads_with_norm_(params, max_norm, total_norm, foreach=foreach)

    return total_norm


@torch.no_grad()
def extra_parallel_fsdp2_clip_grad_norm(
    model, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None
) -> torch.Tensor:
    """
    ExtraParallel-aware gradient clipping for composable FSDP2:

    - Compute local norms for non-ExtraParallel and ExtraParallel parameter groups separately.
    - For finite p: sum p-th powers across the appropriate groups, then take 1/p.
      • non-ExtraParallel: all-reduce over FSDP group.
      • ExtraParallel: all-reduce over Para-FSDP (e.g. ep_fsdp, emb_fsdp) group, then over Para (e.g. ep, emb) group.
    - For inf-norm: take elementwise MAX with the same reduction groups (MAX).
    - Use a single global clip coefficient for both groups.
    """
    ps = get_parallel_state()
    extra_parallel_group = {
        para: ps.extra_parallel_group(para) if ps.extra_parallel_enabled(para) else None
        for para in ps.extra_parallel_names
    }
    # For Para (e.g. ep, emb) params sharded by FSDP2 along hidden dimension
    extra_parallel_fsdp_group = {
        para: ps.extra_parallel_fsdp_device_mesh[para][f"{para}_fsdp"].get_group()
        if ps.extra_parallel_enabled(para) and ps.extra_parallel_fsdp_device_mesh[para] is not None
        else None
        for para in ps.extra_parallel_names
    }

    # Build param groups for ExtraParallel params and non-ExtraParallel params (filter out params without grads)
    extra_parallel_params = {
        para: [p for p in model._extra_parallel_param_groups.get(para, []) if p.grad is not None]
        for para in ps.extra_parallel_names
    }
    non_extra_parallel_params: List[torch.nn.Parameter] = [
        p for p in model._extra_parallel_param_groups.get("non_extra_parallel", []) if p.grad is not None
    ]

    # Compute and reduce non-ExtraParallel
    non_extra_parallel_total = _fsdp2_reduce_group(
        params=non_extra_parallel_params,
        norm_type=norm_type,
        reduce_groups=_fsdp_grad_norm_reduce_groups(ps),
    )
    logger.debug_rank0(f"non_extra_parallel total grad norm: {non_extra_parallel_total}")

    for para in ps.extra_parallel_names:
        logger.debug_rank0(
            f"{para}_params reduces groups: {extra_parallel_fsdp_group[para]=}, {extra_parallel_group[para]=}"
        )

    # Compute and reduce ExtraParallel: first across para_fsdp (e.g. ep_fsdp, emb_fsdp), then across para (e.g. ep, emb)
    extra_parallel_total = {
        para: torch.tensor(0.0, device=torch.device(get_device_type()), dtype=torch.float32)
        for para in ps.extra_parallel_names
    }
    for para in ps.extra_parallel_names:
        if len(extra_parallel_params[para]) > 0:
            para_total = _fsdp2_reduce_group(
                params=extra_parallel_params[para],
                norm_type=norm_type,
                reduce_groups=[
                    (f"{para}_fsdp", extra_parallel_fsdp_group[para]),
                    (f"{para}", extra_parallel_group[para]),
                ],
            )
            extra_parallel_total[para] = para_total
            logger.debug_rank0(f"{para} total grad norm: {para_total}")

    if math.isinf(norm_type):
        total_norm = torch.maximum(non_extra_parallel_total, *extra_parallel_total.values())
    else:
        total_norm = _finalize_total_norm(non_extra_parallel_total + sum(extra_parallel_total.values()), norm_type)

    _raise_if_nonfinite(total_norm, norm_type, error_if_nonfinite)

    # Apply the same clip coefficient to both groups
    for para in ps.extra_parallel_names:
        torch.nn.utils.clip_grads_with_norm_(extra_parallel_params[para], max_norm, total_norm, foreach=foreach)
    torch.nn.utils.clip_grads_with_norm_(non_extra_parallel_params, max_norm, total_norm, foreach=foreach)

    return total_norm


def _finalize_total_norm(total_norm_or_pth_sum: torch.Tensor, norm_type: float) -> torch.Tensor:
    if math.isinf(norm_type):
        return total_norm_or_pth_sum
    return total_norm_or_pth_sum ** (1.0 / float(norm_type))


def _raise_if_nonfinite(total_norm: torch.Tensor, norm_type: float, error_if_nonfinite: bool) -> None:
    if not error_if_nonfinite:
        return
    if bool((~torch.isfinite(total_norm)).item()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from `parameters` is non-finite, "
            "so it cannot be clipped. To disable this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )


def _local_pth_sum(params: List[torch.nn.Parameter], p: float) -> torch.Tensor:
    """Compute the local p-th norm sum with bounded fp32 grad materialization."""
    reduce_device = torch.device(get_device_type())
    res = torch.tensor(0.0, device=reduce_device, dtype=torch.float32)

    # Materializing every grad as fp32 at once can OOM on large models, so keep
    # foreach acceleration local to bounded same-device chunks.
    chunks: dict[torch.device, list[torch.Tensor]] = {}

    def flush_chunk(device: torch.device) -> None:
        nonlocal res
        chunk = chunks.get(device)
        if not chunk:
            return
        chunk_fp32 = [g.to(torch.float32) for g in chunk]
        if _has_foreach_support(chunk_fp32, device) or _device_has_foreach_support(device):
            norm_pows = torch._foreach_pow_(torch._foreach_norm(chunk_fp32, p), p)
        else:
            norm_pows = [torch.linalg.vector_norm(g, p).pow(p) for g in chunk_fp32]
        res = res + torch.sum(torch.stack(norm_pows)).to(reduce_device)
        chunk.clear()

    with torch.no_grad():
        for param in params:
            g = param.grad
            if g is None:
                continue
            if isinstance(g, DTensor):
                g = g.to_local()
            g = g.detach()
            chunk = chunks.setdefault(g.device, [])
            chunk.append(g)
            if len(chunk) >= _LOCAL_NORM_CHUNK_SIZE:
                flush_chunk(g.device)

        for device in list(chunks):
            flush_chunk(device)
    return res


def _local_max(params: List[torch.nn.Parameter]) -> torch.Tensor:
    reduce_device = torch.device(get_device_type())
    mx = None
    for q in params:
        g = q.grad
        if g is None:
            continue
        if isinstance(g, DTensor):
            g_local = g.to_local()
        else:
            g_local = g
        if mx is None:
            mx = torch.tensor(0.0, device=reduce_device, dtype=torch.float32)
        gn = torch.max(torch.abs(g_local.detach().to(torch.float32)))
        mx = torch.maximum(mx, gn.to(reduce_device))
    if mx is None:
        mx = torch.tensor(0.0, device=reduce_device, dtype=torch.float32)
    return mx


def _fsdp2_reduce_group(
    params: List[torch.nn.Parameter],
    norm_type: float,
    reduce_groups: List[tuple[str, dist.ProcessGroup | None]],
) -> torch.Tensor:
    """Compute local group statistic and reduce over provided groups.

    For finite p, returns the globally-reduced sum of p-th powers (not the final norm).
    For inf, returns the globally-reduced max.
    """
    if math.isinf(norm_type):
        val = _local_max(params)
        for _, group in reduce_groups:
            if group is not None:
                dist.all_reduce(val, op=dist.ReduceOp.MAX, group=group)
        return val
    else:
        p = float(norm_type)
        val = _local_pth_sum(params, p)
        logger.debug_rank0(f"local total grad norm: {val}. ProcessGroups to sum {reduce_groups}")
        for name, group in reduce_groups:
            if group is not None:
                dist.all_reduce(val, op=dist.ReduceOp.SUM, group=group)
                logger.debug_rank0(f"After Sum of group {name} total grad norm is {val}")
        return val
