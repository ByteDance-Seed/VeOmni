import math
from typing import List

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)

from ...utils.device import get_device_type
from ...utils.logging import get_logger
from ..parallel_state import get_parallel_state


logger = get_logger(__name__)


def clip_grad_norm(
    model, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None
) -> torch.Tensor:
    # ExtraParallel-aware path (FSDP2 + ExtraParallel): maintain mathematical parity with FSDP1 clipper

    if hasattr(model, "_extra_parallel_param_groups"):
        return extra_parallel_fsdp2_clip_grad_norm(
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


@torch.no_grad()
def extra_parallel_fsdp2_clip_grad_norm(
    model, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None
) -> torch.Tensor:
    """
    ExtraParallel-aware gradient clipping for composable FSDP2 with reductions mirroring FSDP1:

    - Compute local norms for non-ExtraParallel and ExtraParallel parameter groups separately.
    - For finite p: sum p-th powers across the appropriate groups, then take 1/p.
      • non-ExtraParallel: all-reduce over FSDP group.
      • ExtraParallel: all-reduce over Para-FSDP (e.g. ep_fsdp, emb_fsdp) group, then over Para (e.g. ep, emb) group.
    - For inf-norm: take elementwise MAX with the same reduction groups (MAX).
    - Use a single global clip coefficient for both groups.
    """
    ps = get_parallel_state()
    fsdp_group = ps.fsdp_group
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
        reduce_groups=[("fsdp", fsdp_group)],
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
        total_norm = (non_extra_parallel_total + sum(extra_parallel_total.values())) ** (1.0 / float(norm_type))

    # Apply the same clip coefficient to both groups
    for para in ps.extra_parallel_names:
        torch.nn.utils.clip_grads_with_norm_(extra_parallel_params[para], max_norm, total_norm, foreach=foreach)
    torch.nn.utils.clip_grads_with_norm_(non_extra_parallel_params, max_norm, total_norm, foreach=foreach)

    return total_norm


# compute local sum of param gard norm
def _local_pth_sum(params: List[torch.nn.Parameter], p: float) -> torch.Tensor:
    grads = [p.grad for p in params if p.grad is not None]
    grads_local = [
        g.to_local().detach().to(torch.float32) if isinstance(g, DTensor) else g.detach().to(torch.float32)
        for g in grads
    ]

    default_device = grads_local[0].device if len(grads_local) > 0 else torch.device(get_device_type())
    res = torch.tensor(0.0, device=default_device, dtype=torch.float32)
    with torch.no_grad():
        grouped_grads_local = _group_tensors_by_device_and_dtype([grads_local])
        for (device, _), ([device_grads_local], _) in grouped_grads_local.items():
            if _has_foreach_support(device_grads_local, device) or _device_has_foreach_support(device):
                out = torch._foreach_pow_(torch._foreach_norm(device_grads_local, p), p)
                res += torch.sum(torch.stack(out)).to(default_device)
            else:
                for grad_local in device_grads_local:
                    gn = torch.norm(grad_local, p=p)
                    res = res + (gn**p).to(default_device)
    return res


def _local_max(params: List[torch.nn.Parameter]) -> torch.Tensor:
    dev = None
    mx = None
    for q in params:
        g = q.grad
        if g is None:
            continue
        if isinstance(g, DTensor):
            g_local = g.to_local()
        else:
            g_local = g
        if dev is None:
            dev = g_local.device
            mx = torch.tensor(0.0, device=dev, dtype=torch.float32)
        gn = torch.max(torch.abs(g_local.detach().to(torch.float32)))
        mx = torch.maximum(mx, gn)
    if mx is None:
        dev = torch.device(get_device_type())
        mx = torch.tensor(0.0, device=dev, dtype=torch.float32)
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
