import math

import torch

from .fsdp2 import clip_grad_norm as fsdp2_clip_grad_norm
from .fsdp2.clip_grad_norm import _finalize_total_norm, _fsdp2_reduce_group, _fsdp_grad_norm_reduce_groups
from .parallel_state import get_parallel_state


def veomni_clip_grad_norm(
    model, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None
):
    parallel_state = get_parallel_state()
    dp_mode = parallel_state.dp_mode
    if dp_mode == "fsdp2":
        grad_norm = fsdp2_clip_grad_norm(model, max_norm, norm_type, error_if_nonfinite, foreach)
    elif dp_mode == "ddp":
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, foreach=foreach)
    else:
        raise RuntimeError(f"Unknown dp mode {dp_mode}")

    grad_norm = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
    return grad_norm


def veomni_omni_module_clip_grad_norm(
    model,
    parallel_state,
    max_norm: float,
    norm_type: float = 2.0,
) -> float:
    """Gradient-norm clipping for a single OmniModule under its own parallelism.

    An ``OmniModule`` may be wrapped as FSDP2, FSDP2 + ExtraParallel, or DDP, so
    the world-complete sum of pᵗʰ-powers is reduced over the right process group
    for this module's topology, finalized into the module norm, then used to clip
    this module's params:

    * FSDP2: local shard pᵗʰ-sum, all-reduce SUM over the FSDP grad-norm group
      (``dp_shard`` under HSDP, otherwise ``fsdp_group``).
    * FSDP2 + ExtraParallel: non-ExtraParallel params over ``fsdp_group``;
      ExtraParallel params over ``{ep}_fsdp`` then ``{ep}`` (mirrors
      ``extra_parallel_fsdp2_clip_grad_norm``).
    * DDP: local pᵗʰ-sum, **no** reduction — grads are replicated and already
      all-reduced across the DP group by DDP's backward, so each rank's value is
      the full module norm.

    The reduced scalar is identical across ranks, so the returned norm is
    rank-consistent.
    """
    norm_type = float(norm_type)
    ps = parallel_state
    pth_sums: list[torch.Tensor] = []
    groups_to_clip: list[list[torch.nn.Parameter]] = []

    ep_param_groups = getattr(model, "_extra_parallel_param_groups", None)
    if ep_param_groups is not None and ps.any_extra_parallel_enabled:
        non_ep = [p for p in ep_param_groups.get("non_extra_parallel", []) if p.grad is not None]
        if non_ep:
            pth_sums.append(_fsdp2_reduce_group(non_ep, norm_type, _fsdp_grad_norm_reduce_groups(ps)))
            groups_to_clip.append(non_ep)
        for para in ps.extra_parallel_names:
            if not ps.extra_parallel_enabled(para):
                continue
            ep_params = [p for p in ep_param_groups.get(para, []) if p.grad is not None]
            if not ep_params:
                continue
            ep_fsdp_group = ps.extra_parallel_fsdp_device_mesh[para][f"{para}_fsdp"].get_group()
            pth_sums.append(
                _fsdp2_reduce_group(
                    ep_params,
                    norm_type,
                    [(f"{para}_fsdp", ep_fsdp_group), (para, ps.extra_parallel_group(para))],
                )
            )
            groups_to_clip.append(ep_params)
    else:
        params = [p for p in model.parameters() if p.grad is not None]
        if params:
            # FSDP2 grads are sharded -> reduce local shard sums over the grad-norm group.
            # DDP grads are replicated and already all-reduced -> no further reduce.
            reduce_groups = _fsdp_grad_norm_reduce_groups(ps)
            pth_sums.append(_fsdp2_reduce_group(params, norm_type, reduce_groups))
            groups_to_clip.append(params)

    if not pth_sums:
        return 0.0

    if math.isinf(norm_type):
        total_norm = torch.stack(pth_sums).amax()
    else:
        total_norm = _finalize_total_norm(torch.stack(pth_sums).sum(), norm_type)

    for params in groups_to_clip:
        torch.nn.utils.clip_grads_with_norm_(params, max_norm, total_norm)

    return total_norm.item()
