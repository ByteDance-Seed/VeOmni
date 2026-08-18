import math

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor

from .fsdp2 import clip_grad_norm as fsdp2_clip_grad_norm
from .fsdp2.clip_grad_norm import _finalize_total_norm, _fsdp2_reduce_group
from .parallel_state import get_parallel_state


def _allreduce_ddp_sp_grads(model: torch.nn.Module, parallel_state) -> None:
    """Average DDP-module grads over ``fsdp_group`` (``dp_sp``) when ``sp_size > 1``.

    DDP's ``process_group`` is ``dp_group``. Enabling Ulysses/SP shrinks ``dp``
    (``dp_size = world / sp_size``), so that allreduce is often a no-op while each
    rank still holds only a shard's worth of param grads (token-dim Ulysses, or
    batch-dim Omni SP where a DDP vision tower slices the replicated image batch).
    AVG over ``fsdp_group`` matches FSDP2's effective sync surface — including
    HSDP, where ``dp_sp`` already contains ``dp_replicate``.
    """
    group = parallel_state.fsdp_group
    if group is None or dist.get_world_size(group) <= 1:
        return
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=group)


def veomni_clip_grad_norm(
    model, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None
):
    parallel_state = get_parallel_state()
    dp_mode = parallel_state.dp_mode
    if dp_mode == "fsdp2":
        grad_norm = fsdp2_clip_grad_norm(model, max_norm, norm_type, error_if_nonfinite, foreach)
    elif dp_mode == "ddp":
        if parallel_state.sp_size > 1:
            _allreduce_ddp_sp_grads(model, parallel_state)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, foreach=foreach)
    else:
        raise RuntimeError(f"Unknown dp mode {dp_mode}")

    grad_norm = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
    return grad_norm


def veomni_omni_module_clip_grad_norm(
    model,
    max_norm: float,
    norm_type: float = 2.0,
) -> float:
    """Gradient-norm clipping for a single OmniModule under its own parallelism.

    An ``OmniModule`` may be wrapped as FSDP2, FSDP2 + ExtraParallel, or DDP, so
    the world-complete sum of p-th powers is reduced over the right process group
    for this module's topology, finalized into the module norm, then used to clip
    this module's params:

    * FSDP2: local shard p-th-sum, all-reduce SUM over ``fsdp_group``.
    * FSDP2 + ExtraParallel: non-ExtraParallel params over ``fsdp_group``;
      ExtraParallel params over ``{ep}_fsdp`` then ``{ep}`` (mirrors
      ``extra_parallel_fsdp2_clip_grad_norm``).
    * DDP (``sp_size == 1``): local p-th-sum — grads already all-reduced on
      ``dp_group`` by DDP.
    * DDP + SP (``sp_size > 1``): first average grads over ``fsdp_group``
      (see :func:`_allreduce_ddp_sp_grads`), then local p-th-sum.

    The reduced scalar is identical across ranks, so the returned norm is
    rank-consistent.
    """
    norm_type = float(norm_type)
    ps = get_parallel_state()
    if ps.dp_mode == "ddp" and ps.sp_size > 1:
        _allreduce_ddp_sp_grads(model, ps)

    pth_sums: list[torch.Tensor] = []
    groups_to_clip: list[list[torch.nn.Parameter]] = []

    ep_param_groups = getattr(model, "_extra_parallel_param_groups", None)
    if ep_param_groups is not None and ps.any_extra_parallel_enabled:
        non_ep = [p for p in ep_param_groups.get("non_extra_parallel", []) if p.grad is not None]
        if non_ep:
            pth_sums.append(_fsdp2_reduce_group(non_ep, norm_type, [("fsdp", ps.fsdp_group)]))
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
            # FSDP2 DTensor grads are sharded -> SUM local p-th powers over fsdp_group.
            # Plain Tensor grads are full replicas (e.g. Omni connector under a process
            # whose ParallelState is still fsdp2) — do NOT world-SUM those.
            dt_params = [p for p in params if isinstance(p.grad, DTensor)]
            dense_params = [p for p in params if not isinstance(p.grad, DTensor)]
            if dt_params:
                reduce_groups = [("fsdp", ps.fsdp_group)] if ps.dp_mode == "fsdp2" else []
                pth_sums.append(_fsdp2_reduce_group(dt_params, norm_type, reduce_groups))
                groups_to_clip.append(dt_params)
            if dense_params:
                pth_sums.append(_fsdp2_reduce_group(dense_params, norm_type, []))
                groups_to_clip.append(dense_params)

    if not pth_sums:
        return 0.0

    if math.isinf(norm_type):
        total_norm = torch.stack(pth_sums).amax()
    else:
        total_norm = _finalize_total_norm(torch.stack(pth_sums).sum(), norm_type)

    for params in groups_to_clip:
        torch.nn.utils.clip_grads_with_norm_(params, max_norm, total_norm)

    return total_norm.item()


def omni_clip_grad_norm(
    module_runtimes: dict,
    max_grad_norm: float,
    grad_clip_scope: str = "per_module",
) -> float:
    """Clip grads across OmniModule runtimes according to ``grad_clip_scope``.

    * ``per_module`` (default): each module clips against **its own**
      ``args.optimizer.max_grad_norm`` — every OmniModule carries its own
      optimizer config, so *max_grad_norm* here is only the model-level value
      they inherit from. Returns ``sqrt(sum n_i^2)`` of the per-module (pre-clip)
      norms for logging.
    * ``global``: measure each module with ``max_norm=inf`` (no scale),
      ``total = sqrt(sum n_i^2)``, then if ``total > max_grad_norm`` scale **all**
      module grads by one coefficient — single-model / seedream
      ``gradient_clip_val`` semantics. A single threshold is inherent to this
      scope, so the per-module values do not apply.

    Each ``_clip_grad_norm`` enters the module's own ``ParallelState``; the
    ``global`` rescale re-enters it via ``_scoped()`` for the same reason.
    """
    runtimes = list(module_runtimes.values()) if isinstance(module_runtimes, dict) else list(module_runtimes)
    if not runtimes:
        return 0.0

    scope = grad_clip_scope or "per_module"
    if scope == "per_module":
        module_norms = [rt._clip_grad_norm(rt.args.optimizer.max_grad_norm) for rt in runtimes]
        return math.sqrt(sum(g * g for g in module_norms))

    if scope != "global":
        raise ValueError(f"Unknown grad_clip_scope={scope!r}; expected 'per_module' or 'global'")

    module_norms = [rt._clip_grad_norm(float("inf")) for rt in runtimes]
    total = math.sqrt(sum(g * g for g in module_norms))
    if max_grad_norm is not None and max_grad_norm > 0 and total > float(max_grad_norm):
        coeff = float(max_grad_norm) / (total + 1e-6)
        for rt in runtimes:
            with rt._scoped():
                for p in rt.model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(coeff)
    return total


def global_clip_grad_norm_modules(modules, max_grad_norm: float) -> float:
    """Global L2 clip over a list of ``nn.Module`` (align harness / unit tests).

    Same math as ``omni_clip_grad_norm(..., grad_clip_scope='global')`` but for
    plain modules that share the current :func:`get_parallel_state` (no per-module
    trainer scope).
    """
    mods = [m for m in modules if m is not None]
    norms = [float(veomni_omni_module_clip_grad_norm(m, float("inf"))) for m in mods]
    total = math.sqrt(sum(g * g for g in norms)) if norms else 0.0
    if max_grad_norm is not None and max_grad_norm > 0 and total > float(max_grad_norm):
        coeff = float(max_grad_norm) / (total + 1e-6)
        for m in mods:
            for p in m.parameters():
                if p.grad is not None:
                    p.grad.mul_(coeff)
    return total
