import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor

from .fsdp2 import clip_grad_norm as fsdp2_clip_grad_norm
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
        # Defensive: DDP replicates plain tensors, so a DTensor grad only shows
        # up if TP was applied under it -- a combination the local norm below
        # does not handle either, since it would cover this rank's shard alone.
        # ``dp_sp`` excludes ``tp``, so reducing the local shards is at least the
        # right thing for the part this function owns.
        grad = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
        # SUM + divide rather than ReduceOp.AVG, which the NPU backend rejects.
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=group)
        grad.div_(group_size)


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
