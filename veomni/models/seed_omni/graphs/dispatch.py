"""Shared graph endpoint dispatch helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from ..mixins.modulemixin import ModuleMixin


def unwrap_graph_module(wrapped: nn.Module, *, module_name: str) -> ModuleMixin:
    """Return the raw SeedOmni module behind a graph-callable module.

    ``wrapped`` is the object that must be called so DDP/FSDP hooks run.  The
    raw :class:`ModuleMixin` owns graph endpoint methods and pre/post hooks.
    FSDP2 is composable and leaves the module itself as the callable object;
    DDP-style wrappers expose the raw module through ``.module``.
    """
    if isinstance(wrapped, ModuleMixin):
        return wrapped

    raw = getattr(wrapped, "module", None)
    if isinstance(raw, ModuleMixin):
        return raw

    raise TypeError(
        f"Graph module '{module_name}' must be a ModuleMixin or wrap one on `.module`; got {type(wrapped).__name__}."
    )


def call_graph_endpoint(
    wrapped: nn.Module,
    raw: ModuleMixin,
    *,
    method: str,
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Call a graph endpoint through ``wrapped.__call__``.

    Non-``forward`` graph methods are temporarily installed as ``raw.forward``
    so the call still enters via ``wrapped(**kwargs)``.  This preserves graph
    endpoint semantics while allowing wrappers such as FSDP2 to run their
    pre/post-forward hooks.

    The trampoline restores ``raw.forward`` to the module's original forward
    while the endpoint body runs.  This lets endpoint implementations call
    ``self.forward(...)`` to reuse their normal model forward without recursing
    back into the endpoint.  They should still avoid ``self(...)`` inside an
    endpoint: the outer graph dispatch is already running through
    ``wrapped.__call__``.
    """
    if method == "forward":
        return wrapped(**dict(kwargs))

    fn = getattr(raw, method, None)
    if fn is None:
        raise AttributeError(f"Node method {type(raw).__name__}.{method}() is not implemented.")

    original_forward = raw.forward
    endpoint = fn

    def endpoint_forward(*args: Any, **forward_kwargs: Any) -> dict[str, Any]:
        # The outer replacement below makes ``wrapped.__call__`` enter the graph
        # endpoint, which is what lets FSDP/DDP pre-forward hooks run. Once we
        # are inside that endpoint, restore the module's real ``forward`` so
        # endpoint code can call ``self.forward(...)`` and get the model's normal
        # forward instead of the temporary endpoint override.
        raw.forward = original_forward
        try:
            return endpoint(*args, **forward_kwargs)
        finally:
            raw.forward = endpoint_forward

    try:
        raw.forward = endpoint_forward
        return wrapped(**dict(kwargs))
    finally:
        raw.forward = original_forward


def _sp_zero_link(d: Mapping[str, Any]) -> torch.Tensor | None:
    """A zero-magnitude scalar reachable from the first grad-carrying tensor in ``d``."""
    for v in d.values():
        if torch.is_tensor(v) and v.is_floating_point() and v.requires_grad:
            return v.reshape(-1)[:1].sum() * 0.0
    return None


def _sp_add_link(d: Mapping[str, Any], link: torch.Tensor) -> dict[str, Any]:
    """Add ``link`` (scalar 0.0) to the first grad-carrying tensor in ``d`` (copy)."""
    out = dict(d)
    for key, v in out.items():
        if torch.is_tensor(v) and v.is_floating_point() and v.requires_grad:
            out[key] = v + link
            break
    return out


def _sp_broadcast_sample(kwargs: Mapping[str, Any], src_group_rank: int, group: Any) -> dict[str, Any]:
    """Broadcast one sample's ``pre_forward`` return to the whole SP group.

    Every tensor value is broadcast from ``src_group_rank`` (autograd-aware); after
    this, all ranks hold that sample's exact ``pre_forward`` output. Non-tensor
    values are passed through unchanged — modules must derive anything
    sample-specific (e.g. varlen ``cu_seqlens`` / ``max_length``) from the
    broadcast tensors inside ``sp_pre_forward`` rather than relying on those
    pass-through leftovers.
    """
    from veomni.distributed.sequence_parallel import sp_broadcast_from_rank

    return {
        k: (sp_broadcast_from_rank(v, src_group_rank=src_group_rank, group=group) if torch.is_tensor(v) else v)
        for k, v in kwargs.items()
    }


def _ddp_wrapped(module: nn.Module) -> bool:
    """True if ``module`` is a ``DistributedDataParallel`` wrapper."""
    return module.__class__.__name__ == "DistributedDataParallel"


def run_sp_looped_endpoint(
    wrapped: nn.Module,
    raw: ModuleMixin,
    *,
    method: str,
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Drive a module's per-module SP endpoint over the SP group's sample list.

    The orchestrator pins outer SP to 1, so every rank holds a DISTINCT packed
    sample. Instead of concatenating the SP group's ``sp_size`` samples into one
    forward (an ``sp_size``× activation spike), materialise the group's samples
    one at a time and run the same loop on every rank::

        for sample_idx in range(sp_size):          # identical order on all ranks
            data      = broadcast(rank sample_idx's pre_forward)  # ≡ list[all-gather][i]
            sp_kwargs = raw.sp_pre_forward(**data)  # Ulysses / batch slice
            out       = <plain forward>
            out       = raw.sp_post_forward(**out)  # all-gather shards → full sample

    Every rank therefore builds the **same** autograd topology (broadcast →
    slice → forward → all-gather) per sample. Returning only this rank's sample
    for ``post_forward`` / the conversation carrier, while folding a fixed-order
    zero-magnitude link over *all* sample outs, keeps every iteration reachable
    from the loss with a rank-identical backward collective order.

    **DDP:** the loop issues ``sp_size`` forwards before one backward. DDP's
    reducer is built for a single forward/backward pair and mis-handles that
    pattern (observed: some SP ranks get all-zero grads). Call the raw module
    inside the loop and let :func:`veomni_omni_module_clip_grad_norm` all-reduce
    grads over ``fsdp_group`` (``dp_sp``) after backward — same sync surface as
    FSDP2. FSDP2 stays on ``wrapped`` so its per-forward hooks still run.

    Callers MAY wrap this in ``no_reshard_after_forward(wrapped)`` so the
    ``sp_size`` forwards all-gather FSDP2 params once — opt-in via
    ``train.accelerator.fsdp_config.sp_keep_params_unsharded``.
    """
    from veomni.distributed.parallel_state import get_parallel_state

    ps = get_parallel_state()
    sp_size = ps.sp_size
    my_rank = ps.sp_rank
    group = ps.sp_group
    # Bypass DDP hooks for the multi-forward burst (see docstring). FSDP2 /
    # unwrapped modules keep ``wrapped``.
    endpoint_module = raw if _ddp_wrapped(wrapped) else wrapped

    outs: list[dict[str, Any]] = []
    for sample_idx in range(sp_size):
        sample_kwargs = _sp_broadcast_sample(kwargs, src_group_rank=sample_idx, group=group)
        sp_kwargs = raw.sp_pre_forward(method=method, **sample_kwargs)
        out = call_graph_endpoint(endpoint_module, raw, method=method, kwargs=sp_kwargs)
        out = raw.sp_post_forward(method=method, **out)
        outs.append(out)

    # Carrier / post_forward only needs this rank's sample, but every rank must
    # attach the *same* zero-link chain (sample 0 … sp_size-1) so backward visits
    # collectives in lockstep.
    local_out = dict(outs[my_rank])
    for sample_out in outs:
        link = _sp_zero_link(sample_out)
        if link is not None:
            local_out = _sp_add_link(local_out, link)
    return local_out


__all__ = ["call_graph_endpoint", "run_sp_looped_endpoint", "unwrap_graph_module"]
