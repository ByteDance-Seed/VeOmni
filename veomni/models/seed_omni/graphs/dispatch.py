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


def _sp_broadcast_owner(kwargs: Mapping[str, Any], src_group_rank: int, group: Any) -> dict[str, Any]:
    """Broadcast one owner's ``pre_forward`` return to the whole SP group.

    Every tensor value is broadcast from ``src_group_rank`` (autograd-aware); after
    this, all ranks hold the owner's exact ``pre_forward`` output. Non-tensor values
    are passed through unchanged — modules must derive anything owner-specific
    (e.g. varlen ``cu_seqlens`` / ``max_length``) from the broadcast tensors inside
    ``sp_pre_forward`` rather than relying on those pass-through leftovers.
    """
    from veomni.distributed.sequence_parallel import sp_broadcast_from_owner

    return {
        k: (sp_broadcast_from_owner(v, src_group_rank=src_group_rank, group=group) if torch.is_tensor(v) else v)
        for k, v in kwargs.items()
    }


def run_sp_looped_endpoint(
    wrapped: nn.Module,
    raw: ModuleMixin,
    *,
    method: str,
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Drive a module's per-module SP endpoint one SP-group member at a time.

    The orchestrator pins outer SP to 1, so every rank owns a DISTINCT packed
    sample. Instead of concatenating the SP group's ``sp_size`` samples into one
    forward (an ``sp_size``× activation spike), loop over the group and process
    ONE owner's sample per iteration (micro-batch stays a single sample)::

        for owner in range(sp_size):
            data      = broadcast(owner's pre_forward output to the group)  # generic, here
            sp_kwargs = raw.sp_pre_forward(**data)  # SLICE the shared data (module hook)
            out       = <plain forward>             # modeling stays SP-unaware
            out       = raw.sp_post_forward(owner)  # gather shard back to owner (module hook)

    then the graph runs ``post_forward`` once on this rank's OWN-owner output. The
    broadcast is done HERE (whole-dict, axis-agnostic) so ``sp_pre_forward`` just
    receives data every rank already shares and only slices it; the gather stays in
    ``sp_post_forward`` because its concat axis is module-specific.

    A zero-magnitude dependency link is threaded iteration→iteration (and folded
    into the returned own-output). It keeps every iteration reachable from the loss
    on EVERY rank and pins the backward collective order (iteration ``sp_size-1`` →
    ``0``) identically across ranks — the per-owner broadcast/gather collectives
    must fire in matching order or they deadlock. Its value and gradient are
    exactly zero. Callers MAY wrap this in ``no_reshard_after_forward(wrapped)`` so
    the ``sp_size`` forwards all-gather FSDP2 params once — an opt-in memory/comm
    tradeoff the graph gates on the YAML knob ``train.accelerator.fsdp_config.
    sp_keep_params_unsharded`` (a large backbone kept unsharded for the whole burst
    can OOM).
    """
    from veomni.distributed.parallel_state import get_parallel_state

    ps = get_parallel_state()
    sp_size = ps.sp_size
    my_rank = ps.sp_rank
    group = ps.sp_group

    own_out: dict[str, Any] | None = None
    link: torch.Tensor | None = None
    for owner in range(sp_size):
        owner_kwargs = _sp_broadcast_owner(kwargs, src_group_rank=owner, group=group)
        sp_kwargs = raw.sp_pre_forward(method=method, **owner_kwargs)
        if link is not None:
            sp_kwargs = _sp_add_link(sp_kwargs, link)
        out = call_graph_endpoint(wrapped, raw, method=method, kwargs=sp_kwargs)
        out = raw.sp_post_forward(method=method, owner=owner, **out)
        link = _sp_zero_link(out)
        if owner == my_rank:
            own_out = out

    assert own_out is not None, "SP loop produced no own-owner output (sp_size < 1?)."
    if link is not None:
        own_out = _sp_add_link(own_out, link)
    return own_out


__all__ = ["call_graph_endpoint", "run_sp_looped_endpoint", "unwrap_graph_module"]
