"""Shared graph endpoint dispatch helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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


__all__ = ["call_graph_endpoint", "unwrap_graph_module"]
