"""Shared graph endpoint dispatch helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from ..mixins.base_mixin import BaseMixin
from ..omni_pretrained_model import OmniPreTrainedModel


def unwrap_module_chain(wrapped: nn.Module) -> nn.Module:
    """Strip DDP-style ``.module`` and LoRA ``get_base_model()`` wrappers.

    FSDP2 composes in place, so a bare :class:`BaseMixin` is usually returned
    unchanged.
    """
    seen: set[int] = set()
    current = wrapped
    while id(current) not in seen:
        seen.add(id(current))

        if isinstance(current, BaseMixin):
            return current

        inner = getattr(current, "module", None)
        if isinstance(inner, nn.Module) and inner is not current:
            current = inner
            continue

        advanced = False
        for candidate in (current, inner):
            if candidate is None:
                continue
            get_base = getattr(candidate, "get_base_model", None)
            if not callable(get_base):
                continue
            base = get_base()
            if isinstance(base, nn.Module) and base is not current:
                current = base
                advanced = True
                break
        if advanced:
            continue
        break

    return current


def unwrap_graph_module(wrapped: nn.Module, *, module_name: str) -> nn.Module:
    """Return the raw graph-callable module behind a (possibly wrapped) module.

    ``wrapped`` is the object that must be called so DDP/FSDP hooks run.  The
    common case is a :class:`BaseMixin`, which owns graph endpoint methods and
    the shared :meth:`~veomni.models.seed_omni.mixins.base_mixin.BaseMixin._omni_hook_name`
    registry.  Training call-sites dispatch ``pre_forward`` / ``post_forward``
    through :class:`TrainingModuleMixin`; inference call-sites call the endpoint
    directly, with no hooks (see :func:`.executor.execute_generation_node`).
    FSDP2 is composable and leaves the module itself as the
    callable object; DDP-style wrappers expose the mixin through ``.module``.

    Parallel/acceleration hooks (``customized_build_parallelize_model``,
    ``get_parallel_plan``, …) are **not** part of :class:`BaseMixin`; they
    live on :class:`~veomni.models.seed_omni.accelerator.module_runtime.ModuleRuntime`
    or optional family mixins on the wrapped model.

    A LoRA-wrapped module (:class:`veomni.lora.VeOmniLoraModel`, possibly
    FSDP2-composed in place so it is still that instance, or DDP-wrapped on
    ``.module``) exposes the raw base model through ``get_base_model()``
    (PEFT-aligned ``base_model.model``).  Its ``forward`` chain still bottoms out
    at ``base_model.model.forward``, so the :func:`call_graph_endpoint`
    trampoline (which swaps that module's ``forward``) keeps working — we only
    need to return the inner :class:`BaseMixin` here.

    A ``fsdp_mode: eager`` module (``ModuleRuntime._init_eager_inference`` in
    ``module_runtime.py``) is a bare native :class:`OmniPreTrainedModel` from
    plain ``from_pretrained`` — no FSDP/DDP wrap, no ``BaseMixin``. It is only
    reachable from generation nodes (training always builds the wrapped
    ``BaseMixin`` path — eager is gated behind ``for_inference``), which need
    no ``pre_forward``/``post_forward``, so it dispatches straight through.
    """
    raw = unwrap_module_chain(wrapped)
    if isinstance(raw, (BaseMixin, OmniPreTrainedModel)):
        return raw
    raise TypeError(
        f"Graph module '{module_name}' must be a BaseMixin (or eager "
        f"OmniPreTrainedModel) or wrap one on `.module` / LoRA base; got "
        f"{type(wrapped).__name__} (resolved {type(raw).__name__})."
    )


def call_graph_endpoint(
    wrapped: nn.Module,
    raw: nn.Module,
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


__all__ = ["call_graph_endpoint", "unwrap_graph_module", "unwrap_module_chain"]
