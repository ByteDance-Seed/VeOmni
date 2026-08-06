"""VeOmni node executors for SeedOmni graphs.

The graphs only *select* which node runs next; these functions are the
**VeOmni executor** — they unwrap FSDP/DDP wrappers, scope each module's
:class:`ParallelState`, optionally profile the node, and run the training
pre/forward/post chain (including metric metering) or the eager generation
endpoint.

For a profiler-free eager path (HF ``from_pretrained`` / single-process), use
:class:`~veomni.models.seed_omni.modeling_omni.OmniModel` directly — its
``forward`` / ``generate`` call ``pre_forward`` → method → ``post_forward``
without any of the machinery here.
"""

from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Dict, Optional

from ..graphs.base import NodeDef
from ..utils.graph_profiler import GraphProfiler
from .dispatch import call_graph_endpoint, unwrap_graph_module


def execute_train_node(
    modules: Dict[str, Any],
    node: NodeDef,
    batch: Dict[str, Any],
    *,
    profiler: Optional[GraphProfiler] = None,
    scope_fn: Optional[Callable[[str], ContextManager]] = None,
) -> Dict[str, Any]:
    """Run one training node under VeOmni (unwrap + scope + profile + meter).

    Resolves ``node.module``, scopes its :class:`ParallelState` (via ``scope_fn``
    — vocab-parallel ``emb`` / MoE EP groups), and runs ``pre_forward`` → method
    → ``post_forward``, dispatching through the **wrapped** module so DDP/FSDP
    hooks fire (non-``forward`` methods via the ``call_graph_endpoint``
    trampoline).     ``raw`` (the graph hook owner) owns ``pre_forward`` / ``post_forward``;
    FSDP2 is in-place (``raw is wrapped``) while DDP wraps (``raw =
    wrapped.module``).

    Returns the (mutated) ``batch``.
    """
    method = node.method
    wrapped = modules.get(node.module)
    if wrapped is None:
        raise KeyError(
            f"execute_train_node: module '{node.module}' (node '{node.name}') missing "
            f"from modules dict. Provided: {sorted(modules)}."
        )
    raw = unwrap_graph_module(wrapped, module_name=node.module)

    module_context = scope_fn(node.module) if scope_fn is not None else nullcontext()
    profile_context = profiler.node(f"forward:{node.name}") if profiler is not None else nullcontext()
    with module_context, profile_context:
        kwargs = raw.pre_forward(method=method, **batch)

        if hasattr(raw, "metric_meter_add"):
            raw.metric_meter_add(method, kwargs)

        out = call_graph_endpoint(wrapped, raw, method=method, kwargs=kwargs)
        out = raw.post_forward(method=method, **out)

    batch.update(out)
    return batch


def execute_generation_node(
    modules: Dict[str, Any],
    node: NodeDef,
    ctx: Dict[str, Any],
    *,
    state_name: str,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    profiler: Optional[GraphProfiler] = None,
    scope_fn: Optional[Callable[[str], ContextManager]] = None,
) -> Dict[str, Any]:
    """Run one generation node under VeOmni (unwrap + scope + profile).

    Unlike training, inference has no ``pre_forward`` / ``post_forward`` and no
    metric meter — the eager endpoint is called directly with the full ``ctx``
    plus ``generation_kwargs``. ``state_name`` only labels the profiler node.

    Returns the (mutated) ``ctx``.
    """
    method = node.method
    wrapped = modules.get(node.module)
    if wrapped is None:
        raise KeyError(
            f"execute_generation_node: module '{node.module}' (node '{node.name}') missing "
            f"from modules dict. Provided: {sorted(modules)}."
        )
    raw = unwrap_graph_module(wrapped, module_name=node.module)

    module_context = scope_fn(node.module) if scope_fn is not None else nullcontext()
    profile_context = (
        profiler.node(f"[State|{state_name}] {node.name}: {node.module}.{method}")
        if profiler is not None
        else nullcontext()
    )
    with module_context, profile_context:
        out = call_graph_endpoint(
            wrapped,
            raw,
            method=method,
            kwargs={**ctx, "generation_kwargs": generation_kwargs},
        )
    if not isinstance(out, dict):
        raise TypeError(f"FSM node '{node.name}'.{method} must return a dict; got {type(out).__name__}.")
    ctx.update(out)
    return ctx


__all__ = ["execute_train_node", "execute_generation_node"]
