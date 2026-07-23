"""Node executors for the SeedOmni graphs.

The graphs (:class:`~veomni.models.seed_omni.graphs.training_graph.TrainingGraph`
and :class:`~veomni.models.seed_omni.graphs.generation_graph.GenerationGraph`)
only *select* which node runs next — they no longer run model forwards
themselves. The functions here are the **external executor**: given a single
selected :class:`~veomni.models.seed_omni.graphs.graph.NodeDef` (yielded by a
graph's ``iter_nodes`` generator), they resolve the module, scope its
parallel state, and run the node, merging its output back into the shared
carrier in place.

Two executors, mirroring the two views (and their different execution
semantics):

* :func:`execute_train_node` — the *training* forward: ``pre_forward`` (packing /
  Ulysses SP slicing) → optional metric-meter → the graph endpoint (via
  ``call_graph_endpoint`` so FSDP2/DDP hooks fire) → ``post_forward`` (SP
  all-gather). Output is merged into ``batch`` (``conversation_list`` carrier
  and/or a scalar ``_loss``).
* :func:`execute_generation_node` — the *inference* step: the eager endpoint
  call only (no pre/post hooks), with ``generation_kwargs`` threaded in. Output
  is merged into ``ctx``.

Both share the per-node boilerplate — resolve + unwrap the module, enter its
``module_context`` (``scope_fn``) and the profiler node — but keep their bodies
separate because training and inference genuinely differ (this is also why a
future infra-free / eager model can call :func:`execute_generation_node` without
any wrapper hooks). ``scope_fn`` is the graph-agnostic per-module scope hook
(``OmniModel.module_context`` at runtime, or a test stub); ``None`` means no
scoping (eager single-process inference / print-flow tests).
"""

from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Dict, Optional

from .dispatch import call_graph_endpoint, unwrap_graph_module
from .graph import NodeDef
from .profiling import GraphProfiler


def execute_train_node(
    modules: Dict[str, Any],
    node: NodeDef,
    batch: Dict[str, Any],
    *,
    profiler: Optional[GraphProfiler] = None,
    scope_fn: Optional[Callable[[str], ContextManager]] = None,
) -> Dict[str, Any]:
    """Run one training node (selected by :meth:`TrainingGraph.iter_nodes`) over ``batch``.

    Resolves ``node.module``, scopes its :class:`ParallelState` (via ``scope_fn``
    — vocab-parallel ``emb`` / MoE EP groups), and runs ``pre_forward`` → method
    → ``post_forward``, dispatching through the **wrapped** module so DDP/FSDP
    hooks fire (non-``forward`` methods via the ``call_graph_endpoint``
    trampoline). ``raw`` (the unwrapped :class:`ModuleMixin`) owns the hooks;
    FSDP2 is in-place (``raw is wrapped``) while DDP wraps (``raw =
    wrapped.module``).

    Edges are pure topology — no per-node input routing. Every node receives the
    same shared ``batch``; cross-node state flows through the single
    ``conversation_list`` carrier, so the node's return dict (only ever
    ``conversation_list`` and/or ``_loss``) is merged back into ``batch`` in
    place for downstream nodes. Loss draining lives in ``OmniModel``
    (``_collect_training_loss``).

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
        # ``pre_forward`` does packing / conversation extraction and, when the
        # module's scoped ``sp_size > 1``, slices its inputs to this rank's
        # ``1/sp`` shard (classic single-pass Ulysses: every SP rank holds the
        # SAME replicated sample; the model's attention all-to-alls run over the
        # SP group internally). ``post_forward`` all-gathers the output shard
        # back to the full sample on every rank, so downstream nodes run
        # identically on replicated full data. SP is thus fully contained in the
        # module's own pre/post hooks — the graph and executor stay SP-unaware.
        kwargs = raw.pre_forward(method=method, **batch)

        # Opt-in metric meter (only modules multi-inheriting a MetricMeterMixin
        # have ``metric_meter_add``). It drains the FULL pre-slice seqlens the
        # module stashed inside ``pre_forward`` (via ``metric_meter_set_seqlens``,
        # before any SP slice), so metering is SP-invariant regardless of the
        # sharded ``kwargs`` handed here.
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
    """Run one generation node (selected by :meth:`GenerationGraph.iter_nodes`).

    Unlike training, inference has no ``pre_forward`` / ``post_forward`` and no
    metric meter — the eager endpoint is called directly with the full ``ctx``
    plus ``generation_kwargs`` (bare nodes default to ``generate``; dotted nodes
    dispatch verbatim). ``scope_fn`` scopes the module's :class:`ParallelState`
    for distributed inference; eager single-process inference passes ``None``.
    ``state_name`` is only used to label the profiler node.

    The node's return dict is merged into ``ctx`` in place; returns ``ctx``.
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
