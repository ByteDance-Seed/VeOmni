"""
OmniModel V2 — composable multi-modal model driven by config-specified graphs.

This file holds the *minimal* runtime — graph traversal for training, FSM
walk for inference, and a single ``_loss`` aggregation step.  It deliberately
contains **no** build / weight-loading / FSDP wiring; that lives in
:class:`~veomni.trainer.omni.omni_trainer.OmniTrainer`, which builds and
FSDP-wraps each module independently and attaches them to :class:`OmniModel`.

Architecture
------------
``OmniModel`` carries:

* sub-modules           — each named :class:`ModuleMixin` is attached as a
                          **direct attribute** of ``OmniModel``, so
                          ``model.named_children()`` enumerates them in the
                          declared order and parameter fqns flatten to
                          ``<module_name>.<rest>`` (no ``modules_dict.``
                          middle prefix).  ``model.modules_dict`` remains as
                          a read-only dict view for back-compat.
* ``training_graph``    — :class:`TrainingGraph` (DAG over the flat edge list).
* ``generation_graph``  — :class:`GenerationGraph` (FSM, optional).

Loss protocol (single ``_loss`` key per module)
-----------------------------------------------
Each module's ``forward`` returns at most one ``_loss`` scalar — a *token-
level* mean already reduced across every micro-batch the module consumed
internally.  ``OmniModel.forward`` simply sums those scalars across nodes::

    losses[node] = out["_loss"]   # if present
    total = sum(losses.values())  # zero-dim tensor

No aliasing, no per-batch averaging at the OmniModel level — that responsibility
sits with each module's ``post_forward`` (so token counts stay correct when
micro-batch sizes differ across modules).

Training
--------
``forward(**batch)`` walks the DAG one node at a time, fully symmetric with
:meth:`generate` (it resets the graph cursor at entry — each forward is
independent):

  * ``batch = training_graph.step(modules, batch)`` — run the node at the cursor
    (``pre_forward`` → method → ``post_forward``, every method routed through the
    module's ``__call__`` so FSDP2/DDP hooks fire). Edges declare execution order
    only — no per-field routing; every node receives the same shared ``batch``
    and reads / mutates the ``conversation_list`` carrier in place.
  * ``_collect_training_loss(batch)`` — drain the node's optional ``_loss``.
  * ``training_graph.maybe_transition()`` — advance the cursor.

Returns ``{"loss": scalar_or_None, "losses": {node: scalar}}``.

Inference
---------
``generate(request, trace, generation_kwargs)`` loops (it does **not** reset
the FSM — the caller owns request boundaries via :meth:`reset`):

  * ``ctx = fsm.step(modules, ctx)`` — one iteration of the current state.
  * ``fsm.maybe_transition(ctx)`` — first matching condition wins.
  * Stop when ``fsm.is_done()`` or the ``generation_kwargs["max_new_tokens"]``
    safety cap (default 2048) is reached.

Modules emit one-shot ``generated`` payloads (``{type, value}``) from their
FSM step return dict when a span ends; :meth:`OmniModel.generate` drains
those into :attr:`OmniModel.generated` via :meth:`_collect_generated` and
does not persist them on ``ctx``.

Both ``step`` and ``maybe_transition`` accept an optional ``trace`` list —
print-driven flow tests collect the visit log from there to assert the
expected node order and transition timing.

``OmniModel.generate`` always emits a coarse progress trail via
:meth:`logger.info_rank0` — one line per FSM state entry
(``[FSM] step <N>: <state_name>``) so CLI users can follow long-running
spans (e.g. Janus T2I's 576-step ``image_vq`` loop).  Rank-0 gating is
handled by the logger.
"""

from contextlib import nullcontext
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import torch.nn as nn

from ...distributed.parallel_state import use_parallel_state
from ...utils import helper
from .configuration_omni import OmniConfig
from .graphs.generation_graph import GenerationGraph
from .graphs.training_graph import TrainingGraph
from .mixins.modulemixin import ModuleMixin


logger = helper.create_logger(__name__)

# Must match the ``_loss`` key every OmniModule's ``post_forward`` emits.
_LOSS_KEY = "_loss"


class OmniModel(nn.Module):
    """Pure runtime over already-built sub-modules.

    Parameters
    ----------
    config:
        :class:`OmniConfig` with ``modules`` / ``training_graph`` /
        ``generation_graph`` populated.
    modules:
        ``{module_name: ModuleMixin-mixin instance}`` — already constructed
        (and parallelised, if running under FSDP).  ``OmniTrainer`` is
        responsible for building these via ``build_foundation_model`` /
        ``build_parallelize_model``.  The print-flow tests pass plain
        :class:`ModuleMixin` subclasses here.

    Notes
    -----
    The runtime never instantiates modules itself — that contract belongs
    to the trainer.  Keeping the runtime free of build logic means it stays
    importable in cpu-only / torch-free contexts (the tests rely on this).
    """

    config_class = OmniConfig

    # Names that ``OmniModel`` itself uses as attributes — sub-module names
    # coming in from YAML must not collide with these or PyTorch's nn.Module
    # ``add_module`` would silently overwrite framework state.  Listed
    # explicitly so the failure mode is loud and obvious.
    _RESERVED_ATTR_NAMES: Tuple[str, ...] = (
        "config",
        "training_graph",
        "generation_graph",
        "modules_dict",
        "_module_names",
        "_RESERVED_ATTR_NAMES",
    )

    def __init__(self, config: OmniConfig, modules: Mapping[str, nn.Module]):
        super().__init__()
        self.config = config

        missing = [n for n in config.module_names if n not in modules]
        if missing:
            raise KeyError(
                f"OmniModel: modules dict missing entries declared in config: {missing}. "
                f"Provided: {sorted(modules)}; expected: {config.module_names}."
            )
        clashes = [n for n in config.module_names if n in self._RESERVED_ATTR_NAMES]
        if clashes:
            raise ValueError(
                f"OmniModel: sub-module name(s) {clashes} collide with framework "
                f"attribute(s).  Rename these in your YAML's `modules:` section."
            )

        # Sub-modules are attached as **direct attributes** of OmniModel (not
        # via an `nn.ModuleDict` middle layer) so that:
        #   * ``model.named_children()`` directly yields ``[(name, mod), ...]``
        #     — needed by ``build_parallelize_model`` to dispatch a
        #     per-sub-module ``weights_path`` mapping;
        #   * ``model.named_parameters()`` fqns shape as ``<name>.<rest>``
        #     instead of ``modules_dict.<name>.<rest>`` — so per-module
        #     checkpoint subfolders map 1:1 to parameter-fqn prefixes;
        #   * downstream save/load callbacks that target subfolder names can
        #     reuse those names verbatim without stripping a prefix.
        # ``_module_names`` is a plain list (not an ``nn.Module``) so it
        # doesn't show up under ``children()`` / ``modules()``.  The
        # back-compat ``modules_dict`` view below preserves existing call
        # sites that index/iterate by name.
        self._module_names: List[str] = list(config.module_names)
        for name in self._module_names:
            self.add_module(name, modules[name])

        # Both views are optional: an inference-only config may carry just a
        # ``generation_graph`` (no ``training_graph``), and vice versa.
        self.training_graph = TrainingGraph(edges=config.training_graph) if config.has_training_graph() else None

        self.generation_graph = (
            GenerationGraph(fsm_config=config.generation_graph) if config.has_generation_graph() else None
        )

        # Last FSM state printed by :meth:`_emit_progress` — its private
        # dedup cursor (reset on each fresh ``generate`` run).
        self._last_printed_state: Optional[str] = None
        # Per-``generate`` artefacts emitted by modules as one-shot
        # ``ctx["generated"] = {type, value}`` payloads — drained into this
        # list and never persisted back onto ``ctx``.
        self._generated: List[Dict[str, Any]] = []

        # Per-``forward`` training losses, ``{node_name: scalar}`` — drained from
        # each node's ``batch["_loss"]`` by ``_collect_training_loss`` (the
        # training analogue of ``_generated`` / ``_collect_generated``).
        self._losses: Dict[str, Any] = {}

        # ``{module_name: ParallelState}`` — set by the trainer so each node's
        # forward runs under its module's own device mesh / extra-parallel
        # groups (see :meth:`set_module_parallel_states`).  Empty by default so
        # the runtime stays importable without a trainer (the print-flow tests).
        self._module_parallel_states: Dict[str, Any] = {}

        # Prime per-request inference runtime state (FSM at its initial
        # state).  :meth:`generate` deliberately does NOT reset, so a future
        # multi-turn conversation can keep cache across turns and only wipe
        # it when the caller explicitly invokes :meth:`reset`.
        self.reset()

    @property
    def modules_dict(self) -> Dict[str, nn.Module]:
        """Back-compat dict view of the sub-modules.

        Read-only — sub-modules are real attributes; mutating this dict has
        no effect on the model.  Returned as a fresh ``dict`` (not an
        ``nn.ModuleDict``) so callers indexing / iterating it never see
        the deprecated middle-attribute path.
        """
        return {name: getattr(self, name) for name in self._module_names}

    @property
    def generated(self) -> List[Dict[str, Any]]:
        """Artefacts collected during the latest :meth:`generate` run.

        Each entry is ``{"type": <str>, "value": <any>}`` — e.g.
        ``{"type": "image", "value": PIL.Image}``.  Not mirrored on ``ctx``.
        """
        return list(self._generated)

    def set_module_parallel_states(self, module_parallel_states: Mapping[str, Any]) -> None:
        """Register ``{module_name: ParallelState}`` for per-node forward scoping.

        Called by :class:`~veomni.trainer.omni.omni_trainer.OmniTrainer` after the
        modules are built so each node's pre/forward/post runs under its own
        module's device mesh / extra-parallel groups — needed when modules use
        different parallelism (e.g. a vocab-parallel ``emb`` embedding whose
        forward all-reduces over the ``emb`` group, or an EP MoE module whose
        kernels read ``get_parallel_state().ep_group``).
        """
        self._module_parallel_states = dict(module_parallel_states)

    def _module_scope(self, module_name: str):
        """Context manager scoping ``module_name``'s ParallelState as current.

        Used by both training ``forward`` and the inference ``generate`` FSM
        (passed as ``scope_fn`` to :meth:`GenerationGraph.step`).  No-op when no
        per-module states are registered (e.g. eager single-process inference).
        """
        ps = self._module_parallel_states.get(module_name)
        return use_parallel_state(ps) if ps is not None else nullcontext()

    # ── Training ──────────────────────────────────────────────────────────────

    def _collect_training_loss(self, batch: Dict[str, Any], trace: Optional[List[str]] = None) -> None:
        """Drain ``batch["_loss"]`` for the just-stepped node into :attr:`_losses`.

        The training analogue of :meth:`_collect_generated`: the node's
        ``post_forward`` merges an optional scalar ``_loss`` into the carrier;
        we pop it (so it never leaks to the next node) and key it by the node
        that produced it for per-node logging.
        """
        loss = batch.pop(_LOSS_KEY, None)
        if loss is not None:
            node = self.training_graph.current_node_name
            self._losses[node] = loss
            if trace is not None:
                trace.append(f"loss:{node}")

    def forward(
        self,
        *,
        trace: Optional[List[str]] = None,
        **batch: Any,
    ) -> Dict[str, Any]:
        """Execute the training DAG once over the full ``batch``.

        Fully symmetric with :meth:`generate`: walk the graph one node at a time
        — ``step`` runs the node at the cursor, ``_collect_training_loss`` drains
        its ``_loss``, ``maybe_transition`` advances the cursor — until
        ``is_done()``. Each module is invoked exactly once and owns its internal
        micro-batch chunking (token-mean reduction included). The wrapped
        sub-modules are passed to ``step`` so it can fire the FSDP2/DDP hooks.

        Parameters
        ----------
        trace:
            Optional list to which one ``"forward:<node>"`` token is
            appended per executed node — used by the print-flow tests.
        **batch:
            Raw batch fields, transparently visible to every node.

        Returns
        -------
        dict with keys:
        * ``"loss"``    : scalar tensor (sum of all node ``_loss`` values),
                          or ``None`` if no node emitted a loss.
        * ``"losses"``  : ``{node_name: scalar tensor}``.
        """
        if self.training_graph is None:
            raise RuntimeError(
                "OmniModel.forward requires a `training_graph`, but this config declares none "
                "(inference-only). Use `generate` instead, or load a training YAML."
            )

        self.training_graph.reset()
        self._losses.clear()
        modules = {name: getattr(self, name) for name in self._module_names}

        while not self.training_graph.is_done():
            batch = self.training_graph.step(modules, batch, trace=trace, scope_fn=self._module_scope)
            self._collect_training_loss(batch, trace)
            self.training_graph.maybe_transition(trace=trace)

        return {"loss": _sum_losses(self._losses), "losses": dict(self._losses)}

    # ── Inference ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear per-conversation inference runtime state.

        Re-points the generation FSM to its initial state (and is the future
        home for clearing per-module KV / VQ caches).  Called once at
        construction; afterwards the *caller* drives it at request
        boundaries — :class:`OmniInferencer` resets before each independent
        request, while a multi-turn conversation would reset only on an
        explicit ``clear`` so cache survives across turns.  :meth:`generate`
        never resets on its own.
        """
        if self.generation_graph is not None:
            self.generation_graph.reset()
        self._generated.clear()

    @staticmethod
    def _normalize_generated(item: Any) -> Optional[Dict[str, Any]]:
        """Normalize a one-shot module ``generated`` payload to ``{type, value}``."""
        if item is None:
            return None
        if isinstance(item, dict) and "type" in item and "value" in item:
            normalized: Dict[str, Any] = {"type": item["type"], "value": item["value"]}
            if item.get("meta") is not None:
                normalized["meta"] = item["meta"]
            return normalized
        return None

    def _append_generated(self, item: Any) -> None:
        """Append a normalized ``{type, value}`` entry to :attr:`_generated`."""
        normalized = self._normalize_generated(item)
        if normalized is not None:
            self._generated.append(normalized)

    def _collect_generated(self, ctx: Dict[str, Any], trace: Optional[List[str]] = None) -> None:
        """Drain ``ctx["generated"]`` into :attr:`_generated` (one-shot)."""
        generated = ctx.pop("generated", None)
        self._append_generated(generated)
        if trace is not None and generated is not None:
            trace.append(f"generated:{generated['type']}")

    def _invoke_module_finalize(
        self,
        ctx: Dict[str, Any],
        trace: Optional[List[str]] = None,
    ) -> None:
        """Call :meth:`ModuleMixin.finalize` and drain any ``generated`` payload.

        Runs on every module when ``max_new_tokens`` trips before ``done``.
        """
        for name, raw in self.named_omni_modules():
            out = raw.finalize(ctx=ctx)
            if not isinstance(out, dict):
                raise TypeError(f"{type(raw).__name__}.finalize must return a dict, got {type(out).__name__}.")
            generated = out.pop("generated", None)
            self._append_generated(generated)
            if trace is not None and generated is not None:
                trace.append(f"finalize:{name} | generated:{generated['type']}")

    def _emit_progress(self, total_steps: int) -> None:
        """Log one ``[FSM] step <N>: <state>`` line on a state change.

        Owns its own dedup cursor (:attr:`_last_printed_state`) so the caller
        doesn't thread it around: emits via :meth:`logger.info_rank0` only
        when the FSM's current state differs from the last printed one.  The
        cursor is reset at ``total_steps == 0`` (the first call of every
        :meth:`generate` run) so a fresh run always re-prints its initial
        state.
        """
        if total_steps == 0:
            self._last_printed_state = None
        current = self.generation_graph.current_state_name
        if current != self._last_printed_state:
            logger.info_rank0(f"[FSM] step {total_steps:>4}: {current}")
            self._last_printed_state = current

    def generate(
        self,
        request: Dict[str, Any],
        trace: Optional[List[str]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run inference using the FSM.

        Parameters
        ----------
        request:
            Generation request dict — used directly as the initial ``ctx``.
            Built by :class:`OmniInferencer` and contains the seed
            ``conversation_list`` (see :func:`build_conversation`).  Modules
            read and mutate it in place each FSM step.
        trace:
            Optional list — receives FSM step / transition log for testing.
        generation_kwargs:
            Opaque per-request knobs forwarded to every module's FSM step
            (e.g. ``temperature`` / ``top_p`` / ``guidance_scale``).  The
            framework only reads ``max_new_tokens`` (default 2048) as the
            hard safety cap on total FSM iterations.

        Termination is fully FSM-driven: a state reaches ``done`` only when a
        module raises the ``module_signal`` its transition is waiting on
        (e.g. the text encoder emits ``text_done`` on ``</s>``, the VQ
        decoder emits ``image_complete`` when the grid is full).  There is no
        token-level stop list — EOS handling lives inside the emitting module,
        not here.  ``max_new_tokens`` is only a hard safety cap on total FSM
        iterations.

        A coarse progress trail is always emitted via
        :meth:`logger.info_rank0` — one ``[FSM] step <N>: <state>`` line
        each time the FSM enters a new state (the initial state at step 0,
        every state reached via :meth:`GenerationGraph.maybe_transition`,
        and the terminal ``done`` state).  It fires once more after the loop
        exits so the final resting state is always reported regardless of
        how the loop terminated (normal ``done`` or ``max_new_tokens`` cap).
        This lets the user follow which FSM
        span is in flight for long-running inference — Janus T2I e.g. shows
        ``prompt_encode → image_vq → image_vq_end → done``, where the
        576-step ``image_vq`` loop dominates and is read off the step deltas
        between consecutive lines.

        Note this does **not** reset the FSM — the caller owns request
        boundaries via :meth:`reset` (so multi-turn conversations can keep
        cache across turns).  The FSM runs from whatever state it is in.
        """
        ctx: Dict[str, Any] = request
        self._generated.clear()

        modules = {name: _unwrap_module(getattr(self, name)) for name in self._module_names}

        # Progress trail.  ``maybe_transition`` flips the FSM's current state
        # at the END of each iteration body, so emitting at the START of the
        # NEXT body catches every state change (including the initial state at
        # step 0).  A final emit after the loop catches the transition into
        # ``done`` (the while-cond exits before the body iterates that state).
        # :meth:`_emit_progress` owns the dedup cursor — nothing to thread.
        max_new_tokens = generation_kwargs.get("max_new_tokens", 2048)
        total_steps = 0

        while not self.generation_graph.is_done() and total_steps < max_new_tokens:
            self._emit_progress(total_steps)
            ctx = self.generation_graph.step(
                modules, ctx, trace=trace, generation_kwargs=generation_kwargs, scope_fn=self._module_scope
            )
            total_steps += 1
            self._collect_generated(ctx, trace)
            self.generation_graph.maybe_transition(ctx, trace=trace)

        # Final emit — captures the state the FSM is in after the loop
        # exits.  Usually ``done`` (a module raised its terminating signal);
        # otherwise the state the FSM got stuck in when the ``max_new_tokens``
        # safety cap tripped.
        self._emit_progress(total_steps)

        if not self.generation_graph.is_done():
            self._invoke_module_finalize(ctx, trace=trace)

        return ctx

    # ── Utilities ─────────────────────────────────────────────────────────────

    def named_omni_modules(self) -> Iterator[Tuple[str, nn.Module]]:
        """Yield ``(name, raw_module)`` for every entry whose unwrapped form
        is an :class:`ModuleMixin` mixin instance."""
        for name in self._module_names:
            raw = _unwrap_module(getattr(self, name))
            if _is_omni_module(raw):
                yield name, raw  # type: ignore[misc]

    def get_module(self, name: str) -> nn.Module:
        if name not in self._module_names:
            raise KeyError(f"Module '{name}' not found in OmniModel")
        return getattr(self, name)

    def collect_assets(self) -> List[Any]:
        """Collect per-module assets (vision/audio processors, codebooks).

        Tokenizers (e.g. ``janus_text_encoder``) are module-owned assets assigned
        on ``_tokenizer`` by :class:`~veomni.trainer.omni.omni_module_trainer.OmniModuleTrainer`.
        """
        assets: List[Any] = []
        for _, raw in self.named_omni_modules():
            assets.extend(raw.get_assets())
        return assets


# ── helpers ───────────────────────────────────────────────────────────────────


def _unwrap_module(mod: nn.Module) -> nn.Module:
    """Strip DDP / FSDP wrappers so mixin hooks (pre/post_forward) reach the raw module.

    FSDP2 (DTensor params) does not wrap; DDP exposes ``.module``.  We treat
    anything else as already-raw.
    """
    inner = getattr(mod, "module", None)
    if inner is not None and isinstance(inner, nn.Module) and inner is not mod:
        # DDP-style wrapper.
        return inner
    return mod


def _is_omni_module(mod: nn.Module) -> bool:
    return isinstance(mod, ModuleMixin)


def _sum_losses(losses: Dict[str, Any]) -> Optional[Any]:
    """Sum a per-node ``{name: scalar}`` dict into one scalar (or ``None``).

    Plain ``+`` builds the autograd graph across the per-node loss tensors.
    """
    if not losses:
        return None
    it = iter(losses.values())
    total = next(it)
    for v in it:
        total = total + v
    return total


__all__ = ["OmniModel"]
