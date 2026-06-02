"""
OmniModel V2 — composable multi-modal model driven by config-specified graphs.

This file holds the *minimal* runtime — graph traversal for training, FSM
walk for inference, and a single ``_loss`` aggregation step.  It deliberately
contains **no** build / weight-loading / FSDP wiring; that lives in
``OmniTrainer`` (yet to be migrated) which calls
:func:`veomni.distributed.torch_parallelize.build_foundation_model` and
:func:`...build_parallelize_model` per module and hands the resulting dict
to :class:`OmniModel`.

Architecture
------------
``OmniModel`` carries:

* sub-modules           — each named :class:`OmniModule` is attached as a
                          **direct attribute** of ``OmniModel``, so
                          ``model.named_children()`` enumerates them in the
                          declared order and parameter fqns flatten to
                          ``<module_name>.<rest>`` (no ``modules_dict.``
                          middle prefix).  ``model.modules_dict`` remains as
                          a read-only dict view for back-compat.
* ``training_graph``    — :class:`TrainingGraph` (DAG over node/edge pools).
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
For each node in ``training_graph.execution_order``:

  1. Look up the OmniModule via ``training_graph.module_of(node)``.
  2. ``training_graph.collect_inputs(node, outputs, batch)`` assembles
     kwargs (raw_batch + edge-routed values from earlier nodes).
  3. Dispatch on ``training_graph.method_of(node)``:
       * ``forward`` → call the (possibly FSDP-wrapped) module so backward
         hooks fire correctly.
       * any other  → call ``getattr(module, method)`` directly (FSDP2 with
         DTensor params handles partial-param-use transparently).
  4. Store the output dict under the node name.  If it carries ``_loss``,
     accumulate.

Returns ``{"loss": scalar_or_None, "losses": {node: scalar}, "outputs": {node: dict}}``.

Inference
---------
``generate(request, context, max_new_tokens)`` resets the FSM, then loops:

  * ``ctx = fsm.step(modules_dict, ctx)`` — one iteration of the current state.
  * ``fsm.maybe_transition(ctx)`` — first matching condition wins.
  * Stop when ``fsm.is_done()`` or ``max_new_tokens`` exhausts.

Once a step raises ``module_signal``, ``generate`` calls
:meth:`OmniModule.finalize` on every active module so segment buffers flush
at hand-off time.  If the ``max_new_tokens`` safety cap trips first, every
module gets a second ``finalize`` pass — each module decides from its own
buffer state whether to emit, warn-and-discard, or no-op.  Modules may also
emit a one-shot ``generated`` payload (``{type, value}``) from ``finalize``
or from any FSM step; everything is drained into :attr:`OmniModel.generated`
and is not persisted on ``ctx``.

Both ``step`` and ``maybe_transition`` accept an optional ``trace`` list —
print-driven flow tests collect the visit log from there to assert the
expected node order and transition timing.

``OmniModel.generate`` always emits a coarse progress trail via
:meth:`logger.info_rank0` — one line per FSM state entry
(``[FSM] step <N>: <state_name>``) so CLI users can follow long-running
spans (e.g. Janus T2I's 576-step ``image_vq`` loop).  Rank-0 gating is
handled by the logger.
"""

from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from ...utils import helper
from .configuration_seed_omni import OmniConfig
from .generation_graph import FSM_SIGNAL_KEY, GenerationGraph
from .module import OmniModule
from .training_graph import TrainingGraph


logger = helper.create_logger(__name__)


_LOSS_KEY: str = "_loss"


class OmniModel(nn.Module):
    """Pure runtime over already-built sub-modules.

    Parameters
    ----------
    config:
        :class:`OmniConfig` with ``modules`` / ``nodes`` / ``edges`` /
        ``training_graph`` / ``generation_graph`` populated.
    modules:
        ``{module_name: OmniModule-mixin instance}`` — already constructed
        (and parallelised, if running under FSDP).  ``OmniTrainer`` is
        responsible for building these via ``build_foundation_model`` /
        ``build_parallelize_model``.  The print-flow tests pass plain
        :class:`OmniModule` subclasses here.

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
        #     — needed by ``build_parallelize_model`` (D2.2) to dispatch a
        #     per-sub-module ``weights_path`` mapping;
        #   * ``model.named_parameters()`` fqns shape as ``<name>.<rest>``
        #     instead of ``modules_dict.<name>.<rest>`` — matches the design
        #     contract in design.md § "FQN 视角对齐";
        #   * downstream save/load callbacks that target subfolder names can
        #     reuse those names verbatim without stripping a prefix.
        # ``_module_names`` is a plain list (not an ``nn.Module``) so it
        # doesn't show up under ``children()`` / ``modules()``.  The
        # back-compat ``modules_dict`` view below preserves existing call
        # sites that index/iterate by name.
        self._module_names: List[str] = list(config.module_names)
        for name in self._module_names:
            self.add_module(name, modules[name])

        self.training_graph = TrainingGraph(
            nodes=config.nodes,
            edges=config.edges,
            training_edges=config.training_edges,
        )

        self.generation_graph = (
            GenerationGraph(
                fsm_config=config.generation_graph,
                nodes=config.nodes,
                edges=config.edges,
            )
            if config.has_generation_graph()
            else None
        )

        # Last FSM state printed by :meth:`_emit_progress` — its private
        # dedup cursor (reset on each fresh ``generate`` run).
        self._last_printed_state: Optional[str] = None
        # Per-``generate`` artefacts emitted by modules as one-shot
        # ``ctx["generated"] = {type, value}`` payloads — drained into this
        # list and never persisted back onto ``ctx``.
        self._generated: List[Dict[str, Any]] = []

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

    # ── Training ──────────────────────────────────────────────────────────────

    def forward(
        self,
        *,
        trace: Optional[List[str]] = None,
        **batch: Any,
    ) -> Dict[str, Any]:
        """Execute the training DAG once over the full ``batch``.

        Each module is invoked exactly once and is responsible for any
        internal micro-batch chunking (token-mean reduction included).

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
        * ``"outputs"`` : ``{node_name: full output dict}`` (includes
                          tensors routed by edges plus any extras).
        """
        node_outputs: Dict[str, Dict[str, Any]] = {}
        losses: Dict[str, torch.Tensor] = {}

        for node_name in self.training_graph.execution_order:
            module_name = self.training_graph.module_of(node_name)
            method = self.training_graph.method_of(node_name)
            module = getattr(self, module_name)
            raw_module = _unwrap_module(module)

            kwargs = self.training_graph.collect_inputs(node_name, node_outputs, batch)
            kwargs = raw_module.pre_forward(**kwargs) if isinstance(raw_module, OmniModule) else kwargs

            if method == "forward":
                # Through the FSDP wrapper so unshard + backward hooks fire.
                out = module(**kwargs)
            else:
                fn = getattr(raw_module, method, None)
                if fn is None:
                    raise AttributeError(
                        f"Node '{node_name}' requires {module_name}.{method}() "
                        f"but {type(raw_module).__name__} has no such method."
                    )
                # FSDP2 installs its all-gather (unshard) pre-forward hook and the
                # post-forward/backward reduce-scatter hooks on ``module.__call__``
                # only. Calling a non-``forward`` method directly would run the op
                # against still-sharded DTensor params (-> "mixed Tensor and
                # DTensor" errors) and skip gradient sync. Route the call through
                # ``module(**kwargs)`` by temporarily pointing ``forward`` at the
                # target method so every node — regardless of method name — gets
                # the same FSDP2 lifecycle as a plain ``forward`` node.
                orig_forward = raw_module.forward
                try:
                    raw_module.forward = fn
                    out = module(**kwargs)
                finally:
                    raw_module.forward = orig_forward

            if isinstance(raw_module, OmniModule):
                out = raw_module.post_forward(out)

            if not isinstance(out, dict):
                raise TypeError(
                    f"Node '{node_name}' ({module_name}.{method}) returned {type(out).__name__}; expected dict."
                )
            node_outputs[node_name] = out

            # V2 segment-driven carrier: the single ``conversation_list`` object
            # is mutated/replaced in place as it flows
            # ``siglip → vae → text_encoder → backbone → decoders``.  Write it
            # back into the shared batch so downstream nodes read the evolving
            # carrier directly — edges are pure ``{from, to}`` topology.
            convo = out.get("conversation_list")
            if convo is not None:
                batch["conversation_list"] = convo

            if _LOSS_KEY in out and out[_LOSS_KEY] is not None:
                losses[node_name] = out[_LOSS_KEY]

            if trace is not None:
                trace.append(f"forward:{node_name}")

        total = _sum_losses(losses)
        return {"loss": total, "losses": losses, "outputs": node_outputs}

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

    def _collect_generated(self, ctx: Dict[str, Any]) -> None:
        """Drain ``ctx["generated"]`` into :attr:`_generated` (one-shot)."""
        self._append_generated(ctx.pop("generated", None))

    def _invoke_module_finalize(
        self,
        ctx: Dict[str, Any],
        *,
        force: bool = False,
        trace: Optional[List[str]] = None,
    ) -> None:
        """Call :meth:`OmniModule.finalize` and drain any ``generated`` payload.

        By default only runs when the step just raised ``module_signal``.
        ``force=True`` skips that guard — used when ``max_new_tokens`` trips
        before ``done``.
        """
        if not force and FSM_SIGNAL_KEY not in ctx:
            return
        for name, raw in self.named_omni_modules():
            out = raw.finalize(ctx=ctx)
            if not isinstance(out, dict):
                raise TypeError(f"{type(raw).__name__}.finalize must return a dict, got {type(out).__name__}.")
            self._append_generated(out.pop("generated", None))
            if trace is not None:
                trace.append(f"finalize:{name}")

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
        context: Optional[Dict[str, Any]] = None,
        max_new_tokens: int = 512,
        *,
        trace: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run inference using the FSM.

        Parameters
        ----------
        request:
            Generation request dict — seeds ``ctx`` when ``context`` is
            ``None``.
        context:
            Initial generation context (input_ids, attention_mask, ...).  If
            ``None``, starts from a copy of ``request``.  During generation
            ``input_ids`` grows as a full ``(B, T)`` sequence (HF
            ``generate`` style); modules emit ``(B, 1)`` step tokens which
            the FSM appends each iteration.
        max_new_tokens:
            Hard upper bound on total FSM iterations across all states.
        trace:
            Optional list — receives FSM step / transition log for testing.

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
        ctx: Dict[str, Any] = dict(context if context is not None else request)
        self._generated.clear()

        modules = {name: _unwrap_module(getattr(self, name)) for name in self._module_names}

        # Progress trail.  ``maybe_transition`` flips the FSM's current state
        # at the END of each iteration body, so emitting at the START of the
        # NEXT body catches every state change (including the initial state at
        # step 0).  A final emit after the loop catches the transition into
        # ``done`` (the while-cond exits before the body iterates that state).
        # :meth:`_emit_progress` owns the dedup cursor — nothing to thread.
        total_steps = 0
        while not self.generation_graph.is_done() and total_steps < max_new_tokens:
            self._emit_progress(total_steps)
            ctx = self.generation_graph.step(modules, ctx, trace=trace)
            total_steps += 1

            self._collect_generated(ctx)
            self._invoke_module_finalize(ctx, trace=trace)

            self.generation_graph.maybe_transition(ctx, trace=trace)

        # Final emit — captures the state the FSM is in after the loop
        # exits.  Usually ``done`` (a module raised its terminating signal);
        # otherwise the state the FSM got stuck in when the ``max_new_tokens``
        # safety cap tripped.
        self._emit_progress(total_steps)

        if not self.generation_graph.is_done():
            self._invoke_module_finalize(ctx, force=True, trace=trace)

        return ctx

    # ── Utilities ─────────────────────────────────────────────────────────────

    def named_omni_modules(self) -> Iterator[Tuple[str, OmniModule]]:
        """Yield ``(name, raw_module)`` for every entry whose unwrapped form
        is an :class:`OmniModule` mixin instance."""
        for name in self._module_names:
            raw = _unwrap_module(getattr(self, name))
            if isinstance(raw, OmniModule):
                yield name, raw  # type: ignore[misc]

    def get_module(self, name: str) -> nn.Module:
        if name not in self._module_names:
            raise KeyError(f"Module '{name}' not found in OmniModel")
        return getattr(self, name)

    def collect_assets(self) -> List[Any]:
        """Collect per-module assets (vision/audio processors, codebooks).

        The global tokenizer (``OmniConfig.tokenizer_path``) is *not*
        returned here — it is loaded by ``OmniTrainer`` and saved at the
        checkpoint root, separate from per-module subfolders.
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


def _sum_losses(losses: Mapping[str, torch.Tensor]) -> Optional[torch.Tensor]:
    """Sum a per-node ``{name: scalar}`` dict into one scalar (or ``None``)."""
    if not losses:
        return None
    it = iter(losses.values())
    total = next(it)
    for v in it:
        total = total + v
    return total


__all__ = ["OmniModel"]
