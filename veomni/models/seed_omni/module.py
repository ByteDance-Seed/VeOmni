"""
OmniModule — the *mixin* every SeedOmni V2 component multi-inherits from.

Why a mixin?
------------
A real SeedOmni component is almost always a HuggingFace ``transformers`` /
``diffusers`` model — e.g. ``LlamaModel``, ``SiglipVisionModel``, a custom
``DiT``.  Forcing those into a deep abstract base class would mean wrapping or
re-deriving them, which fights the upstream patchgen / monkey-patch flow.

Instead, ``OmniModule`` is a *plain mixin* with **zero required overrides**.
Real classes mix it in alongside their HF base::

    class JanusVisionEncoder(OmniModule, SiglipVisionModel):
        def forward(self, **kwargs) -> dict:
            ...
        def get_parallel_plan(self) -> ParallelPlan | None:
            ...

The graph runtime calls a small list of *optional* hooks (described below).
Each hook has a sensible default so a minimal module — say a pure feature
extractor — can implement only ``forward`` and skip everything else.

Optional hooks
--------------
``forward(**kwargs) -> dict``
    Training entry.  Return a dict whose keys feed downstream graph edges.
    May contain at most one ``_loss`` key (scalar, already token-mean-reduced
    across all micro-batches consumed inside this call).  See
    "Loss protocol" below.

``generate_step(**kwargs) -> dict``
    Inference entry.  Default delegates to :meth:`forward`.  Override when
    sampling logic differs (e.g. a DiT runs a denoising loop here but a
    diffusion-loss in ``forward``; an LM head samples a token here but
    computes CE in ``forward``).

``pre_forward(method, **kwargs) -> dict``
    Per-micro-batch packing / SP slice / data-routing prep.  ``method`` is
    the graph node entry point (``"forward"``, ``"encode"``, ``"decode"``, …).
    Default: identity.

``post_forward(method, **outputs) -> dict``
    Per-call post-processing — e.g. SP gather, final ``_loss`` mean,
    conversation write-back.  ``method`` matches the active graph node.
    Default: identity.

``freeze_model() -> None``
    Optionally freeze a subset of this module's params (the trainer calls it
    once after build, before the FSDP2 wrap / optimizer build).  There is no
    base default and no generic policy — only modules that actually freeze
    something implement it (e.g. ``JanusVqvae`` freezes its inner codec via
    its own ``config.freeze`` knob; the LLM backbone never overrides it).

``get_parallel_plan() -> Any | None``
    Per-module FSDP / EP / SP plan.  Default: ``None`` (inherit OmniModel
    defaults).

``get_assets() -> list``
    Module-owned tooling that should be saved alongside the module weights
    (vision processor, audio feature extractor, codebook lookup tables, ...).
    Tokenizers belong on the module that needs them (e.g. ``janus_text_encoder``).
    Default: ``[]``.

``finalize(*, ctx) -> dict``
    Inference-only flush hook.  ``OmniModel.generate`` calls it on *every*
    module exactly once, but **only** when the ``max_new_tokens`` safety cap
    trips before the FSM reaches ``done`` (the normal-completion path emits
    via each module's per-step ``generated`` payload instead).  Override to
    flush any partially-accumulated output the module still holds — e.g. the
    text encoder decodes its buffered token cache, the VQVAE decodes a full
    VQ grid.  Return ``{"generated": {"type": ..., "value": ...}}`` to append
    to :attr:`OmniModel.generated`, or ``{}`` for nothing.  Default: ``{}``.

``dummy_inputs(*, batch_size, device, dtype) -> dict``
    Zero-tensor placeholders the trainer fills in **during training** for
    micro-batches that are missing one of this module's inputs (e.g. a
    text-only sample missing ``pixel_values``).  This is the
    "training-side dummy forward" mechanism — every active node must
    forward on every micro-batch to keep FSDP DP/SP graphs aligned (see
    "Training vs. inference no input semantics" below).  Inference runners
    do NOT call this; they let the model fast-skip via
    ``if x is None: return {}``.  Default: ``{}`` (no dummies).

Training vs. inference "no input" semantics
-------------------------------------------
The two runtimes treat a missing optional input differently — by design:

* **Training** (FSDP).  Every active node MUST forward on every
  micro-batch or DP/SP all-reduce hangs.  When a micro-batch lacks one
  of a module's inputs (e.g. a text-only sample with no image), the
  encoder runs its :meth:`dummy_inputs` zero tensors and appends a
  ``role="dummy"`` placeholder item to ``conversation_list`` instead of
  ``return {}``.  The backbone skips dummy rows when packing but folds a
  ``+ dummy.value.mean() * 0.0`` "anchor" term into ``inputs_embeds``
  (see ``JanusLlama._fold_fsdp_dummy_anchors``) so a zero-gradient path
  still flows back through the dummy-producing encoder and FSDP grad-sync
  stays aligned across ranks.

* **Inference** (no FSDP).  No grad sync, no DP alignment.  The runtime
  does **not** fill dummies; ``GenerationGraph.step`` permissively
  skips an edge whose source produced an empty ``{}``.  The destination
  node still executes (with ``None`` for the absent kwarg) so its other
  inputs route normally.

Loss protocol (single ``_loss`` key)
------------------------------------
* A module may emit *at most one* loss term per node.
* The loss key MUST be exactly ``"_loss"`` (no ``"text_loss"``,
  ``"gen_loss"`` etc — module identity is already disambiguating because
  ``OmniModel`` indexes by node name).
* The value MUST be a **token-level mean** computed inside ``forward``
  /``post_forward`` — i.e. the module is responsible for summing per-token
  CE across all its micro-batches and dividing by the matching valid-token
  count.  Returning a per-batch mean breaks gradient correctness when token
  counts differ across micro-batches.
* ``OmniModel.forward`` collects every node's ``_loss`` and sums them into
  the total scalar.  Modules that produce no loss return a dict without
  ``_loss``.

Build / save lifecycle
----------------------
``build_*`` and ``CheckpointCallback`` are wired by ``OmniModel`` /
``OmniTrainer`` — they walk every module and call:

  1. :func:`veomni.distributed.torch_parallelize.build_foundation_model`
     ``(module_cfg, init_device)`` — meta-device or eager construction.
  2. :func:`veomni.distributed.torch_parallelize.build_parallelize_model`
     ``(model, weights_path=cfg["weights_path"], plan=module.get_parallel_plan(), ...)``
  3. Per-module :class:`~veomni.trainer.callbacks.CheckpointCallback` writes
     to ``<ckpt_root>/<module_name>/{model.safetensors, config.json,
     <assets...>}`` — each module's directory is self-contained.

These functions are imported lazily by the trainer; the mixin itself stays
import-safe in a torch-free / cpu-only environment so graph-runtime
tests can exercise the FSM without GPUs.
"""

from typing import Any, Dict, List, Optional, Type


class OmniModule:
    """Mixin for SeedOmni V2 modules.

    Multi-inherit alongside the real HF / diffusers backbone class::

        class JanusVisionEncoder(OmniModule, SiglipVisionModel):
            ...

    Only the hooks the module actually needs must be overridden.  All
    defaults below are safe identity passes.
    """

    # ── Per-module asset wiring ───────────────────────────────────────────────
    #
    # A subclass that consumes raw PIL / waveform inputs at inference time
    # declares its processor class here (e.g.
    # ``processor_class = JanusSiglipProcessor``).  :meth:`from_pretrained`
    # then loads it from the same weights folder and stashes it on
    # ``self._processor`` so the module's ``generate`` can tensorise its
    # own inputs — no external wiring step required.
    #
    # Likewise, a module that owns its **own** tokenizer (e.g. Janus
    # ``janus_text_encoder``) keeps ``self._tokenizer = None`` in ``__init__``
    # a ``tokenizer`` property setter (``OmniModuleTrainer`` assigns
    # ``model.tokenizer = build_tokenizer(...)``, same slot as SigLIP ``_processor``).
    #
    # Leave both as ``None`` (default) for modules that only consume already-
    # tensorised inputs (SigLIP, VQVAE, LLaMA backbone).
    processor_class: Optional[Type[Any]] = None
    tokenizer_class: Optional[Type[Any]] = None

    # ── Training hooks ────────────────────────────────────────────────────────

    def pre_forward(self, method: str, **kwargs: Any) -> Dict[str, Any]:
        """Pre-process inputs before the graph node's call-site method.

        ``method`` names the entry point configured on the active node
        (``"forward"``, ``"encode"``, ``"decode"``, …).  Override to route
        packing / SP slice / conversation extraction per call site.

        Default: identity pass-through.
        """
        return kwargs

    def forward(self, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
        """Training forward pass.

        Override to provide module-specific behaviour.  The default raises —
        any module that participates in the *training* graph must implement
        this.  Inference-only modules may override only :meth:`generate_step`
        and leave ``forward`` un-implemented.

        Returns
        -------
        dict
            Arbitrary keys consumed by downstream edges.  May contain at
            most one ``_loss`` scalar (token-mean reduced across all
            micro-batches consumed in this call).
        """
        raise NotImplementedError(
            f"{type(self).__name__}.forward(**kwargs) is not implemented. "
            "Override it on the OmniModule mixin if this module appears in the training graph."
        )

    def post_forward(self, method: str, **outputs: Any) -> Dict[str, Any]:
        """Per-call post-processing — e.g. SP gather, final ``_loss`` mean.

        Default: identity pass-through of the call-site return dict.
        """
        return outputs

    def generate_step(self, **kwargs: Any) -> Dict[str, Any]:
        """Single FSM-driven generation step.

        Default: delegate to :meth:`forward`.  Override when inference logic
        differs from training — e.g. a DiT runs its denoising loop here, an
        LM head samples a token here.
        """
        return self.forward(**kwargs)

    # ── HF lifecycle override ─────────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Any, *args: Any, **kwargs: Any):
        """Load weights, then auto-load the per-module processor / tokenizer if declared.

        Loads weights via the next-in-MRO ``from_pretrained`` (the
        concrete HF base — :class:`PreTrainedModel` or
        :class:`ModelMixin`).  When :attr:`processor_class` / :attr:`tokenizer_class`
        is set, also loads that asset from the same path and stashes it on
        ``model._processor`` / ``model._tokenizer`` so the module's
        :meth:`generate` can tensorise its own inputs — there's no external
        wiring step.

        The assets are loaded via the registry-aware ``build_processor`` /
        ``build_tokenizer`` — the *same* loaders the training path
        (``OmniModuleTrainer._build_model_assets``) uses — so the inference and
        training paths produce identical processor / tokenizer objects.

        A missing / unreadable asset folder is a silent no-op; the module's
        ``generate`` is responsible for raising a clear error if it actually
        needs the asset at call time.  This keeps stripped training checkpoints
        (no preprocessor / tokenizer JSON shipped) loadable.
        """
        # Lazy import to avoid an import cycle (``veomni.models.auto`` pulls in
        # the loader / ops stack at import time, while this module is imported
        # while that stack is still initialising).
        from ..auto import build_processor, build_tokenizer

        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if cls.processor_class is not None:
            try:
                model._processor = build_processor(pretrained_model_name_or_path)
            except Exception:
                # Best-effort: defer the "missing processor" error to ``generate``
                # where the message can reference the actual call site.
                model._processor = None
        if cls.tokenizer_class is not None:
            try:
                model._tokenizer = build_tokenizer(pretrained_model_name_or_path)
            except Exception:
                model._tokenizer = None
        return model

    def finalize(self, *, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Flush module-private generation buffers into a one-shot ``generated`` payload.

        ``OmniModel.generate`` calls this on *every* module exactly once, but
        **only** when the ``max_new_tokens`` safety cap trips before the FSM
        reaches ``done`` (normal completion emits per-step instead).  The
        module inspects its own buffers to decide whether to emit or no-op.

        Return ``{"generated": {"type": ..., "value": ...}}`` to append to
        :attr:`~veomni.models.seed_omni.modeling_omni.OmniModel.generated`,
        or ``{}`` when there is nothing to emit.  Modules must clear their
        private caches inside this hook so artefacts do not linger across
        later FSM spans or multi-turn turns.

        Inference-only.  Default: ``{}`` (no-op).
        """
        return {}

    # ── Parallelism / assets ──────────────────────────────────────────────────

    def get_parallel_plan(self) -> Optional[Any]:
        """Return a per-module VeOmni parallel plan, or ``None`` for default."""
        return None

    def get_assets(self) -> List[Any]:
        """Module-owned auxiliary artefacts to save alongside the weights.

        Vision / audio processors, tokenizers, codebooks, etc.  Default: ``[]``.
        """
        return []

    def dummy_inputs(self, *, batch_size: int, device: Any, dtype: Any) -> Dict[str, Any]:
        """Zero-tensor placeholders for training-side dummy forward.

        Called by :class:`~veomni.trainer.omni_trainer.OmniTrainer` when a
        micro-batch is missing one of this module's required inputs.
        Override to return zero tensors with the **right shape** so the
        full forward path runs and FSDP DP/SP graphs stay aligned across
        ranks (see module-doc "Training vs. inference no input
        semantics").

        Inference runners do NOT call this; they let the model
        ``return {}`` and rely on permissive edge routing.

        Default: ``{}`` (no dummies — appropriate for modules whose
        inputs always come from upstream nodes inside the graph; the
        upstream node's own ``dummy_inputs`` populates the kwargs that
        eventually reach this module).
        """
        return {}


__all__ = ["OmniModule"]
