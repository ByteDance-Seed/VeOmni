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

``pre_forward(**kwargs) -> dict``
    Per-micro-batch packing / SP slice / data-routing prep.  Default:
    identity.  Called by the runtime *before* ``forward``.

``post_forward(outputs: dict) -> dict``
    Per-call post-processing — e.g. SP gather of routed tensors, computing
    the final ``_loss`` mean across micro-batches.  Default: identity.

``get_parallel_plan() -> Any | None``
    Per-module FSDP / EP / SP plan.  Default: ``None`` (inherit OmniModel
    defaults).

``get_assets() -> list``
    Module-owned tooling that should be saved alongside the module weights
    (vision processor, audio feature extractor, codebook lookup tables, ...).
    The global tokenizer lives at ``OmniConfig.tokenizer_path`` and is NOT
    returned here.  Default: ``[]``.

``finalize(*, ctx, request) -> dict``
    Inference-only post-processing hook called *once* when the FSM
    enters the framework-injected ``done`` state.  Override to dump
    accumulated outputs — e.g. tokenizer-decode all generated text,
    save accumulated VQ patches as images, write audio waveforms.
    The framework does not impose an accumulation scheme; modules that
    need cross-step history are responsible for maintaining it inside
    ``ctx`` during ``generate_step`` (typical shape: append the current
    step's ``input_ids`` / ``vq_token_id`` into a running list).
    Return value is collected under ``ctx['finalize'][<module_name>]``
    so callers can read decoded text, image paths, etc.  Default:
    ``{}`` (no-op — module has nothing to finalize).

``dummy_inputs(*, batch_size, device, dtype) -> dict``
    Zero-tensor placeholders the trainer fills in **during training** for
    micro-batches that are missing one of this module's inputs (e.g. a
    text-only sample missing ``pixel_values``).  This is the
    "training-side dummy forward" mechanism — every active node must
    forward on every micro-batch to keep FSDP DP/SP graphs aligned (see
    invariant 10).  Inference runners do NOT call this; they let the
    model fast-skip via ``if x is None: return {}`` and rely on
    permissive edge routing (an absent ``ctx[output]`` simply skips the
    edge).  Default: ``{}`` (no dummies — the trainer raises if a
    required input is missing).

Training vs. inference "no input" semantics
-------------------------------------------
The two runtimes treat a missing optional input differently — by design:

* **Training** (FSDP).  Every active node MUST forward on every
  micro-batch or DP/SP all-reduce hangs.  The trainer asks each module
  for ``dummy_inputs(...)`` and fills missing kwargs with zero tensors
  *before* dispatch.  The model's ``forward``/``encode``/``decode``
  therefore never sees ``None`` for a required input during training —
  the ``if x is None: return {}`` short-circuit is an inference-only
  fast path.  Important corner case: when the dummy zero output flows
  through ``masked_scatter`` with an all-False mask (no real placeholder
  positions), autograd drops the gradient back to the upstream module
  and FSDP grad-sync may mismatch.  The downstream backbone must add a
  ``+ x.sum() * 0.0`` "anchor" term in its ``pre_forward`` to force a
  zero-gradient path through the upstream module; see ``JanusLlama``.

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
import-safe in a torch-free / cpu-only environment so ``test_print_flow.py``
can exercise the graph runtime without GPUs.
"""

from typing import Any, Dict, List, Optional


class OmniModule:
    """Mixin for SeedOmni V2 modules.

    Multi-inherit alongside the real HF / diffusers backbone class::

        class JanusVisionEncoder(OmniModule, SiglipVisionModel):
            ...

    Only the hooks the module actually needs must be overridden.  All
    defaults below are safe identity passes.
    """

    # ── Training hooks ────────────────────────────────────────────────────────

    def pre_forward(self, **kwargs: Any) -> Dict[str, Any]:
        """Pre-process inputs before :meth:`forward`.

        Override to add packing (reshape multi-image pixel_values, compute
        cu_seqlens, build chat-template-aware position_ids, ...) and/or SP
        slicing.

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

    def post_forward(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process :meth:`forward` outputs.

        Override to add SP gather, final token-mean reduction of ``_loss``,
        cleanup of book-keeping fields, etc.

        Default: identity.
        """
        return outputs

    def generate_step(self, **kwargs: Any) -> Dict[str, Any]:
        """Single FSM-driven generation step.

        Default: delegate to :meth:`forward`.  Override when inference logic
        differs from training — e.g. a DiT runs its denoising loop here, an
        LM head samples a token here.
        """
        return self.forward(**kwargs)

    def set_tokenizer(self, tokenizer: Any) -> None:
        """Wire the global tokenizer and resolve vocabulary-specific token ids.

        Optional.  Text-side modules use this to learn special-token ids
        (boi / eoi / eos / image placeholder) at runtime instead of storing
        them in ``config.json``.  Default: no-op.
        """
        return None

    def finalize(self, *, ctx: Dict[str, Any], request: Dict[str, Any]) -> Dict[str, Any]:
        """Post-generation hook called once when the FSM enters ``done``.

        ``OmniModel.generate`` invokes this on every active module after the
        FSM loop terminates, then merges any non-empty return values into
        ``ctx['finalize'][<module_name>]``.  Use it to dump the module's
        accumulated outputs to a usable form — e.g. tokenizer-decode
        ``input_ids`` to text, save VQ patch sequences as images on disk,
        write audio waveforms.

        The framework imposes **no accumulation scheme**: modules that need
        per-step history (which most generative modules do — text, images,
        audio) are responsible for appending into a running list inside
        ``ctx`` during their ``generate_step``, then reading it back here.

        Inference-only.  Default: ``{}`` (no-op — module has nothing to
        finalize).
        """
        return {}

    # ── Parallelism / assets ──────────────────────────────────────────────────

    def get_parallel_plan(self) -> Optional[Any]:
        """Return a per-module VeOmni parallel plan, or ``None`` for default."""
        return None

    def get_assets(self) -> List[Any]:
        """Module-owned auxiliary artefacts to save alongside the weights.

        Vision / audio processors, codebooks, BPE pieces, etc.  The global
        tokenizer is stored at ``OmniConfig.tokenizer_path`` and is *not*
        returned here.  Default: ``[]``.
        """
        return []

    def dummy_inputs(self, *, batch_size: int, device: Any, dtype: Any) -> Dict[str, Any]:
        """Zero-tensor placeholders for training-side dummy forward.

        Called by the trainer (Step 2 ``OmniTrainer`` hook) when a
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
