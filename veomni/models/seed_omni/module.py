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


__all__ = ["OmniModule"]
