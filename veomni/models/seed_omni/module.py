"""
ModuleMixin — base hooks for every SeedOmni V2 sub-model.

Layout
------
* ``module.py`` — :class:`ModuleMixin` (shared defaults).
* ``modules/<family>/<sub>/modulemixin.py`` — ``XxxModuleMixin(ModuleMixin)``
  with train/infer hooks and :meth:`init_omni_state`.
* ``modules/<family>/<sub>/modeling.py`` — HF ``PreTrainedModel`` body
  (``__init__``, ``forward``, weight layout).

Concrete classes combine mixin + HF base::

    class JanusSiglip(JanusSiglipModuleMixin, PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)   # → PreTrainedModel + init_omni_state
            ... submodules ...
            self.post_init()

``init_omni_state`` sets per-module caches (conversation carrier, KV / VQ
buffers, …).  Do **not** override ``post_init`` on mixins — keep HF
``post_init()`` in ``modeling.py`` after submodule construction.

Hooks (all optional except training-graph ``forward``)
------------------------------------------------------
``pre_forward`` / ``post_forward`` — read/write ``conversation_list``.
``forward`` — training compute; may return scalar ``_loss``.
``generate`` / ``generate_step`` — one FSM inference step.
``dummy_inputs`` — FSDP-aligned zero tensors when a modality is absent.
``reset_*_inference_state`` / ``finalize`` — inference lifecycle.
``get_parallel_plan`` / ``get_assets`` — build and checkpoint.

Training nodes must emit at most one token-mean ``_loss``; ``OmniModel``
sums them.  See ``docs/design/seed_omni_v2.md`` for the full contract.
"""

from typing import Any, Dict, List, Optional, Type


class ModuleMixin:
    """Unified SeedOmni V2 mixin for both training and inference hooks."""

    processor_class: Optional[Type[Any]] = None
    tokenizer_class: Optional[Type[Any]] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Route construction through the HF base, then init omni state.

        ``ModuleMixin`` sits *before* the co-inherited ``PreTrainedModel`` in
        the MRO, so a concrete module's ``super().__init__(config)`` lands
        here first.  We forward to ``PreTrainedModel.__init__`` (which sets up
        ``self.config`` and the ``nn.Module`` machinery) and then run
        :meth:`init_omni_state`, so subclasses never need a separate
        ``init_omni_state()`` call — just ``super().__init__(config)`` and the
        standard HuggingFace ``self.post_init()`` after building submodules.
        """
        super().__init__(*args, **kwargs)
        self.init_omni_state()

    def init_omni_state(self) -> None:
        """Initialize per-module runtime state (training/inference caches).

        Override this on a module mixin to set up instance attributes such as
        ``self._conversation_carrier`` / KV caches / sampling buffers.  It is
        invoked automatically by :meth:`__init__`.  This is a **leaf hook**:
        do *not* call ``super().init_omni_state()`` unless a parent mixin
        (e.g. the base text encoder) defines extra shared state worth chaining.
        """
        return None

    # ── Training hooks ────────────────────────────────────────────────────────

    def pre_forward(self, method: str, **kwargs: Any) -> Dict[str, Any]:
        """Pre-process inputs before the graph node's call-site method.

        ``method`` names the entry point configured on the active node
        (``"forward"``, ``"encode"``, ``"decode"``, ...).  Override to route
        packing / SP slice / conversation extraction per call site.

        Default: identity pass-through.
        """
        return kwargs

    def forward(self, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
        """Training forward pass.

        Override to provide module-specific behaviour. The default raises:
        every module that participates in the training graph must implement it.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.forward(**kwargs) is not implemented. "
            "Override it on the module mixin if this module appears in the training graph."
        )

    def post_forward(self, method: str, **outputs: Any) -> Dict[str, Any]:
        """Per-call post-processing — e.g. SP gather, final ``_loss`` mean.

        Default: identity pass-through of the call-site return dict.
        """
        return outputs

    def get_parallel_plan(self) -> Optional[Any]:
        """Return a per-module VeOmni parallel plan, or ``None`` for default."""
        return None

    def get_assets(self) -> List[Any]:
        """Module-owned auxiliary artefacts to save alongside the weights."""
        return []

    def dummy_inputs(self, *, batch_size: int, device: Any, dtype: Any) -> Dict[str, Any]:
        """Zero-tensor placeholders for training-side dummy forward."""
        del batch_size, device, dtype
        return {}

    # ── Inference hooks ───────────────────────────────────────────────────────

    def generate_step(self, **kwargs: Any) -> Dict[str, Any]:
        """Single FSM-driven generation step.

        Default: delegate to :meth:`forward`.  Override when inference logic
        differs from training — e.g. a DiT runs its denoising loop here, an
        LM head samples a token here.
        """
        return self.forward(**kwargs)

    def reset_local_inference_state(self) -> None:
        """Reset per-turn state inside an ongoing conversation.

        Local reset is used when starting a new user query while keeping
        conversation-level state (e.g. BOS/session flags) intact.
        """
        return None

    def reset_global_inference_state(self) -> None:
        """Reset the full conversation-level inference state.

        Global reset starts a fresh conversation from BOS; default delegates
        to local reset for modules without extra global state.
        """
        self.reset_local_inference_state()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Any, *args: Any, **kwargs: Any):
        """Load weights, then auto-load the per-module processor / tokenizer if declared."""
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
        """Flush module-private generation buffers into a one-shot ``generated`` payload."""
        del ctx
        return {}


__all__ = ["ModuleMixin"]
