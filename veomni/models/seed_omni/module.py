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
sums them.  See ``docs/seed_omni/seed_omni_v2.md`` for the full contract.
"""

from typing import Any, Callable, Dict, List, Optional, Type


def pre_forward(context: str) -> Callable[[Callable], Callable]:
    """Decorator: register a **pre-hook** for the graph call-site ``context``.

    Instead of one ``pre_forward(method, ...)`` that branches on ``method``, a
    module with multiple call-sites declares one hook per call-site, each tagged
    with its ``context`` (the method name — ``"encode"`` / ``"decode"`` /
    ``"forward"``)::

        @pre_forward("encode")
        def encode_pre(self, conversation_list=None): ...

        @pre_forward("decode")
        def decode_pre(self, conversation_list=None): ...

    The framework keeps calling :meth:`ModuleMixin.pre_forward` (the dispatcher),
    which routes to the hook whose ``context`` matches the node's method. A
    single-call-site module may still just override ``pre_forward`` directly.
    """

    def decorator(fn: Callable) -> Callable:
        fn._omni_pre_context = context
        return fn

    return decorator


def post_forward(context: str) -> Callable[[Callable], Callable]:
    """Decorator: register a **post-hook** for the graph call-site ``context``.

    The post counterpart of :func:`pre_forward` — see it for the rationale::

        @post_forward("encode")
        def encode_post(self, **outputs): ...
    """

    def decorator(fn: Callable) -> Callable:
        fn._omni_post_context = context
        return fn

    return decorator


class CPUPreprocessor:
    """Picklable, weight-free CPU input-prep run inside DataLoader workers.

    A module whose ``pre_forward`` does heavy **CPU** input preparation (e.g. a
    text encoder's chat-template + tokenize, a vision tower's image normalize)
    can move that work off the main/GPU process by returning one of these from
    :meth:`ModuleMixin.build_cpu_preprocessor`.  The :class:`OmniModuleTrainer`
    orchestrator collects the active graph-node modules' preprocessors and runs
    them inside :class:`~veomni.data.data_collator.SeedOmniCollator` — which
    executes in the DataLoader worker — so the work overlaps with GPU compute via
    prefetch instead of blocking the main process inside ``pre_forward``.

    Contract:

    * **No model weights.** It is pickled / fork-inherited into worker processes,
      so it must hold only CPU-safe, picklable assets (tokenizer / image
      processor / special-token ids / config ints) — never the ``nn.Module``.
    * **CPU only.** Workers must not touch the training CUDA device; build CPU
      tensors (no ``device=``).  The main process's thin ``pre_forward`` does the
      single ``.to(device)``.
    * **In-place mutation.** ``__call__`` receives the batched
      ``conversation_list`` (``list[list[ConversationItem]]``) and mutates items'
      ``value`` / ``meta`` in place, tagging a sentinel in ``meta`` so the thin
      ``pre_forward`` knows the heavy work is already done (and falls back to the
      full self-contained path when the sentinel is absent, e.g. eager inference
      with no worker collator).
    """

    def __call__(self, conversation_list: List[List[Any]]) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} must implement __call__(conversation_list) and mutate it in place."
        )


class ModuleMixin:
    """Unified SeedOmni V2 mixin for both training and inference hooks.

    A module opts into the optional per-module training trace separately, by
    multi-inheriting its own ``XxxTraceMixin(TraceMixin)`` on the concrete model
    (``ModuleMixin`` itself does **not** inherit ``TraceMixin``).  See
    :class:`~veomni.models.seed_omni.tracemixin.TraceMixin`.
    """

    # Generic / combined processor (e.g. an HF ``XxxProcessor`` wrapping several
    # modalities). Single-modality modules instead declare the specific slots
    # below (``image_processor_class`` / ``video_processor_class`` / ...).
    processor_class: Optional[Type[Any]] = None
    image_processor_class: Optional[Type[Any]] = None
    video_processor_class: Optional[Type[Any]] = None
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

    @classmethod
    def _omni_hook_name(cls, marker: str, context: str) -> Optional[str]:
        """Resolve the method name tagged ``marker`` for call-site ``context``.

        Scans the MRO base-first so a subclass's hook overrides a base hook for
        the same ``context``; the result is cached on the class.
        """
        cache_attr = f"__omni_hooks_{marker}__"
        registry: Optional[Dict[str, str]] = cls.__dict__.get(cache_attr)
        if registry is None:
            registry = {}
            for klass in reversed(cls.__mro__):
                for name, attr in vars(klass).items():
                    ctx = getattr(attr, marker, None)
                    if ctx is not None:
                        registry[ctx] = name
            setattr(cls, cache_attr, registry)
        return registry.get(context)

    def pre_forward(self, method: str, **kwargs: Any) -> Dict[str, Any]:
        """Dispatch to the ``@pre_forward(method)``-decorated hook for this node's
        call-site (``"forward"`` / ``"encode"`` / ``"decode"`` / ...).

        Routes packing / SP slice / conversation extraction per call site. A
        module with multiple call-sites declares one ``@pre_forward(<method>)``
        hook each; a single-call-site module may instead override this method
        directly. Default (no hook, no override): identity pass-through.
        """
        name = type(self)._omni_hook_name("_omni_pre_context", method)
        if name is None:
            return kwargs
        return getattr(self, name)(**kwargs)

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
        """Dispatch to the ``@post_forward(method)``-decorated hook for this node's
        call-site — e.g. SP gather, final ``_loss`` mean, conversation write-back.

        Mirrors :meth:`pre_forward`. Default (no hook, no override): identity
        pass-through of the call-site return dict.
        """
        name = type(self)._omni_hook_name("_omni_post_context", method)
        if name is None:
            return outputs
        return getattr(self, name)(**outputs)

    def build_cpu_preprocessor(self) -> Optional["CPUPreprocessor"]:
        """Optional: return a picklable, weight-free :class:`CPUPreprocessor`.

        Default ``None`` = this module does no worker-side input-prep.  Override
        on a module whose ``pre_forward`` has heavy **CPU** work (tokenize /
        image normalize): build a :class:`CPUPreprocessor` from this module's
        already-loaded assets (``self._tokenizer`` / ``self._image_processor`` /
        config ints — never ``self`` / weights) and return it.  The orchestrator
        collects these from the active graph-node modules and runs them inside
        the worker-side collator, so the work overlaps with GPU compute and the
        module's ``pre_forward`` becomes a thin consumer.
        """
        return None

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
        from ..auto import build_tokenizer

        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        # Each per-module asset is loaded only when its class slot is declared,
        # via the declared class (e.g. the image processor reads
        # ``preprocessor_config.json`` rather than auto-detecting — a module dir
        # may also hold a ``video_preprocessor_config.json`` which would confuse
        # auto-resolution). The tokenizer is built by ``build_tokenizer``.
        # On failure the attr is set to ``None`` (best-effort; surfaced lazily by
        # the module when the modality is actually used).
        # ``set attr`` is the public name so the tokenizer goes through its
        # property setter (which may build chat markers / token ids); ``none attr``
        # is the private storage zeroed on failure. For processors the two match.
        #   (set attr, none attr, class attr, build_via_tokenizer)
        asset_specs = [
            ("_processor", "_processor", "processor_class", False),
            ("_image_processor", "_image_processor", "image_processor_class", False),
            ("_video_processor", "_video_processor", "video_processor_class", False),
            ("tokenizer", "_tokenizer", "tokenizer_class", True),
        ]
        for set_attr, none_attr, class_attr, build_via_tokenizer in asset_specs:
            if getattr(cls, class_attr, None) is None:
                continue
            try:
                if build_via_tokenizer:
                    asset = build_tokenizer(pretrained_model_name_or_path)
                else:
                    asset = getattr(cls, class_attr).from_pretrained(pretrained_model_name_or_path)
                setattr(model, set_attr, asset)
            except Exception:
                setattr(model, none_attr, None)
        return model

    def finalize(self, *, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Flush module-private generation buffers into a one-shot ``generated`` payload."""
        del ctx
        return {}


__all__ = [
    "ModuleMixin",
    "CPUPreprocessor",
    "pre_forward",
    "post_forward",
]
