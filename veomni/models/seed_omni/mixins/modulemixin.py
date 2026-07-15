"""
ModuleMixin — base hooks for every SeedOmni V2 sub-model.

Layout
------
* ``modulemixin.py`` — :class:`ModuleMixin` (shared defaults).
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


def pre_forward(*contexts: str) -> Callable[[Callable], Callable]:
    """Decorator: register a **pre-hook** for one or more graph call-sites.

    Instead of one ``pre_forward(method, ...)`` that branches on ``method``, a
    module with multiple call-sites declares one hook per call-site, each tagged
    with its ``context`` (the method name — ``"encode"`` / ``"decode"`` /
    ``"forward"``)::

        @pre_forward("encode")
        def encode_pre(self, conversation_list=None): ...

        @pre_forward("decode")
        def decode_pre(self, conversation_list=None): ...

        @pre_forward("encode", "offline_encode")
        def encode_pre(self, conversation_list=None): ...

    The framework keeps calling :meth:`ModuleMixin.pre_forward` (the dispatcher),
    which routes to the hook whose ``context`` matches the node's method. A
    single-call-site module may still just override ``pre_forward`` directly.
    """

    if not contexts:
        raise ValueError("@pre_forward requires at least one context.")

    def decorator(fn: Callable) -> Callable:
        fn._omni_pre_context = tuple(contexts)
        return fn

    return decorator


def post_forward(*contexts: str) -> Callable[[Callable], Callable]:
    """Decorator: register a **post-hook** for one or more graph call-sites.

    The post counterpart of :func:`pre_forward` — see it for the rationale::

        @post_forward("encode")
        def encode_post(self, **outputs): ...
    """

    if not contexts:
        raise ValueError("@post_forward requires at least one context.")

    def decorator(fn: Callable) -> Callable:
        fn._omni_post_context = tuple(contexts)
        return fn

    return decorator


def sp_pre_forward(*contexts: str) -> Callable[[Callable], Callable]:
    """Decorator: register a **per-module sequence-parallel loop-head hook**.

    A module opts into per-module SP (Ulysses) for a call-site by declaring this
    hook. The graph runs the endpoint ``sp_size`` times (one SP-group member per
    iteration); the driver first broadcasts the active owner's ``pre_forward``
    output to the whole group, THEN calls this hook with that shared data. So the
    hook only slices its inputs to this rank's ``1/sp_size`` chunk (+ any per-shard
    precompute such as ``cu_seqlens``), returning the plain ``forward`` kwargs::

        @sp_pre_forward("forward")
        def forward_sp_pre(self, inputs_embeds, ...): ...

    There is no ``owner`` argument and no broadcast here — every rank already
    holds the owner's data; the slice is owner-agnostic. Its :func:`sp_post_forward`
    counterpart gathers the output shard back to the owner (that one IS
    owner-/axis-aware). See :meth:`ModuleMixin.sp_pre_forward`.
    """

    if not contexts:
        raise ValueError("@sp_pre_forward requires at least one context.")

    def decorator(fn: Callable) -> Callable:
        fn._omni_sp_pre_context = tuple(contexts)
        return fn

    return decorator


def sp_post_forward(*contexts: str) -> Callable[[Callable], Callable]:
    """Decorator: register a **per-module sequence-parallel loop-tail hook**.

    The output counterpart of :func:`sp_pre_forward`: called at the END of each SP
    iteration with the active ``owner`` and the plain ``forward`` output, it
    gathers this rank's output shard to the owner (``sp_gather_to_owner``) so the
    owner reconstructs its full sample::

        @sp_post_forward("forward")
        def forward_sp_post(self, owner, hidden_states, ...): ...

    ``owner`` is the gather DESTINATION the driver dispatches — the hook must NOT use
    it to decide which rank keeps data (that is the driver's ``own_out`` selection).
    Stripping any SP padding the loop-head added IS this hook's job (SP-specific
    cleanup, done on the destination rank) — keep it out of the SP-agnostic
    ``post_forward``.
    """

    if not contexts:
        raise ValueError("@sp_post_forward requires at least one context.")

    def decorator(fn: Callable) -> Callable:
        fn._omni_sp_post_context = tuple(contexts)
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
      ``value`` / ``meta`` in place, tagging the module ``source`` so the thin
      ``pre_forward`` / ``generate`` reads the heavy work back uniformly.
    * **Shared by training + inference.** Training runs it inside
      :class:`~veomni.data.data_collator.SeedOmniCollator` (DataLoader worker);
      inference runs it once over the request in
      :meth:`~veomni.trainer.omni.omni_inferencer.OmniInferencer._preprocess_request`,
      before the FSM. The ``inference`` flag flips the train/infer-only behaviour:
      image modules **skip dummy injection** (no FSDP anchor at inference) and
      text encoders **append the assistant generation prompt**. Extra request
      options (e.g. ``generation_kwargs``) arrive via ``**kwargs`` so a module
      *could* vary its input-prep by them (classifier-free guidance duplicating the
      prompt, …); no current module needs them, but the hook is plumbed through.
    """

    def __call__(self, conversation_list: List[List[Any]], inference: bool = False, **kwargs: Any) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} must implement "
            "__call__(conversation_list, inference=False, **kwargs) and mutate it in place."
        )


class ModuleMixin:
    """Unified SeedOmni V2 mixin for both training and inference hooks.

    A module opts into the optional per-module training trace separately, by
    multi-inheriting its own ``XxxMetricMeterMixin(MetricMeterMixin)`` on the concrete model
    (``ModuleMixin`` itself does **not** inherit ``MetricMeterMixin``).  See
    :class:`~veomni.models.seed_omni.mixins.metric_meter_mixin.MetricMeterMixin`.
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
                    contexts = getattr(attr, marker, None)
                    if contexts is not None:
                        for ctx in contexts:
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

    def metric_meter_set_seqlens(self, method: str, seqlens: List[int]) -> None:
        """Stash the FULL (pre-SP-slice) per-sample token lengths for call-site ``method``.

        **Call this inside a ``pre_forward`` hook, BEFORE any SP gather/slice.** It
        is the single, uniform way a module reports its tokens for the optional
        per-module meter — even AR backbones use it (built from their
        ``cu_seqlens``) instead of a custom reader. It lives on ``ModuleMixin``
        (not ``MetricMeterMixin``) purely so the ``pre_forward`` hooks that call it
        resolve statically; the value is only ever *consumed* by
        :meth:`~veomni.models.seed_omni.mixins.metric_meter_mixin.MetricMeterMixin.metric_meter_token_lengths`
        (a no-op stash on a non-metered module — nothing drains it).

        Why pre-slice / own-data: ``metric_meter_add`` runs *after* ``pre_forward``
        (see ``TrainingGraph.step``), so its ``data`` is already this SP rank's
        shard — measuring it under-counts by ~``sp``. But the value must be this
        rank's **own** lengths (before ``sp_gather_seqs``), NOT the
        post-gather aggregate: ``OmniEnvironMeter`` sums tokens+FLOPs over the
        ``dp_group``, and in per-module SP the gather spans ``module_sp`` distinct
        DP ranks, so a post-gather value would be counted ``module_sp`` times.
        "Own data (full) + DP-sum" reconstructs the true global total, matching the
        non-SP run. (See constraints.md 7c.)
        """
        if not hasattr(self, "_metric_full_seqlens"):
            self._metric_full_seqlens: Dict[str, List[int]] = {}
        self._metric_full_seqlens[method] = [int(s) for s in seqlens]

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

    def supports_sp(self, method: str) -> bool:
        """True when this module declares a per-module SP loop for call-site ``method``.

        Opt-in: a module supports the looped Ulysses path for ``method`` iff it has
        an ``@sp_pre_forward(method)``-decorated hook. The graph only drives the SP
        loop (:func:`~veomni.models.seed_omni.graphs.dispatch.run_sp_looped_endpoint`)
        when this is ``True`` and the module's scoped ``sp_size > 1``; otherwise the
        plain ``method`` runs once.
        """
        return type(self)._omni_hook_name("_omni_sp_pre_context", method) is not None

    def sp_pre_forward(self, method: str, **kwargs: Any) -> Dict[str, Any]:
        """Dispatch to the ``@sp_pre_forward(method)`` loop-head hook.

        The graph driver has already broadcast the active owner's ``pre_forward``
        output to the whole SP group, so ``**kwargs`` is data every rank shares; the
        hook only slices it to this rank's ``1/sp_size`` shard (+ any per-shard
        precompute). No ``owner`` — the slice is owner-agnostic."""
        name = type(self)._omni_hook_name("_omni_sp_pre_context", method)
        if name is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no @sp_pre_forward('{method}') hook; "
                "check `supports_sp(method)` before driving the SP loop."
            )
        return getattr(self, name)(**kwargs)

    def sp_post_forward(self, method: str, owner: int, **outputs: Any) -> Dict[str, Any]:
        """Dispatch to the ``@sp_post_forward(method)`` loop-tail hook (gather this
        rank's output shard back to the ``owner``). Default: identity pass-through."""
        name = type(self)._omni_hook_name("_omni_sp_post_context", method)
        if name is None:
            return outputs
        return getattr(self, name)(owner=owner, **outputs)

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

    def customized_build_parallelize_model(
        self, *, weights_path: Optional[str], args: Any, **kwargs: Any
    ) -> Optional[Any]:
        """Optional override: the module owns its OWN parallelize + weight-load
        (+ optional param offload), bypassing VeOmni's generic
        ``build_parallelize_model`` / weight loader.

        Motivation
        ----------
        The generic path (``BaseTrainer._build_parallelized_model`` ->
        ``build_parallelize_model`` -> FSDP2 wrap + ``load_model_weights`` /
        ``rank0_load_and_broadcast_weights``) materializes every parameter on GPU
        (as an FSDP shard or a rank-0 broadcast buffer) and has no hook for
        bespoke loading such as per-layer streaming CPU offload of very large
        (e.g. EP-sharded MoE expert) weights that do not fit on GPU even when
        sharded. A module with such a need implements this hook to do its own
        meta-init-aware load / shard / offload and return the ready model.

        Contract
        --------
        * Called by :class:`OmniModuleTrainer` AFTER meta-init, INSIDE this
          module's ``use_parallel_state`` scope — so ``get_parallel_state()``
          returns THIS module's device mesh, and ``self.config`` /
          ``self.get_parallel_plan()`` are available.
        * When it returns a module, that module is used verbatim: the override
          owns EVERYTHING (parallelize/FSDP-or-not, weight load, param offload,
          gradient checkpointing, dtype/mixed-precision). VeOmni does not
          post-process it.
        * When it returns ``None`` (the default), the trainer falls back to the
          generic ``build_parallelize_model`` path — behavior is unchanged for
          every module that does not override this.

        Args:
            weights_path: This module's split-checkpoint snapshot dir
                (``args.model.model_path``), or ``None`` for random init.
            args: The (per-module) ``VeOmniArguments`` — read fsdp / mixed
                precision / gradient-checkpointing config from
                ``args.train.accelerator`` as needed.

        Returns:
            A fully parallelized + weight-loaded ``nn.Module`` ready to
            train/infer, or ``None`` to use the generic path.
        """
        del weights_path, args, kwargs
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
        from ...auto import build_tokenizer

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
    "sp_pre_forward",
    "sp_post_forward",
]
