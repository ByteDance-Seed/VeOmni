"""
OmniModule: abstract base class for all composable modules in OmniModel.

Each OmniModule:
  - Implements `forward(**kwargs) -> dict` for training (DAG execution).
  - Optionally overrides `generate_step(**kwargs) -> dict` for inference (FSM execution).
  - Optionally overrides `get_parallel_plan()` for per-module FSDP/EP/SP config.
  - Optionally overrides `pre_forward` / `post_forward` for packing and SP slice/gather.

Return convention:
  - Any key ending with ``_loss`` in the return dict is treated as a training loss
    and collected by OmniModel.forward().
  - Keys used as outputs in connections must be present in the return dict.

Build lifecycle
---------------
Each OmniModule subclass is responsible for its own build lifecycle, identical
to what a BaseTrainer does for a monolithic model:

  1. ``_build_nn_module(cfg, init_device)`` — parse the raw config dict,
     construct the ``nn.Module`` on the given device (cpu / meta / cuda).
  2. ``build(cfg, build_args)``            — call ``_build_nn_module``, then
     call :func:`build_parallelize_model` to apply DDP / FSDP1 / FSDP2
     based on the global ``parallel_state.dp_mode``.  Returns the wrapped
     module ready for training.
  3. ``get_assets()``                      — return any model assets
     (tokenizer, processor, chat template) produced during build.
     Called by :meth:`OmniModel.collect_assets` after all modules are built.

pre_forward / post_forward contract
-------------------------------------
``pre_forward(**kwargs) -> dict``
    Called with a single *micro-batch* of inputs (already sliced from the full
    batch by OmniModel).  Responsible for:

    * **Packing** — reshaping variable-arity inputs (e.g. flattening
      ``(B, N_images, C, H, W)`` → ``(B*N_images, C, H, W)`` for vision
      encoders, or computing ``cu_seqlens`` for packed text sequences).
    * **SP slice** (LLM only) — splitting the sequence dimension across SP
      ranks using :func:`veomni.distributed.sequence_parallel.data.sp_pad_and_slice`.

    Default implementation: identity (returns kwargs unchanged).

``post_forward(outputs: dict) -> dict``
    Called immediately after ``forward``.  Responsible for:

    * **SP gather** — all-gathering the sequence dimension from SP ranks so
      that connection-routed tensors (e.g. ``hidden_states``) are always
      *full* sequences when consumed by the next module.

    Default implementation: identity.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch.nn as nn


@dataclass
class OmniBuildArgs:
    """Lightweight build config passed to :meth:`OmniModule.build`.

    Created by :class:`~veomni.trainer.omni_trainer.OmniTrainer` from the
    full ``VeOmniArguments`` and forwarded to every module's ``build()``.
    Mirrors the subset of arguments used by ``build_parallelize_model``.
    """

    # Device / dtype used to allocate the raw nn.Module before parallelisation.
    # ``"meta"`` is recommended so no memory is wasted before weight loading.
    init_device: str = "cpu"
    torch_dtype: str = "bfloat16"

    # ── FSDP / DDP ────────────────────────────────────────────────────────────
    enable_full_shard: bool = True
    enable_reshard_after_forward: bool = True
    mixed_precision: Any = None  # MixedPrecisionConfig instance or None
    enable_gradient_checkpointing: bool = True
    enable_fsdp_offload: bool = False
    enable_reentrant: bool = False
    enable_forward_prefetch: bool = True
    broadcast_model_weights_from_rank0: bool = False
    max_load_broadcast_size: float = 20.0
    cpu_load_param_name: Optional[str] = None

    # Extra module class names added to every module's ``_no_split_modules``
    # list (merged with the module's own list before passing to
    # ``build_parallelize_model``).
    extra_basic_modules: List[str] = field(default_factory=list)


class OmniModule(nn.Module, ABC):
    """Abstract base class for all OmniModel sub-modules.

    Subclasses must implement :meth:`forward`.  The ``generate_step`` method
    defaults to delegating to ``forward``, which is correct for encoder-style
    modules and for the AR-LLM in teacher-forcing mode.  Modules that need
    different sampling logic (e.g. a DiT denoising step) should override
    ``generate_step``.

    Build lifecycle
    ---------------
    Override :meth:`_build_nn_module` to parse the raw config dict and
    construct the typed ``OmniModule`` instance.  The default :meth:`build`
    classmethod then wraps it with :func:`build_parallelize_model`.  Override
    :meth:`build` for non-standard build patterns (e.g. partial weight tying,
    custom freeze logic).

    Loss convention
    ---------------
    ``forward`` should return a :class:`dict`.  Any key whose name ends with
    ``_loss`` (e.g. ``"lm_loss"``, ``"diffusion_loss"``) is automatically
    aggregated by :class:`~veomni.models.seed_omni.modeling_omni.OmniModel`.

    Parallel plan
    -------------
    Override :meth:`get_parallel_plan` to return a VeOmni ``ParallelPlan``
    (or equivalent) if this module requires non-default FSDP / SP / EP
    sharding.  Returning ``None`` means default sharding inherited from the
    OmniModel level.

    Micro-batch execution
    ---------------------
    OmniModel splits the global batch into micro-batches sized by
    ``module_config["micro_batch_size"]`` and calls each module as::

        inputs = module.pre_forward(**micro_batch_inputs)
        outputs = module(**inputs)
        outputs = module.post_forward(outputs)

    Override ``pre_forward`` / ``post_forward`` to add packing and SP logic.
    """

    # ── Build lifecycle ────────────────────────────────────────────────────────

    @classmethod
    def build(cls, cfg: Dict[str, Any], build_args: OmniBuildArgs) -> nn.Module:
        """Build and parallelize this module from its raw config dict.

        Default implementation:

        1. Allocates the raw ``OmniModule`` on ``build_args.init_device`` by
           calling :meth:`_build_nn_module`.
        2. Wraps it with :func:`~veomni.distributed.torch_parallelize.build_parallelize_model`,
           which selects DDP / FSDP1 / FSDP2 based on the global
           ``parallel_state.dp_mode``.  ``cfg["weights_path"]`` is used for
           per-module weight loading.

        Subclasses may override to add custom logic such as partial freezing,
        weight tying, or multi-source weight loading.

        Parameters
        ----------
        cfg:
            Raw module config dict from ``OmniConfig.modules[name]``.
            Must contain at least ``"model_type"``.  May contain
            ``"weights_path"`` for weight loading.
        build_args:
            Parallelisation settings forwarded from the trainer.

        Returns
        -------
        The wrapped (DDP / FSDP2) module, ready for training.
        """
        from veomni.distributed.torch_parallelize import build_parallelize_model

        module = cls._build_nn_module(cfg, build_args.init_device)

        basic_modules = list(
            set(getattr(module, "_no_split_modules", None) or []) | set(build_args.extra_basic_modules)
        )
        cpu_load_param_name = build_args.cpu_load_param_name
        if cpu_load_param_name is None and hasattr(module, "get_parallel_plan"):
            plan = module.get_parallel_plan()
            cpu_load_param_name = getattr(plan, "cpu_load_param_name", None)

        wrapped = build_parallelize_model(
            module,
            init_device=build_args.init_device,
            weights_path=cfg.get("weights_path"),
            enable_full_shard=build_args.enable_full_shard,
            enable_reshard_after_forward=build_args.enable_reshard_after_forward,
            mixed_precision=build_args.mixed_precision,
            enable_gradient_checkpointing=build_args.enable_gradient_checkpointing,
            enable_fsdp_offload=build_args.enable_fsdp_offload,
            basic_modules=basic_modules,
            enable_reentrant=build_args.enable_reentrant,
            enable_forward_prefetch=build_args.enable_forward_prefetch,
            broadcast_model_weights_from_rank0=build_args.broadcast_model_weights_from_rank0,
            max_load_broadcast_size=build_args.max_load_broadcast_size,
            cpu_load_param_name=cpu_load_param_name,
        )
        wrapped.train()
        return wrapped

    @classmethod
    def _build_nn_module(cls, cfg: Dict[str, Any], init_device: str = "cpu") -> "OmniModule":
        """Parse the raw config dict and instantiate the OmniModule.

        Called by :meth:`build` before parallelisation.  Subclasses must
        override this to convert ``cfg`` into a typed config object and call
        ``__init__``.  Use ``init_device`` to control allocation::

            with torch.device(init_device):
                return cls(MyModuleConfig(...))

        Using ``init_device="meta"`` avoids allocating actual memory before
        ``build_parallelize_model`` loads and shards the weights.
        """
        raise NotImplementedError(f"{cls.__name__} must implement _build_nn_module(cfg, init_device)")

    def get_assets(self) -> List[Any]:
        """Return model assets produced during :meth:`build`.

        Override in subclasses that own a tokenizer or processor, e.g.::

            def get_assets(self):
                return [self._tokenizer]

        The list is collected by :meth:`OmniModel.collect_assets` and exposed
        via :attr:`OmniTrainer.base.model_assets`.
        """
        return []

    # ── Training hooks ─────────────────────────────────────────────────────────

    def pre_forward(self, **kwargs) -> Dict[str, Any]:
        """Pre-process a single micro-batch before ``forward``.

        Default: identity pass-through.  Override to add packing (reshape
        multi-image pixel_values, compute cu_seqlens, etc.) and/or SP slice.

        Args:
            **kwargs: Micro-batch inputs as assembled by OmniGraph.

        Returns:
            Processed kwargs passed directly to :meth:`forward`.
        """
        return kwargs

    @abstractmethod
    def forward(self, **kwargs) -> Dict[str, Any]:
        """Training forward pass.

        Args:
            **kwargs: Processed micro-batch fields from ``pre_forward``.

        Returns:
            dict with arbitrary keys.  Keys ending in ``_loss`` are treated
            as scalar loss terms and summed into the total training loss.
        """

    def post_forward(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process ``forward`` outputs before they are stored.

        Default: identity pass-through.  Override to add SP gather so that
        downstream connections always receive full-sequence tensors.

        Args:
            outputs: The dict returned by :meth:`forward`.

        Returns:
            Processed outputs dict (full-sequence tensors).
        """
        return outputs

    def generate_step(self, **kwargs) -> Dict[str, Any]:
        """Single auto-regressive or diffusion generation step.

        Defaults to calling :meth:`forward`.  Override for modules where
        inference and training behave differently (e.g. a DiT that runs a
        full denoising loop during generation but computes diffusion loss
        during training, or a sampling-based next-token predictor).

        Args:
            **kwargs: Generation context accumulated by the FSM, including
                the raw request dict and all outputs produced so far in the
                current state body.

        Returns:
            dict that is merged back into the FSM context for the next step.
        """
        return self.forward(**kwargs)

    def get_parallel_plan(self) -> Optional[Any]:
        """Return a per-module VeOmni parallel plan, or ``None`` for default."""
        return None
