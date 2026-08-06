# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ModuleRuntime — one OmniModule's training unit (model + opt + ckpt + FSDP2).

A single OmniModule sub-model's training unit (model + optimizer + lr_scheduler +
FSDP2) and the per-module checkpoint manager it owns.  The orchestrator
(:class:`~veomni.trainer.omni.omni_trainer.OmniTrainer`) builds one of these per
declared module, composes their models into one ``OmniModel`` and cascades the
``on_*`` lifecycle into each so every module checkpoints itself.
"""

import os
from contextlib import nullcontext
from dataclasses import fields
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from ....distributed.clip_grad_norm import veomni_omni_module_clip_grad_norm
from ....distributed.parallel_state import (
    get_parallel_state_by_name,
    init_parallel_state,
    use_parallel_state,
)
from ....distributed.torch_compile import CompileConfig
from ....distributed.torch_parallelize import build_parallelize_model
from ....models import build_foundation_model
from ....optim import build_lr_scheduler, build_optimizer
from ....trainer.base import _collect_muon_kwargs
from ....utils import helper, logging
from ....utils.device import get_device_type
from ....utils.model_utils import pretty_print_trainable_parameters
from ..mixins.metric_meter_mixin import MetricMeterMixin, MetricMeterResult
from ..utils.checkpoint import OmniModuleCheckpointManager
from .dispatch import unwrap_module_chain


if TYPE_CHECKING:
    from ....omni_arguments.arguments_types import OmniModuleRuntimeArguments
    from ....trainer.callbacks import TrainerState


logger = logging.get_logger(__name__)


def unwrap_module(mod: nn.Module) -> nn.Module:
    """Strip DDP/LoRA/FSDP wrappers so callers reach the inner :class:`ModuleMixin`."""
    return unwrap_module_chain(mod)


class ModuleRuntime:
    """One OmniModule's training unit (model + optimizer + lr_scheduler + FSDP2).

    Composition over inheritance (mirrors :class:`OmniTrainer`): rather than
    subclassing :class:`BaseTrainer`, it holds a bare ``BaseTrainer`` instance
    in ``self.base`` and drives only that trainer's **per-model build helpers**
    for a single OmniModule sub-model:

    * ``base._build_model``            — meta-init the sub-model from its
      ``config.json`` via the shared (OMNI-registry-aware) loader.
    * ``base._freeze_model_module`` / ``base._setup_lora`` — LoRA wrap +
      trainable-param report.  Freeze itself is **delegated to the module**: we
      call the module's :meth:`OmniModule.freeze_model`, which reads its own
      ``config.freeze`` and decides what to freeze (e.g. JanusVqvae freezes only
      its codec; most modules don't define it and train in full).
    * ``base._build_parallelized_model`` — wrap in its own FSDP2 unit + load the
      module's on-disk weights.
    * :meth:`_build_optimizer` / :meth:`_build_lr_scheduler` — one each, over this
      module's still-trainable params. Optimizer is built during :meth:`__init__`
      after FSDP wrap + freeze; the lr-scheduler is built later by
      :func:`~veomni.trainer.omni.omni_trainer.build_module_lr_schedulers` once
      ``train_steps`` is fixed from the dataset — this class holds no reference to
      the global :class:`OmniTrainingArguments` itself, only the ``total_steps``
      the caller passes to :meth:`_build_lr_scheduler`.
    * :meth:`_init_checkpoint` — builds :class:`OmniModuleCheckpointManager` for
      DCP / HF / LoRA save-load; trace / metering callbacks belong to the
      orchestrator, never here.

    The *global* concerns (distributed ``_setup``, data pipeline, trace
    metering, the train loop) are **never** run here — they are owned once by
    :class:`OmniTrainer`.  The orchestrator's ``on_{train,epoch,step}_*`` cascade
    into this trainer's matching :meth:`on_step_end` & co. so each module
    checkpoints itself.

    ``args`` is a per-module copy of the global arguments whose
    ``model.model_path`` point at this module's split-checkpoint
    subfolder, and whose ``model.model_config`` carries the module's YAML
    ``model_config:`` overrides — so the shared loader resolves the
    right OmniModule classes and the standard meta-init → FSDP2 → weight-load
    path is reused verbatim.
    """

    args: "OmniModuleRuntimeArguments"
    module_name: str
    model: Any
    model_config: Any
    optimizer: Optional[Any] = None
    lr_scheduler: Optional[Any] = None
    _has_trainable_parameters: Optional[bool] = None
    _checkpoint: Optional[OmniModuleCheckpointManager] = None

    def __init__(
        self,
        args: "OmniModuleRuntimeArguments",
        module_name: str,
        *,
        for_inference: bool = False,
    ):
        self.args = args
        self.module_name = module_name
        self.optimizer = None
        self.lr_scheduler = None

        if for_inference and args.accelerator.fsdp_config.fsdp_mode == "eager":
            self._init_eager_inference()
            return

        if for_inference:
            args.accelerator.fsdp_config.mixed_precision.enable = False

        self._setup()

        with use_parallel_state(self.module_name):
            self.model = self._build_module_model()
            self.model_config = self.model.config
            self._load_module_assets()
            if not for_inference:
                self._freeze_model_module()
            self._build_parallelized_model()
            if not for_inference:
                self._scope_recompute_to_parallel_state()
            if not for_inference and self.has_trainable_parameters:
                self._build_optimizer()
            if not for_inference:
                self._init_checkpoint()

        if for_inference:
            self.model.eval()

    def _init_eager_inference(self) -> None:
        """Single-process eager load via ``from_pretrained`` + ``device_map``."""
        args = self.args
        assert args.accelerator.fsdp_config.fsdp_mode == "eager"
        from .. import OMNI_MODEL_REGISTRY, read_model_type

        model_path = args.model_path
        overrides = dict(args.model_config or {})
        model_type = read_model_type(model_path)
        cls = OMNI_MODEL_REGISTRY[model_type]()
        if dist.is_initialized():
            device_map = {"": f"{get_device_type()}:{int(os.getenv('LOCAL_RANK', 0))}"}
        else:
            device_map = "auto"
        logger.info_rank0(
            f"ModuleRuntime '{self.module_name}': eager inference load "
            f"(model_type={model_type}, cls={cls.__name__}, device_map={device_map}) from {model_path}"
        )
        self.model = cls.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            **overrides,
        ).eval()
        self.model_config = self.model.config
        self._load_module_assets()

    def customized_build_parallelize_model(
        self, *, weights_path: Optional[str], args: "OmniModuleRuntimeArguments", **kwargs: Any
    ) -> Optional[Any]:
        """Optional override on a **custom runtime** for bespoke parallelize + load.

        When this returns a module, that module is used verbatim — the override
        owns FSDP/DDP wrap, weight load, param offload, gradient checkpointing,
        and mixed precision.  When it returns ``None`` (the default on
        :class:`ModuleRuntime`), the generic
        :meth:`_parallelize_module_model` path runs against :attr:`self.model`.

        Called inside this module's ``use_parallel_state`` scope after meta-init,
        so ``get_parallel_state()`` returns this module's device mesh.
        """
        del weights_path, args, kwargs
        return None

    def _build_parallelized_model(self):
        """FSDP2-wrap this module's model and load its weights (or defer to runtime).

        FSDP2 (and the meta-init weight load) preserve ``requires_grad``: the
        shard carries it (torch ``_fsdp_param.py``: ``sharded_param.requires_grad_(
        param.requires_grad)``) and the loader writes weights in-place
        (``param.data.copy_``), so the freeze applied in ``_freeze_model_module``
        survives the wrap — no need to re-assert it here.

        A **custom runtime subclass** may fully own parallelize + weight-load by
        overriding :meth:`customized_build_parallelize_model` — e.g. a huge MoE
        backbone that streams EP-sharded experts to CPU, which the generic
        GPU-materializing loader has no hook for. When it returns a model we use
        it verbatim; ``None`` (the default) keeps the generic path. Called within
        this module's ``use_parallel_state`` scope so the FSDP2/DDP wrap reads
        this module's mesh via ``get_parallel_state()``.
        """
        customized_model = self.customized_build_parallelize_model(
            weights_path=self.args.model_path,
            args=self.args,
        )
        if customized_model is not None:
            self.model = customized_model
        else:
            self.model = self._parallelize_module_model(self.model)

    def _build_module_model(self) -> torch.nn.Module:
        """Meta-init one OmniModule sub-model from its ``config.json``."""
        args = self.args
        logger.info_rank0("Build module model")
        return build_foundation_model(
            config_path=args.model_path,
            weights_path=args.model_path,
            torch_dtype="float32" if args.accelerator.fsdp_config.mixed_precision.enable else "bfloat16",
            init_device=args.accelerator.init_device,
            ops_implementation=args.ops_implementation,
            config_kwargs=args.model_config,
        )

    def _setup_module_lora(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap ``model`` with VeOmni LoRA when configured."""
        lora_config = self.args.lora_config
        if not bool(lora_config):
            return model

        customized = getattr(model, "customized_setup_lora", None)
        if callable(customized):
            return customized(lora_config)

        from ....lora import VeOmniLoraConfig, VeOmniLoraModel, resolve_fused_moe_lora_targets

        lora_adapter_path = lora_config.get("lora_adapter", None)
        if lora_adapter_path is not None:
            logger.info_rank0(f"Wrapping model with VeOmniLoraModel from {lora_adapter_path}.")
            return VeOmniLoraModel.from_pretrained(
                model,
                lora_adapter_path,
                is_trainable=lora_config.get("is_trainable", True),
            )

        resolved_config = resolve_fused_moe_lora_targets(model, lora_config)
        cfg = VeOmniLoraConfig.from_yaml(resolved_config)
        logger.info_rank0(f"Initialising VeOmni LoRA adapter from scratch: {cfg}.")
        return VeOmniLoraModel(model, cfg)

    def _parallelize_module_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """FSDP2-wrap ``model`` and load its weights."""
        args = self.args
        kwargs: Dict[str, Any] = {}
        cpu_load_param_name = None
        if hasattr(model, "get_parallel_plan"):
            cpu_load_param_name = getattr(model.get_parallel_plan(), "cpu_load_param_name", None)
        kwargs["cpu_load_param_name"] = cpu_load_param_name
        if bool(args.lora_config):
            lora_adapter_path = args.lora_config.get("lora_adapter", None)
            kwargs["adapter_path"] = lora_adapter_path
            kwargs["is_peft_model"] = True

        muon_expert_zero_comm = self.args.optimizer.type == "muon" and self.args.optimizer.muon_expert_zero_comm

        if args.fqn_to_index_mapping is not None:
            kwargs["fqn_to_index_mapping"] = args.fqn_to_index_mapping
        if args.accelerator.chunk_mbs_config.enable:
            kwargs["chunk_mbs_config"] = args.accelerator.chunk_mbs_config

        model = build_parallelize_model(
            model,
            init_device=args.accelerator.init_device,
            weights_path=args.model_path,
            enable_reshard_after_forward=args.accelerator.fsdp_config.reshard_after_forward,
            mixed_precision=args.accelerator.fsdp_config.mixed_precision,
            enable_gradient_checkpointing=args.accelerator.gradient_checkpointing.enable,
            basic_modules=list(set(getattr(model, "_no_split_modules", None) or []) | set(args.basic_modules)),
            enable_reentrant=args.accelerator.gradient_checkpointing.enable_reentrant,
            early_stop=args.accelerator.gradient_checkpointing.early_stop,
            enable_forward_prefetch=args.accelerator.fsdp_config.forward_prefetch,
            enable_fsdp_offload=args.accelerator.fsdp_config.offload,
            fsdp_offload_pin_memory=args.accelerator.fsdp_config.offload_pin_memory,
            broadcast_model_weights_from_rank0=args.accelerator.broadcast_model_weights_from_rank0,
            ep_sharded_stream_load=args.accelerator.ep_sharded_stream_load,
            max_load_broadcast_size=args.accelerator.fsdp_config.max_load_broadcast_size,
            muon_expert_zero_comm=muon_expert_zero_comm,
            compile_config=CompileConfig(
                **{field.name: getattr(args.accelerator.torch_compile, field.name) for field in fields(CompileConfig)}
            ),
            **kwargs,
        )
        return model.train()

    # ── Parallel state (per-module device mesh) ────────────────────────────────

    @property
    def parallel_state(self):
        """This module's :class:`ParallelState`, resolved from the global registry.

        A read-only lookup (no stored handle) keyed by :attr:`module_name` — the
        registry is the single source of truth. Encapsulates the module's private
        parallelism so callers (the orchestrator, clip/validate) never look it up
        by name themselves; the module owns entering it (see :meth:`_scoped`).
        """
        return get_parallel_state_by_name(self.module_name)

    def _scoped(self):
        """Context manager making this module's ParallelState current.

        The module owns its parallelism: every method that reads
        ``get_parallel_state()`` (optimizer / lr-scheduler build, gradient clip)
        enters this itself, so the orchestrator can call them plainly without
        knowing (or wrapping) the module's private state.
        """
        return use_parallel_state(self.module_name)

    def _setup(self):
        """Build this module's own :class:`ParallelState` and set it current.

        Mirrors the parallel-state half of :meth:`BaseTrainer._setup`.  The
        distributed process group / device / seed are already initialised once
        by the orchestrator (``OmniTrainer.base._setup``), so here we only build
        **this** module's own device mesh from its (merged) ``accelerator``
        and register it.  It is NOT made current here — the build sites scope to
        it explicitly via ``use_parallel_state(self.module_name)`` so the
        immediately-following meta-init + _build_parallelized_model (FSDP wrap)
        read this module's mesh rather than the orchestrator's.  The accelerator
        is already merged + validated by ``build_module_runtime_args``.

        The state is registered in the global ``_PARALLEL_STATE_REGISTRY`` under
        this module's name (``module_name``, unique per OmniConfig and distinct
        from the orchestrator's ``"base"``), so every scope site re-enters it by
        name via ``use_parallel_state(self.module_name)`` (the registry is the
        single source of truth — the module-trainer keeps no local handle).
        ``init_parallel_state`` never overwrites the orchestrator's current global
        state — it only adds to the registry / topology cache.
        """
        acc = self.args.accelerator
        init_parallel_state(
            dp_size=acc.dp_size,
            dp_replicate_size=acc.dp_replicate_size,
            dp_shard_size=acc.dp_shard_size,
            tp_size=acc.tp_size,
            pp_size=acc.pp_size,
            cp_size=acc.cp_size,
            ulysses_size=acc.ulysses_size,
            extra_parallel_sizes=acc.extra_parallel_sizes,
            extra_parallel_placement_innermost=acc.extra_parallel_placement_innermost,
            extra_parallel_names=acc.extra_parallel_names,
            dp_mode=acc.fsdp_config.fsdp_mode,
            async_enabled=acc.enable_async,
            name=self.module_name,
        )

    def _scope_recompute_to_parallel_state(self) -> None:
        """Make gradient-checkpoint recompute re-enter this module's ParallelState.

        torch ``checkpoint``'s ``context_fn`` returns ``(forward_ctx, recompute_ctx)``;
        the forward is already wrapped in :meth:`OmniModelRuntime.module_context`, but the
        recompute (in backward) escapes it. Setting ``recompute_ctx`` to
        :func:`use_parallel_state` keeps reads of the free ``get_parallel_state()``
        (EP groups, vocab-parallel ``emb`` group, …) resolving to this module's mesh
        during recompute. ``use_reentrant=True`` does not honour ``context_fn`` — but
        the omni path runs non-reentrant (``accelerator.gradient_checkpointing.enable_reentrant``
        defaults to ``False``).
        """
        name = self.module_name
        gc = self.args.accelerator.gradient_checkpointing

        def _recompute_context_fn():
            return nullcontext(), use_parallel_state(name)

        # DDP wraps the model (``.module``) and does not expose
        # ``gradient_checkpointing_enable``; FSDP2 wraps in place. Unwrap so the
        # call reaches the raw HF model regardless of dp_mode.
        if gc.enable:
            unwrap_module(self.model).gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": gc.enable_reentrant,
                    "context_fn": _recompute_context_fn,
                }
            )

    # ── Optimizer / lr-scheduler (optimizer in __init__; scheduler after train_steps) ─

    @property
    def has_trainable_parameters(self) -> bool:
        if self._has_trainable_parameters is None:
            self._has_trainable_parameters = any(param.requires_grad for param in self.model.parameters())
        return self._has_trainable_parameters

    def _build_optimizer(self):
        """Build this module's optimizer over its still-trainable params.

        Scoped to this module's own ParallelState: a distributed optimizer (e.g.
        Muon) reads ``get_parallel_state()`` at build time, so it must resolve to
        this module's mesh, not the orchestrator's.
        """
        with self._scoped():
            opt = self.args.optimizer
            self.optimizer = build_optimizer(
                self.model,
                lr=opt.lr,
                weight_decay=opt.weight_decay,
                fused=True,
                optimizer_type=opt.type,
                no_decay_modules=opt.no_decay_modules,
                no_decay_params=opt.no_decay_params,
                muon_kwargs=_collect_muon_kwargs(opt),
            )

    def _build_lr_scheduler(self, total_steps: int):
        """Build this module's lr-scheduler over ``total_steps`` (train_steps * num_train_epochs).

        The orchestrator (:func:`~veomni.trainer.omni.omni_trainer.build_module_lr_schedulers`)
        computes ``total_steps`` once the dataset-derived ``train_steps`` (already clamped by the
        global ``train.max_steps`` debug cap) is known, mirroring :meth:`BaseTrainer._build_lr_scheduler`.
        A no-op for a fully-frozen module (no ``self.optimizer`` to schedule).
        """
        if not self.has_trainable_parameters:
            return
        with self._scoped():
            opt = self.args.optimizer
            self.lr_scheduler = build_lr_scheduler(
                self.optimizer,
                train_steps=total_steps,
                lr=opt.lr,
                lr_min=opt.lr_min,
                lr_decay_style=opt.lr_decay_style,
                lr_decay_ratio=opt.lr_decay_ratio,
                lr_warmup_ratio=opt.lr_warmup_ratio,
                lr_start=opt.lr_start,
            )

    def _clip_grad_norm(self, max_norm: float, norm_type: float = 2.0) -> float:
        """Clip this module's grads under its own parallelism; return the module norm.

        Owns entering its own state: ``veomni_omni_module_clip_grad_norm`` reads
        the current ParallelState via ``get_parallel_state()`` (like
        ``veomni_clip_grad_norm``), so run it inside ``self._scoped()``. The
        orchestrator only sums the returned per-module norms — it never handles
        the module's private mesh.
        """
        with self._scoped():
            return veomni_omni_module_clip_grad_norm(self.model, max_norm, norm_type)

    def _model_reshard(self, reshard: bool) -> None:
        """Set ``set_reshard_after_backward`` on this module's FSDP2 units.

        A gradient-accumulation optimization owned per-module: the orchestrator
        decides *when* (``reshard=False`` on the first micro-step to keep params
        gathered across the window and skip the reshard → re-all-gather churn;
        ``reshard=True`` on the last so the final backward frees the full
        params). This method only *applies* that intent to this module — trading
        param memory for communication.

        Read the **module's own** ``fsdp_config`` (not the orchestrator's): each
        OmniModule has its own merged ``accelerator``, so ``fsdp_mode`` /
        ``reshard_after_backward`` may differ per module — a DDP module has no
        FSDP2 units to toggle (skipped by the ``isinstance`` check), and a module
        that keeps ``reshard_after_backward=True`` opts out here. No
        ``ParallelState`` is read (unlike :meth:`_clip_grad_norm`), so this needs no
        scoping — ``set_reshard_after_backward`` just flips a flag on the unit.
        """
        fsdp_cfg = self.args.accelerator.fsdp_config
        if fsdp_cfg.fsdp_mode != "fsdp2" or fsdp_cfg.reshard_after_backward:
            return
        try:
            from torch.distributed.fsdp import FSDPModule
        except ImportError:
            return
        # ``set_reshard_after_backward`` recurses into every nested FSDP unit by
        # default, so one call on the root-sharded model covers them all (the
        # generic ``parallelize_model_fsdp2`` ``fully_shard``s the root). A module
        # that owns its parallelize via a custom runtime's
        # ``customized_build_parallelize_model`` (contract: "FSDP-or-not") may leave the root un-sharded — it then owns
        # its own reshard policy, so skip rather than assume a root FSDP unit.
        model = self.model
        if isinstance(model, FSDPModule):
            model.set_reshard_after_backward(reshard)

    # ── Metric metering ────────────────────────────────────────────────────────

    def collect_step_metrics(self) -> Optional[MetricMeterResult]:
        """Drain this module's optional metric meter after one training step."""
        model = unwrap_module(self.model)
        if isinstance(model, MetricMeterMixin):
            return model.metric_meter_collect()
        return None

    # ── Checkpoint manager (I/O only; scheduling lives in trainer callbacks) ───

    @property
    def checkpoint(self) -> Optional[OmniModuleCheckpointManager]:
        return self._checkpoint

    @property
    def checkpoint_subfolder(self) -> str:
        if self._checkpoint is None:
            return self.module_name
        return self._checkpoint.checkpoint_subfolder

    @checkpoint_subfolder.setter
    def checkpoint_subfolder(self, value: str) -> None:
        if self._checkpoint is not None:
            self._checkpoint.checkpoint_subfolder = value

    def load(self) -> None:
        """Resume this module's DCP checkpoint, if one is configured."""
        if self._checkpoint is not None:
            self._checkpoint.load()

    def save_dcp(self, state: "TrainerState") -> None:
        """Write this module's distributed checkpoint (train resume)."""
        ckpt = self._checkpoint
        if ckpt is None:
            return
        # Only epoch_end / train_end can revisit a global_step that step_end already
        # wrote; step_end is never deduplicated because DCP and HF share one counter.
        if state.stage in ("epoch_end", "train_end") and ckpt.last_saved_step == state.global_step:
            logger.info_rank0(
                f"Skipping duplicate dcp save for module '{self.module_name}' at {state.stage} "
                f"(global_step {state.global_step} already saved)."
            )
            return
        ckpt.save_dcp(state)

    def save_hf_or_lora(self, state: "TrainerState") -> None:
        """Export this module's HF weights, or its LoRA adapter when LoRA is enabled."""
        ckpt = self._checkpoint
        if ckpt is None:
            return
        # Only epoch_end / train_end can revisit a global_step that step_end already
        # wrote; step_end is never deduplicated because DCP and HF share one counter.
        if state.stage in ("epoch_end", "train_end") and ckpt.last_saved_step == state.global_step:
            logger.info_rank0(
                f"Skipping duplicate hf save for module '{self.module_name}' at {state.stage} "
                f"(global_step {state.global_step} already saved)."
            )
            return
        ckpt.save_hf_or_lora(state)

    def _init_checkpoint(self) -> None:
        """Build this module's DCP / HF / LoRA checkpoint manager.

        Fully-frozen modules (no ``requires_grad`` params) skip checkpointing:
        there is nothing to train, no optimizer to snapshot, and weights stay at
        the released checkpoint (e.g. offline_cache OE/ViT/VAE).
        """
        if not any(p.requires_grad for p in self.model.parameters()):
            logger.info_rank0(f"ModuleRuntime[{self.module_name}]: fully frozen — skipping DCP/HF checkpoint.")
            self._has_trainable_parameters = False
            self._checkpoint = None
            return
        self._checkpoint = OmniModuleCheckpointManager(self)

    def _freeze_model_module(self):
        """Let the module freeze itself, then apply LoRA + trainable-param report."""
        if hasattr(self.model, "freeze_model"):
            self.model.freeze_model()
        self.model = self._setup_module_lora(self.model)
        logger.info_rank0(f"ModuleRuntime '{self.module_name}': trainable parameters after freeze/LoRA")
        pretty_print_trainable_parameters(self.model)
        helper.print_device_mem_info(f"ModuleRuntime '{self.module_name}': VRAM after build")

    def _load_module_assets(self):
        """Bind this module's preprocessor (processor / tokenizer / chat template)
        onto the runtime model for training.

        Meta-init skips ``from_pretrained``, so vision modules and text encoders
        that need a processor or tokenizer at train time bind them here from this
        module's weights path — via ``preprocessor_class.from_pretrained``
        (see :class:`~veomni.models.seed_omni.mixins.module_processor_mixin.Preprocessor`),
        which builds with no model instance involved, then
        :meth:`ModuleMixin.bind_preprocessor` copies its assets onto ``model``.
        A no-op when the module declares no ``preprocessor_class``, or when an
        earlier eager ``from_pretrained`` already bound one. HF export collects
        ``config`` + attached assets from the live model at save time via
        :meth:`collect_hf_export_assets` — nothing is cached on the runtime as a
        separate ``model_assets`` list.
        """
        model = self.model
        label = type(model).__name__
        preprocessor_cls = getattr(type(model), "preprocessor_class", None)
        if preprocessor_cls is None:
            return
        if any(
            getattr(model, attr, None) is not None for attr in ("_image_processor", "_video_processor", "_tokenizer")
        ):
            return  # already bound by an earlier `from_pretrained` (e.g. eager inference)
        weights_path = self.args.model_path
        try:
            # `args.model_config` is the same per-module YAML `model_config:` override
            # dict already threaded into the live model's `config_kwargs` (see
            # `_build_module_model`) — forward it so the preprocessor's config-derived
            # behavior (e.g. `enable_image`, `cache_mode`) agrees with `model.config`.
            preprocessor = preprocessor_cls.from_pretrained(weights_path, config_overrides=self.args.model_config)
        except Exception as e:  # noqa: BLE001 — surfaced lazily by the module if the modality is used
            logger.warning_once(f"ModuleRuntime '{label}': could not build preprocessor from {weights_path}: {e}.")
            return
        model.bind_preprocessor(preprocessor)
        logger.info_rank0(f"ModuleRuntime '{label}': bound preprocessor.")

    def collect_hf_export_assets(self) -> List[Any]:
        """Return this module's config + processor/tokenizer sidecars for HF export.

        ``self.model`` may still be DDP-wrapped here (FSDP2 composes in place and
        exposes the raw model's attributes, but ``DistributedDataParallel`` does
        not forward unknown attribute lookups to ``.module``) — unwrap first so
        ``config`` / processor / tokenizer resolve regardless of ``dp_mode``.
        """
        model = unwrap_module(self.model)
        assets: List[Any] = []
        cfg = getattr(model, "config", None)
        if cfg is not None:
            assets.append(cfg)
        for attr in ("_processor", "_image_processor", "_video_processor", "_tokenizer"):
            asset = getattr(model, attr, None)
            if asset is not None:
                assets.append(asset)
        return assets


__all__ = ["ModuleRuntime"]
