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
from typing import TYPE_CHECKING, Any, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule

from ....distributed.clip_grad_norm import veomni_omni_module_clip_grad_norm
from ....distributed.parallel_state import is_parallel_state_registered, use_parallel_state
from ....utils import logging
from ....utils.checkpoint_utils import should_skip_hf_weight_load
from ....utils.device import get_device_type
from ...model_runtime import VeOmniModelRuntime
from ..mixins.metric_meter_mixin import MetricMeterMixin, MetricMeterResult
from ..utils.checkpoint import OmniModuleCheckpointManager
from .dispatch import unwrap_module_chain


if TYPE_CHECKING:
    from ....arguments.arguments_types import AcceleratorConfig
    from ....arguments.omni_arguments_types import OmniModuleRuntimeArguments, OmniTrainingArguments
    from ....trainer.callbacks import TrainerState


logger = logging.get_logger(__name__)


def unwrap_module(mod: nn.Module) -> nn.Module:
    """Strip DDP/LoRA/FSDP wrappers so callers reach the inner :class:`BaseMixin`."""
    return unwrap_module_chain(mod)


class ModuleRuntime(VeOmniModelRuntime):
    """One OmniModule's training unit (model + optimizer + lr_scheduler + FSDP2).

    A :class:`~veomni.models.model_runtime.VeOmniModelRuntime` whose "one model"
    happens to be one module of a composed Omni model. The base already owns the
    whole build sequence — meta-init, freeze/LoRA, FSDP2/DDP wrap + weight load,
    optimizer, lr-scheduler, gradient clip, and the per-model ``ParallelState``
    every one of those reads — so what is written here is only what a *module*
    does differently from a standalone model:

    * its config lives beside its weights, not at the composed checkpoint root
      (:meth:`build_model`);
    * its preprocessor is bound onto the model itself rather than held by the
      runtime, because the graph calls the module and the module needs it
      (:meth:`build_model_assets`);
    * it may be **fully frozen**, in which case it has no optimizer, no
      lr-scheduler and no checkpoint at all (:attr:`has_trainable_parameters`);
    * it can be built **for inference**, including a single-process eager path
      that skips the distributed build entirely;
    * it clips and checkpoints per-module (:class:`OmniModuleCheckpointManager`,
      ``<save_path>/global_step_N/<module>/``), and the orchestrator only sums
      the norms it returns.

    ``args`` is a per-module :class:`OmniModuleRuntimeArguments` — the module's
    slice of the launcher YAML merged over the model-level defaults, with
    ``model_path`` pointing at this module's split-checkpoint subfolder and its
    own ``accelerator`` / ``optimizer``. That is what lets sibling modules hold
    genuinely different topologies ("freeze the ViT, EP-shard the LLM, DDP the
    VAE") while sharing one build sequence.

    Job-wide concerns (process-group init, data pipeline, trace metering, the
    train loop) are **never** run here — :class:`OmniTrainer` owns them once, and
    cascades its ``on_{train,epoch,step}_*`` hooks into each module so every
    module checkpoints itself.

    Inherited from the base: an attribute this class does not define resolves on
    the wrapped module (``runtime.config``, ``runtime.parameters()``), so a
    mistyped runtime attribute silently reads the model rather than raising.
    """

    # Class default so ``__new__``-constructed tests and ``__getattr__``
    # forwarding to the inner ``nn.Module`` never confuse this flag with a
    # missing model attribute.
    _defer_parallelize: bool = False
    _global_accelerator: Optional["AcceleratorConfig"] = None

    args: "OmniModuleRuntimeArguments"
    train: Optional["OmniTrainingArguments"] = None
    _has_trainable_parameters: Optional[bool] = None

    def __init__(
        self,
        args: "OmniModuleRuntimeArguments",
        module_name: str,
        *,
        train: Optional["OmniTrainingArguments"] = None,
        for_inference: bool = False,
        global_accelerator: Optional["AcceleratorConfig"] = None,
    ):
        self.args = args
        self.model_name = module_name
        self.train = train
        self.optimizer = None
        self.lr_scheduler = None
        self._defer_parallelize = False
        self._global_accelerator = global_accelerator

        if for_inference:
            if args.accelerator.fsdp_config.fsdp_mode == "eager":
                self._init_eager_inference()
            else:
                args.accelerator.fsdp_config.mixed_precision.enable = False
                self._defer_parallelize = args.accelerator.fsdp_config.fsdp_scope == "model"
                self.setup()
                with self._scoped():
                    self.build_model()
                    self.build_model_assets()
                    self.build_parallelized_model()
                self.model.eval()
        else:
            self._defer_parallelize = args.accelerator.fsdp_config.fsdp_scope == "model"
            self.setup()
            with self._scoped():
                self.build_model()
                self.build_model_assets()
                self.freeze_model()
                self.build_parallelized_model()
                if not self._defer_parallelize:
                    self._scope_recompute_to_parallel_state()
                    self.build_optimizer()
                    self.build_checkpoint()

    @property
    def mesh_accelerator(self) -> "AcceleratorConfig":
        """The top-level accelerator once the composed model owns the wrap.

        Under ``fsdp_scope='model'`` the mesh, init device and wrap all belong to
        :class:`OmniModelRuntime`, so this module's YAML overlay (its own DDP /
        emb-parallel block) must not decide them — a module meta-initialized on
        a different mesh than the one it is later sharded over would not load.
        """
        if self._defer_parallelize and self._global_accelerator is not None:
            return self._global_accelerator
        return self.args.accelerator

    @property
    def module_name(self) -> str:
        """This module's name — the same identity the base calls ``model_name``.

        It is the registry key for the module's :class:`ParallelState`, the
        ``<module>/`` checkpoint subdir, and how the graph addresses it, so the
        omni-side name is kept rather than making every call site say
        ``model_name`` about a module.
        """
        return self.model_name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run this module's forward inside its own ``ParallelState``.

        The graph normally reaches the module through
        :meth:`OmniModelRuntime.module_context`, which scopes it. Calling the
        runtime directly has to scope it too: attention resolves its all-to-all
        group from the *current* state, so an unscoped forward under Ulysses SP
        would communicate over the orchestrator's ranks instead of this module's.

        Gated on registration for the same reason ``module_context`` is: an
        eager-inference module never runs :meth:`setup`, so it has no state to
        enter and must not be refused a forward over one.
        """
        scope = self._scoped() if is_parallel_state_registered(self.module_name) else nullcontext()
        with scope:
            return super().__call__(*args, **kwargs)

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
        self.build_model_assets()

    # ── Build (model, assets, parallelize) ────────────────────────────────────

    def build_model(self) -> None:
        """Meta-init this module's sub-model from the config beside its weights.

        Unlike a standalone model, a module reads ``model_path`` rather than
        ``config_path``: the latter is inherited from the composed model's
        arguments and points at the Omni checkpoint *root*, whose ``config.json``
        is the :class:`OmniConfig`, not this module's architecture.
        """
        args = self.args
        logger.info_rank0(f"ModuleRuntime '{self.module_name}': build module model")
        from ....models import build_foundation_model

        acc = self.mesh_accelerator
        self.model = build_foundation_model(
            config_path=args.model_path,
            weights_path=args.model_path,
            torch_dtype="float32" if acc.fsdp_config.mixed_precision.enable else "bfloat16",
            init_device=acc.init_device,
            ops_implementation=args.ops_implementation,
            config_kwargs=args.model_config,
        )
        self.model_config = self.model.config

    def build_model_assets(self) -> None:
        """Bind this module's preprocessor onto the model itself.

        A standalone runtime keeps the tokenizer/processor beside the model
        because the trainer's data pipeline reads through it. A module's caller
        is the graph, which addresses the *module* — so the assets are attached
        to the model (``bind_module_assets``) and travel with it, and HF export
        collects them back off the live model in
        :meth:`collect_hf_export_assets` rather than from a cached list.

        Meta-init skips ``from_pretrained``, so vision modules and text encoders
        that need a processor or tokenizer at train time get one here from this
        module's weights path. A no-op when the module declares no
        ``preprocessor_class``, or when an earlier eager ``from_pretrained``
        already bound one.
        """
        # The base caches its sidecars here for a whole-model export. A module's
        # are read off the live model at save time instead, so this stays empty
        # rather than aliasing the class-level default.
        self.model_assets = []

        model = self.model
        label = type(model).__name__
        if getattr(type(model), "preprocessor_class", None) is None:
            return
        if any(
            getattr(model, attr, None) is not None for attr in ("_image_processor", "_video_processor", "_tokenizer")
        ):
            return  # already bound by an earlier `from_pretrained` (e.g. eager inference)
        try:
            from ..processing.binding import bind_module_assets

            bind_module_assets(
                model,
                checkpoint_path=self.args.model_path,
                config_overrides=self.args.model_config,
            )
        except Exception as e:  # noqa: BLE001 — surfaced lazily by the module if the modality is used
            logger.warning_once(
                f"ModuleRuntime '{label}': could not bind module assets from {self.args.model_path}: {e}."
            )
            return
        logger.info_rank0(f"ModuleRuntime '{label}': bound module assets.")

    def freeze_model(self) -> None:
        """Freeze + LoRA this module, heading the base's report with its name.

        Which parameters freeze is the *module's* decision (``JanusVqvae`` freezes
        only its codec, most modules train in full), so the base's
        model-declares-its-own-policy hook is what runs. Only the logging differs:
        N modules build in sequence, so the parameter table and VRAM reading the
        base prints would otherwise say nothing about which module moved the
        number. A header rather than a second reading, so the log carries one
        VRAM line per module instead of two.
        """
        logger.info_rank0(f"ModuleRuntime '{self.module_name}': freeze + LoRA")
        super().freeze_model()

    def on_lora_matched_nothing(self) -> None:
        """A module the LoRA config did not target simply stays frozen.

        The composed model's ``lora_config`` reaches **every** module (it is a
        ``BaseModelArguments`` field, fanned out by ``_to_module_global_args``),
        so "no targets here" is the normal way to say *adapt the LLM, leave the
        ViT and the VAE alone* — not the misconfiguration it would be for a
        single-model job. The module then has no trainable parameters, which the
        rest of this class already handles: no optimizer, no lr-scheduler, no
        checkpoint manager, and an HF weight load rather than a DCP restore.

        A config that targets *nothing anywhere* **and** leaves the composed
        model with no trainable parameters is still an error; only the composer
        can see that. It is raised in :meth:`OmniModelRuntime.from_model_runtime`.
        """
        logger.info_rank0(
            f"ModuleRuntime '{self.module_name}': the LoRA config matched no parameters here; "
            "training this module frozen."
        )

    def customized_build_parallelize_model(
        self, *, weights_path: Optional[str], args: "OmniModuleRuntimeArguments", **kwargs: Any
    ) -> Optional[Any]:
        """Optional override on a **custom runtime** for bespoke parallelize + load.

        When this returns a module, that module is used verbatim — the override
        owns FSDP/DDP wrap, weight load, param offload, gradient checkpointing,
        and mixed precision. When it returns ``None`` (the default here), the
        generic :meth:`VeOmniModelRuntime.build_parallelized_model` path runs.

        This is the *runtime's* hook, which is why it carries the ``customized_``
        prefix: a **model** that wants to own its own wrap declares
        ``build_parallelize_model`` instead, and the base honours that too.

        Called inside this module's ``use_parallel_state`` scope after meta-init,
        so ``get_parallel_state()`` returns this module's device mesh.
        """
        del weights_path, args, kwargs
        return None

    def build_parallelized_model(self) -> None:
        """FSDP2/DDP-wrap this module and load its weights, unless a runtime owns it.

        A **custom runtime subclass** may fully own parallelize + weight-load by
        overriding :meth:`customized_build_parallelize_model` — e.g. a huge MoE
        backbone that streams EP-sharded experts to CPU, which the generic
        GPU-materializing loader has no hook for.

        When ``fsdp_scope='model'``, this is a no-op: the module stays on meta
        (freeze already applied) so :class:`OmniModelRuntime` can wrap the
        composed parent once, then :meth:`finish_deferred_parallelize` builds
        the optimizer on the now-DTensor parameters.
        """
        if self._defer_parallelize:
            logger.info_rank0(
                f"ModuleRuntime '{self.module_name}': deferring FSDP wrap to the composed "
                "OmniModel (accelerator.fsdp_config.fsdp_scope='model')."
            )
            return
        customized_model = self.customized_build_parallelize_model(
            weights_path=self.args.model_path,
            args=self.args,
        )
        if customized_model is not None:
            self.model = customized_model
            return
        super().build_parallelized_model()

    def finish_deferred_parallelize(self, *, for_inference: bool = False) -> None:
        """Optimizer / checkpoint / GC recompute after the parent OmniModel wrap.

        No-op when this module wrapped itself, or when this is an eager-inference
        module that never entered :meth:`setup`.
        """
        if not self._defer_parallelize:
            return
        if for_inference:
            with self._scoped():
                self._scope_recompute_to_parallel_state()
                self.model.eval()
        else:
            with self._scoped():
                self._scope_recompute_to_parallel_state()
                self.build_optimizer()
                self.build_checkpoint()

    # ── Parallel state (per-module device mesh) ────────────────────────────────

    def _scoped(self):
        """Context manager making this module's ParallelState current.

        The module owns its parallelism: every method that reads
        ``get_parallel_state()`` (optimizer / lr-scheduler build, gradient clip)
        enters this itself, so the orchestrator can call them plainly without
        knowing (or wrapping) the module's private state.
        """
        return use_parallel_state(self.module_name)

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

    @property
    def skip_hf_weight_load(self) -> bool:
        """Whether this module's initial HF weight materialization can be skipped.

        A full non-LoRA resume already carries model weights, so materializing the
        HF checkpoint first only to overwrite it costs a second memory peak that
        can OOM large MoE modules. Skipping is safe **only** if a DCP checkpoint
        will actually restore this module — and for a fully-frozen module
        :meth:`build_checkpoint` installs no checkpoint manager at all, so nothing
        would ever write its weights and it must load the released HF ones.
        """
        load_path = self.train.checkpoint.load_path if self.train is not None else None
        if not should_skip_hf_weight_load(load_path, self.args.lora_config):
            return False

        # A parameterless module has no persistent state for HF or DCP to restore
        # (e.g. a process-only stage), so there is nothing to materialize.
        if not self.model.state_dict():
            return True

        if not self.has_trainable_parameters:
            logger.info_rank0(
                f"ModuleRuntime[{self.module_name}]: fully frozen module has persistent model state; "
                "loading HF weights because no module DCP checkpoint will restore it."
            )
            return False

        return True

    def build_optimizer(self, param_groups: Optional[List[dict]] = None) -> None:
        """Build this module's optimizer, scoped to its own ParallelState.

        A distributed optimizer (Muon) reads ``get_parallel_state()`` at build
        time, so it must resolve to this module's mesh, not the orchestrator's.
        A no-op for a fully-frozen module: there is nothing to step.
        """
        if not self.has_trainable_parameters:
            return
        with self._scoped():
            super().build_optimizer(param_groups)

    def build_lr_scheduler(self, total_steps: int) -> None:
        """Build this module's lr-scheduler over ``total_steps``.

        The orchestrator (:meth:`~veomni.trainer.omni.omni_trainer.OmniTrainer._build_multi_lr_scheduler`)
        computes ``total_steps`` once the dataset-derived ``train_steps`` (already
        clamped by the global ``train.max_steps`` debug cap) is known. A no-op for
        a fully-frozen module: there is no optimizer to schedule.
        """
        if not self.has_trainable_parameters:
            return
        with self._scoped():
            super().build_lr_scheduler(total_steps)

    def clip_grad_norm(self, max_norm: Optional[float] = None, norm_type: float = 2.0) -> float:
        """Clip this module's grads under its own parallelism; return the module norm.

        Uses the omni-module clipper rather than the base's whole-model one: the
        orchestrator combines the per-module norms itself (see
        :func:`~veomni.distributed.clip_grad_norm.omni_clip_grad_norm`), so this
        must return *this* module's norm without reducing across modules.
        """
        if max_norm is None:
            max_norm = self.args.optimizer.max_grad_norm
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

        Read the wrap-time ``fsdp_config`` (not the orchestrator's): under
        ``fsdp_scope='model'`` that is the top-level accelerator so a module
        YAML ``ddp`` overlay does not skip reshard on an FSDP-wrapped child;
        otherwise it is the module's own merged ``accelerator``. A DDP module
        has no FSDP2 units to toggle (skipped by the ``isinstance`` check), and
        a module that keeps ``reshard_after_backward=True`` opts out here. No
        ``ParallelState`` is read (unlike :meth:`clip_grad_norm`), so this needs no
        scoping — ``set_reshard_after_backward`` just flips a flag on the unit.
        """
        fsdp_cfg = self.mesh_accelerator.fsdp_config
        if fsdp_cfg.fsdp_mode != "fsdp2" or fsdp_cfg.reshard_after_backward:
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
    def checkpoint_subfolder(self) -> str:
        if self.checkpoint is None:
            return self.module_name
        return self.checkpoint.checkpoint_subfolder

    @checkpoint_subfolder.setter
    def checkpoint_subfolder(self, value: str) -> None:
        if self.checkpoint is not None:
            self.checkpoint.checkpoint_subfolder = value

    def build_checkpoint(self) -> None:
        """Build this module's DCP / HF / LoRA checkpoint manager.

        Fully-frozen modules (no ``requires_grad`` params) get **no** manager:
        there is nothing to train, no optimizer to snapshot, and weights stay at
        the released checkpoint (e.g. offline_cache OE/ViT/VAE). That is why
        every save/load below tolerates a missing manager, where the base can
        assume one.
        """
        if not any(p.requires_grad for p in self.model.parameters()):
            logger.info_rank0(f"ModuleRuntime[{self.module_name}]: fully frozen — skipping DCP/HF checkpoint.")
            self._has_trainable_parameters = False
            self.checkpoint = None
            return
        self.checkpoint = OmniModuleCheckpointManager(self)

    def load(self) -> None:
        """Resume this module's DCP checkpoint, if one is configured."""
        if self.checkpoint is not None:
            self.checkpoint.load()

    def save_dcp(self, state: "TrainerState") -> None:
        """Write this module's distributed checkpoint (train resume)."""
        ckpt = self.checkpoint
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

    def save_hf_or_lora(self, state: "TrainerState", stage: str = "step_end") -> None:
        """Export this module's HF weights, or its LoRA adapter when LoRA is enabled.

        ``stage`` is part of the base signature; the omni path reads
        ``state.stage``, which the orchestrator sets before every save.
        """
        del stage
        ckpt = self.checkpoint
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

    def save_model_assets(self) -> None:
        """Not a module's job — the composed model writes the shared sidecars.

        The base writes ``model_assets`` beside a whole model's weights. A module
        has none to cache (its processor/tokenizer live on the model itself and
        are exported per-module by :meth:`collect_hf_export_assets`), and the
        job-level sidecars belong to the composed checkpoint root, which
        :meth:`OmniTrainer.save_model_assets` owns.
        """
        raise NotImplementedError(
            f"ModuleRuntime '{self.module_name}' does not write model assets; "
            "per-module sidecars go through collect_hf_export_assets(), and the composed "
            "checkpoint root's assets through OmniTrainer.save_model_assets()."
        )

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
