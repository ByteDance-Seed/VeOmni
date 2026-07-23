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

"""OmniModuleTrainer — one OmniModule's training unit + its checkpoint callbacks.

A single OmniModule sub-model's training unit (model + optimizer + lr_scheduler +
FSDP2) and the per-module checkpoint callbacks it owns.  The orchestrator
(:class:`~veomni.trainer.omni.omni_trainer.OmniTrainer`) builds one of these per
declared module, composes their models into one ``OmniModel`` and cascades the
``on_*`` lifecycle into each so every module checkpoints itself.

Why one trainer per module (vs. one wrapper over ``OmniModel``)?

* Each module is a self-contained HF model with its own ``_no_split_modules``
  and on-disk snapshot — a per-module ``BaseTrainer`` loads its weights from
  the module ``weights_path`` and reuses base build/optimizer/lora logic.
* The training DAG chains hidden states across modules; a **single**
  ``loss.backward()`` still propagates across every FSDP2 unit.  Each module's
  FSDP2 reduce-scatter fires from its own backward hooks.
* Gradient clipping is **global** (over ``OmniModel``'s full DTensor param
  set); the optimizer step iterates every per-module optimizer.
"""

import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch.distributed as dist

from ...distributed.clip_grad_norm import veomni_omni_module_clip_grad_norm
from ...distributed.parallel_state import (
    get_parallel_state_by_name,
    init_parallel_state,
    use_parallel_state,
)
from ...models import build_tokenizer
from ...models.seed_omni.mixins.metric_meter_mixin import MetricMeterMixin, MetricMeterResult
from ...models.seed_omni.mixins.offline_encoding import OfflineEncodingMixin
from ...models.seed_omni.modeling_omni import _unwrap_module
from ...utils import logging
from ..base import BaseTrainer
from ..callbacks import (
    Callback,
    CheckpointerCallback,
    HFLoraCkptCallback,
    HuggingfaceCkptCallback,
)


if TYPE_CHECKING:
    from ...arguments import VeOmniArguments
    from ..callbacks import TrainerState


logger = logging.get_logger(__name__)


class _FrozenModuleNoOpCkptCallback(Callback):
    """No-op checkpoint hooks for fully-frozen modules (no optimizer / DCP save)."""


# ── Per-module checkpoint callbacks (reuse the single-model callbacks) ──────────
#
# Rather than re-implement DCP / HF / LoRA save-load, each OmniModule reuses the
# shared ``CheckpointerCallback`` / ``HuggingfaceCkptCallback`` / ``HFLoraCkptCallback``
# bound to its own ``module_trainer.base``.  The only per-module differences are
# captured by the overridable seams those classes expose:
#
#   * directory  → ``…/global_step_{N}/<module>/`` (DCP + HF) and the LoRA export,
#   * extra_state → **per-model only** (``lr_scheduler``); the global resume state
#     (step / dataloader / environ-meter / rng) is owned once by the orchestrator
#     (:class:`OmniGlobalStateCallback`).
#
# On-disk layout::
#
#     <save_path>/global_step_{N}/
#     ├── <module_a>/        # DCP {model, optimizer, extra_state={lr_scheduler}} (+ hf_ckpt/)
#     ├── <module_b>/        # …
#     └── trainer_state.pt   # global: step / dataloader / environ-meter / rng


class _OmniModulePayloadMixin:
    """Retarget a single-model checkpoint callback at one OmniModule sub-tree.

    Mixed in **before** the concrete base callback so these overrides win.
    ``self.module_name`` (the module's YAML key, passed in at construction)
    is the ``<module>/`` subdir every save / load path is nested under.
    """

    def __init__(self, trainer: "BaseTrainer", module_name: str) -> None:
        self.module_name = module_name
        super().__init__(trainer)

    def _module_subdir(self, root: str, state: "TrainerState") -> str:
        return os.path.join(root, f"global_step_{state.global_step}", self.module_name)

    def _save_dir(self, state: "TrainerState") -> str:
        return self._module_subdir(self.trainer.args.train.checkpoint.save_path, state)

    def _output_dir(self, state: "TrainerState") -> str:
        return self._module_subdir(self.trainer.args.train.checkpoint.output_dir, state)

    def _load_dir(self) -> Optional[str]:
        load_path = self.trainer.args.train.checkpoint.load_path
        return None if load_path is None else os.path.join(load_path, self.module_name)

    def _model_assets_dir(self) -> str:
        return os.path.join(self.trainer.args.train.checkpoint.model_assets_dir, self.module_name)

    def _extra_state(self, state: "TrainerState") -> Dict[str, Any]:
        # Per-model only — the global step / dataloader / environ-meter / rng are
        # saved once by OmniGlobalStateCallback on the orchestrator.
        lr_scheduler = getattr(self.trainer, "lr_scheduler", None)
        return {"lr_scheduler": None if lr_scheduler is None else lr_scheduler.state_dict()}

    def _load_extra_state(self, extra_state: Dict[str, Any]) -> None:
        lr_sd = extra_state.get("lr_scheduler")
        lr_scheduler = getattr(self.trainer, "lr_scheduler", None)
        if lr_sd is not None and lr_scheduler is not None:
            lr_scheduler.load_state_dict(lr_sd)

    def _offline_cache_model(self) -> Optional[OfflineEncodingMixin]:
        model = _unwrap_module(self.trainer.model)
        if not isinstance(model, OfflineEncodingMixin) or model.cache_mode == "full":
            return None
        return model

    def _load_partial_dcp_checkpoint(self, model: OfflineEncodingMixin) -> None:
        load_dir = self._load_dir()
        if load_dir is None:
            return
        self.trainer.checkpointer.wait_for_pending_save()
        model.load_partial_dcp_checkpoint(load_dir, trainer=self.trainer)
        if dist.is_initialized():
            dist.barrier()
        logger.info_rank0(f"Load partial offline-cache checkpoint from {load_dir} successfully!")

    def _save_partial_dcp_checkpoint(self, model: OfflineEncodingMixin, state: "TrainerState") -> None:
        model.save_partial_dcp_checkpoint(self._save_dir(state), trainer=self.trainer, state=state)
        self._last_saved_step = state.global_step

    def _save_full_hf_checkpoint(self, model: OfflineEncodingMixin, state: "TrainerState") -> None:
        hf_weights_path = os.path.join(self._save_dir(state), "hf_ckpt")
        if self.trainer.args.train.global_rank == 0:
            model.save_full_hf_checkpoint(
                hf_weights_path,
                source_path=self.trainer.args.model.model_path,
                trainer=self.trainer,
                state=state,
            )
        if dist.is_initialized():
            dist.barrier()

        self._last_saved_step = state.global_step


class OmniModuleDcpCallback(_OmniModulePayloadMixin, CheckpointerCallback):
    """Per-module DCP resume checkpoint (model + optimizer + lr_scheduler).

    Non-``full`` offline-cache modules own their partial runtime DCP behavior:
    modules without online state can no-op, while modules with online state can
    save/load only the runtime subset they need.
    """

    def _load_checkpoint(self):
        model = self._offline_cache_model()
        if model is None:
            return super()._load_checkpoint()
        self._load_partial_dcp_checkpoint(model)

    def _save_checkpoint(self, state: "TrainerState"):
        model = self._offline_cache_model()
        if model is None:
            return super()._save_checkpoint(state)
        self._save_partial_dcp_checkpoint(model, state)


class OmniModuleHfCallback(_OmniModulePayloadMixin, HuggingfaceCkptCallback):
    """Per-module HuggingFace safetensors export.

    Non-``full`` offline-cache modules own how to materialize a complete HF
    artifact from source weights plus any partial runtime state.
    """

    def _save_checkpoint(self, state: "TrainerState", stage: str = "step_end"):
        model = self._offline_cache_model()
        if model is None:
            return super()._save_checkpoint(state, stage=stage)
        self._save_full_hf_checkpoint(model, state)


class OmniModuleLoraCallback(_OmniModulePayloadMixin, HFLoraCkptCallback):
    """Per-module LoRA-adapter export."""


# ── Per-module trainer ──────────────────────────────────────────────────────────


class OmniModuleTrainer:
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
      module's still-trainable params.  These wrap ``base._build_*`` so the
      *build* lives on the module-trainer; the orchestrator only *calls* them
      (after the shared dataset has fixed ``args.train_steps``).
    * :meth:`_init_callbacks` — builds this module's **own** checkpoint callback
      (:class:`OmniModuleHfCallback` / :class:`OmniModuleLoraCallback`, per-module
      DCP); trace / metering callbacks belong to the orchestrator, never here.

    The *global* concerns (distributed ``_setup``, data pipeline, trace
    metering, the train loop) are **never** run here — they are owned once by
    :class:`OmniTrainer`.  The orchestrator's ``on_{train,epoch,step}_*`` cascade
    into this trainer's matching :meth:`on_step_end` & co. so each module
    checkpoints itself.

    ``args`` is a per-module copy of the global arguments whose
    ``model.{config_path,model_path}`` point at this module's split-checkpoint
    subfolder, and whose ``model.model_config`` carries the module's YAML
    ``model_config:`` overrides — so the shared loader resolves the
    right OmniModule classes and the standard meta-init → FSDP2 → weight-load
    path is reused verbatim.
    """

    base: BaseTrainer
    _has_trainable_parameters: bool = None

    def __init__(
        self,
        args: "VeOmniArguments",
        module_name: str,
    ):
        # Composition (mirrors OmniTrainer): a bare BaseTrainer whose global
        # _setup() is deliberately skipped (owned by OmniTrainer); we call only
        # its per-model build helpers, in order.
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args
        # Single identity for this module: the registry key for its ParallelState
        # AND the ``<module>/`` checkpoint subdir (one name, no aliases).
        self.module_name = module_name

        # Build this module's own ParallelState (does not mutate the global
        # current state — that stays the orchestrator's) and register it in the
        # global parallel-state registry under this module's name.
        self._setup()

        # The meta-init + FSDP2/DDP wrap read the *current* global ParallelState
        # via ``get_parallel_state()`` (``build_parallelize_model`` /
        # ``parallelize_model_fsdp2`` / ``torch_parallelize``), so scope them to
        # this module's state (by registry name).
        with use_parallel_state(self.module_name):
            self.base._build_model()  # meta-init the sub-model from its config.json

            # Load this module's own processor / tokenizer and assemble
            # ``base.model_assets`` (mirrors ``BaseTrainer._build_model_assets``).
            self._build_model_assets()
            self._freeze_model_module()  # module self-freezes + lora + pretty-print + mem

            self._build_parallelized_model()  # FSDP2 wrap + per-module weight load (or module-owned)

            # Gradient-checkpoint recompute runs during backward — OUTSIDE the
            # per-module ``use_parallel_state`` scope that wraps the forward — so it
            # would recompute under the orchestrator's (restored) global state and
            # read the wrong groups (e.g. an EP MoE backbone falls back to the non-EP
            # kernel path on its EP-sharded experts → shape mismatch). Re-enter this
            # module's state during recompute.
            self._scope_recompute_to_parallel_state()

            # This module's own checkpoint callbacks (DCP resume + HF/LoRA export),
            # reusing the shared single-model callbacks.  ``module_name`` (the
            # module's YAML key) is the ``<module>/`` checkpoint subdir.  Optimizer /
            # lr-scheduler are built later via :meth:`_build_optimizer` /
            # :meth:`_build_lr_scheduler` (the orchestrator calls them once
            # ``args.train_steps`` is known).
            #
            # Built inside this module's ``use_parallel_state`` scope so each
            # ``Callback`` captures it (``Callback.__init__`` → ``get_parallel_state()``)
            # as its own ``self.parallel_state``; the runtime ``on_*`` dispatch then
            # needs no wrapper (explicit ownership, mirroring ``BaseTrainer``).
            self._init_callbacks()

        # Make ``base`` look enough like a single-model trainer for the reused
        # checkpoint callbacks: the dataloader is global (owned by the
        # orchestrator, never here).  ``base.model_assets`` was already
        # assembled in :meth:`_build_model_assets` above.
        self.base.train_dataloader = None

        # Last-step per-module metric contribution (theoretical_flops, seqlens),
        # computed at on_step_end from the optional MetricMeterMixin on
        # ``self.base.model``.  No per-module timing — the whole-graph delta is
        # owned by the orchestrator (a module's own wall-clock is meaningless).
        self._metric_meter_result: Optional[MetricMeterResult] = None

    def _build_parallelized_model(self):
        """FSDP2-wrap this module's model and load its weights (or defer to the module).

        FSDP2 (and the meta-init weight load) preserve ``requires_grad``: the
        shard carries it (torch ``_fsdp_param.py``: ``sharded_param.requires_grad_(
        param.requires_grad)``) and the loader writes weights in-place
        (``param.data.copy_``), so the freeze applied in ``_freeze_model_module``
        survives the wrap — no need to re-assert it here.

        A module may fully own its parallelize + weight-load (+ param offload) by
        implementing ``customized_build_parallelize_model`` — e.g. a huge MoE
        backbone that streams EP-sharded experts to CPU, which the generic
        GPU-materializing loader has no hook for. When it returns a model we use
        it verbatim (the override owns fsdp/load/offload/grad-ckpt); ``None`` (the
        default) keeps the generic :meth:`BaseTrainer._build_parallelized_model`
        path. Called within this module's ``use_parallel_state`` scope so the
        FSDP2/DDP wrap reads this module's mesh via ``get_parallel_state()``.
        """
        customized_builder = getattr(
            self.base.model,
            "customized_build_parallelize_model",
            lambda **kwargs: None,
        )
        customized_model = customized_builder(
            weights_path=self.base.args.model.model_path,
            args=self.base.args,
        )
        if customized_model is not None:
            self.base.model = customized_model
        else:
            self.base._build_parallelized_model()  # FSDP2 wrap + per-module weight load

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
        **this** module's own device mesh from its (merged) ``train.accelerator``
        and register it.  It is NOT made current here — the build sites scope to
        it explicitly via ``use_parallel_state(self.module_name)`` so the
        immediately-following meta-init + _build_parallelized_model (FSDP wrap)
        read this module's mesh rather than the orchestrator's.  The accelerator
        is already merged + validated by ``OmniConfig.module_config``.

        The state is registered in the global ``_PARALLEL_STATE_REGISTRY`` under
        this module's name (``module_name``, unique per OmniConfig and distinct
        from the orchestrator's ``"base"``), so every scope site re-enters it by
        name via ``use_parallel_state(self.module_name)`` (the registry is the
        single source of truth — the module-trainer keeps no local handle).
        ``init_parallel_state`` never overwrites the orchestrator's current global
        state — it only adds to the registry / topology cache.
        """
        acc = self.base.args.train.accelerator
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
        the forward is already wrapped in :meth:`OmniModel.module_context`, but the
        recompute (in backward) escapes it. Setting ``recompute_ctx`` to
        :func:`use_parallel_state` keeps reads of the free ``get_parallel_state()``
        (EP groups, vocab-parallel ``emb`` group, …) resolving to this module's mesh
        during recompute. ``use_reentrant=True`` does not honour ``context_fn`` — but
        the omni path runs non-reentrant (``train.gradient_checkpointing.enable_reentrant``
        defaults to ``False``).
        """
        name = self.module_name
        gc = self.base.args.train.gradient_checkpointing

        def _recompute_context_fn():
            return nullcontext(), use_parallel_state(name)

        # DDP wraps the model (``.module``) and does not expose
        # ``gradient_checkpointing_enable``; FSDP2 wraps in place. Unwrap so the
        # call reaches the raw HF model regardless of dp_mode.
        if gc.enable:
            _unwrap_module(self.base.model).gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": gc.enable_reentrant,
                    "context_fn": _recompute_context_fn,
                }
            )

    # ── Optimizer / lr-scheduler (built here; the orchestrator only calls) ─────

    @property
    def has_trainable_parameters(self) -> bool:
        if self._has_trainable_parameters is None:
            self._has_trainable_parameters = any(param.requires_grad for param in self.base.model.parameters())
        return self._has_trainable_parameters

    def _build_optimizer(self):
        """Build this module's optimizer over its still-trainable params.

        Scoped to this module's own ParallelState: a distributed optimizer (e.g.
        Muon) reads ``get_parallel_state()`` at build time, so it must resolve to
        this module's mesh, not the orchestrator's.
        """
        with self._scoped():
            self.base._build_optimizer()

    def _build_lr_scheduler(self):
        """Build this module's lr-scheduler (needs ``base.args.train_steps`` set)."""
        with self._scoped():
            self.base._build_lr_scheduler()

    def _clip_grad_norm(self, max_norm: float, norm_type: float = 2.0) -> float:
        """Clip this module's grads under its own parallelism; return the module norm.

        Owns entering its own state: ``veomni_omni_module_clip_grad_norm`` reads
        the current ParallelState via ``get_parallel_state()`` (like
        ``veomni_clip_grad_norm``), so run it inside ``self._scoped()``. The
        orchestrator only sums the returned per-module norms — it never handles
        the module's private mesh.
        """
        with self._scoped():
            return veomni_omni_module_clip_grad_norm(self.base.model, max_norm, norm_type)

    def _model_reshard(self, reshard: bool) -> None:
        """Set ``set_reshard_after_backward`` on this module's FSDP2 units.

        A gradient-accumulation optimization owned per-module: the orchestrator
        decides *when* (``reshard=False`` on the first micro-step to keep params
        gathered across the window and skip the reshard → re-all-gather churn;
        ``reshard=True`` on the last so the final backward frees the full
        params). This method only *applies* that intent to this module — trading
        param memory for communication.

        Read the **module's own** ``fsdp_config`` (not the orchestrator's): each
        OmniModule has its own merged ``train.accelerator``, so ``fsdp_mode`` /
        ``reshard_after_backward`` may differ per module — a DDP module has no
        FSDP2 units to toggle (skipped by the ``isinstance`` check), and a module
        that keeps ``reshard_after_backward=True`` opts out here. No
        ``ParallelState`` is read (unlike :meth:`_clip_grad_norm`), so this needs no
        scoping — ``set_reshard_after_backward`` just flips a flag on the unit.
        """
        fsdp_cfg = self.base.args.train.accelerator.fsdp_config
        if fsdp_cfg.fsdp_mode != "fsdp2" or fsdp_cfg.reshard_after_backward:
            return
        try:
            from torch.distributed.fsdp import FSDPModule
        except ImportError:
            return
        # ``set_reshard_after_backward`` recurses into every nested FSDP unit by
        # default, so one call on the root-sharded model covers them all (the
        # generic ``parallelize_model_fsdp2`` ``fully_shard``s the root). A module
        # that owns its parallelize via ``customized_build_parallelize_model``
        # (contract: "FSDP-or-not") may leave the root un-sharded — it then owns
        # its own reshard policy, so skip rather than assume a root FSDP unit.
        model = self.base.model
        if isinstance(model, FSDPModule):
            model.set_reshard_after_backward(reshard)

    # ── Metric metering ────────────────────────────────────────────────────────

    def _collect_metric_meter(self) -> Optional[MetricMeterResult]:
        """The per-module metrics computed at the last :meth:`on_step_end`.

        Returns ``(theoretical_flops, seqlens)`` for a metered module, else
        ``None``.  The orchestrator reads this once per step and rolls every
        module's contribution into the overall throughput / MFU.
        """
        return self._metric_meter_result

    # ── Callbacks (checkpoint + metric meter; both per-module) ─────────────────

    def _init_callbacks(self):
        """Build this module's DCP resume + HF/LoRA export callbacks.

        Mirrors :meth:`BaseTrainer._init_callbacks` (the DCP + HF/LoRA half),
        bound to ``self.base`` so the shared callbacks save / load **this**
        module's weights to its ``<module_name>/`` subdir.

        Fully-frozen modules (no ``requires_grad`` params) get no-op callbacks:
        there is nothing to train, no optimizer to snapshot, and weights stay
        at the released checkpoint (e.g. offline_cache OE/ViT/VAE).
        """
        base = self.base
        module_name = self.module_name
        if not any(p.requires_grad for p in base.model.parameters()):
            logger.info_rank0(
                f"OmniModuleTrainer[{module_name}]: fully frozen — skipping DCP/HF checkpoint callbacks."
            )
            self._has_trainable_parameters = False
            self.checkpointer_callback = _FrozenModuleNoOpCkptCallback(base)
            self.hf_ckpt_callback = _FrozenModuleNoOpCkptCallback(base)
            return
        self.checkpointer_callback = OmniModuleDcpCallback(base, module_name)
        if base.args.model.lora_config:
            self.hf_ckpt_callback = OmniModuleLoraCallback(base, module_name)
        else:
            self.hf_ckpt_callback = OmniModuleHfCallback(base, module_name)

    def on_train_begin(self, state):
        self.checkpointer_callback.on_train_begin(state)
        self.hf_ckpt_callback.on_train_begin(state)

    def on_train_end(self, state):
        self.checkpointer_callback.on_train_end(state)
        self.hf_ckpt_callback.on_train_end(state)

    def on_epoch_begin(self, state):
        self.checkpointer_callback.on_epoch_begin(state)
        self.hf_ckpt_callback.on_epoch_begin(state)

    def on_epoch_end(self, state):
        self.checkpointer_callback.on_epoch_end(state)
        self.hf_ckpt_callback.on_epoch_end(state)

    def on_step_begin(self, state, **kwargs):
        self.checkpointer_callback.on_step_begin(state, **kwargs)
        self.hf_ckpt_callback.on_step_begin(state, **kwargs)

    def on_step_end(self, state, **kwargs):
        # Stash this module's time-independent metric contribution
        # (theoretical_flops, seqlens). The orchestrator applies the whole-graph
        # delta to derive achieved FLOPs / MFU; there is no per-module timing
        # (this fires only after the whole graph's fwd+bwd, so a module-local
        # wall-clock would be the whole-step time, not its own).
        # Metering is opt-in: only modules that multi-inherit a MetricMeterMixin report.
        # Unwrap the DDP wrapper (FSDP2 is in-place) so a DDP-wrapped module's
        # MetricMeterMixin is still seen.
        model = _unwrap_module(self.base.model)
        self._metric_meter_result = model.metric_meter_collect() if isinstance(model, MetricMeterMixin) else None
        self.checkpointer_callback.on_step_end(state, **kwargs)
        self.hf_ckpt_callback.on_step_end(state, **kwargs)

    def _freeze_model_module(self):
        """Let the module freeze itself (its policy), then run the base report (+ lora)."""
        model = self.base.model
        if hasattr(model, "freeze_model"):
            model.freeze_model()
        self.base._freeze_model_module()

    def _build_model_assets(self):
        """Load this module's **own** processor / tokenizer and assemble ``base.model_assets``.

        Mirrors :meth:`BaseTrainer._build_model_assets` (which sets
        ``self.model_assets``), but for a sub-module — here it sets
        ``self.base.model_assets`` so the reused HF/asset-export callbacks ship the
        right files to the module's ``<module>/`` subdir.

        Meta-init skips ``from_pretrained``, so the module's own assets are loaded
        here: vision modules (SigLIP / VQVAE) need their processor at train time to
        normalise the raw uint8 images carried in ``conversation_list``; a module
        that owns its own tokenizer (e.g. a T5 text encoder for a DiT) needs that
        too.  Both are loaded from this module's weights path via the
        registry-aware :func:`build_processor` / :func:`build_tokenizer` — the same
        loaders used everywhere else.

        A missing / unreadable asset folder is a best-effort no-op; the module's
        ``generate`` / ``forward`` raises a clear error later if it truly needs it.
        """
        model = self.base.model
        cfg = getattr(model, "config", None)
        label = type(model).__name__
        weights_path = self.base.args.model.model_path

        # Per-module assets, tried in order. Assets that can be derived from the
        # runtime module config use ``from_config`` so CLI overrides are honored.
        # Others are loaded from the module's own checkpoint dir via the class
        # declared on the model. The tokenizer is the exception — it has no class
        # slot (``class_attr is None``) and is built by ``build_tokenizer``.
        # A module that doesn't declare a kind is skipped; a load failure is only
        # a warning (the module raises lazily if that modality is actually used).
        model_type = type(model)
        asset_specs = [
            # (human label, set attr, check attr, class attr | None)
            # ``set attr`` is the public name so the tokenizer goes through its
            # property setter (which may build chat markers / token ids); ``check
            # attr`` is the private storage used for the already-loaded / asset
            # collection. ``class attr`` None => load via ``build_tokenizer``.
            ("processor", "_processor", "_processor", "processor_class"),
            ("image processor", "_image_processor", "_image_processor", "image_processor_class"),
            ("video processor", "_video_processor", "_video_processor", "video_processor_class"),
            ("tokenizer", "tokenizer", "_tokenizer", None),
        ]
        for kind, set_attr, check_attr, class_attr in asset_specs:
            if getattr(model, check_attr, None) is not None:
                continue
            try:
                if class_attr is None:
                    asset = build_tokenizer(weights_path)
                else:
                    asset_class = getattr(model_type, class_attr, None)
                    if asset_class is None:
                        continue
                    if cfg is not None and callable(getattr(asset_class, "from_config", None)):
                        asset = asset_class.from_config(cfg)
                    else:
                        asset = asset_class.from_pretrained(weights_path)
                setattr(model, set_attr, asset)
            except Exception as e:  # noqa: BLE001 — surfaced lazily by the module if the modality is used
                logger.warning_once(f"OmniModuleTrainer '{label}': could not load {kind} from {weights_path}: {e}.")
                continue
            logger.info_rank0(f"OmniModuleTrainer '{label}': loaded {kind}.")

        # Assemble the savable assets (config + own processors + tokenizer).
        assets: List[Any] = []
        if cfg is not None:
            assets.append(cfg)
        for attr in ("_processor", "_image_processor", "_video_processor", "_tokenizer"):
            asset = getattr(model, attr, None)
            if asset is not None:
                assets.append(asset)
        self.base.model_assets = assets


__all__ = [
    "OmniModuleTrainer",
    "OmniModuleDcpCallback",
    "OmniModuleHfCallback",
    "OmniModuleLoraCallback",
]
