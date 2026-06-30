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

from ...distributed.parallel_state import (
    init_parallel_state,
    use_parallel_state,
)
from ...models import build_tokenizer
from ...models.seed_omni.mixins.metric_meter_mixin import MetricMeterMixin, MetricMeterResult
from ...models.seed_omni.modeling_omni import _unwrap_module
from ...utils import logging
from ..base import BaseTrainer
from ..callbacks import (
    CheckpointerCallback,
    HFLoraCkptCallback,
    HuggingfaceCkptCallback,
)


if TYPE_CHECKING:
    from ...arguments import VeOmniArguments
    from ...distributed.parallel_state import ParallelState
    from ..callbacks import TrainerState


logger = logging.get_logger(__name__)


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
    ``self.subfolder_name`` (the module's YAML key, passed in at construction)
    is the ``<module>/`` subdir every save / load path is nested under.
    """

    def __init__(self, trainer: "BaseTrainer", subfolder_name: str) -> None:
        self.subfolder_name = subfolder_name
        super().__init__(trainer)

    def _module_subdir(self, root: str, state: "TrainerState") -> str:
        return os.path.join(root, f"global_step_{state.global_step}", self.subfolder_name)

    def _save_dir(self, state: "TrainerState") -> str:
        return self._module_subdir(self.trainer.args.train.checkpoint.save_path, state)

    def _output_dir(self, state: "TrainerState") -> str:
        return self._module_subdir(self.trainer.args.train.checkpoint.output_dir, state)

    def _load_dir(self) -> Optional[str]:
        load_path = self.trainer.args.train.checkpoint.load_path
        return None if load_path is None else os.path.join(load_path, self.subfolder_name)

    def _model_assets_dir(self) -> str:
        return os.path.join(self.trainer.args.train.checkpoint.model_assets_dir, self.subfolder_name)

    def _extra_state(self, state: "TrainerState") -> Dict[str, Any]:
        # Per-model only — the global step / dataloader / environ-meter / rng are
        # saved once by OmniGlobalStateCallback on the orchestrator.
        return {"lr_scheduler": self.trainer.lr_scheduler.state_dict()}

    def _load_extra_state(self, extra_state: Dict[str, Any]) -> None:
        lr_sd = extra_state.get("lr_scheduler")
        if lr_sd is not None:
            self.trainer.lr_scheduler.load_state_dict(lr_sd)


class OmniModuleDcpCallback(_OmniModulePayloadMixin, CheckpointerCallback):
    """Per-module DCP resume checkpoint (model + optimizer + lr_scheduler)."""


class OmniModuleHfCallback(_OmniModulePayloadMixin, HuggingfaceCkptCallback):
    """Per-module HuggingFace safetensors export."""


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
    parallel_state: "ParallelState"

    def __init__(
        self,
        args: "VeOmniArguments",
        subfolder_name: str = "",
    ):
        # Composition (mirrors OmniTrainer): a bare BaseTrainer whose global
        # _setup() is deliberately skipped (owned by OmniTrainer); we call only
        # its per-model build helpers, in order.
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args

        # Build this module's own ParallelState (does not mutate the global
        # current state — that stays the orchestrator's).
        self._setup()

        # The meta-init + FSDP2/DDP wrap read the *current* global ParallelState
        # via ``get_parallel_state()`` (``build_parallelize_model`` /
        # ``parallelize_model_fsdp2`` / ``torch_parallelize``), so scope them to
        # this module's state.
        with use_parallel_state(self.parallel_state):
            self.base._build_model()  # meta-init the sub-model from its config.json

            # Load this module's own processor / tokenizer and assemble
            # ``base.model_assets`` (mirrors ``BaseTrainer._build_model_assets``).
            self._build_model_assets()
            self._freeze_model_module()  # module self-freezes + lora + pretty-print + mem

            # FSDP2 (and the meta-init weight load) preserve ``requires_grad``: the
            # shard carries it (torch ``_fsdp_param.py``: ``sharded_param.requires_grad_(
            # param.requires_grad)``) and the loader writes weights in-place
            # (``param.data.copy_``), so the freeze applied in ``_freeze_model_module``
            # above survives the wrap — no need to re-assert it here.
            self.base._build_parallelized_model()  # FSDP2 wrap + per-module weight load

            # Gradient-checkpoint recompute runs during backward — OUTSIDE the
            # per-module ``use_parallel_state`` scope that wraps the forward — so it
            # would recompute under the orchestrator's (restored) global state and
            # read the wrong groups (e.g. an EP MoE backbone falls back to the non-EP
            # kernel path on its EP-sharded experts → shape mismatch). Re-enter this
            # module's state during recompute.
            if self.base.args.train.gradient_checkpointing.enable:
                self._scope_recompute_to_parallel_state()

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

        # This module's own checkpoint callbacks (DCP resume + HF/LoRA export),
        # reusing the shared single-model callbacks.  ``subfolder_name`` (the
        # module's YAML key) is the ``<module>/`` checkpoint subdir.  Optimizer /
        # lr-scheduler are built later via :meth:`_build_optimizer` /
        # :meth:`_build_lr_scheduler` (the orchestrator calls them once
        # ``args.train_steps`` is known).
        self._init_callbacks(subfolder_name)

    # ── Parallel state (per-module device mesh) ────────────────────────────────

    def _setup(self):
        """Build this module's own :class:`ParallelState` and set it current.

        Mirrors the parallel-state half of :meth:`BaseTrainer._setup`.  The
        distributed process group / device / seed are already initialised once
        by the orchestrator (``OmniTrainer.base._setup``), so here we only build
        **this** module's own device mesh from its (merged) ``train.accelerator``
        and make it the current global state — so the immediately-following
        meta-init + _build_parallelized_model (FSDP wrap) read this module's mesh
        rather than the orchestrator's.  The accelerator is already merged +
        validated by ``OmniConfig.module_config``; the orchestrator restores its
        default state after the build loop.
        """
        acc = self.base.args.train.accelerator
        self.parallel_state = init_parallel_state(
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
        )

    def _scope_recompute_to_parallel_state(self) -> None:
        """Make gradient-checkpoint recompute re-enter this module's ParallelState.

        torch ``checkpoint``'s ``context_fn`` returns ``(forward_ctx, recompute_ctx)``;
        the forward is already wrapped in :meth:`OmniModel._module_scope`, but the
        recompute (in backward) escapes it. Setting ``recompute_ctx`` to
        :func:`use_parallel_state` keeps reads of the free ``get_parallel_state()``
        (EP groups, vocab-parallel ``emb`` group, …) resolving to this module's mesh
        during recompute. ``use_reentrant=True`` does not honour ``context_fn`` — but
        the omni path runs non-reentrant (``train.gradient_checkpointing.enable_reentrant``
        defaults to ``False``).
        """
        ps = self.parallel_state
        gc = self.base.args.train.gradient_checkpointing

        def _recompute_context_fn():
            return nullcontext(), use_parallel_state(ps)

        # DDP wraps the model (``.module``) and does not expose
        # ``gradient_checkpointing_enable``; FSDP2 wraps in place. Unwrap so the
        # call reaches the raw HF model regardless of dp_mode.
        _unwrap_module(self.base.model).gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": gc.enable_reentrant,
                "context_fn": _recompute_context_fn,
            }
        )

    # ── Optimizer / lr-scheduler (built here; the orchestrator only calls) ─────

    def _build_optimizer(self):
        """Build this module's optimizer over its still-trainable params."""
        self.base._build_optimizer()

    def _build_lr_scheduler(self):
        """Build this module's lr-scheduler (needs ``base.args.train_steps`` set)."""
        self.base._build_lr_scheduler()

    # ── Metric metering ────────────────────────────────────────────────────────

    def collect_metric_meter(self) -> Optional[MetricMeterResult]:
        """The per-module metrics computed at the last :meth:`on_step_end`.

        Returns ``(theoretical_flops, seqlens)`` for a metered module, else
        ``None``.  The orchestrator reads this once per step and rolls every
        module's contribution into the overall throughput / MFU.
        """
        return self._metric_meter_result

    # ── Callbacks (checkpoint + metric meter; both per-module) ─────────────────

    def _init_callbacks(self, subfolder_name: str):
        """Build this module's DCP resume + HF/LoRA export callbacks.

        Mirrors :meth:`BaseTrainer._init_callbacks` (the DCP + HF/LoRA half),
        bound to ``self.base`` so the shared callbacks save / load **this**
        module's weights to its ``<subfolder_name>/`` subdir.
        """
        base = self.base
        self.checkpointer_callback = OmniModuleDcpCallback(base, subfolder_name)
        if base.args.model.lora_config:
            self.hf_ckpt_callback = OmniModuleLoraCallback(base, subfolder_name)
        else:
            self.hf_ckpt_callback = OmniModuleHfCallback(base, subfolder_name)

    # Each checkpoint callback (DCP save/load + HF/LoRA export) runs under this
    # module's ParallelState so the DCP extra-parallel dim
    # preprocessing reads the right meshes via get_parallel_state().

    def on_train_begin(self, state):
        with use_parallel_state(self.parallel_state):
            self.checkpointer_callback.on_train_begin(state)
            self.hf_ckpt_callback.on_train_begin(state)

    def on_train_end(self, state):
        with use_parallel_state(self.parallel_state):
            self.checkpointer_callback.on_train_end(state)
            self.hf_ckpt_callback.on_train_end(state)

    def on_epoch_begin(self, state):
        with use_parallel_state(self.parallel_state):
            self.checkpointer_callback.on_epoch_begin(state)
            self.hf_ckpt_callback.on_epoch_begin(state)

    def on_epoch_end(self, state):
        with use_parallel_state(self.parallel_state):
            self.checkpointer_callback.on_epoch_end(state)
            self.hf_ckpt_callback.on_epoch_end(state)

    def on_step_begin(self, state, **kwargs):
        with use_parallel_state(self.parallel_state):
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
        with use_parallel_state(self.parallel_state):
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
        label = type(model).__name__
        weights_path = self.base.args.model.model_path

        # Per-module assets, tried in order. Each is loaded from the module's own
        # checkpoint dir via the class declared on the model (``<kind>_class``,
        # e.g. reads ``preprocessor_config.json`` so a sibling config can't shadow
        # it) into ``_<kind>``. The tokenizer is the exception — it has no class
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
                    asset = asset_class.from_pretrained(weights_path)
                setattr(model, set_attr, asset)
            except Exception as e:  # noqa: BLE001 — surfaced lazily by the module if the modality is used
                logger.warning_once(f"OmniModuleTrainer '{label}': could not load {kind} from {weights_path}: {e}.")
                continue
            logger.info_rank0(f"OmniModuleTrainer '{label}': loaded {kind}.")

        # Assemble the savable assets (config + own processors + tokenizer).
        assets: List[Any] = []
        cfg = getattr(model, "config", None)
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
