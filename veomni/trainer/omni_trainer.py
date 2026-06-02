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

"""OmniTrainer — orchestrator for OmniModel V2 (one trainer per sub-module).

Unlike single-model trainers (BaseTrainer / VLMTrainer), OmniModel is a
*composition* of several independent OmniModule sub-models (Janus: siglip /
vqvae / text_encoder / llama).  Each sub-model is backed by its **own**
:class:`OmniModuleTrainer` — which (by composition over a bare ``BaseTrainer``)
reuses the base per-model build helpers (``_build_model`` / ``_setup_lora`` /
``_build_parallelized_model`` / ``_build_optimizer`` / ``_build_lr_scheduler``)
to give that module its own FSDP2 unit, optimizer, lr-scheduler, **checkpoint
callback** and on-disk snapshot.

:class:`OmniTrainer` then **strings the module-trainers together**: it owns the
*global* concerns once (distributed ``_setup``, the shared data pipeline, trace
metering, the train loop) and drives the graph — composing the sub-models into
one ``OmniModel``, running the DAG forward, a single ``loss.backward()`` (the
autograd graph connects every FSDP2 unit), and the per-module optimizer step.
Its ``on_{train,epoch,step}_*`` cascade into every module-trainer so each runs
its own checkpoint save/resume.

Why one trainer per module (vs. one wrapper over ``OmniModel``)?

* Each module is a self-contained HF model with its own ``_no_split_modules``
  and on-disk snapshot — a per-module ``BaseTrainer`` loads its weights from
  the module ``weights_path`` and reuses base build/optimizer/lora logic.
* The training DAG chains hidden states across modules; a **single**
  ``loss.backward()`` still propagates across every FSDP2 unit.  Each module's
  FSDP2 reduce-scatter fires from its own backward hooks.
* Gradient clipping is **global** (over ``OmniModel``'s full DTensor param
  set); the optimizer step iterates every per-module optimizer.

Division of labour
------------------
* :class:`OmniModuleTrainer` (per module): ``_build_model`` →
  ``_freeze_model_module`` → ``_build_parallelized_model`` (FSDP2 wrap + weight
  load) → ``_init_callbacks`` (its own per-module DCP callback), then
  ``_build_optimizer`` / ``_build_lr_scheduler`` (called by the orchestrator
  once ``args.train_steps`` is known).  Saves / resumes itself when the
  orchestrator cascades ``on_*`` into it.
* :class:`OmniTrainer` (orchestrator): global ``_setup`` + data pipeline + trace
  callbacks + train loop; builds the module-trainers, composes ``OmniModel``,
  aggregates their optimizers / schedulers behind ``MultiOptimizer`` /
  ``MultiLRScheduler``, owns the graph forward/backward + the global optimizer
  step, and cascades the callback lifecycle into each module-trainer.
"""

import copy
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import torch.distributed as dist

from ..arguments import DataArguments, ModelArguments, TrainingArguments, VeOmniArguments
from ..data import SeedOmniCollator
from ..distributed.clip_grad_norm import veomni_clip_grad_norm
from ..models import build_processor, build_tokenizer
from ..models.seed_omni.modeling_omni import OmniModel
from ..ops.batch_invariant_ops import set_batch_invariant_mode
from ..utils import helper, logging
from ..utils.device import synchronize
from ..utils.model_utils import pretty_print_trainable_parameters
from .base import BaseTrainer
from .callbacks import Callback, CheckpointerCallback, HFLoraCkptCallback, HuggingfaceCkptCallback, TrainerState


if TYPE_CHECKING:
    from ..models.seed_omni.configuration_seed_omni import OmniConfig


logger = logging.get_logger(__name__)


# ── Argument dataclasses ────────────────────────────────────────────────────────


@dataclass
class OmniModelArguments(ModelArguments):
    """Model arguments for OmniModel V2 training / inference."""

    omni_train_yaml_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to the OmniModel master training YAML (e.g. "
                "configs/seed_omni/janus_1.3b/train.yaml).  Declares "
                "modules / nodes / edges / training_graph."
            )
        },
    )
    omni_infer_yaml_path: Optional[Dict[str, str]] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Mapping of inference scenario name → inference YAML path.  "
                "The selected scenario's YAML overlays ``omni_train_yaml_path`` "
                "at runtime (flat dict.update; only top-level keys, in practice "
                "generation_graph).  Example keys: infer_gen / infer_und / "
                "infer_interleave."
            )
        },
    )
    omni_infer_type: Optional[str] = field(
        default=None,
        metadata={"help": "Active inference scenario key into omni_infer_yaml_path (inference only)."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.model_path is not None and self.tokenizer_path is None:
            # Global tokenizer lives at the split-checkpoint root.
            self.tokenizer_path = self.model_path

    def load_omni_config(self, *, infer_type: Optional[str] = None) -> "OmniConfig":
        """Build :class:`OmniConfig` with resolved module paths."""
        from ..models.seed_omni.configuration_seed_omni import OmniConfig

        if not self.omni_train_yaml_path:
            raise ValueError("`model.omni_train_yaml_path` is required for OmniModel V2.")
        if not self.model_path:
            raise ValueError("`model.model_path` is required for OmniModel V2.")

        infer_yaml_path = None
        selected = infer_type or self.omni_infer_type
        if selected is not None:
            infer_map = self.omni_infer_yaml_path or {}
            if selected not in infer_map:
                known = ", ".join(sorted(infer_map)) or "(none)"
                raise KeyError(f"Unknown omni_infer_type {selected!r}; expected one of: {known}.")
            infer_yaml_path = infer_map[selected]

        return OmniConfig.from_paths(
            model_path=self.model_path,
            tokenizer_path=self.tokenizer_path,
            train_yaml_path=self.omni_train_yaml_path,
            infer_yaml_path=infer_yaml_path,
        )


@dataclass
class VeOmniOmniArguments(VeOmniArguments):
    model: "OmniModelArguments" = field(default_factory=OmniModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


# ── Multi-optimizer / multi-scheduler proxies ──────────────────────────────────


class MultiOptimizer:
    """Thin proxy over ``{module_name: torch.optim.Optimizer}``.

    Exposes the minimal :class:`torch.optim.Optimizer` surface the metering /
    logging callbacks read (``param_groups``) and the train loop drives
    (``step`` / ``zero_grad``).  Checkpointing is per-module (handled by each
    module-trainer's :class:`OmniModuleCheckpointCallback` against the real
    per-module optimizer), so no ``state_dict`` is needed here.
    """

    def __init__(self, optimizers: Dict[str, torch.optim.Optimizer]):
        self.optimizers = optimizers

    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = []
        for opt in self.optimizers.values():
            groups.extend(opt.param_groups)
        return groups

    def step(self) -> None:
        for opt in self.optimizers.values():
            opt.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=set_to_none)


class MultiLRScheduler:
    """Thin proxy over ``{module_name: LRScheduler}`` (step-all / lr-read)."""

    def __init__(self, schedulers: Dict[str, Any]):
        self.schedulers = schedulers

    def step(self) -> None:
        for sched in self.schedulers.values():
            sched.step()

    def get_last_lr(self) -> List[float]:
        lrs: List[float] = []
        for sched in self.schedulers.values():
            lrs.extend(sched.get_last_lr())
        return lrs or [0.0]

    def state_dict(self) -> Dict[str, Any]:
        return {name: sched.state_dict() for name, sched in self.schedulers.items()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for name, sched in self.schedulers.items():
            if name in state:
                sched.load_state_dict(state[name])


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


# ── Orchestrator's global resume state ──────────────────────────────────────────


class OmniGlobalStateCallback(Callback):
    """Save / resume the **module-agnostic** global state, once, on the orchestrator.

    Holds exactly what is *not* per-model: ``global_step`` + dataloader position
    + environ-meter + the single torch RNG.  Written to
    ``<save_path>/global_step_{N}/trainer_state.pt`` on rank-0 and read back at
    ``on_train_begin`` (the per-module callbacks restore their own weights /
    optimizer / lr_scheduler from their ``<module>/`` DCPs).
    """

    def __init__(self, trainer: "OmniTrainer") -> None:
        super().__init__(trainer)
        args = trainer.base.args
        self.every_n_steps = args.train.checkpoint.save_steps
        self.every_n_epochs = args.train.checkpoint.save_epochs
        self._last_saved_step: int = -1

    def on_step_end(self, state, **kwargs) -> None:
        if self.every_n_steps and state.global_step % self.every_n_steps == 0:
            self._save(state.global_step)

    def on_epoch_end(self, state, **kwargs) -> None:
        if self.every_n_epochs and (state.epoch + 1) % self.every_n_epochs == 0:
            if state.global_step != self._last_saved_step:
                self._save(state.global_step)

    def on_train_begin(self, state, **kwargs) -> None:
        self._load()

    def _state_path(self, root: str, global_step: int) -> str:
        return os.path.join(root, f"global_step_{global_step}", "trainer_state.pt")

    def _save(self, global_step: int) -> None:
        base = self.trainer.base
        args = base.args
        if args.train.global_rank == 0:
            torch.save(
                {
                    "global_step": global_step,
                    "train_dataloader": base.train_dataloader.state_dict()
                    if base.train_dataloader is not None
                    else {},
                    "environ_meter": base.environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
                self._state_path(args.train.checkpoint.save_path, global_step),
            )
        if dist.is_initialized():
            dist.barrier()
        self._last_saved_step = global_step

    def _load(self) -> None:
        base = self.trainer.base
        args = base.args
        load_path = args.train.checkpoint.load_path
        if load_path is None:
            return
        trainer_state_path = os.path.join(load_path, "trainer_state.pt")
        if not os.path.exists(trainer_state_path):
            return
        ts = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
        base.state.global_step = ts["global_step"]
        base.start_epoch = base.state.global_step // args.train_steps
        base.start_step = base.state.global_step % args.train_steps
        if base.train_dataloader is not None and ts.get("train_dataloader"):
            base.train_dataloader.load_state_dict(ts["train_dataloader"])
        if ts.get("environ_meter") is not None:
            base.environ_meter.load_state_dict(ts["environ_meter"])
        torch.set_rng_state(ts["torch_rng_state"])
        if base.start_step == 0 and base.train_dataloader is not None:
            iter(base.train_dataloader)


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
      (:class:`OmniModuleCheckpointCallback`, per-module DCP); trace / metering
      callbacks belong to the orchestrator, never here.

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

    def __init__(
        self,
        args: "VeOmniArguments",
        conversation_tokenizer: Any = None,
        subfolder_name: str = "",
    ):
        # Composition (mirrors OmniTrainer): a bare BaseTrainer whose global
        # _setup() is deliberately skipped (owned by OmniTrainer); we call only
        # its per-model build helpers, in order.
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args

        self.base._build_model()  # meta-init the sub-model from its config.json
        # Wire the global *conversation* tokenizer (resolves special-token ids /
        # tokenises the conversation).  This is **not** a module asset — the
        # orchestrator owns it; the module's *own* assets are loaded below.
        if conversation_tokenizer is not None and hasattr(self.base.model, "set_conversation_tokenizer"):
            self.base.model.set_conversation_tokenizer(conversation_tokenizer)
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

        # Make ``base`` look enough like a single-model trainer for the reused
        # checkpoint callbacks: the dataloader is global (owned by the
        # orchestrator, never here).  ``base.model_assets`` was already
        # assembled in :meth:`_build_model_assets` above.
        self.base.train_dataloader = None

        # This module's own checkpoint callbacks (DCP resume + HF/LoRA export),
        # reusing the shared single-model callbacks.  ``subfolder_name`` (the
        # module's YAML key) is the ``<module>/`` checkpoint subdir.  Optimizer /
        # lr-scheduler are built later via :meth:`_build_optimizer` /
        # :meth:`_build_lr_scheduler` (the orchestrator calls them once
        # ``args.train_steps`` is known).
        self._init_callbacks(subfolder_name)

    # ── Optimizer / lr-scheduler (built here; the orchestrator only calls) ─────

    def _build_optimizer(self):
        """Build this module's optimizer over its still-trainable params."""
        self.base._build_optimizer()

    def _build_lr_scheduler(self):
        """Build this module's lr-scheduler (needs ``base.args.train_steps`` set)."""
        self.base._build_lr_scheduler()

    # ── Callbacks (checkpoint only; trace lives on the orchestrator) ───────────

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
        loaders used everywhere else — and are this module's *own* assets, distinct
        from the global conversation tokenizer (the orchestrator owns / saves that).

        A missing / unreadable asset folder is a best-effort no-op; the module's
        ``generate`` / ``forward`` raises a clear error later if it truly needs it.
        """
        model = self.base.model
        label = type(model).__name__
        weights_path = self.base.args.model.model_path

        if getattr(type(model), "processor_class", None) is not None and getattr(model, "_processor", None) is None:
            try:
                model._processor = build_processor(weights_path)
                logger.info_rank0(f"OmniModuleTrainer '{label}': loaded processor.")
            except Exception as e:  # noqa: BLE001 — surfaced lazily by the module if truly needed
                logger.warning_once(
                    f"OmniModuleTrainer '{label}': could not load processor from {weights_path}: {e}. "
                    "Training will fail if this modality's images are actually present."
                )

        if getattr(type(model), "tokenizer_class", None) is not None and getattr(model, "_tokenizer", None) is None:
            try:
                model._tokenizer = build_tokenizer(weights_path)
                logger.info_rank0(f"OmniModuleTrainer '{label}': loaded own tokenizer.")
            except Exception as e:  # noqa: BLE001 — surfaced lazily by the module if truly needed
                logger.warning_once(
                    f"OmniModuleTrainer '{label}': could not load own tokenizer from {weights_path}: {e}."
                )

        # Assemble the savable assets (config + own processor + own tokenizer).  The
        # global conversation tokenizer is deliberately excluded — it is the
        # orchestrator's asset, saved once, not this module's.
        assets: List[Any] = []
        cfg = getattr(model, "config", None)
        if cfg is not None:
            assets.append(cfg)
        if getattr(model, "_processor", None) is not None:
            assets.append(model._processor)
        if getattr(model, "_tokenizer", None) is not None:
            assets.append(model._tokenizer)
        self.base.model_assets = assets


# ── OmniTrainer ────────────────────────────────────────────────────────────────


class OmniTrainer:
    """Orchestrator for OmniModel V2 — one :class:`OmniModuleTrainer` per module.

    Composition over inheritance (mirrors :class:`VLMTrainer`): instead of
    subclassing :class:`BaseTrainer`, we hold a bare ``BaseTrainer`` instance in
    ``self.base`` for the *global* concerns (distributed ``_setup``, the shared
    data pipeline, callbacks/metering, the train loop) and drive its private
    ``_build_*`` / ``on_*`` helpers one-by-one.

    Each OmniModule sub-model is built + owned by its own
    :class:`OmniModuleTrainer` (``self.module_trainers``); this orchestrator
    composes their models into one ``OmniModel`` (``self.base.model``) and
    aggregates their per-module optimizers / schedulers behind
    :class:`MultiOptimizer` / :class:`MultiLRScheduler` (``self.base.optimizer``
    / ``self.base.lr_scheduler``).

    Canonical ``BaseTrainer`` state (``model`` / ``optimizer`` / ``lr_scheduler``
    / ``state`` / dataloaders / **trace** callbacks) lives on ``self.base``.
    Omni-specific bookkeeping that is **not** part of ``BaseTrainer``'s interface
    (``omni_config`` / ``module_names`` / ``module_trainers`` / per-module
    ``optimizers`` / ``lr_schedulers``) lives directly on ``self``.

    Checkpointing is **not** owned here: each :class:`OmniModuleTrainer` builds
    its own :class:`OmniModuleCheckpointCallback` and the orchestrator's ``on_*``
    cascade drives them; the orchestrator keeps only trace / metering callbacks.
    """

    # Composition: the underlying single-model trainer. Holds all
    # BaseTrainer-declared state (model / optimizer / lr_scheduler / state /
    # checkpointer / dataloaders / callbacks); accessed as ``self.base.*``.
    base: BaseTrainer

    # OmniModel V2 bookkeeping — not part of BaseTrainer's interface, so it
    # lives directly on the wrapper rather than on ``self.base``.
    omni_config: "OmniConfig"  # parsed modules / nodes / edges / training graph
    module_names: List[str]  # ordered names of every declared module
    module_trainers: Dict[str, OmniModuleTrainer]  # one composition wrapper per module
    optimizers: Dict[str, torch.optim.Optimizer]  # one per trainable module (aggregated into base.optimizer)
    lr_schedulers: Dict[str, Any]  # one per trainable module (aggregated into base.lr_scheduler)

    def __init__(self, args: VeOmniOmniArguments):
        # BaseTrainer.__init__ is NOT called here; we call its private
        # helpers one-by-one so the (overridden) build sequence is explicit.
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args

        self.base._setup()
        # Each module-trainer builds + FSDP2-wraps its own sub-model (weights
        # loaded), then we compose them into one OmniModel.
        self._build_model()
        self._freeze_model_module()  # aggregate trainable-param report
        self._build_model_assets()  # expose [omni_config, tokenizer]
        self.base._build_data_transform()
        self.base._build_dataset()  # fixes args.train_steps (needed by schedulers)
        self._build_collate_fn()  # seedomni → SeedOmniCollator
        self.base._build_dataloader()
        self._build_optimizer()  # drive each module-trainer's optimizer
        self._build_lr_scheduler()  # drive each module-trainer's lr-scheduler
        self.base._build_training_context()
        self._init_callbacks()  # swap single-model ckpt → per-module DCP

    # ── Build: per-module trainers + compose ───────────────────────────────────

    def _module_args(self, weights_path: str, mod_cfg: Dict[str, Any]) -> "VeOmniOmniArguments":
        """Per-module copy of the global args, retargeted at this module.

        The module-trainer reuses ``BaseTrainer._build_model`` /
        ``_build_parallelized_model``, which read ``args.model.{config_path,
        model_path}`` and ``args.model.model_config``.  Point those at the
        module's split-checkpoint subfolder; the module's YAML ``model_config:``
        sub-block becomes the config overrides (forwarded to the OmniModule's
        config — e.g. ``freeze``).  A deep copy keeps per-module mutations
        (e.g. the GC flag) from leaking across modules and avoids re-running
        ``ModelArguments.__post_init__`` (safetensors-index I/O) on every module.
        """
        a = copy.deepcopy(self.base.args)
        a.model.config_path = weights_path
        a.model.model_path = weights_path
        a.model.model_config = dict(mod_cfg.get("model_config") or {})
        return a

    def _build_model(self):
        """Build one :class:`OmniModuleTrainer` per module and compose ``OmniModel``.

        Each module-trainer meta-inits its sub-model (via the shared,
        OMNI-registry-aware loader — ``model_type`` auto-detected from
        ``config.json``), wires the global tokenizer + image processor, and
        FSDP2-wraps + loads its weights.  We then compose the (wrapped)
        sub-models into one ``OmniModel`` on ``base.model``.
        """
        base = self.base
        args: VeOmniOmniArguments = base.args
        self.omni_config: "OmniConfig" = args.model.load_omni_config()
        self.module_names: List[str] = list(self.omni_config.module_names)
        self.module_trainers: Dict[str, OmniModuleTrainer] = {}

        # Global tokenizer (wired into every module-trainer that wants it).
        tokenizer_path = self.omni_config.tokenizer_path or args.model.tokenizer_path
        base.tokenizer = build_tokenizer(tokenizer_path)

        modules: Dict[str, torch.nn.Module] = {}
        for name in self.module_names:
            mod_cfg = self.omni_config.module_config(name)  # deep-copied
            weights_path = mod_cfg.get("weights_path")
            if weights_path is None:
                raise ValueError(f"OmniTrainer: module '{name}' has no `weights_path` in the training YAML.")

            module_trainer = OmniModuleTrainer(
                self._module_args(weights_path, mod_cfg),
                conversation_tokenizer=base.tokenizer,
                subfolder_name=name,  # YAML key → ``<module>/`` checkpoint subdir
            )
            self.module_trainers[name] = module_trainer
            modules[name] = module_trainer.base.model
            logger.info_rank0(f"OmniTrainer: built module-trainer '{name}' from {weights_path}")

        base.model = OmniModel(self.omni_config, modules)
        base.model_config = self.omni_config
        logger.info_rank0(
            f"OmniTrainer: composed OmniModel with {len(self.module_names)} modules ({self.module_names})."
        )

    # ── Freeze (aggregate report) ───────────────────────────────────────────────

    def _freeze_model_module(self):
        """Report aggregate trainable params over the composed model.

        Per-module freeze (``requires_grad=False``) and LoRA wrapping already
        happened inside each :class:`OmniModuleTrainer`; here we only print the
        composed-model view + memory, mirroring the position of
        ``_freeze_model_module`` in the base build order.
        """
        pretty_print_trainable_parameters(self.base.model)
        helper.print_device_mem_info("VRAM usage after building model")

    # ── Build: assets ───────────────────────────────────────────────────────────

    def _build_model_assets(self):
        # Nothing to build here — the global conversation tokenizer wiring
        # (set_conversation_tokenizer) and per-module own-asset loading already
        # happened in :meth:`_build_model`.  Just expose the global assets that
        # BaseTrainer callbacks read.
        self.base.model_assets = [self.omni_config, self.base.tokenizer]

    # ── Build: collator ─────────────────────────────────────────────────────────

    def _build_collate_fn(self):
        """``seedomni`` → list-only ``SeedOmniCollator``; else BaseTrainer default."""
        if self.base.args.data.data_type == "seedomni":
            self.base.collate_fn = SeedOmniCollator()
            logger.info_rank0("OmniTrainer: using SeedOmniCollator (list-only) for data_type='seedomni'")
        else:
            self.base._build_collate_fn()

    # ── Build: per-module optimizers + schedulers (drive the module-trainers) ──

    def _build_optimizer(self):
        """Call each module-trainer's :meth:`OmniModuleTrainer._build_optimizer` + aggregate.

        The build lives on the module-trainer; here we only invoke it and
        collect the result.  ``build_optimizer`` only ever puts ``requires_grad``
        params into the optimizer, so a partially-frozen module just trains its
        remaining params and a fully-frozen one yields a harmless empty optimizer
        (``step()`` is a no-op) — no per-module freeze bookkeeping needed.
        Aggregated behind :class:`MultiOptimizer` so the metering callbacks and
        train loop see the canonical ``base.optimizer`` surface.
        """
        base = self.base
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        for name, module_trainer in self.module_trainers.items():
            module_trainer._build_optimizer()
            self.optimizers[name] = module_trainer.base.optimizer
        base.optimizer = MultiOptimizer(self.optimizers)
        logger.info_rank0(f"OmniTrainer: built {len(self.optimizers)} optimizer(s): {list(self.optimizers)}.")

    def _build_lr_scheduler(self):
        """Call each module-trainer's :meth:`OmniModuleTrainer._build_lr_scheduler` + aggregate.

        Module-trainer schedulers read ``args.train_steps`` (fixed by the shared
        dataset); propagate the global value into each module-trainer's private
        args copy before invoking the build.
        """
        base = self.base
        self.lr_schedulers: Dict[str, Any] = {}
        for name, module_trainer in self.module_trainers.items():
            module_trainer.base.args._train_steps = base.args._train_steps
            module_trainer._build_lr_scheduler()
            self.lr_schedulers[name] = module_trainer.base.lr_scheduler
        base.lr_scheduler = MultiLRScheduler(self.lr_schedulers)

    # ── Callbacks (orchestrator owns trace; each module owns its checkpoint) ───

    def _init_callbacks(self):
        """Build the orchestrator's trace callbacks + the global-state saver.

        Per-model checkpointing (DCP / HF / LoRA) is **not** the orchestrator's
        job: each :class:`OmniModuleTrainer` owns its own callbacks (built in its
        ``__init__``), driven by the ``on_*`` cascade below.  We reuse
        ``base._init_callbacks`` for the metering / logging / profiling stack,
        then repurpose the single-model checkpoint slots: ``checkpointer_callback``
        becomes the module-agnostic :class:`OmniGlobalStateCallback` (step /
        dataloader / environ-meter / rng), and ``hf_ckpt_callback`` is a no-op
        (per-module HF/LoRA lives on each module-trainer).
        """
        base = self.base
        base._init_callbacks()
        base.checkpointer_callback = OmniGlobalStateCallback(self)
        base.hf_ckpt_callback = Callback(base)

    # ── Forward / backward (override single-model path) ────────────────────────

    def forward_backward_step(self, micro_batch: Dict[str, Any]):
        """One gradient-accumulation micro-batch over the training DAG.

        ``OmniModel.forward`` returns ``{"loss", "losses", "outputs"}`` where
        ``loss`` is the summed per-node ``_loss``; a single backward then
        propagates across every FSDP2 unit.
        """
        base = self.base
        micro_batch = base.preforward(micro_batch)

        with base.model_fwd_context, set_batch_invariant_mode(base.args.train.enable_batch_invariant_mode):
            result: Dict[str, Any] = base.model(**micro_batch)

        total_loss: torch.Tensor = result["loss"]
        if total_loss is None:
            raise RuntimeError(
                "OmniModel.forward produced no loss — no training node emitted a `_loss`. "
                "Check that the training data + per-module training forwards are wired (D4/D5)."
            )
        loss_dict: Dict[str, torch.Tensor] = result.get("losses", {})

        with base.model_bwd_context, set_batch_invariant_mode(base.args.train.enable_batch_invariant_mode):
            total_loss.backward()

        del micro_batch
        return total_loss, loss_dict

    def model_reshard(self, micro_step: int, num_micro_steps: int):
        """Toggle ``set_reshard_after_backward`` on every *nested* FSDP2 unit.

        The ``OmniModel`` root is a plain ``nn.Module`` (not an ``FSDPModule``),
        so — unlike the single-model path — we walk its sub-modules and toggle
        each FSDP2 child individually.
        """
        fsdp_cfg = self.base.args.train.accelerator.fsdp_config
        if fsdp_cfg.fsdp_mode != "fsdp2" or fsdp_cfg.reshard_after_backward or num_micro_steps <= 1:
            return
        try:
            from torch.distributed.fsdp import FSDPModule
        except ImportError:
            return
        for mod in self.base.model.modules():
            if isinstance(mod, FSDPModule):
                if micro_step == 0:
                    mod.set_reshard_after_backward(False)
                elif micro_step == num_micro_steps - 1:
                    mod.set_reshard_after_backward(True)

    def train_step(self, data_iterator: Any) -> None:
        base = self.base
        args: VeOmniOmniArguments = base.args
        base.state.global_step += 1

        micro_batches: List[Dict[str, Any]] = next(data_iterator)
        self.on_step_begin(micro_batches=micro_batches)
        synchronize()

        total_loss = 0.0
        total_loss_dict: Dict[str, float] = defaultdict(float)
        num_micro_steps = len(micro_batches)

        for micro_step, micro_batch in enumerate(micro_batches):
            self.model_reshard(micro_step, num_micro_steps)
            loss, loss_dict = self.forward_backward_step(micro_batch)
            total_loss += loss.item() / num_micro_steps
            for k, v in loss_dict.items():
                total_loss_dict[k] += v.item() / num_micro_steps

        # Global gradient clip across every module's DTensor parameters, then
        # step every per-module optimizer + scheduler.
        grad_norm = veomni_clip_grad_norm(base.model, args.train.optimizer.max_grad_norm)
        base.optimizer.step()
        base.lr_scheduler.step()
        base.optimizer.zero_grad()

        self.on_step_end(loss=total_loss, loss_dict=dict(total_loss_dict), grad_norm=grad_norm)

    # ── Callback delegators (trace via base; cascade ckpt into module-trainers) ─
    #
    # ``base.on_*`` fires the orchestrator's trace callbacks (its checkpoint
    # callbacks were neutralised in ``_init_callbacks``); we then cascade the
    # same lifecycle into every module-trainer so each runs its own checkpoint
    # callback (the global ``base.state`` is the single source of step / epoch).

    def on_train_begin(self):
        self.base.on_train_begin()
        for module_trainer in self.module_trainers.values():
            module_trainer.on_train_begin(self.base.state)

    def on_train_end(self):
        self.base.on_train_end()
        for module_trainer in self.module_trainers.values():
            module_trainer.on_train_end(self.base.state)

    def on_epoch_begin(self):
        self.base.on_epoch_begin()
        for module_trainer in self.module_trainers.values():
            module_trainer.on_epoch_begin(self.base.state)

    def on_epoch_end(self):
        self.base.on_epoch_end()
        for module_trainer in self.module_trainers.values():
            module_trainer.on_epoch_end(self.base.state)

    def on_step_begin(self, micro_batches=None):
        self.base.on_step_begin(micro_batches=micro_batches)
        for module_trainer in self.module_trainers.values():
            module_trainer.on_step_begin(self.base.state, micro_batches=micro_batches)

    def on_step_end(self, loss=None, loss_dict=None, grad_norm=None):
        self.base.on_step_end(loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        for module_trainer in self.module_trainers.values():
            module_trainer.on_step_end(self.base.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)

    # ── Train loop (mirrors BaseTrainer.train / VLMTrainer.train) ──────────────

    def train(self):
        base = self.base
        args: VeOmniOmniArguments = base.args
        self.on_train_begin()
        logger.info(
            f"Rank{args.train.local_rank} Start training. "
            f"Start step: {base.start_step}. "
            f"Train steps: {args.train_steps}. "
            f"Start epoch: {base.start_epoch}. "
            f"Train epochs: {args.train.num_train_epochs}."
        )

        for epoch in range(base.start_epoch, args.train.num_train_epochs):
            if hasattr(base.train_dataloader, "set_epoch"):
                base.train_dataloader.set_epoch(epoch)
            base.state.epoch = epoch

            self.on_epoch_begin()

            data_iterator = iter(base.train_dataloader)

            for _ in range(base.start_step, args.train_steps):
                try:
                    self.train_step(data_iterator)
                except StopIteration:
                    logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.dataloader.drop_last}")
                    break

            self.on_epoch_end()

            base.start_step = 0
            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")

        self.on_train_end()

        synchronize()

        base.destroy_distributed()
