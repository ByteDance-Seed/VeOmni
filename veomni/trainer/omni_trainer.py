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

"""OmniTrainer — trainer for OmniModel V2 (per-module FSDP2).

Unlike single-model trainers (BaseTrainer / VLMTrainer), OmniModel is a
*composition* of several independent OmniModule sub-models (Janus: siglip /
vqvae / text_encoder / llama).  This trainer wraps **each sub-module in its
own FSDP2 unit**, builds **one optimizer per trainable module**, and saves /
resumes **each module independently**.

Why per-module FSDP2 (vs. one wrapper over ``OmniModel``)?

* Each module is a self-contained HF model with its own ``_no_split_modules``
  and its own on-disk snapshot — wrapping per child lets us load weights from
  the per-module ``weights_path`` and save/resume each module's DCP folder
  independently.
* The training DAG (``OmniModel.forward``) chains hidden states across
  modules; a **single** ``loss.backward()`` still propagates across every
  FSDP2 unit because the autograd graph connects them.  Each module's FSDP2
  reduce-scatter fires from its own backward hooks.
* Gradient clipping is **global** (over ``OmniModel``'s full DTensor param
  set); the optimizer step iterates every per-module optimizer.

Build flow (``BaseTrainer.__init__`` order, overridden here)
------------------------------------------------------------
1. ``_build_model``             — meta-init each declared module from its
                                  ``OMNI_*_REGISTRY`` class + ``config.json``;
                                  compose into ``OmniModel`` (no weights yet).
                                  Frozen modules get ``requires_grad=False``.
2. ``_build_model_assets``      — load the global tokenizer + wire it into
                                  every module that wants it.
3. ``_build_collate_fn``        — ``seedomni`` → list-only ``SeedOmniCollator``.
4. ``_build_parallelized_model`` — wrap **each child** with
                                  ``build_parallelize_model`` (own FSDP2 unit,
                                  per-module weight load from disk).
5. ``_build_optimizer`` / ``_build_lr_scheduler`` — one per trainable module.
6. ``_init_callbacks``          — swap the single-model checkpoint callback
                                  for a per-module DCP callback.
"""

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import torch.distributed as dist

from ..arguments import DataArguments, ModelArguments, TrainingArguments, VeOmniArguments
from ..data import SeedOmniCollator
from ..distributed.clip_grad_norm import veomni_clip_grad_norm
from ..distributed.torch_parallelize import build_parallelize_model
from ..models import build_foundation_model, build_tokenizer
from ..models.seed_omni.modeling_omni import OmniModel
from ..ops.batch_invariant_ops import set_batch_invariant_mode
from ..optim import build_lr_scheduler, build_optimizer
from ..utils import helper, logging
from ..utils.device import synchronize
from ..utils.model_utils import pretty_print_trainable_parameters
from .base import BaseTrainer, _collect_muon_kwargs
from .callbacks import Callback


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
    (``step`` / ``zero_grad``).  Checkpointing is per-module (handled by
    :class:`OmniPerModuleCheckpointCallback` against the real per-module
    optimizers), so no ``state_dict`` is needed here.
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


# ── Per-module checkpoint callback ──────────────────────────────────────────────


class OmniPerModuleCheckpointCallback(Callback):
    """Save / resume **each FSDP2 module independently**.

    On-disk layout::

        <save_path>/global_step_{N}/
        ├── <module_a>/        # DCP: {model, optimizer, extra_state}
        ├── <module_b>/        # DCP: {model}            (frozen → no optimizer)
        └── trainer_state.pt   # global: step / lr_scheds / dataloader / rng

    Each ``<module>/`` is a self-contained DCP produced by the shared
    :class:`DistributedCheckpointer` over that single child + its optimizer,
    so modules can be resharded / resumed in isolation.
    """

    def __init__(self, trainer: "OmniTrainer") -> None:
        # ``trainer`` is the ``OmniTrainer`` wrapper: base-declared state lives
        # on ``trainer.base`` (model / optimizer / checkpointer / state / ...),
        # while the omni bookkeeping (``module_names`` / ``optimizers``) lives
        # directly on ``trainer``.
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

    def _save(self, global_step: int) -> None:
        trainer: "OmniTrainer" = self.trainer
        base = trainer.base
        args = base.args
        save_dir = os.path.join(args.train.checkpoint.save_path, f"global_step_{global_step}")
        base.checkpointer.wait_for_pending_save()
        helper.empty_cache()

        for name in trainer.module_names:
            child = base.model.get_module(name)
            module_state: Dict[str, Any] = {"model": child, "extra_state": {}}
            if name in trainer.optimizers:
                module_state["optimizer"] = trainer.optimizers[name]
            base.checkpointer.save(os.path.join(save_dir, name), module_state, save_async=False)

        if args.train.global_rank == 0:
            torch.save(
                {
                    "global_step": global_step,
                    "lr_scheduler": base.lr_scheduler.state_dict(),
                    "train_dataloader": base.train_dataloader.state_dict()
                    if base.train_dataloader is not None
                    else {},
                    "torch_rng_state": torch.get_rng_state(),
                },
                os.path.join(save_dir, "trainer_state.pt"),
            )

        helper.empty_cache()
        if dist.is_initialized():
            dist.barrier()
        self._last_saved_step = global_step
        logger.info_rank0(f"OmniTrainer: per-module checkpoint saved at {save_dir}")

    def _load(self) -> None:
        trainer: "OmniTrainer" = self.trainer
        base = trainer.base
        args = base.args
        load_path = args.train.checkpoint.load_path
        if load_path is None:
            return

        base.checkpointer.wait_for_pending_save()
        for name in trainer.module_names:
            child = base.model.get_module(name)
            module_state: Dict[str, Any] = {"model": child, "extra_state": {}}
            if name in trainer.optimizers:
                module_state["optimizer"] = trainer.optimizers[name]
            base.checkpointer.load(os.path.join(load_path, name), module_state)

        trainer_state_path = os.path.join(load_path, "trainer_state.pt")
        if os.path.exists(trainer_state_path):
            ts = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
            base.state.global_step = ts["global_step"]
            base.start_epoch = base.state.global_step // args.train_steps
            base.start_step = base.state.global_step % args.train_steps
            base.lr_scheduler.load_state_dict(ts["lr_scheduler"])
            if base.train_dataloader is not None and ts.get("train_dataloader"):
                base.train_dataloader.load_state_dict(ts["train_dataloader"])
            torch.set_rng_state(ts["torch_rng_state"])

        if dist.is_initialized():
            dist.barrier()
        logger.info_rank0(f"OmniTrainer: resumed per-module checkpoint from {load_path}")


# ── OmniTrainer ────────────────────────────────────────────────────────────────


class OmniTrainer:
    """Trainer for OmniModel V2 — per-module FSDP2 + per-module optimizer.

    Composition over inheritance (mirrors :class:`VLMTrainer`): instead of
    subclassing :class:`BaseTrainer` and relying on ``super().__init__`` +
    selective overrides, we hold a bare ``BaseTrainer`` instance in
    ``self.base`` and drive its private ``_build_*`` / ``on_*`` helpers
    one-by-one.  This makes the (heavily customised) build sequence explicit
    and decoupled from ``BaseTrainer.__init__``'s call order.

    Canonical ``BaseTrainer`` state (``model`` / ``optimizer`` /
    ``lr_scheduler`` / ``state`` / ``checkpointer`` / dataloaders / callbacks)
    lives on ``self.base``.  Omni-specific bookkeeping that is **not** part of
    ``BaseTrainer``'s interface (``omni_config`` / ``module_names`` /
    ``frozen_modules`` / per-module ``optimizers`` / ``lr_schedulers``) lives
    directly on ``self``.      The per-module checkpoint callback therefore binds
    to the ``OmniTrainer`` wrapper and reads base state via ``trainer.base``.
    """

    # Composition: the underlying single-model trainer. Holds all
    # BaseTrainer-declared state (model / optimizer / lr_scheduler / state /
    # checkpointer / dataloaders / callbacks); accessed as ``self.base.*``.
    base: BaseTrainer

    # OmniModel V2 bookkeeping — not part of BaseTrainer's interface, so it
    # lives directly on the wrapper rather than on ``self.base``.
    omni_config: "OmniConfig"  # parsed modules / nodes / edges / training graph
    module_names: List[str]  # ordered names of every declared module
    frozen_modules: set[str]  # modules carrying a truthy freeze / freeze_* knob
    optimizers: Dict[str, torch.optim.Optimizer]  # one per trainable module (aggregated into base.optimizer)
    lr_schedulers: Dict[str, Any]  # one per trainable module (aggregated into base.lr_scheduler)

    def __init__(self, args: VeOmniOmniArguments):
        # BaseTrainer.__init__ is NOT called here; we call its private
        # helpers one-by-one so the (overridden) build sequence is explicit.
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args

        self.base._setup()
        self._build_model()  # build each module via build_foundation_model + compose
        self._freeze_model_module()  # apply per-module freeze + pretty-print params
        self._build_model_assets()  # global tokenizer + per-module wiring
        self.base._build_data_transform()
        self.base._build_dataset()
        self._build_collate_fn()  # seedomni → SeedOmniCollator
        self.base._build_dataloader()
        self._build_parallelized_model()  # per-module FSDP2 wrap + weight load
        self._build_optimizer()  # one optimizer per trainable module
        self._build_lr_scheduler()  # one scheduler per optimizer
        self.base._build_training_context()
        self._init_callbacks()  # swap single-model ckpt → per-module DCP

    # ── Build: model (meta-init + compose) ─────────────────────────────────────

    def _build_model(self):
        """Build every declared module via ``build_foundation_model`` and compose.

        Each module is a self-contained HF model living in its own
        split-checkpoint subfolder.  We reuse the shared loader
        (:func:`build_foundation_model`) per module — exactly like the
        single-model trainers (``BaseTrainer`` / ``VLMTrainer``) — so the
        OmniModule config / modeling classes resolve through the
        registry-aware loader (``get_model_config`` / ``get_model_class``
        consult ``OMNI_*_REGISTRY``).  ``model_type`` is auto-detected from
        each module's ``config.json`` by the loader, so we no longer read it
        explicitly here.

        Weights are **not** loaded at this step: ``weights_path=None`` plus a
        ``meta`` ``init_device`` meta-inits each module (``init_empty_weights``
        inside the loader keeps non-persistent buffers real).  Real per-module
        weights load later in :meth:`_build_parallelized_model`.

        The global tokenizer is built here too, and each module is fully wired
        as it is built — :meth:`set_tokenizer` (so e.g. ``JanusTextEncoder``
        resolves boi/eoi/bos/eos/image ids) and its image ``processor`` (vision
        modules need it to normalise the raw uint8 images carried in
        ``conversation_list``).  :meth:`_build_model_assets` then only exposes
        the already-built assets.
        """
        base = self.base
        args: VeOmniOmniArguments = base.args
        self.omni_config: "OmniConfig" = args.model.load_omni_config()
        self.module_names: List[str] = list(self.omni_config.module_names)

        # Global tokenizer (wired into every module that wants it below).
        tokenizer_path = self.omni_config.tokenizer_path or args.model.tokenizer_path
        base.tokenizer = build_tokenizer(tokenizer_path)

        torch_dtype = "float32" if args.train.accelerator.fsdp_config.mixed_precision.enable else "bfloat16"

        modules: Dict[str, torch.nn.Module] = {}
        for name in self.module_names:
            mod_cfg = self.omni_config.module_config(name)  # deep-copied
            weights_path = mod_cfg.pop("weights_path", None)
            mod_cfg.pop("config_path", None)
            if weights_path is None:
                raise ValueError(f"OmniTrainer: module '{name}' has no `weights_path` in the training YAML.")

            # Remaining ``mod_cfg`` keys are config overrides (freeze flags pass
            # through harmlessly — PretrainedConfig stores unknown kwargs; the
            # freeze knobs are interpreted in :meth:`_freeze_model_module`).
            module = build_foundation_model(
                config_path=weights_path,
                weights_path=None,  # meta-init only; weights load in _build_parallelized_model
                torch_dtype=torch_dtype,
                init_device=args.train.init_device,
                ops_implementation=args.model.ops_implementation,
                config_kwargs=mod_cfg,
            )

            # Wire the global tokenizer into the module (if it wants it).
            if hasattr(module, "set_tokenizer"):
                module.set_tokenizer(base.tokenizer)
            # Meta-init builds from config (not from_pretrained), so the
            # per-module image processor is not auto-loaded — attach it here.
            processor_cls = getattr(type(module), "processor_class", None)
            if processor_cls is not None and getattr(module, "_processor", None) is None:
                try:
                    module._processor = processor_cls.from_pretrained(weights_path)
                    logger.info_rank0(f"OmniTrainer: loaded {processor_cls.__name__} for module '{name}'.")
                except Exception as e:  # noqa: BLE001 — surfaced lazily by the module if truly needed
                    logger.warning_once(
                        f"OmniTrainer: could not load processor for module '{name}' from {weights_path}: {e}. "
                        "Training will fail if this modality's images are actually present."
                    )

            modules[name] = module
            logger.info_rank0(f"OmniTrainer: built module '{name}' from {weights_path}")

        base.model = OmniModel(self.omni_config, modules)
        base.model_config = self.omni_config
        logger.info_rank0(
            f"OmniTrainer: composed OmniModel with {len(self.module_names)} modules ({self.module_names})."
        )

    # ── Freeze (per-module) ─────────────────────────────────────────────────────

    def _freeze_model_module(self):
        """Detect + apply per-module freeze, then report trainable params.

        Mirrors :meth:`VLMTrainer._freeze_model_module` (runs right after
        :meth:`_build_model`, pre-parallelize).  A module is frozen when its
        YAML block carries any truthy ``freeze`` / ``freeze_*`` knob — both the
        detection (``self.frozen_modules``) and the ``requires_grad=False``
        application live here.  :meth:`_build_parallelized_model` re-applies the
        flag after the FSDP2 wrap (and skips gradient checkpointing on them).
        """
        base = self.base
        self.frozen_modules: set[str] = set()
        for name in self.module_names:
            mod_cfg = self.omni_config.module_config(name)  # deep-copied
            if any((k == "freeze" or k.startswith("freeze_")) and bool(v) for k, v in mod_cfg.items()):
                self.frozen_modules.add(name)
                for p in base.model.get_module(name).parameters():
                    p.requires_grad_(False)
        logger.info_rank0(f"OmniTrainer: frozen modules: {sorted(self.frozen_modules) or '(none)'}.")
        pretty_print_trainable_parameters(base.model)
        helper.print_device_mem_info("VRAM usage after building model")

    # ── Build: assets ───────────────────────────────────────────────────────────

    def _build_model_assets(self):
        # Nothing to build here — the tokenizer and per-module set_tokenizer /
        # processor wiring already happened in :meth:`_build_model`.  Just
        # expose the assets that BaseTrainer callbacks read.
        self.base.model_assets = [self.omni_config, self.base.tokenizer]

    # ── Build: collator ─────────────────────────────────────────────────────────

    def _build_collate_fn(self):
        """``seedomni`` → list-only ``SeedOmniCollator``; else BaseTrainer default."""
        if self.base.args.data.data_type == "seedomni":
            self.base.collate_fn = SeedOmniCollator()
            logger.info_rank0("OmniTrainer: using SeedOmniCollator (list-only) for data_type='seedomni'")
        else:
            self.base._build_collate_fn()

    # ── Build: per-module FSDP2 wrap ───────────────────────────────────────────

    def _build_parallelized_model(self):
        """Wrap each OmniModel child in its own FSDP2 unit + load its weights.

        Calling :func:`build_parallelize_model` once per child makes every
        sub-module an independent FSDP2 root over the shared DP mesh.  The
        per-module ``weights_path`` is loaded after the wrap (meta → sharded
        DTensor).  Frozen modules are still wrapped (so weights load) but get
        ``requires_grad`` cleared and grad-checkpointing disabled.
        """
        base = self.base
        args: VeOmniOmniArguments = base.args
        model: OmniModel = base.model

        for name in self.module_names:
            child = model.get_module(name)
            weights_path = self.omni_config.module_config(name).get("weights_path")
            frozen = name in self.frozen_modules
            basic_modules = list(set(getattr(child, "_no_split_modules", None) or []) | set(args.model.basic_modules))
            # Some modules (e.g. SigLIP/VQVAE vision encoders) declare
            # ``supports_gradient_checkpointing = False``; enabling GC on them
            # raises inside ``gradient_checkpointing_enable``. Only enable GC on
            # trainable modules that actually support it.
            supports_gc = bool(getattr(child, "supports_gradient_checkpointing", False))
            enable_gc = args.train.gradient_checkpointing.enable and not frozen and supports_gc
            if args.train.gradient_checkpointing.enable and not frozen and not supports_gc:
                logger.info_rank0(
                    f"OmniTrainer: module '{name}' does not support gradient checkpointing; skipping GC for it."
                )
            logger.info_rank0(f"OmniTrainer: FSDP2-wrapping module '{name}' from {weights_path}...")
            wrapped = build_parallelize_model(
                child,
                init_device=args.train.init_device,
                weights_path=weights_path,
                enable_reshard_after_forward=args.train.accelerator.fsdp_config.reshard_after_forward,
                mixed_precision=args.train.accelerator.fsdp_config.mixed_precision,
                enable_gradient_checkpointing=enable_gc,
                basic_modules=basic_modules,
                enable_reentrant=args.train.gradient_checkpointing.enable_reentrant,
                enable_forward_prefetch=args.train.accelerator.fsdp_config.forward_prefetch,
                enable_fsdp_offload=args.train.accelerator.fsdp_config.offload,
                broadcast_model_weights_from_rank0=args.train.broadcast_model_weights_from_rank0,
                max_load_broadcast_size=args.train.accelerator.fsdp_config.max_load_broadcast_size,
            )
            # Re-attach the (in-place wrapped) child under the same name.
            model.add_module(name, wrapped)
            if frozen:
                for p in wrapped.parameters():
                    p.requires_grad_(False)

        model.train()

    # ── Build: per-module optimizers + schedulers ──────────────────────────────

    def _build_optimizer(self):
        base = self.base
        args: VeOmniOmniArguments = base.args
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        muon_kwargs = _collect_muon_kwargs(args.train.optimizer)
        for name in self.module_names:
            if name in self.frozen_modules:
                continue
            child = base.model.get_module(name)
            self.optimizers[name] = build_optimizer(
                child,
                lr=args.train.optimizer.lr,
                weight_decay=args.train.optimizer.weight_decay,
                fused=True,
                optimizer_type=args.train.optimizer.type,
                no_decay_modules=args.train.optimizer.no_decay_modules,
                no_decay_params=args.train.optimizer.no_decay_params,
                muon_kwargs=muon_kwargs,
            )
        if not self.optimizers:
            raise ValueError("OmniTrainer: no trainable modules — every module is frozen.")
        # ``optimizer`` is a BaseTrainer-declared attr (read by metering /
        # train loop); the per-module dict is omni-specific → kept on ``self``.
        base.optimizer = MultiOptimizer(self.optimizers)
        logger.info_rank0(f"OmniTrainer: built {len(self.optimizers)} optimizer(s): {list(self.optimizers)}.")

    def _build_lr_scheduler(self):
        base = self.base
        args: VeOmniOmniArguments = base.args
        self.lr_schedulers: Dict[str, Any] = {}
        for name, opt in self.optimizers.items():
            self.lr_schedulers[name] = build_lr_scheduler(
                opt,
                train_steps=args.train_steps * args.train.num_train_epochs,
                lr=args.train.optimizer.lr,
                lr_min=args.train.optimizer.lr_min,
                lr_decay_style=args.train.optimizer.lr_decay_style,
                lr_decay_ratio=args.train.optimizer.lr_decay_ratio,
                lr_warmup_ratio=args.train.optimizer.lr_warmup_ratio,
                lr_start=args.train.optimizer.lr_start,
            )
        base.lr_scheduler = MultiLRScheduler(self.lr_schedulers)

    # ── Callbacks (reuse base metering; swap in per-module checkpoint) ─────────

    def _init_callbacks(self):
        base = self.base
        base._init_callbacks()
        # Replace the single-model checkpoint callbacks with per-module DCP.
        # The per-module callback binds to the ``OmniTrainer`` wrapper (``self``)
        # so it can read omni bookkeeping (``module_names`` / ``optimizers``)
        # directly and base state via ``trainer.base``.  ``base.on_step_end`` &
        # co. still drive it through ``base.checkpointer_callback``.
        base.checkpointer_callback = OmniPerModuleCheckpointCallback(self)
        base.hf_ckpt_callback = Callback(base)  # no-op; per-module HF export is a follow-up

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

    # ── Callback delegators (forward to base callbacks bound to ``self.base``) ──

    def on_train_begin(self):
        self.base.on_train_begin()

    def on_train_end(self):
        self.base.on_train_end()

    def on_epoch_begin(self):
        self.base.on_epoch_begin()

    def on_epoch_end(self):
        self.base.on_epoch_end()

    def on_step_begin(self, micro_batches=None):
        self.base.on_step_begin(micro_batches=micro_batches)

    def on_step_end(self, loss=None, loss_dict=None, grad_norm=None):
        self.base.on_step_end(loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)

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
