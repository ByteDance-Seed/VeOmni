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
:class:`~veomni.trainer.omni.omni_module_trainer.OmniModuleTrainer` — which (by
composition over a bare ``BaseTrainer``) reuses the base per-model build helpers
(``_build_model`` / ``_setup_lora`` / ``_build_parallelized_model`` /
``_build_optimizer`` / ``_build_lr_scheduler``) to give that module its own FSDP2
unit, optimizer, lr-scheduler, **checkpoint callback** and on-disk snapshot.

:class:`OmniTrainer` then **strings the module-trainers together**: it owns the
*global* concerns once (distributed ``_setup``, the shared data pipeline, trace
metering, the train loop) and drives the graph — composing the sub-models into
one ``OmniModel``, running the DAG forward, a single ``loss.backward()`` (the
autograd graph connects every FSDP2 unit), and the per-module optimizer step.
Its ``on_{train,epoch,step}_*`` cascade into every module-trainer so each runs
its own checkpoint save/resume.

Division of labour
------------------
* :class:`~veomni.trainer.omni.omni_module_trainer.OmniModuleTrainer` (per
  module): ``_build_model`` → ``_freeze_model_module`` →
  ``_build_parallelized_model`` (FSDP2 wrap + weight load) → ``_init_callbacks``
  (its own per-module DCP callback), then ``_build_optimizer`` /
  ``_build_lr_scheduler`` (called by the orchestrator once ``args.train_steps``
  is known).  Saves / resumes itself when the orchestrator cascades ``on_*``
  into it.
* :class:`OmniTrainer` (orchestrator): global ``_setup`` + data pipeline + trace
  callbacks + train loop; builds the module-trainers, composes ``OmniModel``,
  aggregates their optimizers / schedulers behind :class:`MultiOptimizer` /
  :class:`MultiLRScheduler`, owns the graph forward/backward + the global
  optimizer step, and cascades the callback lifecycle into each module-trainer.
"""

import math
import os
import random
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist

from ...arguments import OmniArguments
from ...data import SeedOmniCollator
from ...data.data_transform import build_data_transform
from ...distributed.clip_grad_norm import veomni_omni_module_clip_grad_norm
from ...distributed.parallel_state import use_parallel_state
from ...models.seed_omni.graphs import GraphProfiler
from ...models.seed_omni.mixins.metric_meter_mixin import MetricMeterResult
from ...models.seed_omni.modeling_omni import OmniModel, _unwrap_module
from ...ops.batch_invariant_ops import set_batch_invariant_mode
from ...utils import helper, logging
from ...utils.device import synchronize
from ...utils.model_utils import pretty_print_trainable_parameters
from ..base import BaseTrainer
from ..callbacks import (
    Callback,
    OmniEnvironMeterCallback,
)
from .omni_module_trainer import OmniModuleTrainer


if TYPE_CHECKING:
    from ...models.seed_omni.configuration_omni import OmniConfig


logger = logging.get_logger(__name__)


# ── Multi-optimizer / multi-scheduler proxies ──────────────────────────────────


class MultiOptimizer:
    """Thin proxy over ``{module_name: torch.optim.Optimizer}``.

    Exposes the minimal :class:`torch.optim.Optimizer` surface the metering /
    logging callbacks read (``param_groups``) and the train loop drives
    (``step`` / ``zero_grad``).  Checkpointing is per-module (handled by each
    module-trainer's :class:`OmniModuleHfCallback` / :class:`OmniModuleLoraCallback`
    against the real per-module optimizer), so no ``state_dict`` is needed here.
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
        # veomni.optim.MultiOptimizer (FSDP2 +ExtraParallel) has ``zero_grad()`` with no args
        # plain torch optimizers default to ``set_to_none=True``.
        for opt in self.optimizers.values():
            opt.zero_grad()


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
            state_path = self._state_path(args.train.checkpoint.save_path, global_step)
            # Create the step dir here: callback order vs. per-module DCP saves
            # is not guaranteed, so we can't assume it already exists.
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            torch.save(
                {
                    "global_step": global_step,
                    "train_dataloader": base.train_dataloader.state_dict()
                    if base.train_dataloader is not None
                    else {},
                    "environ_meter": base.environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "numpy_rng_state": np.random.get_state(),
                    "python_rng_state": random.getstate(),
                },
                state_path,
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
        if ts.get("numpy_rng_state") is not None:
            np.random.set_state(ts["numpy_rng_state"])
        if ts.get("python_rng_state") is not None:
            random.setstate(ts["python_rng_state"])
        if base.start_step == 0 and base.train_dataloader is not None:
            iter(base.train_dataloader)


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
    its own :class:`OmniModuleHfCallback` / :class:`OmniModuleLoraCallback` and the
    orchestrator's ``on_*`` cascade drives them; the orchestrator keeps only trace
    / metering callbacks.
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

    def __init__(self, args: OmniArguments):
        # BaseTrainer.__init__ is NOT called here; we call its private
        # helpers one-by-one so the (overridden) build sequence is explicit.
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args

        self.base._setup()
        self._build_model()
        self._freeze_model_module()
        self._build_model_assets()
        self._build_data_transform()
        self.base._build_dataset()
        self._build_collate_fn()
        self.base._build_dataloader()
        self._build_optimizer()
        self._build_lr_scheduler()
        self.base._build_training_context()
        self._init_callbacks()

    # ── Build: per-module trainers + compose ───────────────────────────────────

    def _build_model(self):
        """Build one :class:`OmniModuleTrainer` (FSDP2) per declared module."""
        base = self.base
        args: OmniArguments = base.args

        self.omni_config = args.load_omni_config()
        self.module_names = self.omni_config.module_names
        self.module_trainers: Dict[str, OmniModuleTrainer] = {}

        modules: Dict[str, torch.nn.Module] = {}
        for name in self.module_names:
            module_config = self.omni_config.module_config(name)
            module_trainer = OmniModuleTrainer(module_config, subfolder_name=name)
            self.module_trainers[name] = module_trainer
            modules[name] = module_trainer.base.model
            logger.info_rank0(f"OmniTrainer: built module-trainer '{name}' from {module_config.model.model_path}")

        model = OmniModel(self.omni_config, modules)
        model.set_module_parallel_states({name: mt.parallel_state for name, mt in self.module_trainers.items()})
        base.model_config = self.omni_config

        base.model = model
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
        # Per-module assets (processor / tokenizer) are loaded in each
        # :class:`OmniModuleTrainer`; the orchestrator only snapshots OmniConfig.
        self.base.model_assets = [self.omni_config]

    # —— Build: data_transform ───────────────────────────────────────────────────────────
    def _build_data_transform(self):
        mm_configs = getattr(self.base.args.data, "mm_configs", None) or {}
        self.base.data_transform = build_data_transform("seedomni", **mm_configs)

    # ── Build: collator ─────────────────────────────────────────────────────────

    def _build_collate_fn(self):
        """list-only ``SeedOmniCollator`` for data_type='seedomni'.

        Collects each active module's optional worker-side CPU preprocessor
        (tokenize / image normalize) and hands them to the collator, so that
        heavy CPU input-prep runs inside the DataLoader worker (overlapping with
        GPU compute via prefetch) instead of blocking the main process inside
        each module's ``pre_forward``. Modules without one contribute nothing.
        Assets (tokenizer / processor) are already loaded by
        ``_build_model_assets`` which runs before this in ``__init__``.
        """
        # Collected (and later applied) in a FIXED, SERIAL order — the config
        # ``modules:`` declaration order, since ``module_trainers`` preserves
        # ``omni_config.module_names`` insertion order. The collator runs them
        # one-by-one in this order, so a module whose prep depends on an earlier
        # module's output (e.g. text chat-template after a vision tower patchifies
        # its image items) must be declared after that module. Inference mirrors
        # this exactly (see ``OmniInferencer._preprocess_request``).
        cpu_preprocessors = []
        for name, module_trainer in self.module_trainers.items():
            model = _unwrap_module(module_trainer.base.model)
            builder = getattr(model, "build_cpu_preprocessor", None)
            preprocessor = builder() if builder is not None else None
            if preprocessor is not None:
                cpu_preprocessors.append(preprocessor)
                logger.info_rank0(
                    f"OmniTrainer: module '{name}' contributes worker-side "
                    f"CPU preprocessor {type(preprocessor).__name__}."
                )
        self.base.collate_fn = SeedOmniCollator(cpu_preprocessors=tuple(cpu_preprocessors))
        logger.info_rank0(
            f"OmniTrainer: SeedOmniCollator with {len(cpu_preprocessors)} worker-side CPU preprocessor(s) "
            "for data_type='seedomni'."
        )

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
            with use_parallel_state(module_trainer.parallel_state):
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
        # The single-model EnvironMeterCallback can't meter OmniModel (no single
        # model_type for FLOPs, entry batch carries only conversation_list).
        # Swap in the per-module metric meter, which delegates token / FLOPs / MFU
        # to each module's MetricMeterMixin and merges them under ``trace/<module>/``.
        base.environ_meter_callback = OmniEnvironMeterCallback(self)
        base.checkpointer_callback = OmniGlobalStateCallback(self)
        base.hf_ckpt_callback = Callback(base)

    # ── Metric metering (gather each module's tokens + theoretical FLOPs) ──────

    def collect_metric_meter(self) -> Dict[str, MetricMeterResult]:
        """Gather every metered module's ``(theoretical_flops, seqlens)``.

        Each :class:`OmniModuleTrainer` stashed its time-independent contribution
        in its own ``on_step_end``; here we only read the results.  The meter
        (:class:`~veomni.utils.omni_helper.OmniEnvironMeter`) sums the FLOPs and merges the token
        lengths, then applies the single whole-graph time to get the overall
        achieved FLOPs / MFU.  Non-metered modules contribute nothing.
        """
        module_metrics: Dict[str, MetricMeterResult] = {}
        for name, module_trainer in self.module_trainers.items():
            result = module_trainer.collect_metric_meter()
            if result is not None:
                module_metrics[name] = result
        return module_metrics

    # ── Forward / backward (override single-model path) ────────────────────────

    def _should_save_graph_profile(self) -> bool:
        profile = self.base.args.graph_profile
        if self.base.args.train.global_rank != 0:
            return False
        if not profile.enable_graph_profiling():
            return False
        return profile.train_start_step <= self.base.state.global_step <= profile.train_end_step

    def _build_graph_profiler(self) -> Optional[GraphProfiler]:
        profile = self.base.args.graph_profile
        if not self._should_save_graph_profile():
            return None

        return GraphProfiler(
            enable_wall_time=profile.enable_wall_time,
            enable_cuda_events=profile.enable_cuda_events,
            enable_memory=profile.enable_memory,
        )

    def _save_graph_profile(self, profiler: Optional[GraphProfiler], *, micro_step: int) -> None:
        if profiler is None:
            return

        args: OmniArguments = self.base.args
        trace_dir = os.path.join(args.train.checkpoint.output_dir, "graph_trace")
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.join(
            trace_dir,
            f"step_{self.base.state.global_step:06d}_micro_{micro_step:02d}_rank_{args.train.global_rank}.txt",
        )
        with open(trace_path, "w", encoding="utf-8") as f:
            f.write("\n".join(profiler.save_records()) + "\n")
        logger.info_rank0(f"OmniTrainer: graph profile trace → {trace_path}")

    def forward_backward_step(self, micro_batch: Dict[str, Any], *, micro_step: int = 0):
        """One gradient-accumulation micro-batch over the training DAG.

        ``OmniModel.forward`` returns ``{"loss", "losses"}`` where ``loss`` is
        the summed per-node ``_loss``; a single backward then propagates across
        every FSDP2 unit.
        """
        base = self.base
        micro_batch = base.preforward(micro_batch)
        profiler = self._build_graph_profiler()

        with base.model_fwd_context, set_batch_invariant_mode(base.args.train.enable_batch_invariant_mode):
            result: Dict[str, Any] = base.model(profiler=profiler, **micro_batch)
        self._save_graph_profile(profiler, micro_step=micro_step)

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
        args: OmniArguments = base.args
        base.state.global_step += 1

        micro_batches: List[Dict[str, Any]] = next(data_iterator)
        self.on_step_begin(micro_batches=micro_batches)
        synchronize()

        total_loss = 0.0
        total_loss_dict: Dict[str, float] = defaultdict(float)
        num_micro_steps = len(micro_batches)

        for micro_step, micro_batch in enumerate(micro_batches):
            self.model_reshard(micro_step, num_micro_steps)
            loss, loss_dict = self.forward_backward_step(micro_batch, micro_step=micro_step)
            total_loss += loss.item() / num_micro_steps
            for k, v in loss_dict.items():
                total_loss_dict[k] += v.item() / num_micro_steps

        max_grad_norm = args.train.optimizer.max_grad_norm
        module_grad_norms = [
            veomni_omni_module_clip_grad_norm(module_trainer.base.model, module_trainer.parallel_state, max_grad_norm)
            for module_trainer in self.module_trainers.values()
        ]
        grad_norm = math.sqrt(sum(g * g for g in module_grad_norms)) if module_grad_norms else 0.0
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
        # Only the orchestrator's meter starts here (records the single
        # whole-graph start time + multi-source ds_idx). Module-trainers do NOT
        # run on_step_begin: per-module token counting happens inside their
        # ``forward`` (right after each module's pre_forward), and per-module
        # timing is meaningless (see OmniEnvironMeter).
        self.base.on_step_begin(micro_batches=micro_batches)

    def on_step_end(self, loss=None, loss_dict=None, grad_norm=None):
        # Module-trainers first: each stashes its per-module metric contribution
        # (``metric_meter_collect`` → theoretical_flops + seqlens) + runs its checkpoint
        # callbacks. The orchestrator's metric callback (inside ``base.on_step_end``)
        # then reads those results, so module-trainers must run *before* it.
        for module_trainer in self.module_trainers.values():
            module_trainer.on_step_end(self.base.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.base.on_step_end(loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)

    # ── Train loop (mirrors BaseTrainer.train / VLMTrainer.train) ──────────────

    def train(self):
        base = self.base
        args: OmniArguments = base.args
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


__all__ = [
    "OmniTrainer",
    "MultiOptimizer",
    "MultiLRScheduler",
    "OmniGlobalStateCallback",
]
