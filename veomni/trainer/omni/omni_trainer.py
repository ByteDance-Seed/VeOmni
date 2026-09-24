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

"""OmniTrainer — orchestrator for OmniModel (one runtime per sub-module).

Unlike single-model trainers (BaseTrainer / VLMTrainer), OmniModel is a
*composition* of several independent OmniModule sub-models (Janus: siglip /
vqvae / text_encoder / llama).  Each sub-model is backed by its **own**
:class:`~veomni.models.seed_omni.accelerated.omni_module.omni_module_runtime.ModuleRuntime` — a
:class:`~veomni.models.model_runtime.VeOmniModelRuntime` subclass, so it inherits
the per-model build sequence and gives that module its own FSDP2 unit,
optimizer, lr-scheduler and on-disk checkpoint subfolder.

:class:`OmniTrainer` owns the *global* concerns once (distributed setup, the
shared data pipeline, step metrics, the train loop) and drives the graph through
its single model handle ``self.model`` — an :class:`OmniModelRuntime` composing
the sub-models — running the DAG forward, a single ``loss.backward()`` (the
autograd graph connects every FSDP2 unit), and the per-module optimizer step.

Division of labour
------------------
* :class:`~veomni.models.seed_omni.accelerated.omni_module.omni_module_runtime.ModuleRuntime` (per
  module): ``_build_model`` → ``_build_model_assets`` → ``_freeze_model_module``
  (freeze + LoRA) → ``_build_parallelized_model`` (FSDP2 wrap + weight load) →
  ``_build_optimizer`` → ``build_checkpoint`` (its own per-module DCP manager).
  The lr-scheduler is built in :meth:`OmniTrainer._build_multi_lr_scheduler`
  once ``train_steps`` is known from the dataset.
* :class:`~veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime.OmniModelRuntime`
  (``self.model``): composes the module runtimes into one :class:`OmniModel`,
  owns the graph loops, ParallelState scoping and graph tracing, and fans
  checkpoint load / save out to every module runtime.
* :class:`OmniTrainer` (orchestrator): distributed setup + data pipeline +
  callbacks + train loop; builds the model handle, aggregates the per-module
  optimizers / schedulers behind :class:`MultiOptimizer` /
  :class:`MultiLRScheduler`, and owns the forward/backward + the optimizer step.
  Checkpoint *cadence* lives in the omni callbacks
  (:class:`OmniModuleDcpCallback` / :class:`OmniModuleHfCallback`) and the
  shared :class:`GlobalStateCallback` (step, dataloader cursor, RNG).
"""

import json
import os
from collections import defaultdict
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, List, Mapping

import torch
import torch.distributed as dist
from torch.utils.checkpoint import set_checkpoint_debug_enabled

from ...arguments.omni_arguments_types import OmniArguments
from ...arguments.parser import save_args
from ...data import SeedOmniCollator, build_dataloader, build_dataset
from ...data.data_transform import build_data_transform
from ...distributed.clip_grad_norm import omni_clip_grad_norm
from ...distributed.offloading import build_activation_offloading_context
from ...distributed.parallel_state import init_parallel_state_from_config, use_parallel_state
from ...models.seed_omni.accelerated import OmniModelRuntime
from ...models.seed_omni.accelerated.omni_module.omni_module_runtime import ModuleRuntime
from ...models.seed_omni.processing_omni import OmniProcessor
from ...ops.batch_invariant_ops import set_batch_invariant_mode
from ...utils import helper, logging
from ...utils.device import get_device_type, get_dist_comm_backend, get_torch_device, synchronize
from ..base import VeOmniIter
from ..callbacks import (
    EvaluateCallback,
    GlobalStateCallback,
    MoERouterMonitorCallback,
    ProfileTraceCallback,
    TqdmCallback,
    TrainerState,
    WandbTraceCallback,
)
from ..callbacks.omni_callbacks import (
    GraphProfileCallback,
    OmniModuleDcpCallback,
    OmniModuleHfCallback,
    OmniRootAssetsCallback,
    OmniStepMetricsCallback,
)


if TYPE_CHECKING:
    from ...models.seed_omni.configuration_omni import OmniConfig

logger = logging.get_logger(__name__)


# ── Multi-optimizer / multi-scheduler proxies ──────────────────────────────────


class MultiOptimizer:
    """Thin proxy over ``{module_name: torch.optim.Optimizer}``.

    Exposes the minimal :class:`torch.optim.Optimizer` surface the logging
    callbacks read (``param_groups``) and the train loop drives
    (``step`` / ``zero_grad``).  Optimizer state is checkpointed per module by
    each :class:`ModuleRuntime`'s own DCP manager, so no ``state_dict`` is needed
    here.
    """

    def __init__(self, optimizers: Dict[str, torch.optim.Optimizer]):
        if not optimizers:
            raise ValueError("OmniTrainer found no trainable module optimizers to build.")
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


def cascade_module_reshard(
    module_runtimes: Mapping[str, ModuleRuntime],
    micro_step: int,
    num_micro_steps: int,
) -> None:
    """Cascade grad-accum FSDP2 reshard intent into every :class:`ModuleRuntime`.

    First micro-step keeps params gathered (``reshard=False``); last micro-step
    reshards (``reshard=True``); middle steps are no-ops. Matches
    :meth:`BaseTrainer.model_reshard` but fans out per omni module.
    """
    if num_micro_steps <= 1:
        return
    if micro_step == 0:
        reshard = False
    elif micro_step == num_micro_steps - 1:
        reshard = True
    else:
        return
    for module_runtime in module_runtimes.values():
        module_runtime._model_reshard(reshard)


def build_omni_model(
    global_args: OmniArguments,
) -> OmniModelRuntime:
    """Build one VeOmni-managed composed model — the trainer's ``self.model``."""
    model_runtime = global_args.resolve_model()
    for name in model_runtime.module_names:
        module_args = model_runtime.modules[name]
        module_args.model_config = dict(module_args.model_config or {})
        module_args.model_config["train_type"] = global_args.train.train_type
    return OmniModelRuntime.from_model_runtime(model_runtime, train=global_args.train, for_inference=False)


# ── OmniTrainer ────────────────────────────────────────────────────────────────


class OmniTrainer:
    """Orchestrator for OmniModel — one :class:`ModuleRuntime` per module.

    Not a :class:`BaseTrainer` subclass: ``BaseTrainer`` assumes one model, one
    optimizer and one checkpoint, while every OmniModule here carries its own.
    The shared pieces (dataset / dataloader builders, trace callbacks,
    :class:`VeOmniIter`) are reused directly.

    There is exactly **one** model handle: ``self.model``, an
    :class:`OmniModelRuntime`.  It composes one :class:`ModuleRuntime` per
    OmniModule (``self.model.module_runtimes``) into a clean :class:`OmniModel`
    and adds the graph loops, ParallelState scoping and graph tracing; the bare
    :class:`OmniModel` surface (``config`` / ``modules_dict`` / ``save_pretrained``)
    is forwarded, so shared callbacks read ``self.model``
    unchanged.  Training requires the runtime — only it exposes the graph loops.

    Canonical training state (``model`` / ``optimizer`` / ``lr_scheduler`` /
    ``state`` / dataloaders / per-step trace metrics) lives on ``self``.  A
    future student+teacher trainer holds two handles by calling
    :func:`build_omni_model` twice.

    Checkpoint I/O is **not** owned here: each :class:`ModuleRuntime` owns its
    DCP manager and :class:`OmniModelRuntime` fans ``load`` / ``save_dcp`` /
    ``save_hf_or_lora`` out to them; the omni callbacks only schedule those calls.
    """

    args: OmniArguments
    state: TrainerState
    start_step: int = 0
    start_epoch: int = 0

    device: torch.device
    model: OmniModelRuntime
    data_transform: Any
    train_dataset: Any
    collate_fn: Any
    train_dataloader: Any
    data_iterator: Any | None = None
    optimizer: MultiOptimizer
    lr_scheduler: MultiLRScheduler

    # ── Per-step trace state (written by omni_callbacks / shared callbacks) ───

    # OmniStepMetricsCallback.on_step_end: training metrics (loss, grad_norm, lr, …).
    # WandbTraceCallback.on_step_end: logs step_env_metrics.
    # TqdmCallback.on_step_end: progress-bar postfix from step_train_metrics.
    step_env_metrics: Dict[str, Any] | None = None
    step_train_metrics: Dict[str, Any] | None = None
    LOG_SAMPLE: bool = True

    @property
    def model_config(self) -> "OmniConfig":
        """Shared trace callbacks read ``trainer.model_config`` — alias of ``model.config``."""
        return self.model.config

    def __init__(self, args: OmniArguments):
        if args.train.train_type == "offline_cache":
            raise NotImplementedError("`train.train_type: offline_cache` is not supported by OmniTrainer yet.")
        self.args = args
        self.device = self.setup_distributed(args)

        self._build_model()
        self._build_step_contexts()
        self._build_data()
        self._build_multi_optimizer()
        self._build_multi_lr_scheduler()
        self._init_callbacks()

    @staticmethod
    def setup_distributed(args: OmniArguments) -> torch.device:
        """Init process group, device, seed, and register orchestrator ParallelState."""
        logger.info_rank0(json.dumps(asdict(args), indent=2))

        device_str = f"{get_device_type()}:{args.train.local_rank}"
        get_torch_device().set_device(device_str)
        device = torch.device(device_str)

        if not dist.is_initialized():
            dist.init_process_group(backend=get_dist_comm_backend())

        logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")

        init_parallel_state_from_config(args.model.accelerator, name="base")

        helper.set_seed(args.train.seed, args.train.enable_full_determinism)
        helper.enable_high_precision_for_bf16()

        if args.train.local_rank == 0:
            helper.enable_third_party_logging()

        if args.train.global_rank == 0:
            save_args(args, args.train.checkpoint.output_dir)

        set_checkpoint_debug_enabled(args.model.accelerator.gradient_checkpointing.debug)
        return device

    def destroy_distributed(self) -> None:
        """Tear down the process group."""
        from ...distributed.parallel_state import clear_parallel_state
        from ...utils.device import is_nccl_backend

        if not dist.is_available() or not dist.is_initialized():
            return

        backend = dist.get_backend()
        helper.empty_cache()
        dist.barrier()

        if is_nccl_backend(backend) and os.getenv("VEOMNI_DESTROY_NCCL_ON_EXIT", "0") != "1":
            logger.info_rank0(
                "Skipping explicit NCCL process-group destroy on normal trainer exit. "
                "Set VEOMNI_DESTROY_NCCL_ON_EXIT=1 to restore the previous teardown behavior."
            )
            return

        synchronize()
        dist.destroy_process_group()
        clear_parallel_state()

    def _build_data(self) -> None:
        """Build transform → dataset (fixes ``train_steps``) → dataloader."""
        with use_parallel_state("base"):
            self._build_data_transform()
            self._build_train_dataset()
            self._build_train_dataloader()

    def _build_data_transform(self) -> None:
        self.data_transform = build_data_transform(self.args.data.data_type, **self.args.data.mm_configs)

    def _build_train_dataset(self) -> None:
        args: OmniArguments = self.args
        train_dataset = build_dataset(
            dataset_name=args.data.dataset_name,
            transform=self.data_transform,
            seed=args.train.seed,
            **asdict(args.data),
        )
        dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
        if args.data.datasets_type == "mapping":
            dataset_length = dataset_length / args.model.accelerator.dp_size
        args.compute_train_steps(dataset_length)
        self.train_dataset = train_dataset

    def _build_train_dataloader(self) -> None:
        args: OmniArguments = self.args
        processor = OmniProcessor.from_config(self.model.config, checkpoint_root=args.model.model_path)
        # FSDP-anchor dummy tensors are only exercised by the training (inference=False)
        # branch. `_build_model` already ran (see `setup`), so every module's own
        # resolved `ModuleRuntime.model_config` is sitting in memory — hand that
        # straight to the processor instead of re-reading each module's config.json
        # from disk. Mirrors the per-module load dtype resolved in
        # ModuleRuntime.build_model; a single dtype is fine since these dummies
        # get re-cast to each module's live self.dtype before reaching its forward.
        dummy_dtype = torch.float32 if args.model.accelerator.fsdp_config.mixed_precision.enable else torch.bfloat16
        module_configs = {name: rt.model_config for name, rt in self.model.module_runtimes.items()}
        processor.bind_dummy_inputs(module_configs, dtype=dummy_dtype)
        logger.info_rank0(f"SeedOmniCollator with {len(processor)} worker-side CPU preprocessor(s).")
        self.collate_fn = SeedOmniCollator(processor=processor)
        dataloader_kwargs = asdict(args.data.dataloader)
        dataloader_type = dataloader_kwargs.pop("type")
        dataloader_kwargs.pop("use_background_prefetcher", None)
        self.train_dataloader = build_dataloader(
            dataloader_type=dataloader_type,
            dataset=self.train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train_steps,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
            dyn_bsz=args.train.dyn_bsz,
            dyn_bsz_runtime=args.train.dyn_bsz_runtime,
            dyn_bsz_count_mode=args.train.dyn_bsz_count_mode,
            dyn_bsz_physical_overflow_ratio=args.train.dyn_bsz_physical_overflow_ratio,
            dyn_bsz_buffer_size=args.data.dyn_bsz_buffer_size,
            seed=args.train.seed,
            collate_fn=self.collate_fn,
            save_steps=args.train.checkpoint.save_steps,
            **dataloader_kwargs,
        )

    # ── Build: per-module trainers + compose ───────────────────────────────────

    def _build_model(self):
        self.model = build_omni_model(global_args=self.args)

    # ── Aggregate per-module optimizers / schedulers ───────────────────────────

    def _build_multi_optimizer(self) -> None:
        """Wrap per-module optimizers in :class:`MultiOptimizer`."""
        optimizers = {
            name: module_runtime.optimizer
            for name, module_runtime in self.model.module_runtimes.items()
            if module_runtime.optimizer is not None
        }
        self.optimizer = MultiOptimizer(optimizers)
        logger.info_rank0(f"OmniTrainer: wired {len(optimizers)} optimizer(s): {list(optimizers)}.")

    def _build_multi_lr_scheduler(self) -> None:
        """Build per-module lr-schedulers and wrap them in :class:`MultiLRScheduler`.

        ``_build_lr_scheduler`` no-ops for a fully-frozen module, so such modules
        contribute no entry to the wrapper.
        """
        total_steps = self.args.train_steps * self.args.train.num_train_epochs
        for module_runtime in self.model.module_runtimes.values():
            module_runtime._build_lr_scheduler(total_steps)
        lr_schedulers = {
            name: module_runtime.lr_scheduler
            for name, module_runtime in self.model.module_runtimes.items()
            if module_runtime.lr_scheduler is not None
        }
        self.lr_scheduler = MultiLRScheduler(lr_schedulers)

    # ── Step-level training contexts (explicit split, mirrors BaseTrainer) ───

    def _build_step_contexts(self) -> None:
        """Build reusable forward/backward context managers from ``args`` (once, at init).

        Each context is entered explicitly in :meth:`forward_backward_step` so
        the train loop reads as a recipe. Grad-accum
        FSDP reshard is handled imperatively via :func:`cascade_module_reshard`.

        The activation-offload contexts are genuinely reusable (``nullcontext`` /
        ``saved_tensors_hooks`` don't consume state on ``__enter__``), so they are
        built once here and entered repeatedly. ``set_batch_invariant_mode`` is a
        ``@contextmanager`` generator CM — single-use by construction (Python
        deletes its ``args``/``kwds``/``func`` on first ``__enter__``) — so it is
        *not* cached here; every call site builds a fresh one per step instead.
        """
        args = self.args
        offload = args.model.accelerator.offload_config
        enable_activation = bool(offload and offload.enable_activation)
        self.fwd_activation_offload_ctx, self.bwd_activation_offload_ctx = build_activation_offloading_context(
            enable_activation=enable_activation,
            enable_gradient_checkpointing=args.model.accelerator.gradient_checkpointing.enable,
            activation_gpu_limit=offload.activation_gpu_limit if offload else 0.0,
        )
        logger.info_rank0(
            "OmniTrainer: step contexts — "
            f"activation_offload={enable_activation}, "
            f"batch_invariant={args.train.enable_batch_invariant_mode}"
        )

    def _cascade_module_reshard(self, micro_step: int, num_micro_steps: int) -> None:
        cascade_module_reshard(self.model.module_runtimes, micro_step, num_micro_steps)

    # ── Callbacks (orchestrator owns trace + per-module checkpoint scheduling) ─

    def _init_callbacks(self):
        """Build orchestrator trace callbacks + global / per-module checkpoint schedulers."""
        self.step_metrics_callback = OmniStepMetricsCallback(self)
        self.tqdm_callback = TqdmCallback(self)
        self.wandb_callback = WandbTraceCallback(self)
        self.profile_callback = ProfileTraceCallback(self)
        self.graph_profile_callback = GraphProfileCallback(self)
        self.omni_root_assets_callback = OmniRootAssetsCallback(self)
        self.module_dcp_callback = OmniModuleDcpCallback(self)
        self.module_hf_ckpt_callback = OmniModuleHfCallback(self)
        self.global_state_callback = GlobalStateCallback(self)
        self.evaluate_callback = EvaluateCallback(self)
        self.moe_monitor_callback = MoERouterMonitorCallback(self)
        self._callback_handlers = [
            self.step_metrics_callback,
            self.tqdm_callback,
            self.wandb_callback,
            self.profile_callback,
            self.graph_profile_callback,
            # Weights first, then the cursor — same order and reasons as
            # ``BaseTrainer._init_callbacks``.
            self.omni_root_assets_callback,
            self.module_dcp_callback,
            self.module_hf_ckpt_callback,
            self.global_state_callback,
            self.evaluate_callback,
            self.moe_monitor_callback,
        ]
        self.state = TrainerState()

    def _callbacks(self, stage: str, **kwargs) -> None:
        # Publish the stage on the state so save paths can branch on it without a
        # ``stage`` argument threaded through every layer.
        self.state.stage = stage
        for callback in self._callback_handlers:
            getattr(callback, f"on_{stage}")(self.state, **kwargs)

    # —— Callback helpers ──────────────────────────────────────────────────────
    def save_model_assets(self) -> None:
        """Export the omni-root HF layout (config + graphs + module sidecars, no weights)."""
        args: OmniArguments = self.args
        if args.train.global_rank == 0:
            save_directory = args.train.checkpoint.model_assets_dir
            self.model.save_pretrained(save_directory, save_module_weights=False)
            logger.info_rank0(f"OmniTrainer: saved OmniModel assets to {save_directory}.")
        if dist.is_initialized():
            dist.barrier()

    def load(self) -> None:
        """Resume every composed model's module checkpoints.

        Today that is ``self.model`` only; a student+teacher trainer extends this
        to ``self.student.load()`` / ``self.teacher.load()``.
        """
        self.model.load()

    def save_dcp(self, state: TrainerState) -> None:
        """Write every composed model's distributed checkpoint (train resume).

        Scheduling (every-N-steps / epochs) stays in the checkpoint callbacks;
        this method only fans out to the model handles.
        """
        self.model.save_dcp(state)

    def save_hf_or_lora(self, state: TrainerState) -> None:
        """Export every composed model's HF weights / LoRA adapter."""
        self.model.save_hf_or_lora(state)

    def init_graph_profile(self) -> None:
        """Open a graph profiler on every composed model handle for this step.

        Scheduling (enabled flags / rank / step window) is owned by
        :class:`GraphProfileCallback`.  Each handle keeps its own
        :attr:`~OmniModelRuntime.step_profiler` so student/teacher do not share one.
        """
        profile = self.args.train.graph_profile
        self.model.begin_step_trace(profile)

    def flush_graph_profile(self, state: TrainerState) -> None:
        """Write and clear every composed model's step graph profiler (no-op if idle)."""
        args = self.args
        self.model.flush_step_trace(
            state.global_step,
            output_dir=args.train.checkpoint.output_dir,
            rank=args.train.global_rank,
            tag="model",
        )

    # ── Grad-norm helpers ─────────────────────────────────────────────────────

    def _clip_grad_norm(self) -> float:
        """Clip grads across OmniModules according to ``grad_clip_scope``.

        * ``per_module`` (default): each :class:`ModuleRuntime` clips the params of
          its own FSDP unit against **its own** ``optimizer.max_grad_norm`` (every
          OmniModule carries its own optimizer config); the whole-model norm is
          their L2 combination. Empty (no module runtimes) → 0.0.
        * ``global``: measure unclipped, ``total=sqrt(sum n_i^2)``, then one scale
          coeff for all modules (seedream ``gradient_clip_val`` semantics), against
          the model-level ``max_grad_norm`` the modules inherit from.
        """
        opt_cfg = self.args.model.optimizer
        return omni_clip_grad_norm(
            self.model.module_runtimes,
            opt_cfg.max_grad_norm,
            grad_clip_scope=opt_cfg.grad_clip_scope,
        )

    def preforward(self, micro_batch: Dict[str, Any]) -> Dict[str, Any]:
        def _to_device(v: Any) -> Any:
            if isinstance(v, torch.Tensor):
                return v.to(self.device, non_blocking=True)
            if isinstance(v, dict):
                return {k: _to_device(vv) for k, vv in v.items()}
            return v

        micro_batch = {k: _to_device(v) for k, v in micro_batch.items()}
        if getattr(self, "LOG_SAMPLE", True):
            helper.print_example(example=micro_batch, rank=self.args.train.local_rank)
            self.LOG_SAMPLE = False
        return micro_batch

    # ── Main entrypoints: forward/backward → step → train loop ─────────────────

    def forward_backward_step(self, micro_batch: Dict[str, Any], *, micro_step: int = 0, num_micro_steps: int = 1):
        """One gradient-accumulation micro-batch over the training DAG.

        ``OmniModelRuntime.forward`` returns ``{"loss", "losses"}`` where ``loss``
        is the summed per-node ``_loss``; a single backward then propagates across
        every FSDP2 unit.

        Each node's loss is a mean over its own micro-batch, so the backward is
        scaled by ``1 / num_micro_steps``: accumulated gradients are then the mean
        over micro-batches, not a sum that grows with the accumulation count. The
        returned loss stays unscaled for logging.
        """
        micro_batch = self.preforward(micro_batch)
        self._cascade_module_reshard(micro_step, num_micro_steps)

        # Forward: spill activations to CPU (if enabled) + batch-invariant ops.
        with self.fwd_activation_offload_ctx, set_batch_invariant_mode(self.args.train.enable_batch_invariant_mode):
            result: Dict[str, Any] = self.model.forward(micro_batch)

        total_loss: torch.Tensor = result["loss"]
        loss_dict: Dict[str, torch.Tensor] = result.get("losses", {})

        # Backward: separate offload hook stack (may differ from forward when GC is on).
        with self.bwd_activation_offload_ctx, set_batch_invariant_mode(self.args.train.enable_batch_invariant_mode):
            (total_loss / num_micro_steps).backward()

        del micro_batch
        return total_loss, loss_dict

    def train_step(self, data_iterator: Any) -> None:
        self.state.global_step += 1

        micro_batches: List[Dict[str, Any]] = next(data_iterator)
        self._callbacks(stage="step_begin", micro_batches=micro_batches)
        if self.args.train.sync_each_train_step:
            synchronize()

        total_loss = 0.0
        total_loss_dict: Dict[str, float] = defaultdict(float)
        num_micro_steps = len(micro_batches)

        for micro_step, micro_batch in enumerate(micro_batches):
            loss, loss_dict = self.forward_backward_step(
                micro_batch, micro_step=micro_step, num_micro_steps=num_micro_steps
            )
            total_loss += loss.item() / num_micro_steps
            for k, v in loss_dict.items():
                total_loss_dict[k] += v.item() / num_micro_steps

        grad_norm = self._clip_grad_norm()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        self._callbacks(stage="step_end", loss=total_loss, loss_dict=dict(total_loss_dict), grad_norm=grad_norm)

    def train(self):
        args: OmniArguments = self.args
        self._callbacks(stage="train_begin")
        logger.info(
            f"Rank{args.train.local_rank} Start training. "
            f"Start step: {self.start_step}. "
            f"Train steps: {args.train_steps}. "
            f"Start epoch: {self.start_epoch}. "
            f"Train epochs: {args.train.num_train_epochs}."
        )

        for epoch in range(self.start_epoch, args.train.num_train_epochs):
            if hasattr(self.train_dataloader, "set_epoch"):
                self.train_dataloader.set_epoch(epoch)
            self.state.epoch = epoch

            self._callbacks(stage="epoch_begin")

            self.data_iterator = VeOmniIter(
                self.train_dataloader,
                use_background_prefetcher=args.data.dataloader.use_background_prefetcher,
            )

            for _ in range(self.start_step, args.train_steps):
                try:
                    self.train_step(self.data_iterator)
                except StopIteration:
                    logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.dataloader.drop_last}")
                    break

            self._callbacks(stage="epoch_end")

            self.start_step = 0
            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")

            if args.data.dataloader.use_background_prefetcher:
                self.data_iterator.stop()

        self._callbacks(stage="train_end")

        if self.data_iterator is not None and args.data.dataloader.use_background_prefetcher:
            self.data_iterator.stop()

        synchronize()

        self.destroy_distributed()


__all__ = [
    "OmniTrainer",
    "MultiOptimizer",
    "MultiLRScheduler",
    "build_omni_model",
    "cascade_module_reshard",
]
