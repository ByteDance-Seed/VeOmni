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
  The lr-scheduler is built by :meth:`OmniModelRuntime._build_lr_scheduler`
  once ``train_steps`` is known from the dataset.
* :class:`~veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime.OmniModelRuntime`
  (``self.model``): composes the module runtimes into one :class:`OmniModel`,
  owns the graph loops, ParallelState scoping and graph tracing, aggregates the
  per-module optimizers / schedulers behind ``MultiOptimizer`` /
  ``MultiLRScheduler``, and fans clipping and checkpoint load / save out to
  every module runtime.
* :class:`OmniTrainer` (orchestrator): distributed setup + data pipeline +
  callbacks + train loop; builds the model handle and owns the
  forward/backward + the optimizer step.
  Checkpoint *cadence* lives in the shared :class:`CheckpointCallback` and
  :class:`GlobalStateCallback` (step, dataloader cursor, RNG).
"""

import json
import math
import os
from dataclasses import asdict
from typing import Any, Dict, List, Mapping

import torch
import torch.distributed as dist
from torch.utils.checkpoint import set_checkpoint_debug_enabled

from ...arguments.omni_arguments_types import OmniArguments, build_omni_model_runtime_args
from ...arguments.parser import save_args
from ...checkpoint import layout
from ...data import build_dataloader, build_dataset
from ...data import seed_omni as _seed_omni_data  # noqa: F401  (registers data_type="seedomni" and its preprocessors)
from ...data.data_transform import build_data_transform
from ...data.seed_omni.collator import SeedOmniCollator
from ...distributed.async_offload import reset_async_activation_offload
from ...distributed.offloading import build_activation_offloading_context
from ...distributed.parallel_state import clear_parallel_state, init_parallel_state_from_config, use_parallel_state
from ...models.seed_omni.accelerated import OmniModelRuntime, build_omni_model_runtime
from ...models.seed_omni.accelerated.omni_module.omni_module_runtime import ModuleRuntime
from ...models.seed_omni.processing_omni import OmniProcessor
from ...models.seed_omni.utils.offline_cache import SeedOmniOfflineCacheWriter
from ...ops.batch_invariant import set_batch_invariant_mode
from ...utils import helper, logging
from ...utils.device import (
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
    is_nccl_backend,
    synchronize,
)
from ...utils.omni_helper import OmniEnvironMeter
from ..base import VeOmniIter
from ..callbacks import (
    CheckpointCallback,
    EvaluateCallback,
    GlobalStateCallback,
    ProfileTraceCallback,
    TqdmCallback,
    TrainerState,
    WandbTraceCallback,
)
from ..callbacks.omni_callbacks import (
    GraphProfileCallback,
    OmniStepMetricsCallback,
)


logger = logging.get_logger(__name__)


def batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move a graph batch's top-level tensors (and those one dict deep) onto ``device``.

    Tensors a module keeps inside its conversation items stay where the
    preprocessor built them; placing those is the module's own ``pre_forward`` /
    ``pre_generate`` job.
    """

    def _to_device(v: Any) -> Any:
        if isinstance(v, torch.Tensor):
            return v.to(device, non_blocking=True)
        if isinstance(v, dict):
            return {k: _to_device(vv) for k, vv in v.items()}
        return v

    return {k: _to_device(v) for k, v in batch.items()}


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

    Job state (``model`` / ``state`` / dataloaders / per-step trace metrics)
    lives on ``self``; the optimizer and lr-scheduler live on ``self.model``,
    as they do on a :class:`BaseTrainer`'s runtime.  A
    future student+teacher trainer holds two handles by calling
    :func:`build_omni_model_runtime` twice.

    Checkpoint I/O is **not** owned here: each :class:`ModuleRuntime` owns its
    DCP manager and :class:`OmniModelRuntime` fans ``load`` / ``save_dcp`` /
    ``save_hf_or_lora`` out to them; :class:`CheckpointCallback` only schedules those calls.
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
    train_steps: int = 0

    # Training context
    model_fwd_context: Any
    model_bwd_context: Any

    environ_meter: OmniEnvironMeter  # see in OmniStepMetricsCallback
    # OmniStepMetricsCallback.on_step_end: training + efficiency metrics (loss, grad_norm, lr, mfu, …).
    # WandbTraceCallback.on_step_end: logs step_env_metrics.
    # TqdmCallback.on_step_end: progress-bar postfix from step_train_metrics.
    step_env_metrics: Dict[str, Any] | None = None
    step_train_metrics: Dict[str, Any] | None = None
    LOG_SAMPLE: bool = True
    offline_cache_writer: SeedOmniOfflineCacheWriter | None = None

    def __init__(self, args: OmniArguments):
        self.args = args
        self.device = self._setup(args)
        self.model = self._build_model_runtime()
        # The collator and dataset builders read the orchestrator's ParallelState.
        with use_parallel_state("base"):
            self._build_data_transform()
            self._build_dataset()
            self._build_collate_fn()
            self._build_dataloader()
        # The dataset fixes train_steps, which the schedule needs.
        self._build_lr_scheduler()
        self._build_training_context()
        self._build_offline_cache_writer()
        self._init_callbacks()

    @staticmethod
    def _setup(args: OmniArguments, *, save_launch_args: bool = True) -> torch.device:
        """Init process group, device, seed, and register orchestrator ParallelState.

        ``save_launch_args`` writes the resolved launcher args into
        ``train.checkpoint.output_dir``; inference turns it off because it
        produces no training artefacts.
        """
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

        if save_launch_args and args.train.global_rank == 0:
            save_args(args, args.train.checkpoint.output_dir)

        set_checkpoint_debug_enabled(args.model.accelerator.gradient_checkpointing.debug)
        return device

    def destroy_distributed(self) -> None:
        """Tear down the process group."""
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

    def _build_model_runtime(self) -> OmniModelRuntime:
        """Build the composed model — every module built, wrapped and given its optimizer.

        ``train.training_task='offline_embedding'`` trains nothing by design, so
        only it may build no optimizer.
        """
        model = build_omni_model_runtime(build_omni_model_runtime_args(self.args), train_args=self.args.train)
        if model.optimizer is None and self.args.train.training_task != "offline_embedding":
            raise ValueError("OmniTrainer has nothing to train: every module is frozen.")
        return model

    def _build_offline_cache_writer(self) -> None:
        if self.args.train.training_task == "offline_embedding":
            self.offline_cache_writer = SeedOmniOfflineCacheWriter(self.args.train.offline_cache_dir)

    def _build_lr_scheduler(self) -> None:
        """Size the run, then let the model schedule over it."""
        self.model._build_lr_scheduler(self.args.train_steps * self.args.train.num_train_epochs)

    def _build_data_transform(self) -> None:
        args: OmniArguments = self.args
        processor = OmniProcessor.from_config(self.model.config, checkpoint_root=args.model.resolved_model_path)
        # FSDP-anchor dummy tensors are only exercised by the training (inference=False)
        # branch. The model runtime is already built, so every module's own
        # resolved `ModuleRuntime.model_config` is sitting in memory — hand that
        # straight to the processor instead of re-reading each module's config.json
        # from disk. Mirrors the per-module load dtype resolved in
        # ModuleRuntime.build_model; a single dtype is fine since these dummies
        # get re-cast to each module's live self.dtype before reaching its forward.
        dummy_dtype = torch.float32 if args.model.accelerator.fsdp_config.mixed_precision.enable else torch.bfloat16
        module_configs = {name: rt.model_config for name, rt in self.model.module_runtimes.items()}
        processor.bind_dummy_inputs(module_configs, dtype=dummy_dtype)
        logger.info_rank0(f"SeedOmni data_transform with {len(processor)} worker-side CPU preprocessor(s).")
        self.data_transform = build_data_transform(
            args.data.data_type,
            processor=processor,
            **args.data.mm_configs,
        )

    def _build_dataset(self) -> None:
        args: OmniArguments = self.args
        self.train_dataset = build_dataset(
            dataset_name=args.data.dataset_name,
            transform=self.data_transform,
            seed=args.train.seed,
            shuffle=args.data.dataloader.shuffle,
            **asdict(args.data),
        )
        dataset_length = None if not hasattr(self.train_dataset, "__len__") else len(self.train_dataset)
        if args.data.datasets_type == "mapping":
            dataset_length = dataset_length / args.model.accelerator.dp_size
        args.compute_train_steps(dataset_length)
        self.train_steps = args.train_steps

    def _build_collate_fn(self) -> None:
        self.collate_fn = SeedOmniCollator()

    def _build_dataloader(self) -> None:
        args: OmniArguments = self.args
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

    def _build_training_context(self) -> None:
        """Build the sync activation-offload fwd/bwd contexts from the global accelerator.

        Async offload needs no trainer-level context: its hooks are attached in
        each module's own parallelize step, and ``enable_activation`` is then
        off, leaving both contexts ``nullcontext``.
        ``set_batch_invariant_mode`` is a single-use generator CM, so it is built
        per step instead of cached here.
        """
        args = self.args
        offload = args.model.accelerator.offload_config
        enable_activation = bool(offload and offload.enable_activation)
        self.model_fwd_context, self.model_bwd_context = build_activation_offloading_context(
            enable_activation=enable_activation,
            enable_gradient_checkpointing=args.model.accelerator.gradient_checkpointing.enable,
            activation_gpu_limit=offload.activation_gpu_limit if offload else 0.0,
        )

    def _init_callbacks(self):
        """Build orchestrator trace callbacks + global / per-module checkpoint schedulers."""
        self.step_metrics_callback = OmniStepMetricsCallback(self)
        self.tqdm_callback = TqdmCallback(self)
        self.wandb_callback = WandbTraceCallback(self)
        self.profile_callback = ProfileTraceCallback(self)
        self.graph_profile_callback = GraphProfileCallback(self)
        self.checkpoint_callback = CheckpointCallback(self)
        self.global_state_callback = GlobalStateCallback(self)
        self.evaluate_callback = EvaluateCallback(self)
        self._callbacks = [
            self.step_metrics_callback,
            self.tqdm_callback,
            self.wandb_callback,
            self.profile_callback,
            self.graph_profile_callback,
            # Weights first, then the cursor — same order and reasons as
            # ``BaseTrainer._init_callbacks``.
            self.checkpoint_callback,
            self.global_state_callback,
            self.evaluate_callback,
        ]
        self.state = TrainerState()

    def save_model_assets(self) -> None:
        """Write every composed model's omni-root sidecars."""
        self.model.save_model_assets()

    def load(self) -> None:
        """Resume every composed model's module checkpoints.

        Today that is ``self.model`` only; a student+teacher trainer extends this
        to ``self.student.load()`` / ``self.teacher.load()``.

        Raises:
            ValueError: ``load_path`` is a step whose manifest disagrees with this
                run on whether a step carries model state -- an ``offline_embedding``
                step resumed by a training run, or the reverse.
        """
        load_path = self.args.train.checkpoint.load_path
        manifest = layout.read_manifest(load_path) if load_path else None
        if manifest is not None and bool(manifest.get("model_state", True)) != self.saves_model_state:
            if self.saves_model_state:
                mismatch = (
                    f"{load_path} holds no model state (an offline_embedding step), but this run trains modules."
                )
            else:
                mismatch = f"{load_path} holds model state, but this offline_embedding run saves none."
            raise ValueError(
                f"{mismatch} An offline_embedding run and a training run must not share "
                "train.checkpoint.output_dir: load_path: auto resumes the newest complete step in it."
            )
        self.model.load()

    @property
    def saves_model_state(self) -> bool:
        """Whether a DCP save writes any module's state.

        False when every module is fully frozen, as in ``offline_embedding``: no
        module has a checkpoint manager, so a step is only the trainer's cursor
        and ``GlobalStateCallback`` marks its manifest so ``load_path: auto``
        still finds it.
        """
        return any(runtime.checkpoint is not None for runtime in self.model.module_runtimes.values())

    def save_dcp(self, state: TrainerState) -> None:
        """Write every composed model's distributed checkpoint (train resume).

        Scheduling (every-N-steps / epochs) stays in the checkpoint callbacks;
        this method only fans out to the model handles.
        """
        self.model.save_dcp(state)

    def save_hf_or_lora(self, state: TrainerState, stage: str = "step_end") -> None:
        """Export every composed model's HF weights / LoRA adapter."""
        self.model.save_hf_or_lora(state, stage=stage)

    def wait_for_pending_save(self) -> None:
        """Drain every composed model's in-flight async checkpoint writes."""
        self.model.wait_for_pending_save()

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

    def _run_callbacks(self, stage: str, **kwargs) -> None:
        for callback in self._callbacks:
            getattr(callback, f"on_{stage}")(self.state, **kwargs)

    def on_train_begin(self):
        self._run_callbacks("train_begin")

    def on_train_end(self):
        self._run_callbacks("train_end")

    def on_epoch_begin(self):
        self._run_callbacks("epoch_begin")

    def on_epoch_end(self):
        self._run_callbacks("epoch_end")

    def on_step_begin(self, micro_batches=None, **kwargs):
        self._run_callbacks("step_begin", micro_batches=micro_batches, **kwargs)

    def on_step_end(self, loss=None, loss_dict=None, grad_norm=None):
        self._run_callbacks("step_end", loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)

    def preforward(self, micro_batch: Dict[str, Any]) -> Dict[str, Any]:
        micro_batch = batch_to_device(micro_batch, self.device)
        if getattr(self, "LOG_SAMPLE", True):
            helper.print_example(example=micro_batch, rank=self.args.train.local_rank)
            self.LOG_SAMPLE = False
        return micro_batch

    @staticmethod
    def _to_int64_scalar(raw_tokens: Any, device: torch.device) -> torch.Tensor:
        if isinstance(raw_tokens, torch.Tensor):
            return raw_tokens.detach().to(device=device, dtype=torch.int64)
        return torch.tensor(raw_tokens, device=device, dtype=torch.int64)

    def forward_backward_step(
        self,
        micro_batch: Dict[str, Any],
        *,
        num_micro_steps: int = 1,
        window_loss_tokens: Mapping[str, torch.Tensor] | None = None,
    ):
        """One gradient-accumulation micro-batch over the training DAG.

        ``OmniModelRuntime.forward`` returns ``{"loss", "losses", "loss_tokens", "conversation_list"}``.
        Under ``sample_mean``, backward scales by ``1 / num_micro_steps``. Under ``token_mean``:

        * Single-loss graphs scale backward by ``_loss_tokens`` and :meth:`train_step` divides
          accumulated gradients once by the window's total tokens before ``clip_grad_norm()``.
        * Multi-loss graphs require every loss key to report its matching ``_loss_tokens``; when
          ``num_micro_steps == 1`` each already-reduced loss keeps weight ``1`` (``0`` if its
          token count is zero), and when ``num_micro_steps > 1`` each loss ``k`` is weighted by
          ``_loss_tokens[k] / window_loss_tokens[k]`` so every loss is independently token-averaged
          across the accumulation window without distorting relative loss scales.
        """
        micro_batch = self.preforward(micro_batch)

        # Forward: spill activations to CPU (if enabled) + batch-invariant ops.
        with self.model_fwd_context, set_batch_invariant_mode(self.args.train.enable_batch_invariant_mode):
            result: Dict[str, Any] = self.model.forward(micro_batch)

        total_loss: torch.Tensor = result["loss"]
        loss_dict: Dict[str, torch.Tensor] = result.get("losses", {})
        loss_tokens: Dict[str, torch.Tensor] = result.get("loss_tokens", {})
        del result
        micro_batch.clear()

        if getattr(self.args.train, "loss_reduction", "sample_mean") == "token_mean":
            if not loss_dict or loss_dict.keys() != loss_tokens.keys():
                raise ValueError(
                    "train.loss_reduction='token_mean' requires every loss to report a matching '_loss_tokens'; "
                    f"got losses={list(loss_dict.keys())} and loss_tokens={list(loss_tokens.keys())}. "
                    "Use train.loss_reduction='sample_mean' instead."
                )
            token_counts = {k: self._to_int64_scalar(v, total_loss.device) for k, v in loss_tokens.items()}
            if len(loss_dict) == 1:
                token_count = next(iter(token_counts.values()))
                scaled_loss = total_loss * token_count.float()
            elif num_micro_steps == 1:
                scaled_loss = None
                for k, v in loss_dict.items():
                    term = v * (token_counts[k] > 0).float()
                    scaled_loss = term if scaled_loss is None else (scaled_loss + term)
            else:
                if window_loss_tokens is None or not (loss_dict.keys() <= window_loss_tokens.keys()):
                    missing = sorted(set(loss_dict.keys()) - set((window_loss_tokens or {}).keys()))
                    raise ValueError(
                        "train.loss_reduction='token_mean' with multiple losses and gradient accumulation "
                        f"(num_micro_steps={num_micro_steps}) requires count_loss_tokens() for every loss key; "
                        f"missing={missing}. Implement count_loss_tokens() on the loss modules or use "
                        "train.loss_reduction='sample_mean'."
                    )
                scaled_loss = None
                for k, v in loss_dict.items():
                    win_k = self._to_int64_scalar(window_loss_tokens[k], v.device)
                    tok_k = token_counts[k].to(device=v.device)
                    zero = torch.zeros((), device=v.device, dtype=torch.float32)
                    weight_k = torch.where(win_k > 0, tok_k.float() / win_k.clamp_min(1).float(), zero)
                    term = v * weight_k
                    scaled_loss = term if scaled_loss is None else (scaled_loss + term)
        else:
            token_counts = None
            scaled_loss = total_loss / num_micro_steps

        # Backward: separate offload hook stack (may differ from forward when GC is on).
        with self.model_bwd_context, set_batch_invariant_mode(self.args.train.enable_batch_invariant_mode):
            scaled_loss.backward()

        del micro_batch
        return total_loss, loss_dict, token_counts

    def _scale_optimizer_grads(self, scale: float) -> None:
        """Rescale all live gradients across every module optimizer in-place."""
        seen: set[int] = set()
        with torch.no_grad():
            for group in self.model.optimizer.param_groups:
                for param in group["params"]:
                    pid = id(param)
                    if pid in seen or param.grad is None:
                        continue
                    seen.add(pid)
                    param.grad.mul_(scale)

    def model_reshard(self, micro_step: int, num_micro_steps: int) -> None:
        """Keep every module's params gathered across the grad-accumulation window."""
        cascade_module_reshard(self.model.module_runtimes, micro_step, num_micro_steps)

    def _reset_async_activation_offload_if_enabled(self) -> None:
        """Reset each module's async-offload managers; its own accelerator decides whether it has any."""
        for module_runtime in self.model.module_runtimes.values():
            if module_runtime.args.accelerator.offload_config.enable_async_activation:
                reset_async_activation_offload(module_runtime.model)

    def sync_before_train_step(self):
        if self.args.train.sync_each_train_step:
            synchronize()

    def train_step(self, data_iterator: Any) -> None:
        # Fetch first: an iterator that runs dry before ``train_steps`` raises
        # here, and must not leave a step counted that never ran.
        micro_batches: List[Dict[str, Any]] = next(data_iterator)
        self.state.global_step += 1

        self._reset_async_activation_offload_if_enabled()
        self.on_step_begin(micro_batches=micro_batches)
        self.sync_before_train_step()

        loss_reduction = getattr(self.args.train, "loss_reduction", "sample_mean")
        total_loss_tensor: torch.Tensor | None = None
        total_loss_dict_tensors: Dict[str, torch.Tensor] = {}
        total_tokens_dict: Dict[str, torch.Tensor] = {}
        num_micro_steps = len(micro_batches)
        window_loss_tokens: Dict[str, torch.Tensor] | None = None
        if loss_reduction == "token_mean" and num_micro_steps > 1:
            count_fn = getattr(self.model, "count_loss_tokens", None)
            if callable(count_fn):
                window_loss_tokens = count_fn(micro_batches)
        single_loss_scaled_in_post = False

        for micro_step, micro_batch in enumerate(micro_batches):
            self.model_reshard(micro_step, num_micro_steps)
            loss, loss_dict, token_counts = self.forward_backward_step(
                micro_batch,
                num_micro_steps=num_micro_steps,
                window_loss_tokens=window_loss_tokens,
            )
            if loss_reduction == "token_mean":
                assert token_counts is not None
                if len(loss_dict) == 1:
                    single_loss_scaled_in_post = True
                for k, v in loss_dict.items():
                    tok_k = token_counts[k]
                    v_w = v.detach().float() * tok_k.float()
                    total_loss_dict_tensors[k] = (
                        v_w if k not in total_loss_dict_tensors else (total_loss_dict_tensors[k] + v_w)
                    )
                    total_tokens_dict[k] = tok_k if k not in total_tokens_dict else (total_tokens_dict[k] + tok_k)
            else:
                loss_w = loss.detach().float()
                total_loss_tensor = loss_w if total_loss_tensor is None else (total_loss_tensor + loss_w)
                for k, v in loss_dict.items():
                    v_w = v.detach().float()
                    total_loss_dict_tensors[k] = (
                        v_w if k not in total_loss_dict_tensors else (total_loss_dict_tensors[k] + v_w)
                    )

        if loss_reduction == "token_mean":
            if single_loss_scaled_in_post and len(total_tokens_dict) > 1:
                raise ValueError(
                    "train.loss_reduction='token_mean' observed inconsistent loss keys across micro-batches "
                    f"in the same step: {list(total_tokens_dict.keys())}."
                )
            if single_loss_scaled_in_post and len(total_tokens_dict) == 1:
                denom = float(next(iter(total_tokens_dict.values())).item())
                if denom > 0.0:
                    self._scale_optimizer_grads(1.0 / denom)
            total_loss_dict = {}
            for k, v in total_loss_dict_tensors.items():
                denom_k = float(total_tokens_dict[k].item()) if k in total_tokens_dict else 0.0
                total_loss_dict[k] = float(v.item()) / denom_k if denom_k > 0.0 else 0.0
            total_loss = sum(total_loss_dict.values())
        else:
            denom = float(num_micro_steps)
            total_loss = (
                float(total_loss_tensor.item()) / denom if (total_loss_tensor is not None and denom > 0.0) else 0.0
            )
            total_loss_dict = {
                k: float(v.item()) / denom if denom > 0.0 else 0.0 for k, v in total_loss_dict_tensors.items()
            }

        grad_norm = self.model.clip_grad_norm()
        if math.isfinite(float(grad_norm)):
            self.model.optimizer.step()
        else:
            logger.warning_rank0(
                f"Step {self.state.global_step}: non-finite grad_norm={grad_norm}, "
                "skipping optimizer.step() to protect model parameters."
            )
        self.model.lr_scheduler.step()
        self.model.optimizer.zero_grad()

        self.on_step_end(loss=total_loss, loss_dict=total_loss_dict, grad_norm=grad_norm)

    def offline_cache_step(self, data_iterator: Any) -> None:
        """One encode-only step: run the training DAG without autograd and cache each conversation."""
        if self.offline_cache_writer is None:
            raise RuntimeError("offline_cache_step requires an initialized SeedOmniOfflineCacheWriter.")

        micro_batches: List[Dict[str, Any]] = next(data_iterator)
        self.state.global_step += 1

        self.on_step_begin(micro_batches=micro_batches)
        self.sync_before_train_step()

        num_micro_steps = len(micro_batches)
        for micro_step, micro_batch in enumerate(micro_batches):
            self.model_reshard(micro_step, num_micro_steps)
            micro_batch = self.preforward(micro_batch)
            with (
                torch.no_grad(),
                self.model_fwd_context,
                set_batch_invariant_mode(self.args.train.enable_batch_invariant_mode),
            ):
                result: Dict[str, Any] = self.model.forward(micro_batch)
            self.offline_cache_writer.save_conversation_list(result["conversation_list"])

        self.on_step_end(loss=0.0, loss_dict={}, grad_norm=0.0)

    def train(self):
        args: OmniArguments = self.args
        self.on_train_begin()
        # After on_train_begin, which hands it the shard position saved with the cursor.
        if self.offline_cache_writer is not None:
            self.offline_cache_writer.open()
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

            self.on_epoch_begin()

            self.data_iterator = VeOmniIter(
                self.train_dataloader,
                use_background_prefetcher=args.data.dataloader.use_background_prefetcher,
            )

            step_fn = self.offline_cache_step if args.train.training_task == "offline_embedding" else self.train_step
            for _ in range(self.start_step, args.train_steps):
                try:
                    step_fn(self.data_iterator)
                except StopIteration:
                    logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.dataloader.drop_last}")
                    break

            self.on_epoch_end()

            self.start_step = 0
            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")

            if args.data.dataloader.use_background_prefetcher:
                self.data_iterator.stop()

        self.on_train_end()
        if self.offline_cache_writer is not None:
            self.offline_cache_writer.finalize()

        if self.data_iterator is not None and args.data.dataloader.use_background_prefetcher:
            self.data_iterator.stop()

        synchronize()

        self.destroy_distributed()


__all__ = [
    "OmniTrainer",
    "batch_to_device",
    "cascade_module_reshard",
]
