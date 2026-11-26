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
from datetime import timedelta
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
from ...utils.constants import IGNORE_INDEX
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
            dist.init_process_group(backend=get_dist_comm_backend(), timeout=timedelta(hours=6))

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
        self._validate_loss_reduction(model)
        return model

    def _validate_loss_reduction(self, model: OmniModelRuntime) -> None:
        """Validate ``train.loss_reduction='token_mean'`` against the active training graph.

        ``token_mean`` scales the summed DAG loss by each micro-batch's share of
        shifted supervised text tokens, which is only valid when the training
        graph's sole loss-producing objective is a single ``TextEncoder.decode``
        node. Multi-objective graphs (e.g. Janus with image VQ loss, BAGEL with
        flow-matching loss, Qwen3-MoE with router auxiliary loss) and non-text
        graphs (e.g. Cosmos3) must use ``sample_mean``.

        Also detects decoders that shift each text item independently
        (``BagelTextEncoder``) rather than once per sample (``TextEncoder``) so
        :meth:`_count_supervised_tokens` uses the decoder's exact target count.
        """
        self._text_label_shift_mode = "per_sample"
        self._token_mean_loss_node: str | None = None
        if getattr(self.args.train, "loss_reduction", "sample_mean") != "token_mean":
            return
        training_graph = getattr(getattr(model, "model", None), "training_graph", None)
        if training_graph is None:
            return

        from ...models.seed_omni.accelerated.utils.dispatch import unwrap_module_chain
        from ...models.seed_omni.modules.bagel.text_encoder.modeling import BagelTextEncoder
        from ...models.seed_omni.modules.text_encoder.modeling import TextEncoder

        sinks = list(training_graph.sinks)
        if len(sinks) != 1:
            raise ValueError(
                f"train.loss_reduction='token_mean' only supports training graphs with a single "
                f"TextEncoder.decode objective; got sinks {sinks}. Use train.loss_reduction='sample_mean' "
                "for multi-objective graphs."
            )
        sink = sinks[0]
        module_name = training_graph.module_of(sink)
        method = training_graph.method_of(sink)
        raw_module = unwrap_module_chain(model.get_module(module_name))
        if method != "decode" or not isinstance(raw_module, TextEncoder):
            raise ValueError(
                f"train.loss_reduction='token_mean' requires a TextEncoder.decode sink; "
                f"got {sink!r} ({type(raw_module).__name__}). Use train.loss_reduction='sample_mean' instead."
            )
        for node in training_graph.execution_order:
            if node == sink:
                continue
            node_method = training_graph.method_of(node)
            node_module = unwrap_module_chain(model.get_module(training_graph.module_of(node)))
            router_aux = getattr(node_module, "_router_aux_loss_enabled", None)
            if node_method in ("decode", "decode_velocity") or (callable(router_aux) and router_aux()):
                raise ValueError(
                    f"train.loss_reduction='token_mean' only supports a single {sink!r} objective, "
                    f"but node {node!r} ({type(node_module).__name__}) also produces a loss. "
                    "Use train.loss_reduction='sample_mean' instead."
                )
        self._token_mean_loss_node = sink
        if isinstance(raw_module, BagelTextEncoder):
            self._text_label_shift_mode = "per_item"

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

    def _loss_reduce_group(self):
        """The group the model's loss was reduced over, or ``None`` when there is none.

        Mirrors ``loss_reduction_group=fsdp_group`` on the text encoder's loss, so
        a ``token_mean`` weight counts the same tokens the loss already averaged.
        """
        from ...distributed.parallel_state import get_parallel_state

        return getattr(get_parallel_state(), "fsdp_group", None)

    def _count_supervised_tokens(self, micro_batch: Mapping[str, Any]) -> int:
        """Supervised-token count of one micro-batch, for ``loss_reduction='token_mean'``.

        Counts the *shifted* labels, matching what the model's `ForCausalLMLoss`
        divides by. The causal shift (`labels[1:]` + one trailing ``IGNORE_INDEX``)
        moves the first label out of the tensor, so counting the unshifted row
        over-counts by one whenever that first position is supervised -- and then
        the micro-batch would be weighted by a denominator the loss never used.
        The shift keeps every other position, so its supervised count differs from
        the unshifted one only in that boundary case.

        Handles packed batches (``packed_labels``), flat/list ``labels``, and
        SeedOmni ``conversation_list`` batches (per-sample text ``meta["labels"]``
        with sample-boundary masking matching ``TextEncoder._prepare_decode_inputs``).
        """
        labels = micro_batch.get("packed_labels")
        if labels is None:
            labels = micro_batch.get("labels")
        if labels is not None:
            if hasattr(labels, "numel"):
                # A packed row arrives as [T] or [1, T]; the shift is over the last dim.
                return int((labels[..., 1:] != IGNORE_INDEX).sum().item())
            total = 0
            for row in labels:
                if hasattr(row, "numel"):
                    total += int((row[..., 1:] != IGNORE_INDEX).sum().item())
                else:
                    total += sum(1 for token in row[1:] if token != IGNORE_INDEX)
            return total

        conversation_list = micro_batch.get("conversation_list")
        if conversation_list is None:
            return 0
        per_item = getattr(self, "_text_label_shift_mode", "per_sample") == "per_item"
        total = 0
        if per_item:
            # Match BagelTextEncoder._prepare_decode_inputs: each non-dummy text
            # item is shifted independently.
            for sample in conversation_list:
                for part in sample:
                    if getattr(part, "is_dummy", False) or getattr(part, "type", "text") != "text":
                        continue
                    meta = getattr(part, "meta", None)
                    part_labels = meta.get("labels") if isinstance(meta, Mapping) else None
                    if part_labels is not None:
                        total += int((part_labels.reshape(-1)[1:] != IGNORE_INDEX).sum().item())
            return total

        for sample in conversation_list:
            sample_labels: List[torch.Tensor] = []
            has_non_text_prefix = False
            for part in sample:
                if getattr(part, "is_dummy", False):
                    continue
                if getattr(part, "type", "text") == "text":
                    meta = getattr(part, "meta", None)
                    part_labels = meta.get("labels") if isinstance(meta, Mapping) else None
                    if part_labels is not None and part_labels.numel() > 0:
                        sample_labels.append(part_labels.reshape(-1))
                else:
                    # Match TextEncoder._prepare_decode_inputs: each non-text
                    # segment contributes a masked placeholder row before the
                    # causal shift so a following supervised text token is kept.
                    if not sample_labels:
                        has_non_text_prefix = True
            if not sample_labels:
                continue
            first_start = 0 if has_non_text_prefix else 1
            total += int((sample_labels[0][first_start:] != IGNORE_INDEX).sum().item())
            for part_labels in sample_labels[1:]:
                total += int((part_labels != IGNORE_INDEX).sum().item())
        return total

    def forward_backward_step(
        self, micro_batch: Dict[str, Any], *, num_micro_steps: int = 1, backward_scale: float | None = None
    ):
        """One gradient-accumulation micro-batch over the training DAG.

        ``OmniModelRuntime.forward`` returns ``{"loss", "losses", "conversation_list"}``
        where ``loss`` is the summed per-node ``_loss``; a single backward then
        propagates across every FSDP2 unit. The result is dropped before the
        backward so the item copies a root wrap's input cast made are not held through it.

        Each node's loss is a mean over its own micro-batch, so ``backward_scale``
        sets the gradient weight of this micro-batch: ``1 / num_micro_steps`` for
        a mean over micro-batches (``loss_reduction='sample_mean'``), or
        ``local_tokens / global_tokens`` for a mean over tokens
        (``'token_mean'``), where ``global_tokens`` already spans every dp rank.
        The returned loss stays unscaled for logging, except that the caller reads
        ``loss * token_count`` back out to rebuild the window mean.
        """
        micro_batch = self.preforward(micro_batch)

        self._log_memory("before forward")

        # Forward: spill activations to CPU (if enabled) + batch-invariant ops.
        with self.model_fwd_context, set_batch_invariant_mode(self.args.train.enable_batch_invariant_mode):
            result: Dict[str, Any] = self.model.forward(micro_batch)

        self._log_memory("after forward")

        total_loss: torch.Tensor = result["loss"]
        loss_dict: Dict[str, torch.Tensor] = result.get("losses", {})
        del result
        micro_batch.clear()

        token_mean_node = getattr(self, "_token_mean_loss_node", None)
        if token_mean_node is not None and loss_dict:
            extra_loss_nodes = [name for name in loss_dict if name != token_mean_node]
            if extra_loss_nodes:
                raise ValueError(
                    f"train.loss_reduction='token_mean' only supports a single {token_mean_node!r} objective, "
                    f"but forward produced additional loss(es) from {extra_loss_nodes}. "
                    "Use train.loss_reduction='sample_mean' instead."
                )

        scale = 1.0 / num_micro_steps if backward_scale is None else backward_scale

        # Backward: separate offload hook stack (may differ from forward when GC is on).
        with self.model_bwd_context, set_batch_invariant_mode(self.args.train.enable_batch_invariant_mode):
            (total_loss * scale).backward()

        self._log_memory("after backward")

        del micro_batch
        return total_loss, loss_dict

    def _log_memory(self, where: str) -> None:
        """Report device memory at a gradient high-water point.

        ``train.log_memory_steps=0`` disables it (the default), because reading
        ``memory_allocated`` can force a device sync. The peak counters are
        process-wide, so a single line after the backward already carries the
        maximum reached during the forward and backward that preceded it; logging
        before the forward too makes an out-of-memory trace say which side of the
        step it died on.

        Skipped on a backend without the allocator counters (CPU), where there is
        no device budget to report and the call would raise.
        """
        steps = getattr(self.args.train, "log_memory_steps", 0)
        if not steps or self.state.global_step % steps != 0 or getattr(self.args.train, "global_rank", 0) != 0:
            return
        device = get_torch_device()
        memory_allocated = getattr(device, "memory_allocated", None)
        if memory_allocated is None:
            return
        logger.info_rank0(
            f"[memory] step {self.state.global_step} {where}: "
            f"allocated {memory_allocated() / 1024**3:.2f} GiB, "
            f"peak {device.max_memory_allocated() / 1024**3:.2f} GiB, "
            f"reserved {device.memory_reserved() / 1024**3:.2f} GiB, "
            f"peak reserved {device.max_memory_reserved() / 1024**3:.2f} GiB"
        )

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

        total_loss_tensor: torch.Tensor | None = None
        total_loss_dict_tensors: Dict[str, torch.Tensor] = {}
        num_micro_steps = len(micro_batches)

        # ``token_mean`` weights each micro-batch by its share of the window's
        # supervised tokens, so counting them up front is what makes a long trace
        # count for more than a short answer. The count is a cheap scan of the
        # packed labels (no forward), then one all-reduce so every rank divides by
        # the same window total. ``sample_mean`` keeps the old equal weighting and
        # skips the scan entirely.
        token_scale = None
        local_token_counts: List[int] = []
        if self.args.train.loss_reduction == "token_mean":
            local_token_counts = [self._count_supervised_tokens(mb) for mb in micro_batches]
            # The model's loss for a micro-batch is already the token mean over the
            # whole reduce group, so the weight needs that same group's count for
            # *each* micro-batch -- not just the window total of this rank's slice.
            # int64 is exact for token counts and supported across NCCL, HCCL, CNCL, and Gloo.
            counts = torch.tensor(local_token_counts, dtype=torch.int64, device=self.device)
            group = self._loss_reduce_group()
            if group is not None and dist.is_available() and dist.is_initialized():
                dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
            local_token_counts = counts.tolist()
            token_scale = sum(local_token_counts)

        for micro_step, micro_batch in enumerate(micro_batches):
            self.model_reshard(micro_step, num_micro_steps)
            backward_scale = None
            if token_scale is not None:
                # A window with no supervised token at all cannot be normalized;
                # fall back to the equal weighting so the step still runs.
                backward_scale = (
                    local_token_counts[micro_step] / token_scale if token_scale > 0 else 1.0 / num_micro_steps
                )
            loss, loss_dict = self.forward_backward_step(
                micro_batch, num_micro_steps=num_micro_steps, backward_scale=backward_scale
            )
            loss_weight = (
                local_token_counts[micro_step] / token_scale
                if (token_scale is not None and token_scale > 0)
                else 1.0 / num_micro_steps
            )
            # ``loss`` is this micro-batch's token mean, so weight by ``loss_weight``
            # before adding: the logged value is then the window's token mean.
            loss_scaled = loss.detach() * loss_weight
            total_loss_tensor = loss_scaled if total_loss_tensor is None else (total_loss_tensor + loss_scaled)
            for k, v in loss_dict.items():
                v_scaled = v.detach() * loss_weight
                total_loss_dict_tensors[k] = (
                    v_scaled if k not in total_loss_dict_tensors else (total_loss_dict_tensors[k] + v_scaled)
                )

        grad_norm = self.model.clip_grad_norm()
        total_loss = float(total_loss_tensor.item()) if total_loss_tensor is not None else 0.0
        total_loss_dict: Dict[str, float] = {k: float(v.item()) for k, v in total_loss_dict_tensors.items()}
        if math.isfinite(float(grad_norm)):
            self.model.optimizer.step()
        else:
            logger.warning_rank0(
                f"Step {self.state.global_step}: non-finite grad_norm={grad_norm}, "
                "skipping optimizer.step() to protect model parameters."
            )
        # The per-micro-batch probe already covers the gradient high-water mark
        # (see `_log_memory`); this one runs after optimizer.step() to catch a
        # step whose peak lands in the optimizer update.
        self._log_memory("after optimizer")
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
