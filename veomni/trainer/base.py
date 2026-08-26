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

"""
Base Trainer class for distributed training.

This module provides the BaseTrainer class which serves as the foundation
for all trainer implementations. Subclasses can override specific methods
to customize training behavior.

Features:
    - Callback system for extensible training hooks
    - Distributed training support
    - Gradient accumulation
    - Checkpointing
"""

import json
import os
import queue
import threading
from abc import ABC
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any, Callable, Dict, List

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.checkpoint import set_checkpoint_debug_enabled
from torch.utils.data import Dataset
from transformers.modeling_outputs import ModelOutput

from ..arguments import VeOmniArguments, save_args
from ..data import (
    DistributedDataloader,
    build_dataloader,
    build_dataset,
)
from ..data.data_collator import DataCollator, MainCollator
from ..data.data_transform import build_data_transform
from ..distributed.offloading import build_activation_offloading_context
from ..distributed.parallel_state import (
    clear_parallel_state,
    init_parallel_state_from_accelerator,
    use_parallel_state,
)
from ..distributed.torch_compile import mark_compile_step_begin
from ..models.model_runtime import VeOmniModelRuntime
from ..ops.batch_invariant_ops import set_batch_invariant_mode
from ..utils import helper, logging
from ..utils.device import (
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
    is_nccl_backend,
    synchronize,
)
from ..utils.loss_utils import count_loss_token, mean_global_loss, reduce_global_loss_token
from .callbacks import (
    ChannelLossCallback,
    EnvironMeterCallback,
    EvaluateCallback,
    GlobalStateCallback,
    ModelDcpCallback,
    ModelHfCallback,
    MoERouterMonitorCallback,
    ProfileTraceCallback,
    RootAssetsCallback,
    TqdmCallback,
    TrainerState,
    WandbTraceCallback,
)


logger = logging.get_logger(__name__)


class BackgroundPrefetcher:
    """
    Prefetches batches from a dataloader in a background thread to overlap data loading
    with GPU computation. Synchronizes dataloader state for correct checkpointing.
    """

    def __init__(self, dataloader, maxsize=1):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.queue = queue.Queue(maxsize=maxsize)
        self.stop_event = threading.Event()
        self.original_state_dict = getattr(dataloader, "state_dict", None)
        self.current_state = None
        self.thread = threading.Thread(target=self._worker)
        self.thread.daemon = True
        self.thread.start()

    def _worker(self):
        try:
            while not self.stop_event.is_set():
                try:
                    item = next(self.iterator)
                except StopIteration:
                    self.queue.put((StopIteration, None))
                    break

                # Ensure we capture the state so that subsequent dataloader advances
                # don't mutate the captured state in-place. The underlying dataloader's
                # state_dict() should handle deepcopying if necessary.
                state = self.original_state_dict() if self.original_state_dict else None
                self.queue.put((item, state))
        except Exception as e:
            self.queue.put((e, None))

    def __iter__(self):
        return self

    def __next__(self):
        res = self.queue.get()
        if isinstance(res, tuple) and len(res) == 2:
            item, state = res
            if item is StopIteration:
                raise StopIteration
            if isinstance(item, Exception):
                raise item
            self.current_state = state
            return item
        else:
            if res is StopIteration:
                raise StopIteration
            if isinstance(res, Exception):
                raise res
            return res

    def state_dict(self):
        if self.current_state is not None:
            return self.current_state
        if self.original_state_dict:
            return self.original_state_dict()
        return {}

    def stop(self, timeout: float = 5.0):
        self.stop_event.set()
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
        except queue.Empty:
            pass
        if self.thread.is_alive():
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                logger.warning("BackgroundPrefetcher worker thread did not terminate within timeout.")


class VeOmniIter:
    """
    A unified iterator wrapper that handles both standard iteration and background prefetching.
    """

    def __init__(self, dataloader, use_background_prefetcher: bool = False, maxsize: int = 1):
        self.dataloader = dataloader
        self.use_background_prefetcher = use_background_prefetcher
        if use_background_prefetcher:
            self.iterator = BackgroundPrefetcher(dataloader, maxsize=maxsize)
        else:
            self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iterator)

    def stop(self, timeout: float = 5.0):
        if self.use_background_prefetcher and hasattr(self.iterator, "stop"):
            self.iterator.stop(timeout=timeout)

    def state_dict(self):
        if self.use_background_prefetcher and hasattr(self.iterator, "state_dict"):
            return self.iterator.state_dict()
        if hasattr(self.dataloader, "state_dict"):
            return self.dataloader.state_dict()
        return {}


class BaseTrainer(Stateful, ABC):
    """
    Base trainer class for distributed model training.

    Holds the model-bound half of a training job as :attr:`model`, a
    :class:`~veomni.models.model_runtime.VeOmniModelRuntime` (build, freeze /
    LoRA, parallelize, optimizer, lr-scheduler, gradient clipping, device mesh,
    checkpoint I/O), and owns the job-bound half itself: distributed init, the
    data pipeline, the train loop, callbacks and checkpoint scheduling. The
    handle forwards the wrapped model's own API, so ``trainer.model.config``
    and ``trainer.model(**batch)`` read as before.

    This class provides the core training infrastructure including:
    - Distributed initialization and parallelism setup
    - Model, optimizer, and scheduler initialization
    - Training step execution with gradient accumulation
    - Checkpointing and fault tolerance
    - Metrics logging

    Subclasses can override the following methods to customize behavior:
    - `post_init()`: Add custom initialization after setup
    - `forward_backward_step()`: Customize forward/backward logic
    - `train_step()`: Customize training step execution
    - `train()`: Train the model

    Callback Hooks:
        The trainer calls callback methods at various stages:
        - evaluate_callback: evaluation callback
        - trace_callback: tracing callback (meter, wandb, tqdm, profile)
        - checkpoint_callback: checkpointing callback
    """

    # Core configs
    args: VeOmniArguments
    device: torch.device

    # Data
    data_transform: Callable
    train_dataset: Dataset
    collate_fn: DataCollator
    train_dataloader: DistributedDataloader

    # Model
    model: VeOmniModelRuntime = None

    # Training context
    model_fwd_context: Any
    model_bwd_context: Any

    # Runtime metrics, controlled by trace_callback
    environ_meter: helper.EnvironMeter  # see in trace_callback.EnvironMeterCallback
    step_env_metrics: Dict[str, Any]  # mfu, flops, tokens, etc
    step_train_metrics: Dict[str, Any]  # loss, grad_norm, lr, etc

    # Checkpointer — built on the runtime, which owns every model/torch boundary.

    # Callback system
    state: TrainerState

    # Training states
    train_steps: int = 0  # total training steps
    start_epoch: int = 0  # start epoch
    start_step: int = 0  # start step

    def __init__(self, args: VeOmniArguments):
        """
        Initialize the trainer.

        Args:
            args: Global Arguments
                Should have attributes: model, data, train
                model: ModelArguments
                data: DataArguments
                train: TrainingArguments
        """

        self.args: VeOmniArguments = args
        self.device = self.setup_distributed(args)
        # Builds the model: meta-init, freeze, parallelize and optimizer all run
        # under the mesh the runtime registered, so nothing below needs a scope.
        self.model = self.build_model_runtime()
        # build dataset and dataloader
        self._build_data_transform()
        self._build_dataset()
        self._build_collate_fn()
        self._build_dataloader()
        # The dataset fixes train_steps, which the schedule needs.
        self.build_lr_scheduler()
        self._build_training_context()
        self._init_callbacks()

    # ── Trainer distributed setup ────────────────────────────────

    @staticmethod
    def setup_distributed(args: VeOmniArguments) -> torch.device:
        """Init process group, device, seed, and register the job's ParallelState.

        Everything here is job-level and runs before any model exists, so it
        takes the config rather than a built trainer.
        """
        # log args
        logger.info_rank0(json.dumps(asdict(args), indent=2))

        # init distributed environment
        device_str = f"{get_device_type()}:{args.train.local_rank}"
        get_torch_device().set_device(device_str)
        device = torch.device(device_str)

        # Initialize distributed process group
        if not dist.is_initialized():
            dist.init_process_group(backend=get_dist_comm_backend())

        logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")

        # Register ParallelState before seed/determinism env vars. Mesh creation
        # must not run under NCCL_DETERMINISTIC=1 on some GPU platforms (L20).
        init_parallel_state_from_accelerator(args.model.accelerator, name="base")

        # Set random seed
        helper.set_seed(args.train.seed, args.train.enable_full_determinism)

        # Enable high precision for bf16
        helper.enable_high_precision_for_bf16()

        # Enable third party logging
        if args.train.local_rank == 0:
            helper.enable_third_party_logging()

        # Save arguments
        if args.train.global_rank == 0:
            save_args(args, args.train.checkpoint.output_dir)

        # Gradient checkpointing debug
        set_checkpoint_debug_enabled(args.model.accelerator.gradient_checkpointing.debug)
        return device

    def destroy_distributed(self):
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

    # ── Trainer build functions ────────────────────────────────

    def build_model_runtime(self) -> VeOmniModelRuntime:
        """Build this job's single model, under the ``"base"`` ParallelState.

        Returns it already built — meta-init, freeze, parallelize and optimizer
        included. A model whose build differs (see :class:`VLMModelRuntime`)
        overrides that step on a runtime subclass returned from here, rather
        than resequencing the build from the trainer.
        """
        return VeOmniModelRuntime(
            self.args.model,
            model_name="base",
            train=self.args.train,
            chat_template_name=self.args.data.chat_template,
        )

    def build_lr_scheduler(self):
        """Size the run, then let the model schedule over it.

        The step count is job-bound — it needs the dataset length and the epoch
        count — so the model takes the number rather than deriving it.
        """
        self.model.build_lr_scheduler(self.args.train_steps * self.args.train.num_train_epochs)

    def _build_data_transform(self):
        self.data_transform = build_data_transform(
            self.args.data.data_type,
            tokenizer=self.model.tokenizer,
            chat_template=self.model.chat_template,
            max_seq_len=self.args.data.max_seq_len,
            text_keys=self.args.data.text_keys,
        )

    def _build_dataset(self):
        args: VeOmniArguments = self.args
        # Build dataset
        self.train_dataset = build_dataset(
            dataset_name=args.data.dataset_name,
            transform=self.data_transform,
            seed=args.train.seed,
            **asdict(args.data),
        )
        dataset_length = None if not hasattr(self.train_dataset, "__len__") else len(self.train_dataset)
        if args.data.datasets_type == "mapping":
            dataset_length = dataset_length / args.model.accelerator.dp_size
        args.compute_train_steps(dataset_length)
        self.train_steps = args.train_steps

    def _build_collate_fn(self):
        seq_classification = self.args.data.data_type == "classification"
        pad_to_length = self.args.train.pad_to_length
        self.collate_fn = MainCollator(
            pad_to_length=pad_to_length,
            seq_classification=seq_classification,
        )

    def _build_dataloader(self):
        args: VeOmniArguments = self.args
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

    def _build_training_context(self):
        """Build training context for distributed training."""
        self.model_fwd_context, self.model_bwd_context = build_activation_offloading_context(
            self.args.model.accelerator.offload_config.enable_activation,
            self.args.model.accelerator.gradient_checkpointing.enable,
            self.args.model.accelerator.offload_config.activation_gpu_limit,
        )

    def _init_callbacks(self):
        """Initialize callbacks."""
        self.environ_meter_callback = EnvironMeterCallback(self)
        self.tqdm_callback = TqdmCallback(self)
        self.wandb_callback = WandbTraceCallback(self)
        self.profile_callback = ProfileTraceCallback(self)
        self.root_assets_callback = RootAssetsCallback(self)
        self.dcp_callback = ModelDcpCallback(self)
        self.global_state_callback = GlobalStateCallback(self)
        # One callback, both export formats: a model that trains only adapters
        # writes the adapter, and the model is what knows.
        self.hf_ckpt_callback = ModelHfCallback(self)
        self.evaluate_callback = EvaluateCallback(self)
        self.moe_monitor_callback = MoERouterMonitorCallback(self)
        self.channel_loss_callback = ChannelLossCallback(self)
        # Ordered dispatch list. Callbacks own their ParallelState explicitly:
        # each captured it at construction (``Callback.parallel_state``), and
        # ChannelLossComputer receives that same cached state. Shared objects
        # (EnvironMeter, DCP checkpointer) are handed the state directly, so
        # no ambient ``use_parallel_state`` scope is needed around hook dispatch.
        #
        # ``channel_loss_callback`` is ordered after the meter (which resets
        # ``step_*_metrics`` in ``on_step_end``) and before ``wandb`` (which
        # logs them), so its per-source metrics survive into the logged payload.
        self._callbacks = [
            self.environ_meter_callback,
            self.tqdm_callback,
            self.channel_loss_callback,
            self.wandb_callback,
            self.profile_callback,
            self.root_assets_callback,
            # Weights first, then the cursor: at resume the DCP load frees its
            # materialization buffers before the dataloader prefetches, and at
            # save a crash between the two leaves weights whose trainer state is
            # merely absent, which resumes with a warning.
            self.dcp_callback,
            self.global_state_callback,
            self.hf_ckpt_callback,
            self.evaluate_callback,
            self.moe_monitor_callback,
        ]
        self.state = TrainerState()

    # ── Trainer checkpoint functions ────────────────────────────────

    def load(self) -> None:
        """Resume this job's model.

        The fan-out is the point: a trainer holding a second model (a DPO
        reference, a distillation teacher) extends this to load both, and the
        callback that triggers it keeps knowing only *when*. Extra state is
        restored inside the checkpoint manager; job cursor lives on
        ``GlobalStateCallback``.
        """
        self.model.load()

    def save_dcp(self, state: TrainerState) -> None:
        """Write this job's resumable checkpoint for ``state.global_step``."""
        self.model.save_dcp(state)

    def save_hf_or_lora(self, state: TrainerState, stage: str = "step_end") -> None:
        """Export this job's weights in whichever format the model was trained in."""
        self.model.save_hf_or_lora(state, stage=stage)

    # ── Trainer callback hooks ────────────────────────────────

    def on_train_begin(self):
        for callback in self._callbacks:
            callback.on_train_begin(self.state)

    def on_train_end(self):
        for callback in self._callbacks:
            callback.on_train_end(self.state)

    def on_epoch_begin(self):
        for callback in self._callbacks:
            callback.on_epoch_begin(self.state)

    def on_epoch_end(self):
        for callback in self._callbacks:
            callback.on_epoch_end(self.state)

    def on_step_begin(self, micro_batches=None, **kwargs):
        for callback in self._callbacks:
            callback.on_step_begin(self.state, micro_batches=micro_batches, **kwargs)

    def on_step_end(self, loss=None, loss_dict=None, grad_norm=None):
        for callback in self._callbacks:
            callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)

    # ── Trainer train step functions ────────────────────────────────

    def preforward(self, micro_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Preprocess micro batches before forward pass.

        Tensors are moved to ``self.device`` non-blockingly. Nested dicts
        (e.g. ``multimodal_metadata`` emitted by ``PackingCollator``) are
        recursed so inner tensor values land on the device too; Python ints
        / lists / etc. pass through unchanged.
        """

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

    def postforward(
        self, outputs: ModelOutput, micro_batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Postprocess model outputs after forward pass."""
        loss_dict: Dict[str, torch.Tensor] = mean_global_loss(
            outputs.loss,
            self.micro_batch_token_len,
            self.micro_batches_token_len,
            getattr(self, "global_micro_batches_token_len", None),
        )
        loss = torch.stack(list(loss_dict.values())).sum()
        return loss, loss_dict

    def forward_backward_step(
        self, micro_batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        channel_loss_callback = getattr(self, "channel_loss_callback", None)
        micro_step_context = (
            channel_loss_callback.micro_step_context(self.state, micro_batch)
            if channel_loss_callback is not None
            else nullcontext()
        )
        with micro_step_context:
            micro_batch = self.preforward(micro_batch)
            if channel_loss_callback is not None:
                channel_loss_callback.strip_model_inputs(micro_batch)

            channel_forward_context = (
                channel_loss_callback.model_forward_context() if channel_loss_callback is not None else nullcontext()
            )
            with (
                use_parallel_state("base"),
                self.model_fwd_context,
                set_batch_invariant_mode(self.args.train.enable_batch_invariant_mode),
                channel_forward_context,
            ):
                outputs: ModelOutput = self.model(**micro_batch, use_cache=False)

            with use_parallel_state("base"):
                loss, loss_dict = self.postforward(outputs, micro_batch)

            with (
                use_parallel_state("base"),
                self.model_bwd_context,
                set_batch_invariant_mode(self.args.train.enable_batch_invariant_mode),
            ):
                loss.backward()

            del micro_batch
            return loss, loss_dict

    def model_reshard(self, micro_step: int, num_micro_steps: int):
        """Reshard model after backward pass."""
        args: VeOmniArguments = self.args
        if (
            args.model.accelerator.fsdp_config.fsdp_mode == "fsdp2"
            and not args.model.accelerator.fsdp_config.reshard_after_backward
            and num_micro_steps > 1
        ):
            if micro_step == 0:
                self.model.set_reshard_after_backward(False)
            elif micro_step == num_micro_steps - 1:
                self.model.set_reshard_after_backward(True)

    def _configure_hsdp_allreduce(self, micro_step: int, num_micro_steps: int):
        args: VeOmniArguments = self.args
        if (
            args.model.accelerator.fsdp_config.fsdp_mode == "fsdp2"
            and args.model.accelerator.dp_replicate_size > 1
            and num_micro_steps > 1
        ):
            if micro_step == 0:
                self.model.set_requires_all_reduce(False)
            elif micro_step == num_micro_steps - 1:
                self.model.set_requires_all_reduce(True)

    def sync_before_train_step(self):
        if self.args.train.sync_each_train_step:
            synchronize()

    def train_step(
        self,
        data_iterator: Any,
    ) -> Dict[str, float]:
        self.state.global_step += 1

        micro_batches: List[Dict[str, Any]] = next(data_iterator)

        self.on_step_begin(micro_batches=micro_batches)

        # Forward and backward for each micro batch
        self.sync_before_train_step()

        total_loss = 0.0
        total_loss_dict = defaultdict(int)

        # token num for fixed_ce_loss in postforward
        self.micro_batches_token_len = count_loss_token(micro_batches)
        self.global_micro_batches_token_len = reduce_global_loss_token(self.micro_batches_token_len)
        num_micro_steps = len(micro_batches)
        # forward and backward pass with gradient_accumulationsteps
        for micro_step, micro_batch in enumerate(micro_batches):
            mark_compile_step_begin(getattr(self.model, "_veomni_compile_uses_cuda_graphs", False))
            self.model_reshard(micro_step, num_micro_steps)
            self._configure_hsdp_allreduce(micro_step, num_micro_steps)
            loss: torch.Tensor
            loss_dict: Dict[str, torch.Tensor]
            # token num for fixed_ce_loss in postforward
            self.micro_batch_token_len = count_loss_token(micro_batch)
            loss, loss_dict = self.forward_backward_step(micro_batch)

            total_loss += loss.item()
            for k, v in loss_dict.items():
                total_loss_dict[k] += v.item()

        # Gradient clipping (reads FSDP/EP groups from this model's ParallelState)
        grad_norm = self.model.clip_grad_norm()

        # Optimizer and scheduler step
        self.model.optimizer.step()
        self.model.lr_scheduler.step()
        self.model.optimizer.zero_grad()

        self.on_step_end(loss=total_loss, loss_dict=total_loss_dict, grad_norm=grad_norm)

    # ── Trainer train loop ────────────────────────────────

    def train(self):
        args: VeOmniArguments = self.args
        self.on_train_begin()
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

            # Create a batch generator
            self.data_iterator = VeOmniIter(
                self.train_dataloader, use_background_prefetcher=args.data.dataloader.use_background_prefetcher
            )

            for _ in range(self.start_step, args.train_steps):
                try:
                    self.train_step(self.data_iterator)
                except StopIteration:
                    logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.dataloader.drop_last}")
                    break

            self.on_epoch_end()

            self.start_step = 0

            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")

            if args.data.dataloader.use_background_prefetcher:
                self.data_iterator.stop()

        self.on_train_end()

        if "data_iterator" in locals() and args.data.dataloader.use_background_prefetcher:
            self.data_iterator.stop()

        synchronize()

        self.destroy_distributed()
