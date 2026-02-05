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
from abc import ABC
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin
from transformers.modeling_outputs import ModelOutput

from ..arguments import VeOmniArguments, save_args
from ..checkpoint import CheckpointerBase
from ..data import (
    DistributedDataloader,
    build_dataloader,
    build_dataset,
)
from ..distributed.clip_grad_norm import veomni_clip_grad_norm
from ..distributed.offloading import build_activation_offloading_context
from ..distributed.parallel_state import init_parallel_state
from ..distributed.torch_parallelize import build_parallelize_model
from ..models import build_foundation_model, build_tokenizer
from ..optim import build_lr_scheduler, build_optimizer
from ..utils import helper, logging
from ..utils.device import (
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
    synchronize,
)
from ..utils.loss_utils import count_loss_token, mean_global_loss
from ..utils.model_utils import pretty_print_trainable_parameters
from .callbacks import (
    Callback,
    CallbackHandler,
    CheckpointerCallback,
    EnvironMeterCallback,
    EvaluateCallback,
    HuggingfaceCkptCallback,
    ProfileTraceCallback,
    TqdmCallback,
    TrainerState,
    WandbTraceCallback,
)


logger = logging.get_logger(__name__)


class BaseTrainer(Stateful, ABC):
    """
    Base trainer class for distributed model training.

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
    - `fit()`: Train the model

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
    train_dataset: Dataset
    val_dataset: Dataset
    train_dataloader: DistributedDataloader
    val_dataloader: DistributedDataloader

    # Model
    model: PreTrainedModel
    model_config: PretrainedConfig
    tokenizer: PreTrainedTokenizerBase
    processor: ProcessorMixin
    model_assets: List[Any]

    # Training components
    optimizers: Optimizer
    lr_schedulers: LRScheduler

    # Training context
    model_fwd_context: Any
    model_bwd_context: Any

    # Runtime metrics, controlled by trace_callback
    environ_meter: helper.EnvironMeter  # see in trace_callback.EnvironMeterCallback
    step_env_metrics: Dict[str, Any]  # mfu, flops, tokens, etc
    step_train_metrics: Dict[str, Any]  # loss, grad_norm, lr, etc

    # Checkpointer
    checkpointer: CheckpointerBase  # see in checkpoint_callback.CheckpointerCallback

    # Callback system
    callbacks: CallbackHandler
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
        logger.info_rank0(json.dumps(asdict(self.args), indent=2))

        self._setup()
        # build model and model assets (config, tokenizer, processor, chat_template)
        self._build_model()
        # build dataset and dataloader
        self._build_data()
        # Parallelize model
        self._build_parallelized_model()
        # Build optimizer and lr scheduler
        self._build_optimizer_and_scheduler()
        # Initialize training states
        self.train_steps = 0
        self.start_epoch = 0
        self.start_step = 0
        # Build training context
        self._build_training_context()
        # Initialize callbacks
        self._init_callbacks()

        # Call post-initialization hook for subclasses
        self.post_init()

    def post_init(self) -> None:
        pass

    def freeze_module(self):
        pass

    def build_param_groups(self):
        return None

    def build_data_transform(self):
        raise NotImplementedError("build_data_transform must be implemented in subclasses")

    def build_model_assets(self):
        self.tokenizer = build_tokenizer(self.args.model.tokenizer_path)
        return [self.tokenizer]

    def build_data_collate_info(self):
        return {}

    def build_collaten_fn_kwargs(self):
        if self.args.data.data_type == "classification":
            seq_classification = True
        else:
            seq_classification = False
        return {
            "data_collate_info": self.build_data_collate_info(),
            "pad_to_length": self.args.train.pad_to_length,
            "seq_classification": seq_classification,
        }

    def _setup(self):
        self._init_distributed()

        # Set random seed
        helper.set_seed(self.args.train.seed, self.args.train.enable_full_determinism)

        # Enable third party logging
        if self.args.train.local_rank == 0:
            helper.enable_third_party_logging()

        # Save arguments
        if self.args.train.global_rank == 0:
            save_args(self.args, self.args.train.output_dir)

    def _init_distributed(self):
        device_str = f"{get_device_type()}:{self.args.train.local_rank}"
        get_torch_device().set_device(device_str)
        self.device = torch.device(device_str)

        # Initialize distributed process group
        if not dist.is_initialized():
            dist.init_process_group(backend=get_dist_comm_backend(), device_id=self.device)

        logger.info(f"Process rank: {self.args.train.global_rank}, world size: {self.args.train.world_size}")

        # Initialize parallel state
        init_parallel_state(
            dp_size=self.args.train.data_parallel_size,
            dp_replicate_size=self.args.train.data_parallel_replicate_size,
            dp_shard_size=self.args.train.data_parallel_shard_size,
            tp_size=self.args.train.tensor_parallel_size,
            ep_size=self.args.train.expert_parallel_size,
            pp_size=self.args.train.pipeline_parallel_size,
            cp_size=self.args.train.context_parallel_size,
            ulysses_size=self.args.train.ulysses_parallel_size,
            dp_mode=self.args.train.data_parallel_mode,
            async_enabled=self.args.train.async_enabled,
        )

    def _build_model(self):
        logger.info_rank0("Build model")
        self.model = build_foundation_model(
            config_path=self.args.model.config_path,
            weights_path=self.args.model.model_path,
            torch_dtype="float32" if self.args.train.enable_mixed_precision else "bfloat16",
            attn_implementation=self.args.model.attn_implementation,
            moe_implementation=self.args.model.moe_implementation,
            init_device=self.args.train.init_device,
        )
        self.model_config = self.model.config

        # model assets
        self.model_assets = [self.model_config]
        self.model_assets.extend(self.build_model_assets())

        # freeze module
        self.freeze_module()
        pretty_print_trainable_parameters(self.model)
        helper.print_device_mem_info("VRAM usage after building model")

    def build_training_dataset(self):
        args: VeOmniArguments = self.args
        data_transform = self.build_data_transform()
        # Build dataset
        self.train_dataset = build_dataset(
            dataset_name=args.data.dataset_name,
            transform=data_transform,
            seed=args.train.seed,
            **asdict(args.data),
        )
        dataset_length = None if not hasattr(self.train_dataset, "__len__") else len(self.train_dataset)
        if args.data.datasets_type == "mapping":
            dataset_length = dataset_length / args.train.data_parallel_size
        args.compute_train_steps(dataset_length)
        self.train_steps = args.train_steps

    def build_training_dataloader(self):
        args: VeOmniArguments = self.args
        collate_fn_kwargs = self.build_collaten_fn_kwargs()
        self.train_dataloader = build_dataloader(
            dataloader_type=args.data.dataloader_type,
            dataset=self.train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train_steps,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
            dyn_bsz=args.train.dyn_bsz,
            dyn_bsz_buffer_size=args.data.dyn_bsz_buffer_size,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor,
            seed=args.train.seed,
            build_collate_fn=True,
            collate_fn_kwargs=collate_fn_kwargs,
        )

    def _build_data(self):
        logger.info_rank0("Build data")
        self.build_training_dataset()
        self.build_training_dataloader()
        # TODO: val dataset & dataloader

    def _build_parallelized_model(self):
        args: VeOmniArguments = self.args

        # Parallelize model
        self.model = build_parallelize_model(
            self.model,
            init_device=args.train.init_device,
            weights_path=args.model.model_path,
            enable_full_shard=args.train.enable_full_shard,
            enable_reshard_after_forward=args.train.enable_reshard_after_forward,
            enable_mixed_precision=args.train.enable_mixed_precision,
            enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
            enable_fsdp_offload=args.train.enable_fsdp_offload,
            basic_modules=self.model._no_split_modules + args.model.basic_modules,
            enable_reentrant=args.train.enable_reentrant,
            enable_forward_prefetch=args.train.enable_forward_prefetch,
        )
        self.model.train()

    def _build_optimizer_and_scheduler(self):
        """Build optimizer and learning rate scheduler."""
        args: VeOmniArguments = self.args

        param_groups = self.build_param_groups()

        # Build optimizer
        self.optimizer = build_optimizer(
            self.model,
            lr=args.train.lr,
            weight_decay=args.train.weight_decay,
            fused=True,
            optimizer_type=args.train.optimizer,
            param_groups=param_groups,
        )

        # Build lr scheduler
        self.lr_scheduler = build_lr_scheduler(
            self.optimizer,
            train_steps=args.train_steps * args.train.num_train_epochs,
            lr=args.train.lr,
            lr_min=args.train.lr_min,
            lr_decay_style=args.train.lr_decay_style,
            lr_decay_ratio=args.train.lr_decay_ratio,
            lr_warmup_ratio=args.train.lr_warmup_ratio,
            lr_start=args.train.lr_start,
        )

    def _build_training_context(self):
        """Build training context for distributed training."""
        self.model_fwd_context, self.model_bwd_context = build_activation_offloading_context(
            self.args.train.enable_activation_offload,
            self.args.train.enable_gradient_checkpointing,
            self.args.train.activation_gpu_limit,
        )

    def _model_reshard(self, micro_step: int, num_micro_steps: int):
        """Reshard model after backward pass."""
        args: VeOmniArguments = self.args
        if (
            args.train.data_parallel_mode == "fsdp2"
            and not args.train.enable_reshard_after_backward
            and num_micro_steps > 1
        ):
            if micro_step == 0:
                self.model.set_reshard_after_backward(False)
            elif micro_step == num_micro_steps - 1:
                self.model.set_reshard_after_backward(True)

    def _init_callbacks(self):
        """Initialize callbacks."""
        callbacks = [
            EnvironMeterCallback(self),  # First in trace
            TqdmCallback(self),
            WandbTraceCallback(self),
            ProfileTraceCallback(self),
            CheckpointerCallback(self),
            HuggingfaceCkptCallback(self),
            EvaluateCallback(self),
        ]
        self.callbacks = CallbackHandler(callbacks)
        self.state = TrainerState()

    def add_callback(self, callback: Callback):
        self.callbacks.add(callback)

    def fit(self):
        args: VeOmniArguments = self.args
        self.callbacks.call("on_train_begin", self.state)
        logger.info(
            f"Rank{args.train.local_rank} Start training. "
            f"Train_steps: {args.train_steps}. "
            f"Epochs: {args.train.num_train_epochs}."
        )

        for epoch in range(self.start_epoch, args.train.num_train_epochs):
            if hasattr(self.train_dataloader, "set_epoch"):
                self.train_dataloader.set_epoch(epoch)
            self.state.epoch = epoch

            self.callbacks.call("on_epoch_begin", self.state)

            # Create a batch generator
            data_iterator = iter(self.train_dataloader)

            for _ in range(self.start_step, args.train_steps):
                try:
                    self.train_step(data_iterator)
                except StopIteration:
                    logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                    break

            self.callbacks.call("on_epoch_end", self.state)

            self.start_step = 0
            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")

        self.callbacks.call("on_train_end", self.state)

        synchronize()
        # Clean up optimizer and lr scheduler
        del self.optimizer, self.lr_scheduler
        helper.empty_cache()

        dist.barrier()
        dist.destroy_process_group()

    def preforward(self, micro_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Preprocess micro batches before forward pass."""
        micro_batch = {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in micro_batch.items()
        }
        if getattr(self, "LOG_SAMPLE", True):
            helper.print_example(example=micro_batch, rank=self.args.train.local_rank)
            self.LOG_SAMPLE = False
        return micro_batch

    def postforward(
        self, outputs: ModelOutput, micro_batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Postprocess model outputs after forward pass."""
        loss_dict: Dict[str, torch.Tensor] = mean_global_loss(
            outputs.loss, self.micro_batch_token_len, self.micro_batches_token_len
        )
        loss = torch.stack(list(loss_dict.values())).sum()
        return loss, loss_dict

    def forward_backward_step(
        self, micro_batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        micro_batch = self.preforward(micro_batch)

        with self.model_fwd_context:
            outputs: ModelOutput = self.model(**micro_batch, use_cache=False)

        loss: torch.Tensor
        loss_dict: Dict[str, torch.Tensor]
        loss, loss_dict = self.postforward(outputs, micro_batch)

        # Backward pass
        with self.model_bwd_context:
            loss.backward()

        del micro_batch
        return loss, loss_dict

    def train_step(
        self,
        data_iterator: Any,
    ) -> Dict[str, float]:
        args = self.args
        self.state.global_step += 1

        micro_batches: List[Dict[str, Any]] = next(data_iterator)

        self.callbacks.call("on_step_begin", self.state, micro_batches=micro_batches)

        # Forward and backward for each micro batch
        synchronize()

        total_loss = 0.0
        total_loss_dict = defaultdict(int)

        # token num for fixed_ce_loss in postforward
        self.micro_batches_token_len = count_loss_token(micro_batches)
        num_micro_steps = len(micro_batches)
        # forward and backward pass with gradient_accumulationsteps
        for micro_step, micro_batch in enumerate(micro_batches):
            self._model_reshard(micro_step, num_micro_steps)
            loss: torch.Tensor
            loss_dict: Dict[str, torch.Tensor]
            # token num for fixed_ce_loss in postforward
            self.micro_batch_token_len = count_loss_token(micro_batch)
            loss, loss_dict = self.forward_backward_step(micro_batch)

            total_loss += loss.item()
            for k, v in loss_dict.items():
                total_loss_dict[k] += v.item()

        # Gradient clipping
        grad_norm = veomni_clip_grad_norm(self.model, args.train.max_grad_norm)

        # Optimizer and scheduler step
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        self.callbacks.call("on_step_end", self.state, loss=total_loss, loss_dict=total_loss_dict, grad_norm=grad_norm)
