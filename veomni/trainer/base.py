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
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin
from transformers.modeling_outputs import ModelOutput

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
from ..utils.arguments import DataArguments, ModelArguments, TrainingArguments, save_args
from ..utils.device import (
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
    synchronize,
)
from ..utils.loss_utils import count_loss_token, mean_global_loss
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
from .preforward import Preforward


logger = logging.get_logger(__name__)


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


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
    args: Arguments
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

    # Callback system
    callbacks: CallbackHandler
    state: TrainerState

    def __init__(self, args: Arguments):
        """
        Initialize the trainer.

        Args:
            args: Global Arguments
                Should have attributes: model, data, train
                model: ModelArguments
                data: DataArguments
                train: TrainingArguments
        """

        self.args: Arguments = args
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
        # preforward & postforward
        self._build_preforward_postforward()

        # Call post-initialization hook for subclasses
        self.post_init()

    def post_init(self) -> None:
        pass

    def freeze_module(self):
        fsdp_kwargs = {}
        return fsdp_kwargs

    def build_param_groups(self):
        return None

    def build_data_transform(self):
        raise NotImplementedError("build_data_transform must be implemented in subclasses")

    def build_model_assets(self):
        self.tokenizer = build_tokenizer(self.args.model.tokenizer_path)
        return [self.tokenizer]

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

        # freeze module and init fsdp_kwargs for building parallelized model
        self.fsdp_kwargs = self.freeze_module()

        helper.print_device_mem_info("VRAM usage after building model")

    def _build_data(self):
        logger.info_rank0("Build data")
        args: Arguments = self.args

        data_transform = self.build_data_transform()
        # Build dataset
        self.train_dataset = build_dataset(
            dataset_name=args.data.dataset_name,
            transform=data_transform,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
            **asdict(args.data),
        )
        dataset_length = None if not hasattr(self.train_dataset, "__len__") else len(self.train_dataset)
        if args.data.datasets_type == "mapping":
            dataset_length = dataset_length / args.train.data_parallel_size
        args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, dataset_length)
        self.train_steps = args.train.train_steps

        # Build dataloader
        self.train_dataloader = build_dataloader(
            dataloader_type=args.data.dataloader_type,
            dataset=self.train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
            collate_fn=[],
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train.train_steps,
            rmpad=args.train.rmpad,
            rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
            dyn_bsz_margin=args.train.dyn_bsz_margin,
            dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor,
        )

        # TODO: val dataset & dataloader

    def _build_parallelized_model(self):
        args: Arguments = self.args

        # Parallelize model
        self.model = build_parallelize_model(
            self.model,
            init_device=args.train.init_device,
            weights_path=args.model.model_path,
            enable_full_shard=args.train.enable_full_shard,
            enable_mixed_precision=args.train.enable_mixed_precision,
            enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
            enable_fsdp_offload=args.train.enable_fsdp_offload,
            fsdp_kwargs=self.fsdp_kwargs,
            basic_modules=self.model._no_split_modules + args.model.basic_modules,
            enable_reentrant=args.train.enable_reentrant,
            enable_forward_prefetch=args.train.enable_forward_prefetch,
        )
        self.model.train()

    def _build_optimizer_and_scheduler(self):
        """Build optimizer and learning rate scheduler."""
        args: Arguments = self.args

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
            train_steps=args.train.train_steps * args.train.num_train_epochs,
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

    def _build_preforward_postforward(self):
        """Build preforward and postforward hooks."""
        self.pre_forward = Preforward(
            rmpad_with_pos_ids=self.args.train.rmpad_with_pos_ids,
            attn_implementation=self.args.model.attn_implementation,
        )
        self.post_forward = lambda x: None  # postforward only used for rl_trainer

    def add_callback(self, callback: Callback):
        self.callbacks.add(callback)

    def fit(self):
        args: Arguments = self.args
        self.callbacks.call("on_train_begin", self.state)
        logger.info(
            f"Rank{args.train.local_rank} Start training. "
            f"Train_steps: {args.train.train_steps}. "
            f"Epochs: {args.train.num_train_epochs}."
        )

        for epoch in range(self.start_epoch, args.train.num_train_epochs):
            if hasattr(self.train_dataloader, "set_epoch"):
                self.train_dataloader.set_epoch(epoch)
            self.state.epoch = epoch

            self.callbacks.call("on_epoch_begin", self.state)

            # Create a batch generator
            data_iterator = iter(self.train_dataloader)

            for _ in range(self.start_step, args.train.train_steps):
                try:
                    self.train_step(data_iterator)
                except StopIteration:
                    logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                    break

            self.callbacks.call("on_epoch_end", self.state)

            self.start_step = 0
            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")

        synchronize()

        # Clean up optimizer and lr scheduler
        del self.optimizer, self.lr_scheduler
        helper.empty_cache()
        self.callbacks.call("on_train_end", self.state)
        dist.barrier()
        dist.destroy_process_group()

    def preforward(self, micro_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess micro batches before forward pass."""
        micro_batch = self.pre_forward(micro_batch)

        micro_batch = {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in micro_batch.items()
        }
        if getattr(self, "LOG_SAMPLE", True):
            helper.print_example(example=micro_batch, rank=self.args.train.local_rank)
            self.LOG_SAMPLE = False
        return micro_batch

    def postforward(self, outputs: ModelOutput, micro_batch: Dict[str, torch.Tensor]) -> None:
        """Postprocess model outputs after forward pass."""
        loss_dict: Dict[str, torch.Tensor] = mean_global_loss(
            outputs.loss, self.micro_batch_token_len, self.micro_batches_token_len
        )
        loss = torch.stack(list(loss_dict.values())).sum()
        return loss, loss_dict

    def forward_backward_step(self, micro_batch: dict[str, torch.Tensor]) -> torch.Tensor:
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

        # token num for fixed_ce_loss
        self.micro_batches_token_len = count_loss_token(micro_batches)

        # forward and backward pass with gradient_accumulationsteps
        for micro_batch in micro_batches:
            loss: torch.Tensor
            loss_dict: Dict[str, torch.Tensor]
            # token num for fixed_ce_loss
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
