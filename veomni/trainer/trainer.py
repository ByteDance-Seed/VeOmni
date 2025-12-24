"""
Trainer implementation based on BaseTrainer.

This module provides a concrete implementation of BaseTrainer that follows
the training logic from train_torch.py.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.distributed as dist
from tqdm import trange

from ..checkpoint import build_checkpointer, ckpt_to_state_dict
from ..data import (
    build_chat_template,
    build_dataloader,
    build_dataset,
)
from ..data.data_transform import process_pretrain_example, process_sft_example
from ..distributed.clip_grad_norm import veomni_clip_grad_norm
from ..distributed.offloading import build_activation_offloading_context
from ..distributed.parallel_state import get_parallel_state, init_parallel_state
from ..distributed.torch_parallelize import build_parallelize_model
from ..models import build_foundation_model, build_tokenizer, save_model_assets, save_model_weights
from ..optim import build_lr_scheduler, build_optimizer
from ..utils import helper
from ..utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from ..utils.device import (
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
    synchronize,
)
from ..utils.dist_utils import all_reduce
from .base import BaseTrainer


logger = helper.create_logger(__name__)


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


class Trainer(BaseTrainer):
    """
    Concrete trainer implementation following train_torch.py logic.

    This trainer handles:
    - Distributed training setup
    - Model, optimizer, and scheduler initialization
    - Training loop with gradient accumulation
    - Checkpoint saving and loading
    - Metrics logging
    - Save and load checkpoint
    """

    def __init__(self, args: Arguments):
        """
        Initialize the trainer with arguments.

        Args:
            args: Arguments object containing model, data, and training configurations.
                Should have attributes: model, data, train
        """
        # Store args as job_config for compatibility with base class
        self.args: Arguments = args

        # Initialize distributed training environment
        self._init_distributed()

        # Initialize fault tolerance manager, TODO: Implement FTManager
        self.ft_manager = None

        logger.info_rank0(json.dumps(asdict(self.args), indent=2))

        # Set random seed
        helper.set_seed(args.train.seed, args.train.enable_full_determinism)

        # Enable third party logging
        if args.train.local_rank == 0:
            helper.enable_third_party_logging()

        # Save arguments
        if self.args.train.global_rank == 0:
            save_args(self.args, self.args.train.output_dir)

        # Build checkpointer instance
        self._build_checkpointer()

        # Prepare model and parallelize model
        self._build_model()

        # Prepare data, tokenizer, transform function, dataset, dataloader and calculate train steps
        self._build_data()

        # Parallelize model
        self._build_parallelized_model()

        # Build optimizer and lr scheduler
        self._build_optimizer_and_scheduler()

        # Initialize training states
        self.global_step = 0
        self.train_steps = 0
        self.ntokens_seen = 0
        self.start_epoch = 0
        self.start_step = 0

        # Build training context
        self._build_training_context()

        # Initialize metrics and environment meter
        self._init_metrics()

        # Load checkpoint if specified
        if args.train.load_checkpoint_path:
            self._load_checkpoint(args.train.load_checkpoint_path)

        # Initialize callbacks
        self._init_callbacks()

        # Call base class post_init hook
        self.post_init()

    def _init_callbacks(self):
        """Initialize callbacks."""
        pass

    def _init_distributed(self):
        """Initialize distributed training environment."""
        # Set torch device
        device_type = get_device_type()
        device_str = f"{device_type}:{self.args.train.local_rank}"
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
        )

    def _build_data(self):
        """Prepare tokenizer and dataloader."""

        logger.info_rank0("Prepare data")
        args: Arguments = self.args

        self.tokenizer = build_tokenizer(args.model.tokenizer_path)

        # Build transform function
        if args.data.data_type == "plaintext":
            transform = partial(
                process_pretrain_example,
                tokenizer=self.tokenizer,
                max_seq_len=args.data.max_seq_len,
                text_keys=args.data.text_keys,
            )
        elif args.data.data_type == "conversation":
            chat_template = build_chat_template(args.data.chat_template, self.tokenizer)
            self.chat_template = chat_template
            transform = partial(
                process_sft_example,
                chat_template=chat_template,
                max_seq_len=args.data.max_seq_len,
                text_keys=args.data.text_keys,
            )
        else:
            raise NotImplementedError(f"Unsupported data type: {args.data.data_type}.")

        # Build dataset
        train_dataset = build_dataset(
            dataset_name=args.data.dataset_name,
            transform=transform,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
            **asdict(args.data),
        )
        dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
        if args.data.datasets_type == "mapping":
            dataset_length = dataset_length / args.train.data_parallel_size
        args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, dataset_length)
        self.train_steps = args.train.train_steps

        # Build dataloader
        self.dataloader = build_dataloader(
            dataloader_type=args.data.dataloader_type,
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
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

        self.gradient_accumulation_steps = args.train.dataloader_batch_size

    def _build_model(self):
        """Build model."""
        logger.info_rank0("Prepare model")
        args = self.args

        # Build model
        self.model = build_foundation_model(
            config_path=args.model.config_path,
            weights_path=args.model.model_path,
            torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
            attn_implementation=args.model.attn_implementation,
            moe_implementation=args.model.moe_implementation,
            init_device=args.train.init_device,
            force_use_huggingface=args.model.force_use_huggingface,
        )
        self.model_config = self.model.config
        helper.print_device_mem_info("VRAM usage after building model")

    def _build_parallelized_model(self):
        args: Arguments = self.args
        # Get optimizer pre hook if available
        get_optimizer_pre_hook = getattr(self.model, "get_optimizer_pre_hook", None)

        # Parallelize model
        self.model = build_parallelize_model(
            self.model,
            init_device=args.train.init_device,
            weights_path=args.model.model_path,
            enable_full_shard=args.train.enable_full_shard,
            enable_mixed_precision=args.train.enable_mixed_precision,
            enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
            enable_fsdp_offload=args.train.enable_fsdp_offload,
            basic_modules=self.model._no_split_modules + args.model.basic_modules,
            enable_reentrant=args.train.enable_reentrant,
            enable_forward_prefetch=args.train.enable_forward_prefetch,
        )

        self.model_parts = [self.model]  # For compatibility with base class
        self.get_optimizer_pre_hook = get_optimizer_pre_hook

        self.model.train()

    def _build_optimizer_and_scheduler(self):
        """Build optimizer and learning rate scheduler."""
        args: Arguments = self.args

        # Build optimizer
        self.optimizer = build_optimizer(
            self.model,
            lr=args.train.lr,
            weight_decay=args.train.weight_decay,
            fused=True,
            optimizer_type=args.train.optimizer,
        )

        # Register optimizer pre hook if available
        if self.get_optimizer_pre_hook is not None:
            optimizer_pre_hook = self.get_optimizer_pre_hook(
                self.model, self.model_config, args.train.data_parallel_mode
            )
            self.optimizer.register_step_pre_hook(optimizer_pre_hook)

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

        # For compatibility with base class
        self.optimizers = self.optimizer
        self.lr_schedulers = self.lr_scheduler

    def _build_checkpointer(self):
        """Build checkpointer instance."""
        # The checkpointer class is already built in __init__
        # This method is for future extension
        self.checkpointer = build_checkpointer(
            dist_backend=self.args.train.data_parallel_mode, ckpt_manager=self.args.train.ckpt_manager
        )

    def _build_training_context(self):
        """Build training context for distributed training."""
        # Build activation offloading context
        self.model_fwd_context, self.model_bwd_context = build_activation_offloading_context(
            self.args.train.enable_activation_offload,
            self.args.train.enable_gradient_checkpointing,
            self.args.train.activation_gpu_limit,
        )

    def _init_metrics(self):
        """Initialize metrics and environment meter."""

        args: Arguments = self.args

        # Initialize environment meter
        self.metrics_processor = helper.EnvironMeter(
            config=self.model_config,
            global_batch_size=args.train.global_batch_size,
            rmpad=args.train.rmpad,
            rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            empty_cache_steps=args.train.empty_cache_steps,
            enable_multisource=args.data.enable_multisource,
            dataloader=self.dataloader,
            data_path=args.data.train_path,
        )

        # Initialize wandb if enabled
        if args.train.global_rank == 0:
            if args.train.use_wandb:
                import wandb

                wandb.init(
                    project=args.train.wandb_project,
                    name=args.train.wandb_name,
                    config={**vars(args.model), **vars(args.data), **vars(args.train)},
                )

            # Save model assets before training
            model_assets = [
                self.model_config,
                self.tokenizer if args.data.data_type == "plaintext" else getattr(self, "chat_template", None),
            ]
            save_model_assets(args.train.model_assets_dir, model_assets)

        # Initialize profiler if enabled
        if args.train.profile_this_rank:
            self.profiler = helper.create_profiler(
                start_step=args.train.profile_start_step,
                end_step=args.train.profile_end_step,
                trace_dir=args.train.profile_trace_dir,
                record_shapes=args.train.profile_record_shapes,
                profile_memory=args.train.profile_profile_memory,
                with_stack=args.train.profile_with_stack,
                global_rank=args.train.global_rank,
            )
            self.profiler.start()
        else:
            self.profiler = None

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint from path."""
        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "extra_state": {},
        }
        self.checkpointer.load(checkpoint_path, state)

        self.global_step = state["extra_state"]["global_step"]
        self.start_epoch = self.global_step // self.args.train.train_steps
        self.start_step = self.global_step % self.args.train.train_steps

        self.lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        self.dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        self.metrics_processor.load_state_dict(state["extra_state"]["metrics_processor"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])

        if self.start_step == 0:  # Resume at the end of epoch
            iter(self.dataloader)  # Clear resume state and prefetch data

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {checkpoint_path} successfully!")

    def batch_generator(
        self, data_loader: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Returns an iterator that processes batches from the data iterator."""

        data_iterator = iter(data_loader)
        while True:
            # data_load_start = time.perf_counter()
            try:
                micro_batches = next(data_iterator)
            except StopIteration as ex:
                # If data runs out during gradient accumulation, that
                # entire step will not be executed.
                raise RuntimeError("Data loader exhausted") from ex

            # Update environment meter
            for micro_batch in micro_batches:
                self.metrics_processor.add(micro_batch)
                # Remove multisource metadata if present
                if self.args.data.enable_multisource:
                    micro_batch.pop("ds_idx", None)
                    micro_batch.pop("source_name", None)

            # TODO: Add data loading time to metrics processor
            # self.metrics_processor.data_loading_times.append(time.perf_counter() - data_load_start)
            yield micro_batches

    def pre_forward(
        self, input_dict: dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        """
        Pre-process data before forward pass.

        This method is called right before the model forward pass to perform
        final data preparation (e.g., device transfer, metadata removal).

        Args:
            input_dict: Dictionary containing input tensors from the dataloader.
            labels: Optional labels (not used in this implementation).

        Returns:
            Processed micro_batch dictionary ready for model forward pass.
        """
        # Move micro_batch to cuda device
        micro_batch = {
            k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in input_dict.items()
        }

        return micro_batch

    def forward_backward_step(
        self, input_dict: dict[str, torch.Tensor], num_micro_batches: int = 1, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perform forward and backward pass for a single micro batch.

        This follows the logic from train_torch.py where each micro_batch
        is processed separately with gradient accumulation.

        Args:
            input_dict: Dictionary containing input tensors for the micro batch.
            num_micro_batches: Number of micro batches in the current step (for loss scaling).
            labels: Optional labels (not used in this implementation).

        Returns:
            Loss tensor for this micro batch.
        """

        # Pre-process data before forward pass
        micro_batch = self.pre_forward(input_dict)

        # Forward pass
        # In train_torch.py: loss.mean() / len(micro_batches)
        with self.model_fwd_context:
            loss: torch.Tensor = self.model(**micro_batch, use_cache=False).loss.mean() / num_micro_batches

        # Backward pass
        with self.model_bwd_context:
            loss.backward()

        del micro_batch
        return loss

    def train_step(
        self,
        data_iterator: Any,  # Iterable that yields List[Dict] (micro_batches)
    ) -> Dict[str, float]:
        """
        Execute a single training step.

        This processes micro_batches (which are already accumulated),
        performs gradient clipping, optimizer step, and logging.

        Args:
            data_iterator: Iterator that yields List[Dict[str, Any]] where each
                Dict is a micro_batch. This matches train_torch.py's dataloader behavior.
        """
        args = self.args
        # Update step counters
        self.global_step += 1

        # Get micro_batches from iterator
        # data_iterator yields List[Dict] (micro_batches)
        micro_batches: List[Dict[str, Any]] = next(data_iterator)

        # Print example on first step
        if self.global_step == 1:
            helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

        # Forward and backward for each micro batch
        synchronize()
        start_time = time.time()
        total_loss = 0.0
        num_micro_batches = len(micro_batches)

        # forward and backward pass with gradient_accumulationsteps
        for micro_batch in micro_batches:
            loss = self.forward_backward_step(micro_batch, num_micro_batches=num_micro_batches)
            total_loss += loss.item()

        # Gradient clipping
        grad_norm = veomni_clip_grad_norm(self.model, args.train.max_grad_norm)

        # Optimizer and scheduler step
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor().item()

        # Collect mean loss and grad_norm across data parallel group
        parallel_state = get_parallel_state()
        total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=parallel_state.fsdp_group)

        delta_time = time.time() - start_time
        lr = max(self.lr_scheduler.get_last_lr())

        # Update metrics
        train_metrics = self.metrics_processor.step(delta_time, global_step=self.global_step)

        # Log metrics
        if args.train.global_rank == 0:
            if args.train.use_wandb:
                import wandb

                train_metrics.update(
                    {
                        "training/loss": total_loss,
                        "training/grad_norm": grad_norm,
                        "training/lr": lr,
                    }
                )
                wandb.log(train_metrics, step=self.global_step)

        return {
            "loss": float(total_loss),
            "grad_norm": float(grad_norm),
            "lr": float(lr),
        }

    def fit(self):
        """
        Main training loop.

        This follows the training loop structure from train_torch.py.
        """
        args: Arguments = self.args

        logger.info(
            f"rank{args.train.local_rank} Start training, "
            f"train_steps: {args.train.train_steps}, "
            f"epochs: {args.train.num_train_epochs}"
        )

        for epoch in range(self.start_epoch, args.train.num_train_epochs):
            if hasattr(self.dataloader, "set_epoch"):
                self.dataloader.set_epoch(epoch)

            data_loader_tqdm = trange(
                args.train.train_steps,
                desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
                total=args.train.train_steps,
                initial=self.start_step,
                disable=args.train.local_rank != 0,
            )

            # Create a batch generator
            data_iterator = self.batch_generator(self.dataloader)

            for _ in range(self.start_step, args.train.train_steps):
                try:
                    step_metrics = self.train_step(data_iterator)
                except RuntimeError as e:
                    if "Data loader exhausted" in str(e):
                        logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                        break
                    raise

                # Update progress bar
                data_loader_tqdm.update()
                if step_metrics:
                    data_loader_tqdm.set_postfix_str(
                        "loss: {loss:.2f}, grad_norm: {grad_norm:.2f}, lr: {lr:.2e}".format(**step_metrics)
                    )

                # Profiling
                if self.profiler and self.global_step <= args.train.profile_end_step:
                    self.profiler.step()
                    if self.global_step == args.train.profile_end_step:
                        self.profiler.stop()

                # Save checkpoint
                if args.train.save_steps and self.global_step % args.train.save_steps == 0:
                    self._save_checkpoint()

                # Evaluate
                if args.train.eval_steps and self.global_step % args.train.eval_steps == 0:
                    self._evaluate()

            data_loader_tqdm.close()
            self.start_step = 0

            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")

            # Save checkpoint at end of epoch
            if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
                self._save_checkpoint()

            # Evaluate
            if args.train.eval_epochs and (epoch + 1) % args.train.eval_epochs == 0:
                self._evaluate()

        synchronize()

        # Save final model in HuggingFace format
        if args.train.global_rank == 0 and args.train.save_hf_weights:
            self._save_hf_model()

        # Clean up optimizer and lr scheduler
        del self.optimizer, self.lr_scheduler
        helper.empty_cache()

        dist.barrier()
        dist.destroy_process_group()

    def _evaluate(self):
        """Evaluate the model."""
        for callback in self.callbacks:
            callback.evaluate(self)

    def _log(self):
        """Log the metrics."""
        for callback in self.callbacks:
            callback.on_log(self)

    def _save_checkpoint(self):
        """Save checkpoint."""
        args: Arguments = self.args
        save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{self.global_step}")

        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "extra_state": {
                "global_step": self.global_step,
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "train_dataloader": self.dataloader.state_dict(),
                "metrics_processor": self.metrics_processor.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
            },
        }

        self.checkpointer.save(args.train.save_checkpoint_path, state, global_steps=self.global_step)

        # Empty cache and barrier
        helper.empty_cache()
        dist.barrier()

        logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

    def _save_hf_model(self):
        """Save model in HuggingFace format."""

        args: Arguments = self.args
        save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{self.global_step}")
        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")

        model_state_dict = ckpt_to_state_dict(
            save_checkpoint_path=save_checkpoint_path,
            ckpt_manager=args.train.ckpt_manager,
        )

        model_assets = [
            self.model_config,
            self.tokenizer if args.data.data_type == "plaintext" else getattr(self, "chat_template", None),
        ]
        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")


if __name__ == "__main__":
    args = parse_args(Arguments)
    trainer = Trainer(args)
    trainer.fit()
