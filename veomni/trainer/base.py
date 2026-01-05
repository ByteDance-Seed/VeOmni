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

from abc import ABC
from collections.abc import Generator, Iterable
from typing import Any, Callable, List, Optional

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from transformers import PretrainedConfig, PreTrainedTokenizerBase

from ..utils.logging import get_logger


logger = get_logger(__name__)


class BaseTrainer(Stateful, ABC):
    """
    Base trainer class for distributed model training.

    This class provides the core training infrastructure including:
    - Distributed initialization and parallelism setup
    - Model, optimizer, and scheduler initialization
    - Training step execution with gradient accumulation
    - Checkpointing and fault tolerance
    - Metrics logging
    - **Callback system for customization**

    Subclasses can override the following methods to customize behavior:
    - `pre_forward()`: Pre-process data before forward pass
    - `forward_backward_step()`: Customize forward/backward logic
    - `train_step()`: Customize training step execution
    - `batch_generator()`: Customize batch processing
    - `post_init()`: Add custom initialization after setup
    - `init_distributed()`: Initialize distributed training environment
    - `fit()`: Train the model

    Callback Hooks:
        The trainer calls callback methods at various stages:

        - evaluate_callback: evaluation callback
        - log_callback: logging callback
    """

    # Core configs
    args: Any  # Global Arguments

    # Swappable training components
    tokenizer: PreTrainedTokenizerBase  # Tokenizer
    train_dataset: Dataset  # Dataset
    val_dataset: Dataset  # Dataset
    train_dataloader: DataLoader  # Dataloader
    val_dataloader: DataLoader  # Dataloader
    model: torch.nn.Module  # model_parts[0], if only one model
    model_config: PretrainedConfig  # ModelConfig
    model_parts: list[torch.nn.Module] = []  # [model], if multiple models
    loss_fn: Any  # LossFunction, e.g. torch.nn.CrossEntropyLoss
    optimizers: Optimizer  # Optimizers, e.g. torch.optim.Adam
    lr_schedulers: LRScheduler  # LRSchedulers, e.g. torch.optim.lr_scheduler.LambdaLR
    validator: Any  # Validator
    metrics_processor: Any  # MetricsProcessor

    # Non-swappable training components
    checkpointer: Any  # CheckpointManager
    ft_manager: Any  # FTManager

    # Runtime utilities
    device: torch.device
    train_context: Generator[None, None, None]
    gradient_accumulation_steps: int

    # training states
    global_step: int
    train_steps: int
    ntokens_seen: int

    # Callback system
    callbacks: List[Callable] = []

    def __init__(self, args: Any):
        """
        Initialize the trainer.

        Args:
            args: Global Arguments
                Should have attributes: model, data, train
                model: ModelArguments
                data: DataArguments
                train: TrainingArguments
        """

        self.args = args

        # Initialize distributed and build meshes
        self.init_distributed()

        # Initialize fault tolerance manager
        # self.ft_manager = FTManager(job_config.fault_tolerance)
        # dp_degree, dp_rank = self.ft_manager.get_dp_info(dp_degree, dp_rank)
        self.ft_manager = None  # Placeholder - replace with actual FTManager

        # Build tokenizer and dataloader
        # self.tokenizer = build_tokenizer(args.model.tokenizer_path)
        self.tokenizer = None  # Placeholder

        # Build train and val datasets
        # self.train_dataset = build_dataset(args.data.train_path)
        # self.val_dataset = build_dataset(args.data.val_path)
        self.train_dataset = None  # Placeholder
        self.val_dataset = None  # Placeholder

        # self.train_dataloader = build_dataloader(train_dataset, tokenizer, args.train.global_batch_size)
        # self.val_dataloader = build_dataloader(val_dataset, tokenizer, args.train.global_batch_size)
        self.train_dataloader = None
        self.val_dataloader = None

        # Build model (using meta init)
        # model = build_model(args.model.model_path)
        self.model = None  # Placeholder
        self.model_config = None  # Placeholder
        self.model_parts = [self.model]

        # Build loss function
        # self.loss_fn = build_loss_function(args.train.loss_function)
        self.loss_fn = None  # Placeholder

        # Build optimizer and LR scheduler
        # self.optimizer = build_optimizer(args.train.optimizer)
        # self.lr_scheduler = build_lr_scheduler(args.train.lr_scheduler)

        self.optimizer = None  # Placeholder
        self.lr_scheduler = None  # Placeholder
        self.optimizers = []  # Placeholder
        self.lr_schedulers = []  # Placeholder

        # Build checkpointer
        # self.checkpointer = build_checkpointer(args.train.ckpt_manager)
        self.checkpointer = None  # Placeholder

        # Build validator
        # self.validator = build_validator(args.train.validator)
        self.validator = None  # Placeholder

        # Build metrics processor
        # self.metrics_processor = helper.EnvironMeter()
        self.metrics_processor = None  # Placeholder

        # Initialize training states
        self.global_step = 0
        self.train_steps = 0
        self.ntokens_seen = 0
        self.pp_has_first_stage = False
        self.pp_has_last_stage = False

        # Build training context
        self.train_context = None  # Placeholder
        self.model_fwd_context = None  # Placeholder
        self.model_bwd_context = None  # Placeholder

        logger.info("Trainer is initialized")

        # Call post-initialization hook for subclasses
        self.post_init()

    def add_callback(self, callback: Callable) -> None:
        """
        Add a callback to the trainer.

        Args:
            callback: TrainerCallback instance to add.
        """
        self.callbacks.append(callback)

    def post_init(self) -> None:
        """
        Post-initialization hook called after all components are set up.

        Subclasses can override this method to perform additional initialization
        after the base trainer setup is complete.
        """
        pass

    def init_distributed(self) -> Any:
        """
        Initialize distributed training environment.

        Returns:
            None
        """
        pass

    def batch_generator(
        self, data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """
        Returns an iterator that processes batches from the data iterator.

        This method handles device transfer and token counting. Subclasses can
        override this method to customize batch processing (e.g., different
        device placement, custom data augmentation, etc.).

        Args:
            data_iterable: Iterable over raw data batches.

        Returns:
            Iterator over processed batches (input_dict, labels).

        Subclasses can override this method to customize batch processing.
        """
        return None  # Placeholder

    def pre_forward(
        self, input_dict: dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
        """
        Pre-process data before model forward pass.

        This method is called right before the model forward pass to perform
        final data preparation. It processes the raw data from the dataloader
        and prepares it for the model's forward pass. It can separate the main
        input tensor from auxiliary inputs and construct additional keyword
        arguments (e.g., attention masks).

        This method can be overridden in subclasses to customize data processing
        for different training strategies (e.g., converting tensors to DTensors,
        applying custom transformations, etc.).

        Args:
            input_dict: Dictionary containing tensors from the dataloader. Must
                contain an "input" key with the main input tensor. May contain
                additional keys for auxiliary inputs (e.g., position ids).
            labels: Target labels for the batch.

        Returns:
            A tuple of (inputs, labels, extra_inputs, extra_kwargs) where:
                - inputs: Main input tensor extracted from input_dict["input"].
                - labels: Target labels (unchanged from input parameter).
                - extra_inputs: Dict of auxiliary input tensors (all keys except
                    "input" from input_dict). These are passed to the model forward
                    but are NOT forwarded across pipeline parallel stages.
                - extra_kwargs: Dict of additional keyword arguments for model forward.
                    These ARE forwarded across pipeline parallel stages. Contains
                    attention_masks if flex attention is enabled.

        Note:
            The distinction between extra_inputs and extra_kwargs is important for
            pipeline parallelism: extra_kwargs are forwarded to all pipeline stages,
            while extra_inputs are only available to the first stage.

        Subclasses should override this method to customize data preprocessing.
        """
        pass

    def forward_backward_step(self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        """
        Perform forward and backward pass for a single batch.

        This method can be overridden in subclasses to customize the forward/backward
        logic (e.g., different loss computation, custom gradient handling, etc.).

        Args:
            input_dict: Dictionary containing input tensors from the dataloader.
            labels: Target labels for the batch.

        Returns:
            The computed loss tensor.

        Subclasses can override this method to implement custom training strategies.
        """

        pass

    def train_step(self, data_iterator: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]) -> None:
        """
        Execute a single training step.

        This method performs gradient accumulation, optimizer step, and metrics logging.
        Subclasses can override this method to customize the training step logic
        (e.g., different gradient clipping, custom logging, etc.).

        Args:
            data_iterator: Iterator over training batches.

        Subclasses can override this method to implement custom training step behavior.
        """
        pass
