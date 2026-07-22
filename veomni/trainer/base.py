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
from dataclasses import asdict, fields
from typing import Any, Callable, Dict, List

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.checkpoint import set_checkpoint_debug_enabled
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
from ..data.chat_template import ChatTemplate
from ..data.data_collator import DataCollator, MainCollator
from ..data.data_transform import build_data_transform
from ..distributed.chunk_mbs import build_chunk_mbs_ranges, chunk_mbs_context
from ..distributed.clip_grad_norm import veomni_clip_grad_norm
from ..distributed.offloading import build_activation_offloading_context
from ..distributed.parallel_state import clear_parallel_state, init_parallel_state, use_parallel_state
from ..distributed.torch_compile import CompileConfig, mark_compile_step_begin
from ..distributed.torch_parallelize import build_parallelize_model
from ..models import build_foundation_model, build_tokenizer
from ..ops.batch_invariant_ops import set_batch_invariant_mode
from ..optim import build_lr_scheduler, build_optimizer
from ..utils import helper, logging
from ..utils.checkpoint_utils import should_skip_hf_weight_load
from ..utils.device import (
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
    is_nccl_backend,
    synchronize,
)
from ..utils.loss_utils import count_loss_token, mean_global_loss
from ..utils.model_utils import pretty_print_trainable_parameters
from .callbacks import (
    ChannelLossCallback,
    CheckpointerCallback,
    EnvironMeterCallback,
    EvaluateCallback,
    HFLoraCkptCallback,
    HuggingfaceCkptCallback,
    MoERouterMonitorCallback,
    ProfileTraceCallback,
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


def _resolve_muon_lr(optimizer_cfg) -> float:
    """Resolve Muon LR, inheriting AdamW lr under match_rms_adamw when unset."""
    if optimizer_cfg.muon_lr is not None:
        return float(optimizer_cfg.muon_lr)
    adamw_lr = float(optimizer_cfg.lr)
    if optimizer_cfg.muon_adjust_lr_fn == "match_rms_adamw":
        return adamw_lr
    # original: Moonlight-style ~25x AdamW lr starting point
    return 25.0 * adamw_lr


def _collect_muon_kwargs(optimizer_cfg) -> Dict[str, Any]:
    """Pull Muon-specific hyperparameters out of ``OptimizerConfig``."""
    return {
        "lr": _resolve_muon_lr(optimizer_cfg),
        "momentum": optimizer_cfg.muon_momentum,
        "nesterov": optimizer_cfg.muon_nesterov,
        "weight_decay": optimizer_cfg.muon_weight_decay,
        "ns_steps": optimizer_cfg.muon_ns_steps,
        "ns_coefficients": tuple(optimizer_cfg.muon_ns_coefficients),
        "eps": optimizer_cfg.muon_eps,
        "adjust_lr_fn": optimizer_cfg.muon_adjust_lr_fn,
        "ns_implementation": optimizer_cfg.muon_ns_implementation,
        "gram_ns_reset_iterations": tuple(optimizer_cfg.muon_gram_ns_reset_iterations),
        # Surface for startup summary only; not a DistributedMuon ctor kwarg.
        "expert_zero_comm": bool(optimizer_cfg.muon_expert_zero_comm),
        "adamw_lr": float(optimizer_cfg.lr),
        "muon_lr_explicit": optimizer_cfg.muon_lr is not None,
    }


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
    model: PreTrainedModel = None
    model_config: PretrainedConfig = PretrainedConfig()
    tokenizer: PreTrainedTokenizerBase = None
    processor: ProcessorMixin = None
    chat_template: ChatTemplate = None
    model_assets: List[Any] = []

    # Training components
    optimizer: Optimizer = None
    lr_scheduler: LRScheduler = None

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
        # ``_setup`` registers ParallelState ("base") before seed/determinism so
        # device-mesh process groups are created with default NCCL settings —
        # matching pre-registry init order (avoids L20 SIGSEGV when
        # NCCL_DETERMINISTIC=1 is set before mesh construction).
        self._setup()
        # Every build step below reads the current ParallelState via
        # ``get_parallel_state()`` (meta-init, FSDP2/TP/EP wrap + weight load,
        # EP-/muon-aware optimizer, SP-aware data pipeline). Scope the whole
        # build under the registered name (a no-op for the single-model case:
        # the global already equals the registered ``"base"`` state).
        with use_parallel_state("base"):
            # build model
            self._build_model()
            # freeze module and print trainable parameters
            self._freeze_model_module()
            # build model assets (config, tokenizer, processor, chat_template)
            self._build_model_assets()
            # build dataset and dataloader
            self._build_data_transform()
            self._build_dataset()
            self._build_collate_fn()
            self._build_dataloader()

            # Parallelize model
            self._build_parallelized_model()
            # Build optimizer and lr scheduler
            self._build_optimizer()
            self._build_lr_scheduler()
            # Build training context
            self._build_training_context()
            # Initialize callbacks
            self._init_callbacks()

    def _setup(self):
        # log args
        logger.info_rank0(json.dumps(asdict(self.args), indent=2))

        # init distributed environment
        device_str = f"{get_device_type()}:{self.args.train.local_rank}"
        get_torch_device().set_device(device_str)
        self.device = torch.device(device_str)

        # Initialize distributed process group
        if not dist.is_initialized():
            dist.init_process_group(backend=get_dist_comm_backend())

        logger.info(f"Process rank: {self.args.train.global_rank}, world size: {self.args.train.world_size}")

        # Register ParallelState before seed/determinism env vars. Mesh creation
        # must not run under NCCL_DETERMINISTIC=1 on some GPU platforms (L20).
        self.register_parallel_state("base")

        # Set random seed
        helper.set_seed(self.args.train.seed, self.args.train.enable_full_determinism)

        # Enable high precision for bf16
        helper.enable_high_precision_for_bf16()

        # Enable third party logging
        if self.args.train.local_rank == 0:
            helper.enable_third_party_logging()

        # Save arguments
        if self.args.train.global_rank == 0:
            save_args(self.args, self.args.train.checkpoint.output_dir)

        # Gradient checkpointing debug
        set_checkpoint_debug_enabled(self.args.train.gradient_checkpointing.debug)

    def register_parallel_state(self, name: str = "base"):
        """Register this trainer's ParallelState under ``name`` in the registry."""
        init_parallel_state(
            dp_size=self.args.train.accelerator.dp_size,
            dp_replicate_size=self.args.train.accelerator.dp_replicate_size,
            dp_shard_size=self.args.train.accelerator.dp_shard_size,
            tp_size=self.args.train.accelerator.tp_size,
            pp_size=self.args.train.accelerator.pp_size,
            cp_size=self.args.train.accelerator.cp_size,
            ulysses_size=self.args.train.accelerator.ulysses_size,
            extra_parallel_sizes=self.args.train.accelerator.extra_parallel_sizes,
            extra_parallel_placement_innermost=self.args.train.accelerator.extra_parallel_placement_innermost,
            extra_parallel_names=self.args.train.accelerator.extra_parallel_names,
            dp_mode=self.args.train.accelerator.fsdp_config.fsdp_mode,
            async_enabled=self.args.train.accelerator.enable_async,
            name=name,
        )

    def _build_model(self):
        logger.info_rank0("Build model")
        self.model = build_foundation_model(
            config_path=self.args.model.config_path,
            weights_path=self.args.model.model_path,
            torch_dtype="float32" if self.args.train.accelerator.fsdp_config.mixed_precision.enable else "bfloat16",
            init_device=self.args.train.init_device,
            ops_implementation=self.args.model.ops_implementation,
            config_kwargs=self.args.model.model_config,
        )
        self.model_config = self.model.config

    def _setup_lora(self):
        """Wrap ``self.model`` with the PEFT-free :class:`veomni.lora.VeOmniLoraModel`.

        A single native path handles both dense ``nn.Linear`` LoRA
        (``lora_modules`` / ``target_modules``) and MoE expert LoRA
        (``target_parameters``, wrapper flavour selected by
        ``share_expert_lora``). On resume (``lora_config['lora_adapter']`` set)
        the wrappers are rebuilt from the on-disk ``adapter_config.json`` (MoE
        mode lives in its ``veomni_lora`` block); otherwise a fresh adapter is
        initialised from the yaml config. Either way the actual adapter
        *weights* are streamed in later during parallelization
        (``build_parallelize_model`` with ``adapter_path``).

        Recognised ``lora_config`` keys (in addition to ``rank`` / ``alpha`` /
        ``lora_adapter`` / ``is_trainable``): ``lora_modules`` (aka
        ``target_modules``), ``target_parameters``, ``share_expert_lora``,
        ``use_rslora``, ``lora_dropout``, ``bias``, ``exclude_modules``,
        ``rank_pattern``, ``alpha_pattern``, ``modules_to_save`` — see
        :class:`veomni.lora.VeOmniLoraConfig`.

        Fused-MoE models (Qwen3-MoE family) may list the semantic expert module
        names ``gate_proj`` / ``up_proj`` / ``down_proj`` in ``lora_modules``;
        these are auto-mapped to the model's fused expert ``target_parameters``
        (see :func:`veomni.lora.resolve_fused_moe_lora_targets`). Dense models
        keep those names as ordinary ``nn.Linear`` LoRA targets.
        """
        lora_config = self.args.model.lora_config
        if not bool(lora_config):
            return

        from ..lora import VeOmniLoraConfig, VeOmniLoraModel, resolve_fused_moe_lora_targets

        lora_adapter_path = lora_config.get("lora_adapter", None)
        if lora_adapter_path is not None:
            logger.info_rank0(f"Wrapping model with VeOmniLoraModel from {lora_adapter_path}.")
            self.model = VeOmniLoraModel.from_pretrained(
                self.model,
                lora_adapter_path,
                is_trainable=lora_config.get("is_trainable", True),
            )
        else:
            # Rewrite semantic MoE module names onto fused expert parameters
            # before building the config (no-op for dense models / plain configs).
            resolved_config = resolve_fused_moe_lora_targets(self.model, lora_config)
            cfg = VeOmniLoraConfig.from_yaml(resolved_config)
            logger.info_rank0(f"Initialising VeOmni LoRA adapter from scratch: {cfg}.")
            self.model = VeOmniLoraModel(self.model, cfg)

    def _freeze_model_module(self):
        self._setup_lora()
        pretty_print_trainable_parameters(self.model)
        helper.print_device_mem_info("VRAM usage after building model")

    def _build_model_assets(self):
        # model assets
        self.tokenizer = build_tokenizer(self.args.model.tokenizer_path)
        self.model_assets = [self.model_config, self.tokenizer]

    def _build_data_transform(self):
        self.data_transform = build_data_transform(
            self.args.data.data_type,
            tokenizer=self.tokenizer,
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
            dataset_length = dataset_length / args.train.accelerator.dp_size
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

    def _build_parallelized_model(self):
        args: VeOmniArguments = self.args
        kwargs = {}
        cpu_load_param_name = None
        if hasattr(self.model, "get_parallel_plan"):
            cpu_load_param_name = getattr(self.model.get_parallel_plan(), "cpu_load_param_name", None)
        kwargs["cpu_load_param_name"] = cpu_load_param_name
        if bool(args.model.lora_config):
            lora_adapter_path = args.model.lora_config.get("lora_adapter", None)
            kwargs["adapter_path"] = lora_adapter_path
            kwargs["is_peft_model"] = True

        muon_expert_zero_comm = args.train.optimizer.type == "muon" and args.train.optimizer.muon_expert_zero_comm

        if args.model.fqn_to_index_mapping is not None:
            kwargs["fqn_to_index_mapping"] = args.model.fqn_to_index_mapping
        if args.train.chunk_mbs_config.enable:
            kwargs["chunk_mbs_config"] = args.train.chunk_mbs_config

        # A full non-LoRA resume already contains model weights. Skip the HF
        # materialization pass to avoid a second peak (HF load then checkpoint
        # overwrite) that can OOM large MoE jobs. LoRA resumes still need the HF base.
        skip_hf_weight_load = should_skip_hf_weight_load(
            args.train.checkpoint.load_path,
            args.model.lora_config,
        )
        if skip_hf_weight_load:
            logger.info_rank0(
                f"Checkpoint resume enabled (load_path={args.train.checkpoint.load_path}); "
                "skipping HF weight materialization before checkpoint restore."
            )

        # Parallelize model
        self.model = build_parallelize_model(
            self.model,
            init_device=args.train.init_device,
            weights_path=args.model.model_path,
            should_skip_hf_weight_load=skip_hf_weight_load,
            enable_reshard_after_forward=args.train.accelerator.fsdp_config.reshard_after_forward,
            mixed_precision=args.train.accelerator.fsdp_config.mixed_precision,
            enable_gradient_checkpointing=args.train.gradient_checkpointing.enable,
            basic_modules=list(
                set(getattr(self.model, "_no_split_modules", None) or []) | set(args.model.basic_modules)
            ),
            enable_reentrant=args.train.gradient_checkpointing.enable_reentrant,
            early_stop=args.train.gradient_checkpointing.early_stop,
            enable_forward_prefetch=args.train.accelerator.fsdp_config.forward_prefetch,
            enable_fsdp_offload=args.train.accelerator.fsdp_config.offload,
            broadcast_model_weights_from_rank0=args.train.broadcast_model_weights_from_rank0,
            ep_sharded_stream_load=args.train.ep_sharded_stream_load,
            max_load_broadcast_size=args.train.accelerator.fsdp_config.max_load_broadcast_size,
            muon_expert_zero_comm=muon_expert_zero_comm,
            compile_config=CompileConfig(
                **{field.name: getattr(args.train.torch_compile, field.name) for field in fields(CompileConfig)}
            ),
            **kwargs,
        )
        self.model.train()

    def _build_optimizer(self):
        args: VeOmniArguments = self.args
        # Build optimizer
        self.optimizer = build_optimizer(
            self.model,
            lr=args.train.optimizer.lr,
            weight_decay=args.train.optimizer.weight_decay,
            fused=True,
            optimizer_type=args.train.optimizer.type,
            no_decay_modules=args.train.optimizer.no_decay_modules,
            no_decay_params=args.train.optimizer.no_decay_params,
            muon_kwargs=_collect_muon_kwargs(args.train.optimizer),
        )

    def _build_lr_scheduler(self):
        args: VeOmniArguments = self.args
        # Build lr scheduler
        self.lr_scheduler = build_lr_scheduler(
            self.optimizer,
            train_steps=args.train_steps * args.train.num_train_epochs,
            lr=args.train.optimizer.lr,
            lr_min=args.train.optimizer.lr_min,
            lr_decay_style=args.train.optimizer.lr_decay_style,
            lr_decay_ratio=args.train.optimizer.lr_decay_ratio,
            lr_warmup_ratio=args.train.optimizer.lr_warmup_ratio,
            lr_start=args.train.optimizer.lr_start,
        )

    def _build_training_context(self):
        """Build training context for distributed training."""
        self.model_fwd_context, self.model_bwd_context = build_activation_offloading_context(
            self.args.train.accelerator.offload_config.enable_activation,
            self.args.train.gradient_checkpointing.enable,
            self.args.train.accelerator.offload_config.activation_gpu_limit,
        )

    def _init_callbacks(self):
        """Initialize callbacks."""
        self.environ_meter_callback = EnvironMeterCallback(self)
        self.tqdm_callback = TqdmCallback(self)
        self.wandb_callback = WandbTraceCallback(self)
        self.profile_callback = ProfileTraceCallback(self)
        self.checkpointer_callback = CheckpointerCallback(self)
        if self.args.model.lora_config:
            self.hf_ckpt_callback = HFLoraCkptCallback(self)
        else:
            self.hf_ckpt_callback = HuggingfaceCkptCallback(self)
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
            self.checkpointer_callback,
            self.hf_ckpt_callback,
            self.evaluate_callback,
            self.moe_monitor_callback,
        ]
        self.state = TrainerState()

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

        chunk_mbs_config = getattr(self.args.train, "chunk_mbs_config", None)
        self._chunk_mbs_ranges = build_chunk_mbs_ranges(micro_batch, chunk_mbs_config)
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
            outputs.loss, self.micro_batch_token_len, self.micro_batches_token_len
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

            chunk_ranges = getattr(self, "_chunk_mbs_ranges", None)
            channel_forward_context = (
                channel_loss_callback.model_forward_context() if channel_loss_callback is not None else nullcontext()
            )
            with (
                use_parallel_state("base"),
                chunk_mbs_context(chunk_ranges),
                self.model_fwd_context,
                set_batch_invariant_mode(self.args.train.enable_batch_invariant_mode),
                channel_forward_context,
            ):
                outputs: ModelOutput = self.model(**micro_batch, use_cache=False)

            with use_parallel_state("base"):
                loss, loss_dict = self.postforward(outputs, micro_batch)

            # Backward pass
            with (
                use_parallel_state("base"),
                chunk_mbs_context(chunk_ranges),
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
            args.train.accelerator.fsdp_config.fsdp_mode == "fsdp2"
            and not args.train.accelerator.fsdp_config.reshard_after_backward
            and num_micro_steps > 1
        ):
            if micro_step == 0:
                self.model.set_reshard_after_backward(False)
            elif micro_step == num_micro_steps - 1:
                self.model.set_reshard_after_backward(True)

    def _configure_hsdp_allreduce(self, micro_step: int, num_micro_steps: int):
        args: VeOmniArguments = self.args
        if (
            args.train.accelerator.fsdp_config.fsdp_mode == "fsdp2"
            and args.train.accelerator.dp_replicate_size > 1
            and num_micro_steps > 1
        ):
            if micro_step == 0:
                self.model.set_requires_all_reduce(False)
            elif micro_step == num_micro_steps - 1:
                self.model.set_requires_all_reduce(True)

    def train_step(
        self,
        data_iterator: Any,
    ) -> Dict[str, float]:
        args = self.args
        self.state.global_step += 1

        micro_batches: List[Dict[str, Any]] = next(data_iterator)

        self.on_step_begin(micro_batches=micro_batches)

        # Forward and backward for each micro batch
        synchronize()

        total_loss = 0.0
        total_loss_dict = defaultdict(int)

        # token num for fixed_ce_loss in postforward
        self.micro_batches_token_len = count_loss_token(micro_batches)
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

        # Gradient clipping (reads FSDP/EP groups from current ParallelState)
        with use_parallel_state("base"):
            grad_norm = veomni_clip_grad_norm(self.model, args.train.optimizer.max_grad_norm)

        # Optimizer and scheduler step
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        self.on_step_end(loss=total_loss, loss_dict=total_loss_dict, grad_norm=grad_norm)

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
