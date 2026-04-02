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

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from ..utils import logging
from ..utils.env import get_env


logger = logging.get_logger(__name__)


# ================================ Training Arguments ======================================
#
# Hierarchy:
#   train.*
#   ├── optimizer.*          → OptimizerConfig
#   ├── wandb.*              → WandbConfig
#   ├── profile.*            → ProfileConfig
#   ├── gradient_checkpointing.*  → GradientCheckpointingConfig
#   ├── accelerator.*        → AcceleratorConfig
#   │   ├── fsdp_config.*    → FSDPConfig
#   │   └── offload_config.* → OffloadConfig
#   └── checkpoint.*         → CheckpointConfig
#


@dataclass
class OptimizerConfig:
    """train.optimizer.* — Optimizer and learning-rate schedule."""

    type: Literal["adamw", "anyprecision_adamw"] = field(
        default="adamw",
        metadata={"help": "Optimizer type. Default to adamw."},
    )
    lr: float = field(
        default=5e-5,
        metadata={"help": "Maximum learning rate or default learning rate, or init learning rate for warmup."},
    )
    lr_min: float = field(
        default=1e-7,
        metadata={"help": "Minimum learning rate."},
    )
    lr_start: float = field(
        default=0.0,
        metadata={"help": "Learning rate for warmup start. Default to 0.0."},
    )
    lr_warmup_ratio: float = field(
        default=0,
        metadata={"help": "Ratio of learning rate warmup steps."},
    )
    lr_decay_style: str = field(
        default="constant",
        metadata={"help": "Name of the learning rate scheduler."},
    )
    lr_decay_ratio: float = field(
        default=1.0,
        metadata={"help": "Ratio of learning rate decay steps."},
    )
    weight_decay: float = field(
        default=0,
        metadata={"help": "L2 regularization strength."},
    )
    no_decay_modules: List[str] = field(
        default_factory=list,
        metadata={"help": "Modules without weight decay, for example, RMSNorm."},
    )
    no_decay_params: List[str] = field(
        default_factory=list,
        metadata={"help": "Parameters without weight decay, for example, bias."},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Clip value for gradient norm."},
    )


@dataclass
class WandbConfig:
    """train.wandb.* — Weights & Biases logging."""

    enable: bool = field(
        default=False,
        metadata={"help": "Enable wandb logging."},
    )
    project: str = field(
        default="VeOmni",
        metadata={"help": "Wandb project name."},
    )
    name: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb experiment name."},
    )
    id: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb run ID for resuming a previous run."},
    )


@dataclass
class ProfileConfig:
    """train.profile.* — Torch profiler settings."""

    enable: bool = field(
        default=False,
        metadata={"help": "Enable profiling."},
    )
    start_step: int = field(
        default=1,
        metadata={"help": "Start step for profiling."},
    )
    end_step: int = field(
        default=2,
        metadata={"help": "End step for profiling."},
    )
    trace_dir: str = field(
        default="./trace",
        metadata={"help": "Directory to save profiling traces."},
    )
    record_shapes: bool = field(
        default=True,
        metadata={"help": "Whether or not to record the shapes of the input tensors."},
    )
    profile_memory: bool = field(
        default=True,
        metadata={"help": "Whether or not to profile the memory usage."},
    )
    with_stack: bool = field(
        default=True,
        metadata={"help": "Whether or not to record the stack traces."},
    )
    with_modules: bool = field(
        default=False,
        metadata={"help": "Whether or not to record module hierarchy in profiling traces."},
    )
    rank0_only: bool = field(
        default=True,
        metadata={
            "help": "whether to profile rank0 only. When false, every rank will be profiled; Please expect many files to save, which can be slow and take a lot of disk space."
        },
    )


@dataclass
class GradientCheckpointingConfig:
    """train.gradient_checkpointing.* — Activation recomputation settings."""

    enable: bool = field(
        default=True,
        metadata={"help": "Enable gradient checkpointing."},
    )
    debug: bool = field(
        default=False,
        metadata={
            "help": "Debug gradient checkpointing: https://docs.pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.set_checkpoint_debug_enabled."
        },
    )
    enable_reentrant: bool = field(
        default=False,
        metadata={"help": "Use reentrant gradient checkpointing."},
    )


@dataclass
class FSDPConfig:
    """train.accelerator.fsdp_config.* — FSDP sharding configuration."""

    fsdp_mode: Literal["ddp", "fsdp1", "fsdp2"] = field(
        default="ddp",
        metadata={"help": "Data parallel mode."},
    )
    reshard_after_forward: bool = field(
        default=True,
        metadata={"help": "Enable reshard after forward for FSDP2."},
    )
    reshard_after_backward: bool = field(
        default=True,
        metadata={"help": "Enable reshard after backward for FSDP2."},
    )
    full_shard: bool = field(
        default=True,
        metadata={"help": "Enable fully shard for FSDP training (ZeRO-3)."},
    )
    forward_prefetch: bool = field(
        default=True,
        metadata={"help": "Enable forward prefetch for FSDP1."},
    )
    offload: bool = field(
        default=False,
        metadata={"help": "Enable CPU offload for FSDP1."},
    )


@dataclass
class OffloadConfig:
    """train.accelerator.offload_config.* — Activation offload settings."""

    enable_activation: bool = field(
        default=False,
        metadata={"help": "Enable activation offload to CPU."},
    )
    activation_gpu_limit: float = field(
        default=0.0,
        metadata={
            "help": "When enabling activation offload, `activation_gpu_limit` GB activations are allowed to reserve on GPU."
        },
    )


@dataclass
class AcceleratorConfig:
    """train.accelerator.* — Parallelism and distributed-training topology."""

    dp_replicate_size: int = field(
        default=-1,
        metadata={"help": "Data parallel replicate size."},
    )
    dp_shard_size: int = field(
        default=-1,
        metadata={"help": "Data parallel shard degree."},
    )
    tp_size: int = field(
        default=1,
        metadata={"help": "Tensor parallel size."},
    )
    ep_size: int = field(
        default=1,
        metadata={"help": "Expert parallel size."},
    )
    ep_outside: bool = field(
        default=False,
        metadata={"help": "Enable expert parallelism outside in ep-fsdp."},
    )
    extra_parallel_sizes: List[int] = field(
        default_factory=list,
        metadata={"help": "Extra parallelism sizes."},
    )
    extra_parallel_placement_innermost: List[bool] = field(
        default_factory=list,
        metadata={"help": "Extra parallelism outside in para-fsdp."},
    )
    extra_parallel_names: List[str] = field(
        default_factory=list,
        metadata={"help": "Extra parallelism names."},
    )
    pp_size: int = field(
        default=1,
        metadata={"help": "Pipeline parallel size."},
    )
    ulysses_size: int = field(
        default=1,
        metadata={"help": "Ulysses sequence parallel size."},
    )
    enable_async: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable async ulysses."},
    )
    cp_size: int = field(
        default=1,
        metadata={"help": "Ring-attn context parallel size."},
    )
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    offload_config: OffloadConfig = field(default_factory=OffloadConfig)

    def __post_init__(self):
        # although expert parallel and extra parallel are both provided in the arguments,
        # the implementation is configuring extra parallelism to include expert parallelism
        self.extra_parallel_sizes.append(self.ep_size)
        self.extra_parallel_names.append("ep")
        self.extra_parallel_placement_innermost.append(self.ep_outside)


@dataclass
class CheckpointConfig:
    """train.checkpoint.* — Checkpoint saving and loading."""

    output_dir: str = field(
        default="output",
        metadata={"help": "Path to save model checkpoints."},
    )
    manager: str = field(
        default="dcp",
        metadata={"help": "Checkpoint manager."},
    )
    save_async: bool = field(
        default=False,
        metadata={"help": "Whether to save checkpoint asynchronously."},
    )
    load_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to checkpoint to resume from."},
    )
    save_steps: int = field(
        default=0,
        metadata={"help": "Number of steps between two checkpoint saves."},
    )
    save_epochs: int = field(
        default=1,
        metadata={"help": "Number of epochs between two checkpoint saves."},
    )
    hf_save_steps: int = field(
        default=0,
        metadata={"help": "Number of steps between two hf model weights save."},
    )
    hf_save_epochs: int = field(
        default=0,
        metadata={"help": "Number of epochs between two hf model weights save."},
    )
    save_hf_weights: bool = field(
        default=True,
        metadata={"help": "Save the huggingface format weights to the last checkpoint dir."},
    )


@dataclass
class TrainingArguments:
    """train.* — Top-level training configuration."""

    train_architecture: Literal["full", "lora"] = field(
        default="full",
        metadata={
            "help": "Specifies the parameter update strategy for training the multi-modal model. 'full' for Standard SFT, lora for LoRA."
        },
    )
    dyn_bsz: bool = field(
        default=True,
        metadata={"help": "Enable dynamic batch size for padding-free training."},
    )
    micro_batch_size: int = field(
        default=1,
        metadata={"help": "Micro batch size. The number of samples per iteration on each device."},
    )
    global_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Global batch size. If None, use `micro_batch_size` * `data_parallel_size`."},
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Epochs to train."},
    )
    pad_to_length: bool = field(
        default=False,
        metadata={"help": "Pad packed sequences to a fixed length when using dynamic batch size."},
    )
    bsz_warmup_ratio: float = field(
        default=0,
        metadata={"help": "Ratio of batch size warmup steps."},
    )
    bsz_warmup_init_mbtoken: int = field(
        default=200,
        metadata={"help": "Initial number of tokens in a batch in warmup phase."},
    )
    enable_mixed_precision: bool = field(
        default=True,
        metadata={"help": "Enable mixed precision training."},
    )
    init_device: Literal["cpu", "cuda", "meta", "npu"] = field(
        default="cuda",
        metadata={
            "help": "Device to initialize model weights. 1. `cpu`: Init parameters on CPU in rank0 only. 2. `cuda`: Init parameters on GPU. 3. `meta`: Init parameters on meta. 4. `npu`: Init parameters on Ascend NPU."
        },
    )
    broadcast_model_weights_from_rank0: bool = field(
        default=True,
        metadata={
            "help": "When enabled, only rank0 reads model weights from HuggingFace safetensor from disk. Other ranks would receive weights through broadcast. This helps to avoid disk I/O bottleneck."
        },
    )
    enable_full_determinism: bool = field(
        default=False,
        metadata={"help": "Enable full determinism."},
    )
    enable_batch_invariant_mode: bool = field(
        default=False,
        metadata={"help": "Enable batch invariant mode."},
    )
    empty_cache_steps: int = field(
        default=500,
        metadata={"help": "Number of steps between two empty cache operations."},
    )
    gc_steps: int = field(
        default=500,
        metadata={"help": "Number of steps between two gc.collect. GC is disabled if it is positive."},
    )
    eval_steps: int = field(
        default=0,
        metadata={"help": "Number of steps between two evaluations. 0 to disable."},
    )
    eval_epochs: int = field(
        default=1,
        metadata={"help": "Number of epochs between two evaluations. 0 to disable."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."},
    )
    enable_compile: bool = field(
        default=False,
        metadata={"help": "Enable torch compile."},
    )
    max_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Max training steps per epoch. (for debug)"},
    )
    moe_load_balance_monitor_interval: int = field(
        default=0,
        metadata={"help": "Log MoE expert load heatmap every N steps. 0 = disabled. Requires wandb.enable=True."},
    )

    # sub-argument groups
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    gradient_checkpointing: GradientCheckpointingConfig = field(default_factory=GradientCheckpointingConfig)
    accelerator: AcceleratorConfig = field(default_factory=AcceleratorConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def __post_init__(self):
        self._train_steps = -1
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.global_rank = int(os.getenv("RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))

        self._validate_accelerator()
        self._derive_batch_config()
        self._resolve_checkpoint_paths()
        self._resolve_profile()

    # -- validation & derivation helpers (called by __post_init__) -----------------------

    def _validate_accelerator(self):
        acc = self.accelerator

        if self.world_size % (acc.pp_size * acc.ulysses_size * acc.cp_size * acc.tp_size) != 0:
            raise ValueError(
                f"World size should be a multiple of pp_size: {acc.pp_size}, "
                f"ulysses_size: {acc.ulysses_size}, cp_size: {acc.cp_size}, "
                f"tp_size: {acc.tp_size}."
            )
        assert acc.tp_size == 1, "Tensor parallel size not supported yet."
        assert acc.pp_size == 1, "Pipeline parallel size not supported yet."

        acc.dp_size = self.world_size // (acc.pp_size * acc.ulysses_size * acc.cp_size * acc.tp_size)

        # resolve dp_replicate_size / dp_shard_size
        if acc.dp_replicate_size > 0 and acc.dp_shard_size > 0:
            assert acc.dp_size == acc.dp_replicate_size * acc.dp_shard_size, (
                f"dp_size should be equal to dp_replicate_size: {acc.dp_replicate_size} "
                f"* dp_shard_size: {acc.dp_shard_size}."
            )
        elif acc.dp_replicate_size > 0:
            if acc.dp_size % acc.dp_replicate_size != 0:
                raise ValueError("dp_size should be a multiple of dp_replicate_size.")
            acc.dp_shard_size = acc.dp_size // acc.dp_replicate_size
        elif acc.dp_shard_size > 0:
            if acc.dp_size % acc.dp_shard_size != 0:
                raise ValueError("dp_size should be a multiple of dp_shard_size.")
            acc.dp_replicate_size = acc.dp_size // acc.dp_shard_size
        else:
            acc.dp_replicate_size = 1
            acc.dp_shard_size = acc.dp_size

        # multi-node warning
        num_nodes = int(os.getenv("WORLD_SIZE", 1)) // int(os.getenv("LOCAL_WORLD_SIZE", 1))
        if num_nodes > 1:
            logger.warning_rank0(
                f"Detected {num_nodes} nodes. "
                "Make sure that `train.checkpoint.output_dir` is shared by all nodes. "
                "Otherwise, each node will save checkpoints to its local directory, which may cause inconsistencies or job failures."
            )

        # init method constraints
        assert acc.ep_size == 1 or self.init_device != "cpu", (
            "cpu init is not supported when enable ep. Please use `init_device = cuda` or `init_device = meta` instead."
        )
        if acc.fsdp_config.fsdp_mode == "fsdp2":
            assert self.init_device == "meta", "Please use init_device: meta for FSDP2 training"
        else:
            if self.broadcast_model_weights_from_rank0:
                logger.warning_rank0(
                    "Ignoring train.broadcast_model_weights_from_rank0=True because it is only "
                    "used with train.accelerator.fsdp_config.fsdp_mode='fsdp2'. "
                    f"Received fsdp_mode={acc.fsdp_config.fsdp_mode!r}. Disable this flag or switch to fsdp2.",
                )

    def _derive_batch_config(self):
        acc = self.accelerator

        # gradient accumulation steps
        if self.global_batch_size is None:
            self.global_batch_size = self.micro_batch_size * acc.dp_size
            self.gradient_accumulation_steps = 1
            logger.info_rank0("`global_batch_size` is None, disable gradient accumulation.")
        elif self.global_batch_size % (self.micro_batch_size * acc.dp_size) == 0:
            self.gradient_accumulation_steps = self.global_batch_size // (self.micro_batch_size * acc.dp_size)
            logger.info_rank0(f"Set gradient accumulation to {self.gradient_accumulation_steps}.")
        else:
            raise ValueError(f"`global_batch_size` should be a multiple of {self.micro_batch_size * acc.dp_size}.")

        if self.gradient_accumulation_steps > 1 and acc.fsdp_config.offload:
            raise ValueError("Gradient accumulation is not supported with FSDP offload.")

        # dataloader batch size
        self.dataloader_batch_size = 1 if self.dyn_bsz else self.global_batch_size // acc.dp_size

    def _resolve_checkpoint_paths(self):
        ckpt = self.checkpoint

        if ckpt.load_path == "auto":
            from ..utils.checkpoint_utils import get_checkpoint_path

            ckpt.load_path = get_checkpoint_path(
                output_dir=ckpt.output_dir,
                is_local_rank0=self.local_rank == 0,
                ckpt_manager=ckpt.manager,
            )

        # output_dir/
        # ├── checkpoints/          # DCP training checkpoints (model + optimizer + extra_state)
        # │   ├── global_step_100/
        # │   └── global_step_200/
        # │       └── hf_ckpt/      # HF safetensors saved under the last checkpoint folder
        # └── model_assets/
        ckpt.save_path = os.path.join(ckpt.output_dir, "checkpoints")
        ckpt.model_assets_dir = os.path.join(ckpt.output_dir, "model_assets")

    def _resolve_profile(self):
        if self.profile.enable:
            if self.profile.rank0_only:
                self.profile.this_rank = self.global_rank == 0
            else:
                logger.warning_rank0(
                    "Profiling on ALL ranks is enabled. This would save a lot of files which takes time and space."
                )
                self.profile.this_rank = True
        else:
            self.profile.this_rank = False


# ================================ Model Arguments ======================================
#
# Hierarchy:
#   model.*
#   └── ops_implementation.* → OpsImplementationConfig
#


@dataclass
class OpsImplementationConfig:
    """model.ops_implementation.* — Attention / MoE kernel implementation."""

    attn_implementation: Optional[
        Literal[
            "eager",
            "sdpa",
            "flash_attention_2",
            "flash_attention_3",
            "flash_attention_4",
            "native-sparse",
        ]
    ] = field(
        default="flash_attention_2",
        metadata={"help": "Attention implementation to use."},
    )
    moe_implementation: Optional[Literal["eager", "fused", "fused_quack"]] = field(
        default=None,
        metadata={
            "help": "MoE implementation to use. "
            "'eager' for reference loop, 'fused' for Triton group-gemm, "
            "'fused_quack' for Quack CUTLASS/CuTe kernels (SM90+)."
        },
    )
    cross_entropy_loss_implementation: str = field(
        default="eager",
        metadata={
            "help": "Cross-entropy loss kernel implementation. "
            "'eager' for standard PyTorch, 'liger_fused' for Liger fused linear CE."
        },
    )
    moe_load_balancing_loss_implementation: str = field(
        default="eager",
        metadata={"help": "MoE load-balancing loss kernel implementation. 'eager' for standard PyTorch."},
    )
    rms_norm_implementation: str = field(
        default="eager",
        metadata={
            "help": "RMSNorm kernel implementation. "
            "'eager' for standard PyTorch, 'liger' for LigerKernel fused RMSNorm."
        },
    )
    causal_conv1d_implementation: str = field(
        default="eager",
        metadata={
            "help": "causal_conv1d kernel implementation. "
            "'eager' for standard PyTorch, 'liger' for LigerKernel fused RMSNorm."
        },
    )
    chunk_gated_delta_rule_implementation: str = field(
        default="eager",
        metadata={
            "help": "chunk_gated_delta_rule kernel implementation. "
            "'eager' for standard PyTorch, 'liger' for LigerKernel fused RMSNorm."
        },
    )
    swiglu_mlp_implementation: str = field(
        default="eager",
        metadata={
            "help": "SwiGLU MLP kernel implementation. "
            "'eager' for standard PyTorch, 'liger' for LigerKernel fused SwiGLU."
        },
    )
    apply_rotary_pos_emb_implementation: str = field(
        default="eager",
        metadata={
            "help": "Rotary positional embedding kernel implementation. "
            "'eager' for standard PyTorch, 'liger' for LigerKernel fused RoPE."
        },
    )

    @property
    def moe_experts_implementation(self) -> str:
        """Bridge ``moe_implementation`` → ``moe_experts_implementation`` for OpSlot lookup."""
        return self.moe_implementation or "eager"

    def __post_init__(self):
        if get_env("MODELING_BACKEND") == "veomni":
            replacements = {
                "flash_attention_2": "veomni_flash_attention_2_with_sp",
                "flash_attention_3": "veomni_flash_attention_3_with_sp",
                "flash_attention_4": "veomni_flash_attention_4_with_sp",
            }
            if self.attn_implementation in replacements:
                new_impl = replacements[self.attn_implementation]
                logger.info_rank0(f"Replacing attn_implementation from '{self.attn_implementation}' to '{new_impl}'")
                self.attn_implementation = new_impl


@dataclass
class ModelArguments:
    """model.* — Model architecture, paths, and multimodal encoder/decoder setup."""

    config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path/HDFS path to the model config. Defaults to `model_path`."},
    )
    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path/HDFS path to the pre-trained model. If unspecified, use random init."},
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path/HDFS path to the tokenizer. Defaults to `config_path`."},
    )
    safetensor_idx_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to model.safetensors.index.json. Defaults to `model_path`/model.safetensors.index.json."
        },
    )
    foundation: Dict[str, str] = field(
        default_factory=dict,
        metadata={"help": "Foundation model extra config."},
    )
    encoders: Dict[Literal["image"], Dict[str, str]] = field(
        default_factory=dict,
        metadata={"help": "Multimodal encoder config and weights."},
    )
    decoders: Dict[Literal["image"], Dict[str, str]] = field(
        default_factory=dict,
        metadata={"help": "Multimodal decoder config and weights."},
    )
    input_encoder: Literal["encoder", "decoder"] = field(
        default="encoder",
        metadata={"help": "Use encoder to encode input images or use decoder.encoder to encode input images."},
    )
    output_encoder: Literal["encoder", "decoder"] = field(
        default="decoder",
        metadata={"help": "Use encoder to encode output images or use decoder.encoder to encode output images."},
    )
    encode_target: bool = field(
        default=False,
        metadata={"help": "Whether to encode target with decoder. Only supports stable diffusion as decoder."},
    )
    basic_modules: Optional[List[str]] = field(
        default_factory=list,
        metadata={"help": "Basic modules beyond model._no_split_modules to be sharded in FSDP."},
    )
    lora_config: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Config for lora."},
    )
    ops_implementation: OpsImplementationConfig = field(default_factory=OpsImplementationConfig)

    def __post_init__(self):
        if self.config_path is None and self.model_path is None:
            raise ValueError("`config_path` must be specified when `model_path` is None.")

        if self.config_path is None:
            self.config_path = self.model_path

        if self.tokenizer_path is None:
            self.tokenizer_path = self.config_path

        # Auto-resolve safetensor_idx_path from model_path if not specified
        if self.safetensor_idx_path is None and self.model_path is not None:
            default_idx_path = os.path.join(self.model_path, "model.safetensors.index.json")
            if os.path.exists(default_idx_path):
                self.safetensor_idx_path = default_idx_path

        # Parse fqn_to_index_mapping from safetensor index json
        self.fqn_to_index_mapping = None
        if self.safetensor_idx_path is not None:
            with open(self.safetensor_idx_path) as f:
                weight_map = json.load(f)["weight_map"]
            self.fqn_to_index_mapping = {fqn: int(filename.split("-")[1]) for fqn, filename in weight_map.items()}
        if self.fqn_to_index_mapping is None:
            logger.warning_rank0(
                "fqn_to_index_mapping is None, saved safetensor will be a single file instead of sharded."
            )

        suppoerted_encoder_types = ["image", "video", "audio"]
        for encoder_type, encoder_args in self.encoders.items():
            if encoder_type not in suppoerted_encoder_types:
                raise ValueError(
                    f"Unsupported encoder type: {encoder_type}. Should be one of {suppoerted_encoder_types}."
                )

            if encoder_args.get("config_path") is None and encoder_args.get("model_path") is None:
                raise ValueError("`config_path` and `model_path` cannot be both empty.")

            if encoder_args.get("config_path") is None:
                encoder_args["config_path"] = encoder_args["model_path"]

        supported_decoder_types = ["image"]
        for decoder_type, decoder_args in self.decoders.items():
            if decoder_type not in supported_decoder_types:
                raise ValueError(
                    f"Unsupported decoder type: {decoder_type}. Should be one of {supported_decoder_types}."
                )

            if decoder_args.get("config_path") is None and decoder_args.get("model_path") is None:
                raise ValueError("`config_path` and `model_path` cannot be both empty.")

            if decoder_args.get("config_path") is None:
                decoder_args["config_path"] = decoder_args["model_path"]


# ================================ Data Arguments ======================================
#
# Hierarchy:
#   data.*
#   └── dataloader.*         → DataloaderConfig
#


@dataclass
class DataloaderConfig:
    """data.dataloader.* — DataLoader construction parameters."""

    type: str = field(
        default="native",
        metadata={"help": "Type of the dataloader."},
    )
    num_workers: int = field(
        default=2,
        metadata={"help": "Number of workers to load data."},
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "Number of batches loaded in advance by each worker."},
    )
    drop_last: bool = field(
        default=True,
        metadata={"help": "Whether to drop the last incomplete batch."},
    )
    pin_memory: bool = field(
        default=True,
        metadata={"help": "Whether to pin memory for dataloader."},
    )


@dataclass
class DataArguments:
    """data.* — Dataset paths, tokenization, and batching."""

    train_path: str = field(
        metadata={"help": "Local path/HDFS path of the training data. Use comma to separate multiple datasets."},
    )
    eval_path: Optional[str] = field(
        default=None,
        metadata={"help": "path of the evaluation data. If None, use a subset of train_path."},
    )
    train_size: int = field(
        default=10_000_000,
        metadata={"help": "Number of tokens for training to compute training steps for dynamic batch dataloader."},
    )
    train_sample: int = field(
        default=10_000,
        metadata={
            "help": "Number of samples for training to compute training steps for non-dynamic batch dataloader."
        },
    )
    data_type: Literal["plaintext", "conversation", "diffusion", "classification", "dpo"] = field(
        default="conversation",
        metadata={"help": "Type of the training data."},
    )
    datasets_type: str = field(
        default="mapping",
        metadata={"help": "Type of the datasets."},
    )
    multisource_datasets_type: str = field(
        default="interleave",
        metadata={"help": "Type of the datasets for multisource training."},
    )
    source_name: str = field(
        default=None,
        metadata={"help": "Dataset name for training. If multisource, dataset name will be loaded from yaml config."},
    )
    dyn_bsz_buffer_size: int = field(
        default=200,
        metadata={"help": "Buffer size for dynamic batch size."},
    )
    text_keys: str = field(
        default=None,
        metadata={"help": "Key to get text from the training data."},
    )
    chat_template: str = field(
        default="default",
        metadata={"help": "Chat template to use."},
    )
    max_seq_len: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length in training."},
    )
    silent_exception: bool = field(  # TODO: add silent_exception feature
        default=False,
        metadata={"help": "Whether to ignore exceptions when loading data. Defaults to ``False``"},
    )
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)

    def __post_init__(self):
        self.enable_multisource = self.train_path.endswith(".yaml")

        if self.enable_multisource:
            self.dataset_name = self.multisource_datasets_type
        else:
            self.dataset_name = self.datasets_type

        if self.text_keys is None:
            if self.data_type == "plaintext":
                self.text_keys = "content_split"
            elif self.data_type == "conversation":
                self.text_keys = "messages"
            elif self.data_type == "classification":
                self.text_keys = "text"
            elif self.data_type == "dpo":
                self.text_keys = "chosen"
            else:
                raise ValueError(f"Unknown data type: {self.data_type}")

        if self.dataloader.num_workers == 0:
            self.dataloader.prefetch_factor = None


# ================================ Top-Level Arguments ======================================


@dataclass
class VeOmniArguments:
    """Root config — assembles model, data, and train."""

    model: ModelArguments = field(default_factory=ModelArguments)
    data: DataArguments = field(default_factory=DataArguments)
    train: TrainingArguments = field(default_factory=TrainingArguments)

    def __post_init__(self):
        if self.train.pad_to_length:
            if not self.train.dyn_bsz:
                logger.warning_rank0(
                    "pad_to_length is enabled without dyn_bsz, which is not supported. "
                    "Please set pad_to_length to False or enable dyn_bsz."
                )
                self.train.pad_to_length = False
            else:
                self.train.pad_to_length = self.train.micro_batch_size * self.data.max_seq_len
                logger.info_rank0(f"set pad_to_length = micro_batch_size * max_seq_len = {self.train.pad_to_length}")

    def compute_train_steps(self, dataset_length: Optional[int] = None):
        if self.train.dyn_bsz:
            assert self.data.max_seq_len is not None and self.data.train_size is not None, (
                "data.max_seq_len and data.train_size are required."
            )
            train_size = int(self.data.train_size * (1 + self.train.bsz_warmup_ratio / 2))
            self._train_steps = math.ceil(train_size / (self.train.global_batch_size * self.data.max_seq_len))
        else:
            if dataset_length is not None:  # mapping dataset
                self._train_steps = math.floor(dataset_length / self.train.dataloader_batch_size)
            else:
                self._train_steps = math.ceil(self.data.train_sample / self.train.dataloader_batch_size)

    @property
    def train_steps(self) -> int:
        if self.train.max_steps is not None and self._train_steps >= self.train.max_steps:
            logger.warning_once(f"Set train_steps to {self.train.max_steps}. It should be for debug purpose only.")
            return self.train.max_steps

        if self._train_steps == -1:
            raise ValueError("Please run `compute_train_steps` first!")

        return self._train_steps


# ================================ Infer Arguments ======================================


@dataclass
class InferArguments:
    """Standalone inference configuration."""

    model_path: str = field(
        metadata={"help": "Local path/HDFS path to the pre-trained model."},
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path/HDFS path to the tokenizer. Defaults to `config_path`."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."},
    )
    do_sample: bool = field(
        default=True,
        metadata={"help": "Whether or not to use sampling in decoding."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "The temperature value of decoding."},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "The top_p value of decoding."},
    )
    max_tokens: int = field(
        default=1024,
        metadata={"help": "Max tokens to generate."},
    )

    def __post_init__(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path
