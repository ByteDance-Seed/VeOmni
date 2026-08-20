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

import math
import os
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import ClassVar, Dict, List, Literal, Optional

from ..utils import logging
from ..utils.env import get_env


logger = logging.get_logger(__name__)


def _resolve_hdfs_path(path: Optional[str]) -> Optional[str]:
    """Copy an ``hdfs://`` path to a local cache and return the local path.

    Non-HDFS paths (local filesystem, HF hub ids, hdfs-fuse mounts) are returned
    unchanged. Concurrent processes on the same node are serialized by the file
    lock inside ``copy_to_local``, so the download happens only once per node.
    """
    from ..utils.fs import copy_to_local, is_non_local

    if path is None or not is_non_local(path):
        return path

    return copy_to_local(path.rstrip("/"), verbose=True)


# ================================ Training Arguments ======================================
#
# Hierarchy:
#   model.*
#   ├── ops_implementation.* → OpsImplementationConfig
#   ├── optimizer.*          → OptimizerConfig
#   └── accelerator.*        → AcceleratorConfig
#       ├── fsdp_config.*    → FSDPConfig
#       |   └── mixed_precision.* → MixedPrecisionConfig
#       ├── offload_config.* → OffloadConfig
#       ├── gradient_checkpointing.*  → GradientCheckpointingConfig
#       ├── torch_compile.*  → TorchCompileConfig
#       └── chunk_mbs_config.*   → ChunkMBSConfig
#   train.*
#   ├── wandb.*              → WandbConfig
#   ├── profile.*            → ProfileConfig
#   ├── channel_loss.*       → ChannelLossConfig
#   └── checkpoint.*         → CheckpointConfig
#
# `accelerator` and `optimizer` sit under `model`, not `train`, because both are
# per-model decisions: an omni model gives each module its own block, so
# `model.modules.<name>.accelerator.*` has the same shape as `model.accelerator.*`
# and the two merge through one code path. What is left under `train.*` is the
# job-level schedule — batch sizes, steps, logging, checkpointing — which stays
# singular no matter how many modules the model has.
#
# `model.*` is split so an omni module can reuse it without inheriting a
# single-model loader's baggage:
#   BaseModelArguments      model fields alone (model_path, lora_config, ops, …)
#   └── ModelRuntimeArguments   + accelerator + optimizer — one training unit
#       └── ModelArguments      + config/tokenizer/safetensor-index paths
#


@dataclass
class OptimizerConfig:
    """model.optimizer.* — Optimizer and learning-rate schedule.

    ``type="muon"`` builds a Muon + AdamW multi-optimizer: 2D hidden weights
    and 3D MoE expert stacks use Muon, while embeddings, lm_head, biases and
    norms use AdamW.
    """

    type: Literal["adamw", "anyprecision_adamw", "muon"] = field(
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
    grad_clip_scope: Literal["per_module", "global"] = field(
        default="per_module",
        metadata={
            "help": (
                "How OmniTrainer applies max_grad_norm across OmniModules. "
                "'per_module' (default): clip each module to max_grad_norm independently, "
                "then report sqrt(sum n_i^2). "
                "'global': measure each module unclipped, total=sqrt(sum n_i^2), then scale "
                "all modules by one coefficient (single-model / seedream gradient_clip_val semantics)."
            ),
        },
    )
    # ---- Muon-specific (only consulted when type == "muon") ---------------
    muon_lr: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Learning rate for the Muon group (2D hidden weights and 3D expert stacks). "
                "If unset: inherits model.optimizer.lr when muon_adjust_lr_fn=match_rms_adamw "
                "(default); uses 25x model.optimizer.lr when muon_adjust_lr_fn=original "
                "(Moonlight-style starting point)."
            )
        },
    )
    muon_momentum: float = field(
        default=0.95,
        metadata={"help": "Momentum factor for the Muon group."},
    )
    muon_nesterov: bool = field(
        default=True,
        metadata={"help": "Use Nesterov momentum in Muon."},
    )
    muon_weight_decay: float = field(
        default=0.0,
        metadata={"help": "Decoupled weight decay for the Muon group."},
    )
    muon_ns_steps: int = field(
        default=5,
        metadata={"help": "Number of Newton-Schulz iteration steps in Muon."},
    )
    muon_ns_coefficients: List[float] = field(
        default_factory=lambda: [3.4445, -4.7750, 2.0315],
        metadata={"help": "Quintic Newton-Schulz polynomial coefficients (a, b, c)."},
    )
    muon_eps: float = field(
        default=1e-7,
        metadata={"help": "Numerical-stability epsilon for the spectral-norm normalization in Muon."},
    )
    muon_adjust_lr_fn: Literal["original", "match_rms_adamw"] = field(
        default="match_rms_adamw",
        metadata={
            "help": (
                "Per-matrix learning-rate adjustment used by Muon. "
                "'original' follows Keller Jordan; 'match_rms_adamw' (default) "
                "matches the RMS of an AdamW update so AdamW-tuned hyperparams transfer."
            )
        },
    )
    muon_head_group_size: int = field(
        default=0,
        metadata={
            "help": (
                "Attention heads per Newton-Schulz block for head-split Muon. "
                "0 (default) orthogonalizes each projection as a single matrix; 1 is fully per-head; "
                "g > 1 puts g heads in each block. Any value >= 1 also requires "
                "muon_head_split_modules."
            )
        },
    )
    muon_head_split_modules: List[str] = field(
        default_factory=list,
        metadata={
            "help": (
                "Leaf module names to head-split, matched exactly against the children of an "
                "attention module, e.g. ['q_b_proj'] for DeepSeek V3/V4 MLA up-projections or "
                "['q_proj', 'k_proj', 'v_proj'] for GQA. Required whenever "
                "muon_head_group_size >= 1; see docs/usage/basic_modules.md."
            )
        },
    )
    muon_expert_zero_comm: bool = field(
        default=False,
        metadata={
            "help": (
                "Use whole-expert Shard(0) for Muon under FSDP+ExtraParallel when "
                "(num_experts/ep_size) %% ep_fsdp_size == 0; otherwise fall back "
                "to the default hidden-dim sharding path."
            )
        },
    )
    muon_ns_implementation: Literal["std", "gram", "gram_quack"] = field(
        default="gram_quack",
        metadata={
            "help": (
                "Newton-Schulz implementation used by Muon. "
                "'std': torch.optim.Muon-compatible Newton-Schulz; "
                "'gram': pure-PyTorch Dao-AILab Gram Newton-Schulz; "
                "'gram_quack' (default): Dao-AILab Gram-NS with quack CuTeDSL GEMM kernels "
                "(Hopper/Blackwell). If quack/package is missing, falls back to gram."
            )
        },
    )
    muon_gram_ns_reset_iterations: List[int] = field(
        default_factory=lambda: [2],
        metadata={
            "help": (
                "Restart indices for Gram Newton-Schulz (applied immediately before "
                "those iteration indices). Default [2] matches Dao-AILab guidance "
                "for 5-step schedules."
            )
        },
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
class ChannelLossConfig:
    """train.channel_loss.* — Per-channel causal-LM loss logging."""

    enable: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable detached per-channel cross-entropy logging. This is an observability-only "
                "side channel and does not change the training objective."
            )
        },
    )
    interval: int = field(
        default=10,
        metadata={
            "help": (
                "Compute and log channel loss every N optimizer steps. The detached fused-loss "
                "fallback recomputes the LM-head projection on sampled steps; use 1 to sample every step."
            )
        },
    )
    source_id_keys: List[str] = field(
        default_factory=lambda: ["channel_id", "source_id", "dataset_id", "ds_idx"],
        metadata={
            "help": (
                "Batch metadata keys to read as channel/source IDs. The first key found in each "
                "micro-batch is used. Values should be one per packed sequence."
            )
        },
    )
    source_name_keys: List[str] = field(
        default_factory=lambda: ["channel_name", "source_name", "dataset_name", "data_name"],
        metadata={
            "help": (
                "Batch metadata keys to read as display names for channel/source IDs. Values should "
                "align with source_id_keys when provided."
            )
        },
    )
    extra_strip_keys: List[str] = field(
        default_factory=lambda: ["cur_token_num"],
        metadata={
            "help": (
                "Extra metadata keys to remove from each micro-batch before model forward when "
                "channel loss is enabled."
            )
        },
    )
    loss_metric_prefix: str = field(
        default="channel_loss",
        metadata={"help": "Metric prefix for per-channel average CE."},
    )
    weighted_loss_metric_prefix: str = field(
        default="channel_loss_weighted",
        metadata={"help": "Metric prefix for per-channel CE weighted by all logged tokens in the step."},
    )
    token_count_metric_prefix: str = field(
        default="channel_tokens",
        metadata={"help": "Metric prefix for per-channel supervised token counts."},
    )
    log_weighted_loss: bool = field(
        default=True,
        metadata={"help": "Log loss_sum / total_step_tokens for each channel."},
    )
    log_token_count: bool = field(
        default=True,
        metadata={"help": "Log supervised token count for each channel."},
    )
    strict: bool = field(
        default=False,
        metadata={
            "help": (
                "Raise when enabled but a micro-batch has no configured source ID, or when "
                "source metadata cannot be aligned with packed segments. Default False skips "
                "micro-batches with missing or mismatched metadata."
            )
        },
    )

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError("train.channel_loss.interval must be at least 1.")


@dataclass
class GradientCheckpointingConfig:
    """model.accelerator.gradient_checkpointing.* — Activation recomputation settings."""

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
    early_stop: bool = field(
        default=True,
        metadata={
            "help": (
                "Stop non-reentrant checkpoint recomputation as soon as all needed tensors are computed. "
                "PyTorch ignores this option when enable_reentrant=True."
            )
        },
    )


@dataclass
class ChunkMBSConfig:
    """model.accelerator.chunk_mbs_config.* — Packed-sequence layer micro-batching."""

    enable: bool = field(
        default=False,
        metadata={"help": "Enable ChunkMBS for packed-sequence decoder layers."},
    )
    chunk_mbs: int = field(
        default=1,
        metadata={"help": "Number of packed samples per layer chunk."},
    )

    def __post_init__(self):
        if self.chunk_mbs < 1:
            raise ValueError(f"chunk_mbs_config.chunk_mbs must be >= 1, got {self.chunk_mbs}.")


@dataclass
class MixedPrecisionConfig:
    """model.accelerator.fsdp_config.mixed_precision.* — Mixed precision settings."""

    enable: bool = field(
        default=True,
        metadata={"help": "Enable mixed precision training."},
    )
    param_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Dtype for the unsharded parameter."},
    )
    reduce_dtype: str = field(
        default="float32",
        metadata={"help": "Dtype for gradient reduction (i.e. reduce-scatter or all-reduce)."},
    )
    output_dtype: str = field(
        default=None,
        metadata={"help": "Dtype for casting floating-point forward outputs (FSDP2)."},
    )
    cast_forward_inputs: bool = field(
        default=True,
        metadata={"help": "Enable mixed precision cast forward inputs (FSDP2)."},
    )

    def __post_init__(self):
        def _check_dtype(dtype: str):
            if dtype is not None and dtype not in ["bfloat16", "float32", "float16"]:
                raise ValueError(f"Invalid dtype {dtype} for mixed precision training.")

        _check_dtype(self.param_dtype)
        _check_dtype(self.reduce_dtype)
        _check_dtype(self.output_dtype)


@dataclass
class FSDPConfig:
    """model.accelerator.fsdp_config.* — FSDP sharding configuration."""

    # eager mode use HF.from_pretrained(..., device_map='auto') for inference
    fsdp_mode: Literal["ddp", "fsdp2", "eager"] = field(
        default="fsdp2",
        metadata={
            "help": (
                "Data parallel mode. 'eager' skips every wrapper for the single-process "
                "inference path an OmniModule takes via _init_eager_inference."
            )
        },
    )
    reshard_after_forward: bool = field(
        default=True,
        metadata={"help": "Enable reshard after forward for FSDP2."},
    )
    reshard_after_backward: bool = field(
        default=True,
        metadata={"help": "Enable reshard after backward for FSDP2."},
    )
    forward_prefetch: bool = field(
        default=True,
        metadata={"help": "Enable forward prefetch."},
    )
    offload: bool = field(
        default=False,
        metadata={"help": "Enable CPU offload for FSDP2."},
    )
    offload_pin_memory: bool = field(
        default=True,
        metadata={
            "help": "When FSDP2 CPU offload is enabled, pin (page-lock) the offloaded CPU param shards. Default True matches torch's CPUOffloadPolicy. Set False to keep shards in pageable memory — for very large models (e.g. MoE experts) pinning is charged as non-reclaimable Shmem and can OOM the memcg; False avoids that at the cost of slightly slower non-pinned H2D per layer."
        },
    )
    max_load_broadcast_size: float = field(
        default=20.0,
        metadata={
            "help": "Maximum size (in GB) of parameters broadcasted from rank 0 during loading weights (FSDP2). Parameters exceeding this threshold will be chunked according to the parallel plan before broadcasting."
        },
    )
    mixed_precision: MixedPrecisionConfig = field(default_factory=MixedPrecisionConfig)

    def __post_init__(self):
        if self.fsdp_mode not in ("ddp", "fsdp2", "eager"):
            raise ValueError(
                f"Unsupported fsdp_mode={self.fsdp_mode!r}. FSDP1 has been removed; "
                "switch to fsdp_mode='fsdp2' (with model.accelerator.init_device='meta'), "
                "'ddp', or 'eager'."
            )


@dataclass
class OffloadConfig:
    """model.accelerator.offload_config.* — Activation offload settings."""

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
class TorchCompileConfig:
    """train.torch_compile.* — Per-block torch.compile options."""

    enable: bool = field(
        default=False,
        metadata={"help": "Enable per-block torch.compile for supported FSDP2 text and VLM training."},
    )
    backend: Optional[str] = field(
        default="inductor",
        metadata={"help": "Backend passed to torch.compile."},
    )
    mode: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Mode passed to torch.compile. Leave as None to use the inductor default. "
                "'reduce-overhead' enables CUDA Graphs on the inductor backend and requires "
                "train.accelerator.fsdp_config.reshard_after_forward=False."
            )
        },
    )
    fullgraph: bool = field(
        default=True,
        metadata={"help": "Whether to pass fullgraph=True to torch.compile."},
    )
    dynamic: bool = field(
        default=False,
        metadata={"help": "Whether to pass dynamic=True to torch.compile."},
    )


@dataclass
class AcceleratorConfig:
    """model.accelerator.* — Parallelism and distributed-training topology.

    Per-module: an omni model builds one of these for each module, so every knob
    here is overridable per module.
    """

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
    init_device: Literal["cuda", "meta", "npu"] = field(
        default="meta",
        metadata={
            "help": "Device to initialize model weights. 1. `cuda`: Init parameters on GPU. 2. `meta`: Init parameters on meta (required for FSDP2). 3. `npu`: Init parameters on Ascend NPU."
        },
    )
    broadcast_model_weights_from_rank0: bool = field(
        default=True,
        metadata={
            "help": "When enabled, only rank0 reads model weights from HuggingFace safetensor from disk. Other ranks would receive weights through broadcast. This helps to avoid disk I/O bottleneck."
        },
    )
    ep_sharded_stream_load: bool = field(
        default=False,
        metadata={
            "help": "Opt-in fast/low-memory weight loader for large MoE checkpoints: each rank reads only its ExtraParallel dim-0 slice of the expert tensors straight from the checkpoint. Requires the every-rank-reads path (`broadcast_model_weights_from_rank0=False`). A model with no ExtraParallel parallel_plan logs and falls back to the standard loader, so a heterogeneous OmniModel can enable this for its MoE module alone; a model that has a plan but an unsupported checkpoint layout raises `NotImplementedError`."
        },
    )
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    offload_config: OffloadConfig = field(default_factory=OffloadConfig)
    gradient_checkpointing: GradientCheckpointingConfig = field(default_factory=GradientCheckpointingConfig)
    torch_compile: TorchCompileConfig = field(default_factory=TorchCompileConfig)
    chunk_mbs_config: ChunkMBSConfig = field(default_factory=ChunkMBSConfig)

    def __post_init__(self):
        # although expert parallel and extra parallel are both provided in the arguments,
        # the implementation is configuring extra parallelism to include expert parallelism.
        # Guarded so re-instantiating from a saved config (asdict round-trip) does not
        # append a second "ep" dimension.
        if "ep" not in self.extra_parallel_names:
            self.extra_parallel_sizes.append(self.ep_size)
            self.extra_parallel_names.append("ep")
            self.extra_parallel_placement_innermost.append(self.ep_outside)

        # world_size and dp_size are plain attributes rather than fields: an omni model
        # builds one of these per module, and a field would let asdict() round-trip a
        # value derived under a different WORLD_SIZE back in as if the user had set it.
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self._resolve_topology()
        self._validate_init_device()

    def _resolve_topology(self):
        non_dp_size = self.pp_size * self.ulysses_size * self.cp_size * self.tp_size
        if self.world_size % non_dp_size != 0:
            raise ValueError(
                f"World size should be a multiple of pp_size: {self.pp_size}, "
                f"ulysses_size: {self.ulysses_size}, cp_size: {self.cp_size}, "
                f"tp_size: {self.tp_size}."
            )
        assert self.tp_size == 1, "Tensor parallel size not supported yet."
        assert self.pp_size == 1, "Pipeline parallel size not supported yet."
        assert self.cp_size == 1, "Context parallel size not supported yet."

        self.dp_size = self.world_size // non_dp_size

        # dp_replicate_size / dp_shard_size stay fields so HSDP is configurable, and
        # are resolved in place: -1 means "derive me".
        if self.dp_replicate_size > 0 and self.dp_shard_size > 0:
            assert self.dp_size == self.dp_replicate_size * self.dp_shard_size, (
                f"dp_size should be equal to dp_replicate_size: {self.dp_replicate_size} "
                f"* dp_shard_size: {self.dp_shard_size}."
            )
        elif self.dp_replicate_size > 0:
            if self.dp_size % self.dp_replicate_size != 0:
                raise ValueError("dp_size should be a multiple of dp_replicate_size.")
            self.dp_shard_size = self.dp_size // self.dp_replicate_size
        elif self.dp_shard_size > 0:
            if self.dp_size % self.dp_shard_size != 0:
                raise ValueError("dp_size should be a multiple of dp_shard_size.")
            self.dp_replicate_size = self.dp_size // self.dp_shard_size
        else:
            self.dp_replicate_size = 1
            self.dp_shard_size = self.dp_size

    def _validate_init_device(self):
        if self.fsdp_config.fsdp_mode == "fsdp2":
            assert self.init_device == "meta", "Please use model.accelerator.init_device: meta for FSDP2 training"
        else:
            # DDP wraps with ``device_ids=[local_rank]``, which torch refuses for a
            # CPU-resident module, and only rank0 would hold weights anyway. Fail
            # here so every rank stops at parse time, rather than let rank0 die in
            # DDP's constructor while the others block in its first collective.
            assert self.init_device != "cpu", (
                "model.accelerator.init_device: cpu is not supported with fsdp_mode: ddp. "
                "Use meta or an accelerator device."
            )

        # ep_sharded_stream_load only runs on the every-rank-reads path, so it is
        # mutually exclusive with broadcast_model_weights_from_rank0. Fail early
        # instead of silently ignoring the flag.
        assert not (self.ep_sharded_stream_load and self.broadcast_model_weights_from_rank0), (
            "model.accelerator.ep_sharded_stream_load requires "
            "model.accelerator.broadcast_model_weights_from_rank0=False "
            "(it reads each rank's ExtraParallel slice directly and cannot run on the broadcast path)."
        )

        # `world_size`/`dp_size`/`dp_shard_size`/`dp_replicate_size` are process-global,
        # runtime-derived state, not CLI/YAML-configurable — so (like `world_size` on
        # `TrainingArguments`) they're plain instance attributes, not dataclass fields.
        # Deriving them here (instead of via an external `_validate_accelerator` call
        # from the owning `TrainingArguments`) makes every `AcceleratorConfig` instance
        # self-resolving regardless of *how* or *where* it's constructed: the top-level
        # `train.accelerator`, or a per-OmniModule accelerator rebuilt from scratch by
        # `build_module_runtime_args` / `build_module_args` (each of those calls
        # `AcceleratorConfig(**field_values)` fresh, dropping any externally-computed
        # `dp_size` since it was never a field to begin with).
        self.world_size = int(os.getenv("WORLD_SIZE", 1))

        if self.world_size % (self.pp_size * self.ulysses_size * self.cp_size * self.tp_size) != 0:
            raise ValueError(
                f"World size should be a multiple of pp_size: {self.pp_size}, "
                f"ulysses_size: {self.ulysses_size}, cp_size: {self.cp_size}, "
                f"tp_size: {self.tp_size}."
            )
        assert self.tp_size == 1, "Tensor parallel size not supported yet."
        assert self.pp_size == 1, "Pipeline parallel size not supported yet."
        assert self.cp_size == 1, "Context parallel size not supported yet."

        self.dp_size = self.world_size // (self.pp_size * self.ulysses_size * self.cp_size * self.tp_size)

        if self.dp_replicate_size > 0 and self.dp_shard_size > 0:
            assert self.dp_size == self.dp_replicate_size * self.dp_shard_size, (
                f"dp_size should be equal to dp_replicate_size: {self.dp_replicate_size} "
                f"* dp_shard_size: {self.dp_shard_size}."
            )
        elif self.dp_replicate_size > 0:
            if self.dp_size % self.dp_replicate_size != 0:
                raise ValueError("dp_size should be a multiple of dp_replicate_size.")
            self.dp_shard_size = self.dp_size // self.dp_replicate_size
        elif self.dp_shard_size > 0:
            if self.dp_size % self.dp_shard_size != 0:
                raise ValueError("dp_size should be a multiple of dp_shard_size.")
            self.dp_replicate_size = self.dp_size // self.dp_shard_size
        else:
            self.dp_replicate_size = 1
            self.dp_shard_size = self.dp_size


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
    dcp_save_to_lowest_rank: bool = field(
        default=False,
        metadata={
            "help": (
                "Route each replicated DCP shard to the lowest global rank that holds it, instead "
                "of load-balancing writes across all replica holders. Useful on a non-shared "
                "filesystem (each node writes to local disk): it concentrates the deduplicated copy "
                "onto the lowest-ranked replica group instead of scattering writes across all "
                "replicas. In the standard HSDP layout (FSDP shard within a node, replication "
                "across nodes) that lowest replica group lives on one node, so that node holds a "
                "complete checkpoint. Only affects replicated data (the FSDP/HSDP replicate dim); "
                "unique expert/tensor/pipeline-parallel shards are never deduplicated and stay "
                "distributed. Trades write parallelism for locality, so leave False when output_dir "
                "is shared."
            )
        },
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
    dyn_bsz_runtime: Literal["main", "worker"] = field(
        default="main",
        metadata={"help": "Which process dynamic batching runs in: main process or DataLoader worker."},
    )
    dyn_bsz_count_mode: Literal["total", "effective"] = field(
        default="total",
        metadata={
            "help": (
                "How dynamic batching counts tokens when packing a micro batch. "
                "'total' (default, legacy) sums attention_mask; 'effective' sums "
                "only loss-contributing tokens (labels != IGNORE_INDEX), which "
                "balances effective tokens across DP ranks at the cost of allowing "
                "controlled physical-token overflow."
            )
        },
    )
    dyn_bsz_physical_overflow_ratio: float = field(
        default=1.5,
        metadata={
            "help": (
                "Physical-token cap multiplier used when dyn_bsz_count_mode='effective'. "
                "The cap is ceil(micro_batch_size * max_seq_len * ratio), so values "
                "> 1.0 let effective-token batching differ from total-token batching "
                "while still bounding prompt-heavy micro batches."
            )
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
    sync_each_train_step: bool = field(
        default=True,
        metadata={
            "help": (
                "Synchronize the accelerator before each training step's forward/backward work. "
                "Disable to allow asynchronous dataloader and H2D work to overlap with the next step."
            )
        },
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
    max_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Max training steps per epoch. (for debug)"},
    )
    moe_load_balance_monitor_interval: int = field(
        default=0,
        metadata={
            "help": (
                "Log MoE expert load heatmap every N steps. 0 = disabled. Counts are "
                "all-reduced across EP and DP groups so the heatmap is global. "
                "Wandb logging is performed only when train.wandb.enable=True."
            )
        },
    )

    # sub-argument groups
    wandb: WandbConfig = field(default_factory=WandbConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    channel_loss: ChannelLossConfig = field(default_factory=ChannelLossConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def __post_init__(self):
        if self.dyn_bsz_physical_overflow_ratio < 1.0:
            raise ValueError(
                f"dyn_bsz_physical_overflow_ratio must be >= 1.0, got {self.dyn_bsz_physical_overflow_ratio}."
            )

        self._train_steps = -1
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.global_rank = int(os.getenv("RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))

        self._warn_multi_node()
        self._resolve_checkpoint_paths()
        self._resolve_profile()

    # -- validation & derivation helpers (called by __post_init__) -----------------------

    def _warn_multi_node(self):
        num_nodes = self.world_size // int(os.getenv("LOCAL_WORLD_SIZE", 1))
        if num_nodes > 1:
            logger.warning_rank0(
                f"Detected {num_nodes} nodes. "
                "Make sure that `train.checkpoint.output_dir` is shared by all nodes. "
                "Otherwise, each node will save checkpoints to its local directory, which may cause inconsistencies or job failures."
            )

    def _derive_batch_config(self, accelerator: AcceleratorConfig):
        """Derive batch/accumulation sizes from the model's accelerator topology.

        Takes the accelerator as an argument rather than reading it off ``self``:
        it lives on ``model`` now, so only the root config can pair the two.
        """
        acc = accelerator

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

        # dataloader batch size
        if self.dyn_bsz:
            if self.dyn_bsz_runtime == "main":
                self.dataloader_batch_size = 1
            else:
                self.dataloader_batch_size = self.global_batch_size // acc.dp_size // self.micro_batch_size
        else:
            self.dataloader_batch_size = self.global_batch_size // acc.dp_size  # = micro bsz * grad accu

    def _resolve_checkpoint_paths(self):
        ckpt = self.checkpoint

        if ckpt.load_path == "auto":
            from ..utils.checkpoint_utils import get_checkpoint_path

            ckpt.load_path = get_checkpoint_path(
                output_dir=ckpt.output_dir,
                is_local_rank0=self.local_rank == 0,
                ckpt_manager=ckpt.manager,
            )

        if ckpt.load_path:
            load_path = Path(os.path.normpath(os.path.abspath(ckpt.load_path)))
            output_dir = Path(os.path.normpath(os.path.abspath(ckpt.output_dir)))

            try:
                load_path.relative_to(output_dir)
            except ValueError:
                logger.warning("load_checkpoint_path should be under output_dir.")

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


# NPU compatibility tables for ``_validate_implementations``. ``"eager"`` is
# always allowed implicitly. A value not in ``_NPU_ALLOWED[field]`` raises on
# NPU; a value in ``_NPU_REQUIRED[field]`` raises off NPU.
#
# Hardcoded (not inferred from ``BackendSpec.requires``) because backend names
# alone do not capture per-model and per-hardware compatibility. The NPU
# default-normalization step runs before this allow-list validation.
_NPU_ALLOWED: Dict[str, frozenset] = {
    "rms_norm_implementation": frozenset({"npu"}),
    "rotary_pos_emb_implementation": frozenset({"npu"}),
    "rotary_pos_emb_vision_implementation": frozenset({"npu"}),
    "swiglu_mlp_implementation": frozenset(),
    "load_balancing_loss_implementation": frozenset({"triton"}),
    "cross_entropy_loss_implementation": frozenset({"chunk_loss", "npu"}),
    "moe_implementation": frozenset({"fused_npu"}),
}

_NPU_REQUIRED: Dict[str, frozenset] = {
    "rms_norm_implementation": frozenset({"npu"}),
    "rotary_pos_emb_implementation": frozenset({"npu"}),
    "rotary_pos_emb_vision_implementation": frozenset({"npu"}),
    "cross_entropy_loss_implementation": frozenset({"npu"}),
    "moe_implementation": frozenset({"fused_npu"}),
}

_NPU_DEFAULT_FALLBACK: Dict[str, str] = {
    "rms_norm_implementation": "npu",
    "rotary_pos_emb_implementation": "npu",
    "rotary_pos_emb_vision_implementation": "npu",
    "swiglu_mlp_implementation": "eager",
    "load_balancing_loss_implementation": "eager",
    "cross_entropy_loss_implementation": "npu",
    "moe_implementation": "fused_npu",
}


@dataclass
class OpsImplementationConfig:
    """model.ops_implementation.* — kernel backend selection per op.

    Defaults are GPU-optimal (Liger / Triton / fused_triton). On NPU, values
    still equal to the dataclass defaults listed in ``_NPU_DEFAULT_FALLBACK``
    are automatically mapped to NPU-compatible or eager implementations;
    explicit non-default overrides are validated and unsupported values raise.
    Per-op fields are ``str`` so third-party backends can register without
    changing this dataclass.

    NPU validation runs at two times:

    - **Config-parse time** (``__post_init__``) for ops registered in the
      legacy per-model registry: ``rms_norm``, ``rotary_pos_emb``,
      ``rotary_pos_emb_vision``, ``swiglu_mlp``, ``load_balancing_loss``, plus
      ``cross_entropy_loss`` and ``moe``. Errors fire immediately with a
      model-agnostic allow-list.
    - **Model-build time** (``OpSlot.bind`` via ``KERNEL_REGISTRY.resolve``)
      for Qwen3.5-only ops: ``rms_norm_gated``, ``causal_conv1d``,
      ``chunk_gated_delta_rule``. These OpSlots only exist in Qwen3.5's
      patched modeling module, so config-parse-time validation would force
      every NPU user to override them even when training non-Qwen3.5 models.
      All three ship both a GPU (``fla``) and an NPU (``npu``) backend; the
      kernel's ``HardwareRequirement`` raises only when the requested value has
      no backend for the current hardware. The varlen (``dyn_bsz=True``) caveat
      is documented in the field metadata.

    Backends: ``"eager"`` (HF reference, always available),
    ``"liger_kernel"`` (GPU, needs ``liger-kernel``), ``"npu"`` (Ascend),
    ``"triton"`` (CUDA ``triton``). Load-balancing loss has a CUDA Triton
    backend; on NPU, values equal to the dataclass default are normalized to
    ``"eager"`` before registry binding.
    """

    attn_implementation: Optional[
        Literal[
            "eager",
            "sdpa",
            "flash_attention_2",
            "flash_attention_3",
            "flash_attention_4",
            "flex_attention",
            "magi_attention",
            "native-sparse",
        ]
    ] = field(
        default="flash_attention_2",
        metadata={"help": "Attention implementation."},
    )
    moe_implementation: str = field(
        default="fused_triton",
        metadata={
            "help": "MoE experts forward. 'fused_triton' (default, GPU SM70+) | "
            "'fused_quack' (GPU SM90+) | 'fused_npu' (NPU) | 'eager'. "
            "On NPU, a default-valued 'fused_triton' selection maps to 'fused_npu'; "
            "incompatible non-default overrides raise. Legacy 'fused' "
            "auto-resolves to fused_quack/fused_npu with a deprecation warning."
        },
    )
    cross_entropy_loss_implementation: str = field(
        default="liger_kernel",
        metadata={
            "help": "Cross-entropy loss. 'liger_kernel' (default, GPU; fused linear+CE — "
            "requires VeOmni-patched modeling that passes hidden_states=/weights= to "
            "self.loss_function; unpatched HF models raise) | 'chunk_loss' (chunked "
            "F.linear+CE, hardware-agnostic) | 'npu' (chunk_loss + torch_npu gate) | "
            "'eager' (PyTorch F.cross_entropy)."
        },
    )
    rms_norm_implementation: str = field(
        default="liger_kernel",
        metadata={
            "help": "RMSNorm. 'liger_kernel' (default, GPU) | 'npu' | "
            "'triton' (DeepSeek-V3 batch-invariant; GPU only) | 'eager'."
        },
    )
    swiglu_mlp_implementation: str = field(
        default="liger_kernel",
        metadata={
            "help": "SwiGLU MLP. 'liger_kernel' (default, GPU) | 'eager'. No NPU backend — NPU users must set 'eager'."
        },
    )
    rotary_pos_emb_implementation: str = field(
        default="liger_kernel",
        metadata={
            "help": "Rotary positional embedding. 'liger_kernel' (default, GPU) | "
            "'npu' | 'triton' (DeepSeek-V3 deterministic; GPU only) | 'eager'."
        },
    )
    rotary_pos_emb_vision_implementation: str = field(
        default="eager",
        metadata={"help": "Rotary positional embedding in vision part. 'npu' | 'eager' (default)."},
    )
    load_balancing_loss_implementation: str = field(
        default="triton",
        metadata={
            "help": "MoE load-balancing loss. 'triton' (default; needs 'triton' on CUDA) | 'eager'. "
            "On NPU, config normalization maps the default 'triton' value to 'eager'."
        },
    )
    rms_norm_gated_implementation: str = field(
        default="fla",
        metadata={
            "help": "Gated RMSNorm implementation (Qwen3.5 GatedDeltaNet `self.norm`). "
            "'fla' (default) uses fla.modules.FusedRMSNormGated (requires flash-linear-attention, GPU). "
            "'eager' uses the HuggingFace Qwen3_5RMSNormGated. "
            "'npu' uses the VeOmni NPUFusedRMSNormGated."
        },
    )
    causal_conv1d_implementation: str = field(
        default="fla",
        metadata={
            "help": "Varlen depthwise causal conv1d implementation (Qwen3.5 GatedDeltaNet pre-mixer). "
            "'fla' (default) uses fla.modules.convolution.causal_conv1d (requires flash-linear-attention, GPU). "
            "'eager' leaves causal_conv1d_fn unset; the varlen training path then raises "
            "because no torch fallback handles cu_seqlens. "
            "'npu' uses the vendored Triton kernel (requires triton-ascend, NPU). "
            "Only affects varlen (dyn_bsz) training; a non-eager value on hardware without a "
            "matching backend raises at OpSlot bind time."
        },
    )
    chunk_gated_delta_rule_implementation: str = field(
        default="fla",
        metadata={
            "help": "Chunk gated delta-rule kernel for Qwen3.5 linear attention. "
            "'fla' (default) uses fla.ops.gated_delta_rule.chunk_gated_delta_rule (requires flash-linear-attention, GPU). "
            "'flash_qla' uses QwenLM FlashQLA (ships under the gpu extra, Hopper SM90 only — "
            "no Ampere/Ada below or Blackwell above; SM10x wheels are WIP upstream). "
            "'eager' uses transformers' torch_chunk_gated_delta_rule, which does NOT support "
            "cu_seqlens; varlen training therefore raises at runtime. "
            "'npu' uses the vendored Triton kernel (requires triton-ascend, NPU). "
            "'npu_ascendc' uses the AscendC fused ops (requires fla_npu + triton-ascend, NPU; "
            "delegates heavy GDN compute to torch.ops.npu.*). "
            "A non-eager value on hardware without a matching backend raises at OpSlot bind time."
        },
    )
    dsa_indexer_implementation: Literal["eager", "cudnn", "tilelang"] = field(
        default="eager",
        metadata={"help": "DeepSeek sparse attention top-k indexer implementation: 'eager', 'cudnn', or 'tilelang'."},
    )
    dsa_attention_implementation: Literal["eager", "flashmla_cudnn", "tilelang"] = field(
        default="eager",
        metadata={"help": "DeepSeek sparse attention implementation: 'eager', 'flashmla_cudnn', or 'tilelang'."},
    )
    mhc_implementation: Literal["eager", "tilelang"] = field(
        default="eager",
        metadata={
            "help": "Manifold-constrained Hyper-Connection implementation. 'tilelang' enables the "
            "DeepSeek V4 TileKernels forward/backward path on NVIDIA SM90+; 'eager' uses PyTorch."
        },
    )

    def __post_init__(self):
        if get_env("MODELING_BACKEND") == "veomni":
            replacements = {
                "flash_attention_2": "veomni_flash_attention_2_with_sp",
                "flash_attention_3": "veomni_flash_attention_3_with_sp",
                "flash_attention_4": "veomni_flash_attention_4_with_sp",
                "flex_attention": "veomni_flex_attention_with_sp",
                "magi_attention": "veomni_magi_attention_with_sp",
            }
            if self.attn_implementation in replacements:
                new_impl = replacements[self.attn_implementation]
                logger.info_rank0(f"Replacing attn_implementation from '{self.attn_implementation}' to '{new_impl}'")
                self.attn_implementation = new_impl

        # Legacy alias: ``moe_implementation='fused'`` resolves to a
        # hardware-appropriate fused kernel — Quack on GPU, NPU group-gemm on
        # Ascend. Kept for back-compat with pre-#678 YAMLs; warn so users
        # migrate to the explicit name.
        if self.moe_implementation == "fused":
            from ..utils.import_utils import is_torch_npu_available

            resolved = "fused_npu" if is_torch_npu_available() else "fused_quack"
            logger.warning_rank0(
                f"moe_implementation='fused' is a deprecated alias; resolving to '{resolved}' on this host. "
                f"Set moe_implementation='{resolved}' explicitly to silence this warning."
            )
            self.moe_implementation = resolved

        self._apply_npu_default_fallback()
        self._validate_implementations()

    def _apply_npu_default_fallback(self):
        """Auto-resolve GPU-only defaults to NPU-compatible alternatives.

        When running on NPU, fields still at their GPU default are automatically
        swapped to the NPU fallback from ``_NPU_DEFAULT_FALLBACK``. Explicit
        user overrides (non-default values) are left untouched and will be
        caught by ``_validate_implementations`` if unsupported.
        """
        from ..utils.import_utils import is_torch_npu_available

        if not is_torch_npu_available():
            return

        gpu_defaults = {f.name: f.default for f in fields(self) if f.default is not MISSING}
        for field_name, npu_value in _NPU_DEFAULT_FALLBACK.items():
            if field_name not in gpu_defaults:
                continue
            current = getattr(self, field_name)
            if current == gpu_defaults[field_name]:
                setattr(self, field_name, npu_value)
                logger.info_rank0(
                    f"{field_name}: auto-resolved GPU default {current!r} -> {npu_value!r} on Ascend NPU."
                )

    def _validate_implementations(self):
        """Fail fast on hardware/op mismatch at config-parse time.

        Only checks things cheaper to catch here than at bind time. Package
        availability (liger / torch_npu) and per-model backend compatibility
        are validated by the resolution sites (``apply_per_model_patches`` /
        ``apply_global_ops`` / ``install_loss_mapping`` /
        ``KERNEL_REGISTRY.resolve``) — not duplicated here.
        """
        from ..ops import config as _ops_config_pkg  # noqa: F401  triggers op registrations
        from ..ops.config.registry import list_ops
        from ..utils.import_utils import is_package_available, is_torch_npu_available

        # Coverage check: every registered op must appear in ``_NPU_ALLOWED``,
        # otherwise a future op addition silently bypasses NPU validation.
        registered_fields = {op.config_field for op in list_ops()}
        missing = registered_fields - _NPU_ALLOWED.keys()
        assert not missing, (
            f"NPU allow-list missing entries for registered ops: {sorted(missing)}. "
            f"Add them to _NPU_ALLOWED in arguments_types.py."
        )

        on_npu = is_torch_npu_available()

        for field_name, npu_ok in _NPU_ALLOWED.items():
            value = getattr(self, field_name)
            if value == "eager":
                continue
            if on_npu and value not in npu_ok:
                allowed = sorted(npu_ok | {"eager"})
                raise ValueError(
                    f"{field_name}={value!r} is not supported on Ascend NPU. "
                    f"Set to one of {allowed}; 'eager' is the universal fallback "
                    f"for ops with no NPU kernel for the current model."
                )
            if not on_npu and value in _NPU_REQUIRED.get(field_name, frozenset()):
                raise ValueError(f"{field_name}={value!r} requires Ascend NPU but none is available.")

        # The Triton load-balancing-loss kernel imports ``triton`` at module
        # top — surface a missing package here with an actionable message
        # instead of a noisy ImportError at apply_global_ops time.
        if self.load_balancing_loss_implementation == "triton" and not is_package_available("triton"):
            raise ValueError(
                "load_balancing_loss_implementation='triton' requires the 'triton' package "
                "on CUDA. Install it or set the field to 'eager'."
            )


@dataclass
class BaseModelArguments:
    """Model fields shared by every trainable unit, whole model or single module.

    Deliberately excludes the config/tokenizer/index paths: an omni module is
    addressed by its subfolder inside a composed checkpoint and never carries
    its own tokenizer, so those belong on :class:`ModelArguments` alone.
    """

    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path/HDFS path to the pre-trained model. If unspecified, use random init."},
    )
    model_config: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Config to overwrite foundation model config."},
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

    _fqn_to_index_mapping_cache: ClassVar[Dict[str, Optional[Dict[str, int]]]] = {}

    def __post_init__(self):
        # Localize here rather than in each owner: every subclass needs a
        # ``model_path`` that exists on disk before any loader touches it, and a
        # composed model resolves its module subfolders against this root.
        self.model_path = _resolve_hdfs_path(self.model_path)

    def _safetensor_idx_path(self) -> Optional[str]:
        """Where to read the HF ``weight_map`` from. Overridden to allow an explicit path."""
        if self.model_path is None:
            return None
        return os.path.join(self.model_path, "model.safetensors.index.json")

    @property
    def fqn_to_index_mapping(self) -> Optional[Dict[str, int]]:
        """Raw HF ``weight_map`` from the safetensor index (MoE key renames happen at runtime).

        Resolved lazily and cached per index path: a composed model builds one of
        these per module and re-instantiates them on every merge, and sibling
        modules routinely share a checkpoint, so parsing eagerly would re-read the
        same index file several times per job.
        """
        idx_path = self._safetensor_idx_path()
        if idx_path is None:
            self._warn_unsharded()
            return None

        cache = BaseModelArguments._fqn_to_index_mapping_cache
        if idx_path not in cache:
            if os.path.exists(idx_path):
                from ..models.checkpoint_tensor_loading import parse_fqn_to_index_mapping_from_json

                cache[idx_path] = parse_fqn_to_index_mapping_from_json(idx_path)
            else:
                cache[idx_path] = None

        mapping = cache[idx_path]
        if mapping is None:
            self._warn_unsharded()
        return mapping

    @staticmethod
    def _warn_unsharded() -> None:
        logger.warning_once("fqn_to_index_mapping is None, saved safetensor will be a single file instead of sharded.")


@dataclass
class ModelRuntimeArguments(BaseModelArguments):
    """Everything one training unit needs: model fields + its own accelerator and optimizer.

    This is the shape a per-module runtime consumes. An omni model builds one of
    these per module and merges it over the model-level defaults, which is why
    the pair is declared here rather than on :class:`ModelArguments`.
    """

    accelerator: AcceleratorConfig = field(default_factory=AcceleratorConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass
class ModelArguments(ModelRuntimeArguments):
    """model.* — One composed model, plus the paths its loaders resolve from."""

    config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path/HDFS path to the model config. Defaults to `model_path`."},
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

    def __post_init__(self):
        if self.config_path is None and self.model_path is None:
            raise ValueError("`config_path` must be specified when `model_path` is None.")

        # Download HDFS-hosted paths to a local cache before resolving defaults so
        # that all downstream loaders (config/tokenizer/safetensors) see local paths.
        super().__post_init__()
        self.config_path = _resolve_hdfs_path(self.config_path)
        self.tokenizer_path = _resolve_hdfs_path(self.tokenizer_path)

        if self.config_path is None:
            self.config_path = self.model_path

        if self.tokenizer_path is None:
            self.tokenizer_path = self.config_path

    def _safetensor_idx_path(self) -> Optional[str]:
        """Honour an explicit index path, else fall back to the one under ``model_path``."""
        return self.safetensor_idx_path or super()._safetensor_idx_path()


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
    worker_num_threads: Optional[int] = field(
        default=None,
        metadata={"help": "Per-worker torch thread count for dataloader subprocesses."},
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "Number of batches loaded in advance by each worker."},
    )
    persistent_workers: bool = field(
        default=False,
        metadata={"help": "Keep DataLoader worker processes alive between iterator recreations."},
    )
    in_order: bool = field(
        default=True,
        metadata={"help": "Return worker-loaded batches in first-in, first-out order."},
    )
    drop_last: bool = field(
        default=True,
        metadata={"help": "Whether to drop the last incomplete batch."},
    )
    pin_memory: bool = field(
        default=True,
        metadata={"help": "Whether to pin memory for dataloader."},
    )
    use_background_prefetcher: bool = field(
        default=False,
        metadata={"help": "Whether to use BackgroundPrefetcher for dataloader."},
    )


@dataclass
class DataArguments:
    """data.* — Dataset paths, tokenization, and batching."""

    supports_torch_compile = True

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
    data_type: Literal[
        "plaintext",
        "conversation",
        "diffusion",
        "classification",
        "dpo",
        "seedomni",
        "seedomni_cached",
    ] = field(default="conversation", metadata={"help": "Type of the training data."})
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
            elif self.data_type in {"seedomni", "seedomni_cached"}:
                # SeedOmni V2 modules own their own tokenization; the transform
                # reads ``conversations`` / ``images`` columns directly and
                # cached rows carry ``conversation_list`` directly. Leave it unset.
                pass
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
        self.train._derive_batch_config(self.model.accelerator)

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

        accelerator = self.model.accelerator
        if accelerator.chunk_mbs_config.enable:
            if self.train.pad_to_length:
                raise ValueError(
                    "model.accelerator.chunk_mbs_config.enable is not supported with train.pad_to_length yet."
                )
            if accelerator.gradient_checkpointing.enable and accelerator.gradient_checkpointing.enable_reentrant:
                raise ValueError(
                    "model.accelerator.chunk_mbs_config.enable requires non-reentrant gradient checkpointing. "
                    "Set model.accelerator.gradient_checkpointing.enable_reentrant=False."
                )
            if self.data.data_type == "dpo":
                raise ValueError("model.accelerator.chunk_mbs_config.enable is not supported by the DPO trainer yet.")

        if accelerator.torch_compile.enable:
            if accelerator.chunk_mbs_config.enable:
                raise ValueError(
                    "model.accelerator.chunk_mbs_config.enable is not supported with "
                    "model.accelerator.torch_compile.enable yet. "
                    "ChunkMBS wraps decoder forwards with per-batch chunk ranges before decoder blocks are compiled."
                )
            if not getattr(self.data, "supports_torch_compile", True):
                raise ValueError(
                    "model.accelerator.torch_compile.enable is not supported by this data pipeline. "
                    "The pipeline must implement pad_to_length for static packed shapes."
                )
            if self.data.data_type not in ("plaintext", "conversation", "classification", "dpo"):
                raise ValueError(
                    "model.accelerator.torch_compile.enable currently supports packed language-model data types only; "
                    f"got data.data_type={self.data.data_type!r}."
                )
            if not self.train.dyn_bsz or not self.train.pad_to_length:
                raise ValueError(
                    "model.accelerator.torch_compile.enable requires train.dyn_bsz=True and "
                    "train.pad_to_length=True. "
                    "Variable packed lengths trigger recompilation and prevent stable CUDA Graph replay when enabled; "
                    "see https://github.com/ByteDance-Seed/VeOmni/issues/401."
                )

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
        self.model_path = _resolve_hdfs_path(self.model_path)
        self.tokenizer_path = _resolve_hdfs_path(self.tokenizer_path)
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path
