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

"""Launcher argument schema for SeedOmni V2 training + inference.

Standalone from the V1 ``VeOmniArguments`` hierarchy.  A single ``base.yaml``
drives both :class:`~veomni.trainer.omni.omni_trainer.OmniTrainer` and
:class:`~veomni.trainer.omni.omni_inferencer.OmniInferencer`.

Omni-specific layout:

* ``model`` is an :class:`~veomni.omni_arguments.model_runtime.OmniModelRuntimeArguments`
  block — ``model_path``, ``model_config``, ``ops_implementation``, ``accelerator``,
  and ``optimizer``.
* Per-module overrides live in ``model.model_config.modules`` YAML;
  :meth:`OmniArguments.resolve_model` merges them into
  :attr:`~veomni.omni_arguments.model_runtime.OmniModelRuntimeArguments.modules`
  (each entry is :class:`OmniModuleRuntimeArguments`: same flat fields).
* ``data`` / ``train`` / ``infer`` remain launcher-wide.
"""

from __future__ import annotations

import math
import os
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from ..arguments.arguments_types import (
    AcceleratorConfig,
    ChannelLossConfig,
    CheckpointConfig,
    ChunkMBSConfig,
    DataloaderConfig,
    GradientCheckpointingConfig,
    OpsImplementationConfig,
    OptimizerConfig,
    ProfileConfig,
    TorchCompileConfig,
    WandbConfig,
)
from ..utils import logging


logger = logging.get_logger(__name__)

OMNI_TRAIN_WORKFLOWS = {"train", "offline_cache", "train_with_cache", "train_and_cache"}
LAUNCHER_CONFIG_KEYS = frozenset({"modules", "train_graph", "train_type", "infer_graph", "infer_type"})


def _hf_module_model_config(model_config: dict | None) -> dict:
    """Drop launcher layout keys before merging or exporting per-module ``model_config``."""
    if not model_config:
        return {}
    return {key: value for key, value in model_config.items() if key not in LAUNCHER_CONFIG_KEYS}


@dataclass
class BaseOmniModelArguments:
    """Shared model fields merged into every :class:`OmniModuleRuntimeArguments`."""

    model_path: str | None = field(
        default=None,
        metadata={"help": "Local path/HDFS path to the pre-trained model. If unspecified, use random init."},
    )
    model_config: dict | None = field(
        default_factory=dict,
        metadata={"help": "HF config overrides for the foundation model."},
    )
    basic_modules: list[str] | None = field(
        default_factory=list,
        metadata={"help": "Basic modules beyond model._no_split_modules to be sharded in FSDP."},
    )
    lora_config: dict | None = field(
        default_factory=dict,
        metadata={"help": "Config for lora."},
    )
    ops_implementation: OpsImplementationConfig = field(default_factory=OpsImplementationConfig)


@dataclass
class OmniModuleRuntimeArguments(BaseOmniModelArguments):
    """Per-module runtime — flat model fields + ``accelerator`` + ``optimizer``."""

    _fqn_to_index_mapping_cache: ClassVar[dict[str, dict[str, int] | None]] = {}

    accelerator: AcceleratorConfig = field(default_factory=AcceleratorConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    def _resolve_fqn_to_index_mapping(self) -> dict[str, int] | None:
        """Parse HF ``weight_map`` from ``{model_path}/model.safetensors.index.json`` when present."""
        model_path = self.model_path
        if model_path is None:
            return None
        cache = type(self)._fqn_to_index_mapping_cache
        if model_path in cache:
            return cache[model_path]

        idx_path = os.path.join(model_path, "model.safetensors.index.json")
        if not os.path.exists(idx_path):
            cache[model_path] = None
            return None
        from ..models.checkpoint_tensor_loading import parse_fqn_to_index_mapping_from_json

        mapping = parse_fqn_to_index_mapping_from_json(idx_path)
        cache[model_path] = mapping
        return mapping

    @property
    def fqn_to_index_mapping(self) -> dict[str, int] | None:
        """Lazy parse of ``model_path/model.safetensors.index.json`` for HF sharded save/load."""
        return self._resolve_fqn_to_index_mapping()

    def to_hf_config(self, module_name: str) -> dict:
        """Project onto this module's slim :class:`OmniConfig` entry."""
        model_block: dict = {
            "ops_implementation": asdict(self.ops_implementation),
        }
        overrides = _hf_module_model_config(self.model_config)
        if overrides:
            model_block["model_config"] = deepcopy(overrides)
        return {
            "subfolder": module_name,
            "model": model_block,
        }


def _is_omni_checkpoint_root(path: str | None) -> bool:
    return bool(path) and os.path.isfile(os.path.join(str(path), "config.json"))


def _try_load_omni_checkpoint_config(path: str | None):
    if not _is_omni_checkpoint_root(path):
        return None
    from ..models.seed_omni.configuration_omni import OmniConfig

    return OmniConfig.from_pretrained(str(path))


@dataclass
class OmniGraphProfileArguments:
    """``train.graph_profile.*`` — SeedOmni graph profiler settings."""

    train_start_step: int = field(default=1, metadata={"help": "First step to save graph profiler records for."})
    train_end_step: int = field(default=2, metadata={"help": "Last step to save graph profiler records for."})
    enable_wall_time: bool = field(default=False, metadata={"help": "Append wall-clock timing to graph records."})
    enable_cuda_events: bool = field(default=False, metadata={"help": "Append CUDA event timing to graph records."})
    enable_memory: bool = field(default=False, metadata={"help": "Append peak device memory to graph records."})

    def enable_graph_profiling(self) -> bool:
        return self.enable_wall_time or self.enable_cuda_events or self.enable_memory


@dataclass
class OmniInferArguments:
    """``infer.*`` — per-call inference knobs (prompt, generation kwargs, output)."""

    generation_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Free-form generation kwargs passed to the generation graph."},
    )
    prompt: str = field(default="", metadata={"help": "User text prompt (required at generate time)."})
    images: list[str] = field(default_factory=list, metadata={"help": "Reference image paths / URLs."})
    output_dir: str = field(default="output", metadata={"help": "Root output directory."})
    seed: int = field(default=42, metadata={"help": "Random seed."})


@dataclass
class OmniDataArguments:
    """``data.*`` for OmniModel V2."""

    train_path: str = field(
        metadata={"help": "Local path/HDFS path of the training data. Use comma to separate multiple datasets."},
    )
    eval_path: str | None = field(
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
    silent_exception: bool = field(
        default=False,
        metadata={"help": "Whether to ignore exceptions when loading data. Defaults to ``False``"},
    )
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    mm_configs: dict | None = field(
        default_factory=dict,
        metadata={"help": "Config for multimodal input (forwarded to the seedomni data transform)."},
    )

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
                pass
            else:
                raise ValueError(f"Unknown data type: {self.data_type}")

        if self.dataloader.num_workers == 0:
            self.dataloader.prefetch_factor = None


@dataclass
class OmniTrainingArguments:
    """``train.*`` for OmniModel V2 — parallelism and optimizer live on ``model``."""

    dyn_bsz: bool = field(
        default=True,
        metadata={"help": "Enable dynamic batch size for padding-free training."},
    )
    micro_batch_size: int = field(
        default=1,
        metadata={"help": "Micro batch size. The number of samples per iteration on each device."},
    )
    global_batch_size: int | None = field(
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
    init_device: Literal["cpu", "cuda", "meta", "npu"] = field(
        default="meta",
        metadata={
            "help": "Device to initialize model weights. 1. `cpu`: Init parameters on CPU in rank0 only. 2. `cuda`: Init parameters on GPU. 3. `meta`: Init parameters on meta (required for FSDP2). 4. `npu`: Init parameters on Ascend NPU."
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
            "help": "Opt-in fast/low-memory weight loader for large MoE checkpoints: each rank reads only its ExtraParallel dim-0 slice of the expert tensors straight from the checkpoint. Requires the every-rank-reads path (`broadcast_model_weights_from_rank0=False`) and a model with an ExtraParallel parallel_plan; unsupported model/checkpoint combinations raise `NotImplementedError`."
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
    max_steps: int | None = field(
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
    train_type: str | None = field(default=None, metadata={"help": "SeedOmni V2 training workflow."})
    offline_cache_dir: str | None = field(
        default=None,
        metadata={"help": "Output directory for train_type='offline_cache'."},
    )
    graph_profile: OmniGraphProfileArguments = field(default_factory=OmniGraphProfileArguments)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    channel_loss: ChannelLossConfig = field(default_factory=ChannelLossConfig)
    gradient_checkpointing: GradientCheckpointingConfig = field(default_factory=GradientCheckpointingConfig)
    torch_compile: TorchCompileConfig = field(default_factory=TorchCompileConfig)
    chunk_mbs_config: ChunkMBSConfig = field(default_factory=ChunkMBSConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def __post_init__(self):
        self.train_type = self.train_type or "train"
        if self.train_type not in OMNI_TRAIN_WORKFLOWS:
            known = ", ".join(sorted(OMNI_TRAIN_WORKFLOWS))
            raise ValueError(f"Unknown train.train_type {self.train_type!r}; expected one of: {known}.")
        if self.train_type == "train_and_cache":
            raise NotImplementedError("`train.train_type: train_and_cache` is reserved and is not implemented yet.")
        if self.train_type == "offline_cache" and not self.offline_cache_dir:
            raise ValueError("`train.offline_cache_dir` is required when `train.train_type` is 'offline_cache'.")

        if self.dyn_bsz_physical_overflow_ratio < 1.0:
            raise ValueError(
                f"dyn_bsz_physical_overflow_ratio must be >= 1.0, got {self.dyn_bsz_physical_overflow_ratio}."
            )
        if self.chunk_mbs_config.chunk_mbs < 1:
            raise ValueError(f"chunk_mbs_config.chunk_mbs must be >= 1, got {self.chunk_mbs_config.chunk_mbs}.")

        self._train_steps = -1
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.global_rank = int(os.getenv("RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self._resolve_checkpoint_paths()
        self._resolve_profile()

    def _validate_accelerator(self, accelerator: AcceleratorConfig) -> None:
        acc = accelerator

        if self.world_size % (acc.pp_size * acc.ulysses_size * acc.cp_size * acc.tp_size) != 0:
            raise ValueError(
                f"World size should be a multiple of pp_size: {acc.pp_size}, "
                f"ulysses_size: {acc.ulysses_size}, cp_size: {acc.cp_size}, "
                f"tp_size: {acc.tp_size}."
            )
        assert acc.tp_size == 1, "Tensor parallel size not supported yet."
        assert acc.pp_size == 1, "Pipeline parallel size not supported yet."
        assert acc.cp_size == 1, "Context parallel size not supported yet."

        acc.dp_size = self.world_size // (acc.pp_size * acc.ulysses_size * acc.cp_size * acc.tp_size)

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

        num_nodes = int(os.getenv("WORLD_SIZE", 1)) // int(os.getenv("LOCAL_WORLD_SIZE", 1))
        if num_nodes > 1:
            logger.warning_rank0(
                f"Detected {num_nodes} nodes. "
                "Make sure that `train.checkpoint.output_dir` is shared by all nodes. "
                "Otherwise, each node will save checkpoints to its local directory, which may cause inconsistencies or job failures."
            )

        assert acc.ep_size == 1 or self.init_device != "cpu", (
            "cpu init is not supported when enable ep. Please use `init_device = cuda` or `init_device = meta` instead."
        )
        if acc.fsdp_config.fsdp_mode == "fsdp2":
            assert self.init_device == "meta", "Please use init_device: meta for FSDP2 training"
        elif self.broadcast_model_weights_from_rank0:
            logger.warning_rank0(
                "Ignoring train.broadcast_model_weights_from_rank0=True because it is only "
                "used with accelerator.fsdp_config.fsdp_mode='fsdp2'. "
                f"Received fsdp_mode={acc.fsdp_config.fsdp_mode!r}. Disable this flag or switch to fsdp2.",
            )

        assert not (self.ep_sharded_stream_load and self.broadcast_model_weights_from_rank0), (
            "train.ep_sharded_stream_load requires train.broadcast_model_weights_from_rank0=False "
            "(it reads each rank's ExtraParallel slice directly and cannot run on the broadcast path)."
        )

    def _derive_batch_config(self, accelerator: AcceleratorConfig) -> None:
        acc = accelerator

        if self.global_batch_size is None:
            self.global_batch_size = self.micro_batch_size * acc.dp_size
            self.gradient_accumulation_steps = 1
            logger.info_rank0("`global_batch_size` is None, disable gradient accumulation.")
        elif self.global_batch_size % (self.micro_batch_size * acc.dp_size) == 0:
            self.gradient_accumulation_steps = self.global_batch_size // (self.micro_batch_size * acc.dp_size)
            logger.info_rank0(f"Set gradient accumulation to {self.gradient_accumulation_steps}.")
        else:
            raise ValueError(f"`global_batch_size` should be a multiple of {self.micro_batch_size * acc.dp_size}.")

        if self.dyn_bsz:
            if self.dyn_bsz_runtime == "main":
                self.dataloader_batch_size = 1
            else:
                self.dataloader_batch_size = self.global_batch_size // acc.dp_size // self.micro_batch_size
        else:
            self.dataloader_batch_size = self.global_batch_size // acc.dp_size

    def _resolve_checkpoint_paths(self) -> None:
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

        ckpt.save_path = os.path.join(ckpt.output_dir, "checkpoints")
        ckpt.model_assets_dir = os.path.join(ckpt.output_dir, "model_assets")

    def _resolve_profile(self) -> None:
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


@dataclass
class OmniArguments:
    """Root launcher config for SeedOmni V2."""

    model: Any = field(default_factory=lambda: _default_model_runtime())
    data: OmniDataArguments = field(default_factory=OmniDataArguments)
    train: OmniTrainingArguments = field(default_factory=OmniTrainingArguments)
    infer: OmniInferArguments = field(default_factory=OmniInferArguments)

    def __post_init__(self):
        self._train_steps = -1

        self.train._validate_accelerator(self.model.accelerator)
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

        if self.train.chunk_mbs_config.enable:
            if self.train.pad_to_length:
                raise ValueError("train.chunk_mbs_config.enable is not supported with train.pad_to_length yet.")
            if self.train.gradient_checkpointing.enable and self.train.gradient_checkpointing.enable_reentrant:
                raise ValueError(
                    "train.chunk_mbs_config.enable requires non-reentrant gradient checkpointing. "
                    "Set train.gradient_checkpointing.enable_reentrant=False."
                )

        if self.train.torch_compile.enable:
            raise ValueError(
                "train.torch_compile.enable is not supported by SeedOmni V2 yet "
                f"(data.data_type={self.data.data_type!r})."
            )

    def resolve_model(self, *, for_inference: bool = False):
        """Build a resolved :class:`~veomni.omni_arguments.model_runtime.OmniModelRuntimeArguments`.

        Set ``for_inference=True`` to apply the all-eager inference accelerator
        default on top of ``model.model_config.modules``.
        """
        from .model_runtime import resolve_omni_model

        return resolve_omni_model(self, for_inference=for_inference)

    def _to_module_global_args(self) -> OmniModuleRuntimeArguments:
        """Project ``model`` defaults onto :class:`OmniModuleRuntimeArguments` for per-module merging."""
        from .model_runtime import _to_module_global_args

        return _to_module_global_args(self.model)

    def compute_train_steps(self, dataset_length: int | None = None):
        if self.train.dyn_bsz:
            assert self.data.max_seq_len is not None and self.data.train_size is not None, (
                "data.max_seq_len and data.train_size are required."
            )
            train_size = int(self.data.train_size * (1 + self.train.bsz_warmup_ratio / 2))
            self._train_steps = math.ceil(train_size / (self.train.global_batch_size * self.data.max_seq_len))
        else:
            if dataset_length is not None:
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


def _default_model_runtime():
    from .model_runtime import OmniModelRuntimeArguments

    return OmniModelRuntimeArguments()


__all__ = [
    "LAUNCHER_CONFIG_KEYS",
    "BaseOmniModelArguments",
    "OMNI_TRAIN_WORKFLOWS",
    "OmniArguments",
    "OmniDataArguments",
    "OmniGraphProfileArguments",
    "OmniInferArguments",
    "OmniModuleRuntimeArguments",
    "OmniTrainingArguments",
    "_hf_module_model_config",
    "_is_omni_checkpoint_root",
]
