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

* ``model`` is an :class:`OmniModelRuntimeArguments` block — ``model_path``,
  ``model_config``, ``ops_implementation``, ``accelerator``, and ``optimizer``.
* Per-module overrides live in ``model.model_config.modules`` YAML;
  :meth:`OmniArguments.resolve_model` merges them into
  :attr:`OmniModelRuntimeArguments.modules` (each entry is
  :class:`OmniModuleRuntimeArguments`: same flat fields).
* ``data`` / ``train`` / ``infer`` remain launcher-wide.

``OmniModelRuntimeArguments`` and its resolution helpers (``resolve_omni_model``,
``build_omni_model_runtime``, ...) live in this same module rather than a separate
``model_runtime.py``: :class:`OmniArguments.model` is typed as
``OmniModelRuntimeArguments`` and :meth:`OmniArguments.resolve_model` /
:meth:`OmniArguments._to_module_global_args` call the resolution helpers directly,
which would otherwise form an import cycle (``OmniArguments`` -> resolver ->
``OmniModelRuntimeArguments`` -> back to ``OmniArguments`` for the ``args:
OmniArguments`` parameter type) across two files.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, Literal

from ..arguments.arguments_types import (
    AcceleratorConfig,
    ChannelLossConfig,
    CheckpointConfig,
    DataloaderConfig,
    OpsImplementationConfig,
    OptimizerConfig,
    ProfileConfig,
    WandbConfig,
)
from ..arguments.parser import _deep_update, _instantiate_recursive
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


DEFAULT_SCENARIO = "default"


@dataclass
class OmniModelRuntimeArguments(BaseOmniModelArguments):
    """One composed Omni model — flat model fields + ``accelerator`` + ``optimizer`` + resolved state.

    YAML supplies ``model_path``, ``model_config``, ``ops_implementation``, ``accelerator``,
    and ``optimizer``. :func:`resolve_omni_model` fills ``modules``, graph scenario maps,
    and scenario keys.
    """

    accelerator: AcceleratorConfig = field(default_factory=AcceleratorConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    modules: dict[str, OmniModuleRuntimeArguments] = field(default_factory=dict)
    training_graphs: dict[str, Any] = field(default_factory=dict)
    generation_graphs: dict[str, Any] = field(default_factory=dict)
    train_type: str | None = None
    infer_type: str | None = None
    generation_kwargs: dict[str, Any] = field(default_factory=dict)

    def launcher_config(self, key: str, default: Any = None) -> Any:
        """Read a launcher layout key from ``model_config`` (``modules``, graphs, …)."""
        return (self.model_config or {}).get(key, default)

    def set_launcher_config(self, key: str, value: Any) -> None:
        if self.model_config is None:
            self.model_config = {}
        self.model_config[key] = value

    @property
    def resolved_model_path(self) -> str:
        path = self.model_path
        if not path:
            raise ValueError("`model.model_path` (split-checkpoint root) is required for OmniModel V2.")
        return path

    @property
    def module_names(self) -> list[str]:
        return list(self.modules)

    @property
    def train_types(self) -> list[str]:
        return list(self.training_graphs)

    @property
    def infer_types(self) -> list[str]:
        return list(self.generation_graphs)

    @property
    def training_graph(self) -> list[dict]:
        from ..models.seed_omni.configuration_omni import select_graph

        graph = select_graph(
            self.training_graphs,
            self.train_type,
            empty_hint="Populate `model.model_config.train_graph` with at least one scenario.",
            unknown_hint="train_type",
        )
        return list(graph)

    @property
    def generation_graph(self) -> dict:
        from ..models.seed_omni.configuration_omni import select_graph

        return select_graph(
            self.generation_graphs,
            self.infer_type,
            empty_hint="Populate `model.model_config.infer_graph` with at least one scenario.",
            unknown_hint="infer_type",
        )

    def module_checkpoint_subfolder(self, name: str) -> str:
        if name not in self.modules:
            known = ", ".join(self.modules) or "(none)"
            raise KeyError(f"Module {name!r} not found in model runtime; known modules: {known}.")
        return name

    def to_hf_config(self):
        """Project onto the checkpoint-shaped HF :class:`~veomni.models.seed_omni.configuration_omni.OmniConfig`."""
        from ..models.seed_omni.configuration_omni import OmniConfig

        module_entries = {name: mod.to_hf_config(name) for name, mod in self.modules.items()}
        return OmniConfig.from_dict(
            {
                "training_graph": deepcopy(self.training_graph),
                "generation_graphs": deepcopy(self.generation_graphs),
                "infer_type": self.infer_type,
                "generation_kwargs": dict(self.generation_kwargs),
                "modules": module_entries,
            }
        )


def resolve_omni_model(args: OmniArguments, *, for_inference: bool = False) -> OmniModelRuntimeArguments:
    """Resolve ``args.model`` launcher fields into a fully populated :class:`OmniModelRuntimeArguments`."""
    model_runtime = args.model
    if not model_runtime.model_path:
        raise ValueError("`model.model_path` (split-checkpoint root) is required for OmniModel V2.")

    model_path = model_runtime.model_path
    omni_cfg = _try_load_omni_checkpoint_config(model_path)

    train_modules = model_runtime.launcher_config("modules")
    if train_modules is None and omni_cfg is not None:
        train_modules = {
            name: {
                "model_path": omni_cfg.module_checkpoint_subfolder(name),
                **({"ops_implementation": ops} if (ops := omni_cfg.module_ops_implementation(name)) else {}),
                **({"model_config": overrides} if (overrides := omni_cfg.module_model_config(name)) else {}),
            }
            for name in omni_cfg.module_names
        }
    if train_modules is None:
        raise ValueError(
            "`model.model_config.modules` (per-module override YAML) is required when "
            "`model_path` is not a self-contained omni checkpoint."
        )

    train_graph = model_runtime.launcher_config("train_graph")
    if train_graph is None and omni_cfg is not None:
        train_graph = {DEFAULT_SCENARIO: omni_cfg.training_graph}
    if train_graph is None:
        raise ValueError(
            "`model.model_config.train_graph` is required when `model_path` has no omni "
            "`config.json` to load the training graph from."
        )

    infer_graph = model_runtime.launcher_config("infer_graph")
    if not infer_graph and omni_cfg is not None:
        infer_graph = omni_cfg.generation_graphs
    if not infer_graph:
        raise ValueError(
            "`model.model_config.infer_graph` is required when `model_path` has no omni "
            "`config.json` generation graphs."
        )

    train_type = model_runtime.launcher_config("train_type")
    infer_type = model_runtime.launcher_config("infer_type")
    if infer_type is None and omni_cfg is not None:
        infer_type = omni_cfg.infer_type

    train_type = _resolve_graph_type(args, model_runtime, train_graph, "train_type", train_type)
    infer_type = _resolve_graph_type(args, model_runtime, infer_graph, "infer_type", infer_type)

    modules = build_module_runtime_args(
        _to_module_global_args(model_runtime),
        model_path,
        train_modules,
        for_inference=for_inference,
    )
    for module_args in modules.values():
        _validate_omni_accelerator(module_args.accelerator, pad_to_length=args.train.pad_to_length)
    training_graphs = _load_graph_map(train_graph)
    generation_graphs = _load_graph_map(infer_graph)
    if train_type is not None and train_type not in training_graphs:
        known = ", ".join(training_graphs)
        raise KeyError(f"Unknown train_type {train_type!r}; expected one of: {known}.")
    if infer_type is not None and infer_type not in generation_graphs:
        known = ", ".join(generation_graphs)
        raise KeyError(f"Unknown infer_type {infer_type!r}; expected one of: {known}.")

    shared_fields = {f.name for f in fields(BaseOmniModelArguments)}
    model_kwargs = {name: getattr(model_runtime, name) for name in shared_fields}
    return OmniModelRuntimeArguments(
        **model_kwargs,
        accelerator=model_runtime.accelerator,
        optimizer=model_runtime.optimizer,
        modules=modules,
        training_graphs=training_graphs,
        generation_graphs=generation_graphs,
        train_type=train_type,
        infer_type=infer_type,
        generation_kwargs=dict(args.infer.generation_kwargs),
    )


def build_omni_model_runtime(
    global_args: OmniModuleRuntimeArguments,
    model_path: str | os.PathLike,
    train_graph: str | os.PathLike | Mapping[str, Any] | list | dict,
    infer_graph: str | os.PathLike | Mapping[str, Any] | list | dict,
    train_modules: str | os.PathLike | dict[str, Any],
    train_type: str | None = None,
    infer_type: str | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    *,
    for_inference: bool = False,
    accelerator: Any = None,
    optimizer: Any = None,
) -> OmniModelRuntimeArguments:
    """Build a resolved :class:`OmniModelRuntimeArguments` from launcher YAML paths (tests / export)."""
    if accelerator is None:
        accelerator = global_args.accelerator
    if optimizer is None:
        optimizer = global_args.optimizer

    modules = build_module_runtime_args(
        global_args,
        model_path,
        train_modules,
        for_inference=for_inference,
    )
    training_graphs = _load_graph_map(train_graph)
    generation_graphs = _load_graph_map(infer_graph)
    if train_type is not None and train_type not in training_graphs:
        known = ", ".join(training_graphs)
        raise KeyError(f"Unknown train_type {train_type!r}; expected one of: {known}.")
    if infer_type is not None and infer_type not in generation_graphs:
        known = ", ".join(generation_graphs)
        raise KeyError(f"Unknown infer_type {infer_type!r}; expected one of: {known}.")

    shared_fields = {f.name for f in fields(BaseOmniModelArguments)}
    model_kwargs = {name: getattr(global_args, name) for name in shared_fields if name != "model_path"}
    return OmniModelRuntimeArguments(
        model_path=str(model_path),
        **model_kwargs,
        accelerator=accelerator,
        optimizer=optimizer,
        modules=modules,
        training_graphs=training_graphs,
        generation_graphs=generation_graphs,
        train_type=train_type,
        infer_type=infer_type,
        generation_kwargs=dict(generation_kwargs or {}),
    )


def build_module_runtime_args(
    global_args: OmniModuleRuntimeArguments,
    model_path: str | os.PathLike,
    modules: str | os.PathLike | dict[str, Any],
    *,
    for_inference: bool = False,
) -> dict[str, OmniModuleRuntimeArguments]:
    """Merge launcher module YAML onto ``global_args`` without loading graphs."""
    modules_overrides = _load_launcher_yaml(modules)
    modules_overrides = _resolve_model_path(model_path, modules_overrides)

    if for_inference:
        modules_overrides = _deep_update(
            _resolve_default_accelerator(modules_overrides, {}),
            modules_overrides,
        )

    base_dict = _module_base(asdict(global_args))
    runtime_modules: dict[str, OmniModuleRuntimeArguments] = {}
    for name, override in modules_overrides.items():
        module_args = _instantiate_recursive(
            OmniModuleRuntimeArguments,
            _deep_update(deepcopy(base_dict), override),
        )
        runtime_modules[name] = module_args
    return runtime_modules


def build_module_args(config, name: str) -> OmniModuleRuntimeArguments:
    """Instantiate :class:`OmniModuleRuntimeArguments` from an ``OmniConfig.modules`` entry."""
    cfg = config.modules.get(name, None)
    if cfg is None:
        raise KeyError(f"Module '{name}' not found in OmniConfig.modules")
    if not isinstance(cfg, dict):
        raise TypeError(f"Module '{name}' must be a mapping for build_module_args().")
    return _instantiate_recursive(OmniModuleRuntimeArguments, _normalize_module_cfg(cfg))


def _normalize_module_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten checkpoint-shaped module entries onto launcher fields."""
    cfg = deepcopy(cfg)
    cfg.pop("subfolder", None)
    model_block = cfg.pop("model", None)
    if isinstance(model_block, dict):
        for key, value in model_block.items():
            if key not in cfg:
                cfg[key] = value
    return cfg


def _to_module_global_args(model_runtime: OmniModelRuntimeArguments) -> OmniModuleRuntimeArguments:
    """Project omni-model defaults onto :class:`OmniModuleRuntimeArguments` for per-module merging."""
    shared_fields = {f.name for f in fields(BaseOmniModelArguments)}
    model_kwargs = {name: getattr(model_runtime, name) for name in shared_fields}
    model_kwargs["model_config"] = _hf_module_model_config(model_kwargs.get("model_config"))
    return OmniModuleRuntimeArguments(
        **model_kwargs,
        accelerator=model_runtime.accelerator,
        optimizer=model_runtime.optimizer,
    )


def _resolve_graph_type(
    args: OmniArguments,
    model_runtime: OmniModelRuntimeArguments,
    graph: str | dict[str, str],
    config_key: str,
    selected: str | None = None,
) -> str:
    graph_field = "train_graph" if config_key == "train_type" else "infer_graph"

    if isinstance(graph, str):
        graph_map = {DEFAULT_SCENARIO: graph}
        if selected is None:
            selected = model_runtime.launcher_config(config_key) or DEFAULT_SCENARIO
    else:
        graph_map = graph or {}
        if not graph_map:
            raise ValueError(f"`model.model_config.{graph_field}` must declare at least one scenario.")
        if selected is None:
            selected = model_runtime.launcher_config(config_key)
        if selected is None:
            selected = next(iter(graph_map))
        if selected not in graph_map:
            known = ", ".join(sorted(graph_map)) or "(none)"
            raise KeyError(f"Unknown model.model_config.{config_key} {selected!r}; expected one of: {known}.")

    model_runtime.set_launcher_config(config_key, selected)
    return selected


def _graph_scenario_specs(spec: Any) -> dict[str, Any]:
    if spec is None:
        return {}
    if isinstance(spec, list):
        return {DEFAULT_SCENARIO: spec}
    if isinstance(spec, Mapping):
        if {"initial", "states"} <= set(spec):
            return {DEFAULT_SCENARIO: spec}
        return dict(spec)
    return {DEFAULT_SCENARIO: spec}


def _load_graph_map(
    spec: str | os.PathLike | Mapping[str, Any] | list | None,
) -> dict[str, Any]:
    specs = _graph_scenario_specs(spec)
    if not specs:
        raise ValueError("Graph spec is required — declare at least one scenario.")
    graphs: dict[str, Any] = {}
    for name, scenario_spec in specs.items():
        graph = _load_launcher_yaml(scenario_spec)
        if not graph:
            raise ValueError(f"Graph scenario {str(name)!r} resolved to an empty graph (from {scenario_spec!r}).")
        graphs[str(name)] = graph
    return graphs


def _load_launcher_yaml(spec: str | os.PathLike | dict | list | None):
    if spec is None:
        return {}
    if isinstance(spec, (str, os.PathLike)):
        from .parser import load_yaml_with_inherit

        return load_yaml_with_inherit(str(spec))
    return deepcopy(spec)


def _resolve_model_path(
    model_path: str | os.PathLike,
    modules_config: dict[str, Any] | None,
) -> dict[str, Any]:
    if not modules_config:
        return {}
    checkpoint_root = str(model_path)
    for mod_cfg in modules_config.values():
        if not isinstance(mod_cfg, dict):
            continue
        resolved = mod_cfg.get("model_path") or mod_cfg.get("weights_path")
        if resolved is None:
            continue
        if not os.path.isabs(resolved):
            resolved = os.path.join(checkpoint_root, resolved)
        mod_cfg["model_path"] = resolved
    return modules_config


def _resolve_default_accelerator(
    train_modules_config: dict[str, Any],
    infer_modules_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    # `broadcast_model_weights_from_rank0` is only meaningful for `fsdp2`; forcing it off here
    # alongside `fsdp_mode: eager` avoids a spurious `_validate_omni_accelerator` warning on every
    # module for the common single-process eager-inference default.
    eager_by_module = {
        name: {
            "accelerator": {
                "fsdp_config": {"fsdp_mode": "eager"},
                "broadcast_model_weights_from_rank0": False,
            }
        }
        for name in train_modules_config
    }
    return _deep_update(eager_by_module, infer_modules_overrides)


def _module_base(global_dict: dict[str, Any]) -> dict[str, Any]:
    acc = global_dict.get("accelerator")
    if isinstance(acc, dict):
        acc["dp_replicate_size"] = -1
        acc["dp_shard_size"] = -1
    return global_dict


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

        self._train_steps = -1
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.global_rank = int(os.getenv("RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self._resolve_checkpoint_paths()
        self._resolve_profile()

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


def _validate_omni_accelerator(accelerator: AcceleratorConfig, *, pad_to_length: bool | int = False) -> None:
    """Cross-field checks for the SeedOmni-V2-only knobs living on ``AcceleratorConfig``.

    Kept Omni-side (not on the shared :class:`AcceleratorConfig` used by V1 trainers too) since
    some of these constraints (e.g. the blanket ``torch_compile`` ban) do not apply to V1, which
    fully supports ``torch_compile`` with its own, different validation. ``accelerator`` is
    self-contained for every check except the ``chunk_mbs_config`` vs ``pad_to_length`` one, which
    needs the launcher-wide ``train.pad_to_length`` passed in explicitly.

    Called once for the top-level default (``model.accelerator``, at ``OmniArguments.__post_init__``
    time, before modules are resolved) and once per module (in :func:`resolve_omni_model`, after
    ``modules`` merges each module's own ``accelerator:`` YAML override) so a per-module override is
    validated too, not just the global default.
    """
    acc = accelerator

    assert acc.ep_size == 1 or acc.init_device != "cpu", (
        "cpu init is not supported when enable ep. Please use `accelerator.init_device = cuda` or "
        "`accelerator.init_device = meta` instead."
    )
    if acc.fsdp_config.fsdp_mode == "fsdp2":
        assert acc.init_device == "meta", "Please use accelerator.init_device: meta for FSDP2 training"
    elif acc.broadcast_model_weights_from_rank0:
        logger.warning_rank0(
            "Ignoring accelerator.broadcast_model_weights_from_rank0=True because it is only "
            "used with accelerator.fsdp_config.fsdp_mode='fsdp2'. "
            f"Received fsdp_mode={acc.fsdp_config.fsdp_mode!r}. Disable this flag or switch to fsdp2.",
        )

    assert not (acc.ep_sharded_stream_load and acc.broadcast_model_weights_from_rank0), (
        "accelerator.ep_sharded_stream_load requires accelerator.broadcast_model_weights_from_rank0=False "
        "(it reads each rank's ExtraParallel slice directly and cannot run on the broadcast path)."
    )

    if acc.chunk_mbs_config.enable:
        if pad_to_length:
            raise ValueError("accelerator.chunk_mbs_config.enable is not supported with train.pad_to_length yet.")
        if acc.gradient_checkpointing.enable and acc.gradient_checkpointing.enable_reentrant:
            raise ValueError(
                "accelerator.chunk_mbs_config.enable requires non-reentrant gradient checkpointing. "
                "Set accelerator.gradient_checkpointing.enable_reentrant=False."
            )

    if acc.torch_compile.enable:
        raise ValueError("accelerator.torch_compile.enable is not supported by SeedOmni V2 yet.")


@dataclass
class OmniArguments:
    """Root launcher config for SeedOmni V2."""

    model: OmniModelRuntimeArguments = field(default_factory=OmniModelRuntimeArguments)
    data: OmniDataArguments = field(default_factory=OmniDataArguments)
    train: OmniTrainingArguments = field(default_factory=OmniTrainingArguments)
    infer: OmniInferArguments = field(default_factory=OmniInferArguments)

    def __post_init__(self):
        self._train_steps = -1

        num_nodes = int(os.getenv("WORLD_SIZE", 1)) // int(os.getenv("LOCAL_WORLD_SIZE", 1))
        if num_nodes > 1:
            logger.warning_rank0(
                f"Detected {num_nodes} nodes. "
                "Make sure that `train.checkpoint.output_dir` is shared by all nodes. "
                "Otherwise, each node will save checkpoints to its local directory, which may cause inconsistencies or job failures."
            )

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

        _validate_omni_accelerator(self.model.accelerator, pad_to_length=self.train.pad_to_length)

    def resolve_model(self, *, for_inference: bool = False) -> OmniModelRuntimeArguments:
        """Build a resolved :class:`OmniModelRuntimeArguments`.

        Set ``for_inference=True`` to apply the all-eager inference accelerator
        default on top of ``model.model_config.modules``.
        """
        return resolve_omni_model(self, for_inference=for_inference)

    def _to_module_global_args(self) -> OmniModuleRuntimeArguments:
        """Project ``model`` defaults onto :class:`OmniModuleRuntimeArguments` for per-module merging."""
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


__all__ = [
    "DEFAULT_SCENARIO",
    "LAUNCHER_CONFIG_KEYS",
    "BaseOmniModelArguments",
    "OMNI_TRAIN_WORKFLOWS",
    "OmniArguments",
    "OmniDataArguments",
    "OmniGraphProfileArguments",
    "OmniInferArguments",
    "OmniModelRuntimeArguments",
    "OmniModuleRuntimeArguments",
    "OmniTrainingArguments",
    "_hf_module_model_config",
    "_is_omni_checkpoint_root",
    "build_module_args",
    "build_module_runtime_args",
    "build_omni_model_runtime",
    "resolve_omni_model",
]
