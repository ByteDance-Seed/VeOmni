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

"""Resolve launcher :class:`OmniModelRuntimeArguments` into merged modules and graphs."""

from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any

from ..arguments.arguments_types import AcceleratorConfig, OptimizerConfig
from ..arguments.parser import _deep_update, _instantiate_recursive
from .arguments_types import (
    BaseOmniModelArguments,
    OmniModuleRuntimeArguments,
    _hf_module_model_config,
    _try_load_omni_checkpoint_config,
)


if TYPE_CHECKING:
    from .arguments_types import OmniArguments


DEFAULT_SCENARIO = "default"


@dataclass
class OmniModelRuntimeArguments(BaseOmniModelArguments):
    """One composed Omni model — flat model fields + ``accelerator`` + ``optimizer`` + resolved state.

    YAML supplies ``model_path``, ``model_config``, ``ops_implementation``, ``accelerator``,
    and ``optimizer``. :meth:`resolve_omni_model` fills ``modules``, graph scenario maps,
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
        """Project onto the checkpoint-shaped HF :class:`OmniConfig`."""
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
    eager_by_module = {name: {"accelerator": {"fsdp_config": {"fsdp_mode": "eager"}}} for name in train_modules_config}
    return _deep_update(eager_by_module, infer_modules_overrides)


def _module_base(global_dict: dict[str, Any]) -> dict[str, Any]:
    acc = global_dict.get("accelerator")
    if isinstance(acc, dict):
        acc["dp_replicate_size"] = -1
        acc["dp_shard_size"] = -1
    return global_dict


__all__ = [
    "DEFAULT_SCENARIO",
    "OmniModelRuntimeArguments",
    "build_module_args",
    "build_module_runtime_args",
    "build_omni_model_runtime",
    "resolve_omni_model",
]
