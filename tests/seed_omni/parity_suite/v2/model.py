"""Shared V2 model behavior helpers for parity drivers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules import OMNI_MODEL_REGISTRY, read_model_type


def graph_active_module_names(case: Any) -> frozenset[str]:
    """Return every module referenced by the case's selected graph."""

    return frozenset(str(node.module) for node in case.nodes)


def load_graph_active_omni_config(case: Any, module_names: Iterable[str] | None = None) -> OmniConfig:
    """Load an OmniConfig narrowed to the modules used by ``case.graph``."""

    names = graph_active_module_names(case) if module_names is None else frozenset(module_names)
    graph = None if case.graph.domain == "training" else case.graph.name
    return load_omni_config_from_dir(
        case.model.v2_model.config_dir,
        graph=graph,
        graph_domain=case.graph.domain,
        module_names=names,
    )


def load_graph_active_omni_model(
    case: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> OmniModel:
    """Load the default graph-active V2 model for a parity case."""

    config = load_graph_active_omni_config(case)
    modules = load_graph_active_omni_modules(case, config.module_names, device=device, dtype=dtype)
    return OmniModel(config, modules).eval()


def load_graph_active_omni_modules(
    case: Any,
    module_names: Iterable[str],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, nn.Module]:
    model_root = case.model.v2_model.model_root
    if model_root is None:
        raise ValueError(f"{case.model.name} V2 model_root is required.")
    return {
        name: load_omni_module_from_pretrained(model_root / name, device=device, dtype=dtype).eval()
        for name in module_names
    }


def load_omni_config_from_dir(
    config_dir: str | Path,
    *,
    graph: str | None = None,
    graph_domain: str | None = None,
    module_names: Iterable[str] | None = None,
) -> OmniConfig:
    root = Path(config_dir)
    modules = _load_yaml(root / "modules_train.yaml")
    if module_names is not None:
        modules = _filter_modules(modules, module_names)
    data: dict[str, Any] = {"modules": modules}
    if graph_domain != "inference":
        data["training_graph"] = _load_yaml(root / "graph_train.yaml")["training_graph"]
    if graph is not None:
        graph_name = graph if graph.endswith(".yaml") else f"graph_{graph}.yaml"
        infer = _load_yaml(root / graph_name)
        data["generation_graph"] = infer["generation_graph"]
        data["generation_kwargs"] = infer.get("generation_kwargs", {})
    return OmniConfig.from_dict(data)


def load_omni_module_from_pretrained(
    module_dir: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    model_type = read_model_type(str(module_dir))
    model_cls = OMNI_MODEL_REGISTRY[model_type]()
    module = model_cls.from_pretrained(module_dir, torch_dtype=dtype)
    return module.to(device=device)


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _filter_modules(modules: Mapping[str, Any], module_names: Iterable[str]) -> dict[str, Any]:
    selected = frozenset(module_names)
    missing = sorted(selected.difference(modules))
    if missing:
        raise KeyError(f"Graph references module(s) missing from modules_train.yaml: {missing}")
    return {name: modules[name] for name in modules if name in selected}


__all__ = [
    "graph_active_module_names",
    "load_graph_active_omni_config",
    "load_graph_active_omni_model",
    "load_graph_active_omni_modules",
    "load_omni_config_from_dir",
    "load_omni_module_from_pretrained",
]
