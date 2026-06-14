"""Shared V2 model behavior helpers for parity drivers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.modules import OMNI_MODEL_REGISTRY, read_model_type


def load_omni_config_from_dir(config_dir: str | Path, *, graph: str | None = None) -> OmniConfig:
    root = Path(config_dir)
    modules = _load_yaml(root / "modules_train.yaml")
    train_graph = _load_yaml(root / "graph_train.yaml")["training_graph"]
    data: dict[str, Any] = {
        "modules": modules,
        "training_graph": train_graph,
    }
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


__all__ = [
    "load_omni_config_from_dir",
    "load_omni_module_from_pretrained",
]
