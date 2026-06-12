"""V2 config loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from veomni.models.seed_omni.configuration_omni import OmniConfig


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_omni_config_from_dir(config_dir: str | Path, *, graph: str | None = None) -> OmniConfig:
    root = Path(config_dir)
    modules = load_yaml(root / "modules_train.yaml")
    train_graph = load_yaml(root / "graph_train.yaml")["training_graph"]
    data: dict[str, Any] = {
        "modules": modules,
        "training_graph": train_graph,
    }
    if graph is not None:
        graph_name = graph if graph.endswith(".yaml") else f"graph_{graph}.yaml"
        infer_path = root / graph_name
        if infer_path.exists():
            infer = load_yaml(infer_path)
            data["generation_graph"] = infer.get("generation_graph")
            data["generation_kwargs"] = infer.get("generation_kwargs", {})
    return OmniConfig.from_dict(data)
