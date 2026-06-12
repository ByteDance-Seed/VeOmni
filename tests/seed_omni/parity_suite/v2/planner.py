"""Probe planning helpers over SeedOmni V2 graph configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.seed_omni.parity_suite.v2.config import load_omni_config_from_dir, load_yaml
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.generation_graph import GenerationGraph
from veomni.models.seed_omni.training_graph import TrainingGraph


@dataclass(frozen=True)
class NodeAnchor:
    name: str
    module: str
    method: str
    graph: str
    state: str | None = None


def build_node_catalog(config: OmniConfig) -> dict[str, NodeAnchor]:
    catalog: dict[str, NodeAnchor] = {}
    if config.training_graph:
        training = TrainingGraph(config.training_graph)
        for node in training.execution_order:
            catalog[node] = NodeAnchor(
                name=node,
                module=training.module_of(node),
                method=training.method_of(node),
                graph="training",
            )
    if config.generation_graph is not None:
        generation = GenerationGraph(config.generation_graph)
        for state_name, state in generation._states.items():
            for node in state.node_sequence:
                key = f"{state_name}:{node}"
                node_def = generation._node_pool[node]
                catalog[key] = NodeAnchor(
                    name=key,
                    module=node_def.module,
                    method=node_def.method,
                    graph="generation",
                    state=state_name,
                )
    return catalog


@dataclass(frozen=True)
class GraphSpec:
    name: str
    path: Path
    domain: str


def discover_graph_specs(config_dir: str | Path) -> list[GraphSpec]:
    root = Path(config_dir)
    specs: list[GraphSpec] = []
    train_path = root / "graph_train.yaml"
    if train_path.exists():
        specs.append(GraphSpec(name="train", path=train_path, domain="training"))
    for path in sorted(root.glob("graph_infer_*.yaml")):
        graph_name = path.stem.removeprefix("graph_")
        specs.append(GraphSpec(name=graph_name, path=path, domain="inference"))
    return specs


def graph_exists(config_dir: str | Path, graph: str) -> bool:
    return any(spec.name == graph for spec in discover_graph_specs(config_dir))


def build_node_catalog_for_graph(config_dir: str | Path, graph: str) -> dict[str, NodeAnchor]:
    if graph == "train":
        config = load_omni_config_from_dir(config_dir)
    else:
        config = load_omni_config_from_dir(config_dir, graph=graph)
    catalog = build_node_catalog(config)
    expected_graph_type = "training" if graph == "train" else "generation"
    return {name: node for name, node in catalog.items() if node.graph == expected_graph_type}


def graph_generation_kwargs(config_dir: str | Path, graph: str) -> dict[str, Any]:
    if graph == "train":
        return {}
    path = Path(config_dir) / f"graph_{graph}.yaml"
    if not path.exists():
        return {}
    data = load_yaml(path)
    return dict(data.get("generation_kwargs") or {})
