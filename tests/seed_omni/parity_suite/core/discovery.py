"""Discovery of SeedOmni V2 parity cases from model contracts and graph YAML."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from veomni.models.seed_omni.generation_graph import GenerationGraph
from veomni.models.seed_omni.training_graph import TrainingGraph

from .spec import (
    PARITY_ENABLE_ENV,
    GateSpec,
    ModelSpec,
    ScenarioSpec,
    load_model_spec,
    load_yaml_file,
    repository_root,
)


@dataclass(frozen=True)
class GraphSpec:
    name: str
    path: Path
    domain: str


@dataclass(frozen=True)
class NodeSpec:
    name: str
    module: str
    method: str
    graph: str
    state: str | None = None

    @property
    def anchor(self) -> str:
        if self.state is None:
            return self.name
        return f"{self.state}:{self.name}"


@dataclass(frozen=True)
class ParityCase:
    model: ModelSpec
    scenario: ScenarioSpec
    tier: str
    graph: GraphSpec
    nodes: tuple[NodeSpec, ...]

    @property
    def node_id(self) -> str:
        return f"{self.model.name}.{self.tier}.{self.graph.name}.{self.scenario.id}"

    @property
    def requires_cuda(self) -> bool:
        return self.model.gate.requires_cuda

    @property
    def min_cuda_devices(self) -> int:
        return self.model.gate.min_cuda_devices

    def static_skip_reason(self) -> str | None:
        if os.environ.get(PARITY_ENABLE_ENV) != "1":
            return f"Set {PARITY_ENABLE_ENV}=1 to run {self.node_id}."
        if self.model.reference.checkpoint is not None and not self.model.reference.checkpoint.exists():
            return f"Reference checkpoint does not exist: {self.model.reference.checkpoint}"
        if self.model.v2_model.model_root is not None and not self.model.v2_model.model_root.exists():
            return f"V2 model root does not exist: {self.model.v2_model.model_root}"
        return None


def default_model_dirs() -> tuple[Path, ...]:
    seed_omni_tests = repository_root() / "tests" / "seed_omni"
    required_files = ("base.yaml", "scenarios.yaml", "mapping.yaml")
    return tuple(
        path
        for path in sorted(seed_omni_tests.iterdir())
        if path.is_dir() and all((path / file_name).exists() for file_name in required_files)
    )


def discover_cases(model_dirs: Iterable[str | Path] | None = None) -> tuple[ParityCase, ...]:
    """Discover all configured parity cases without executing reference or V2 models."""

    cases: list[ParityCase] = []
    seen: set[tuple[str, str, str, str]] = set()
    for model_dir in model_dirs or default_model_dirs():
        model = load_model_spec(model_dir)
        graph_by_name = {graph.name: graph for graph in discover_graph_specs(model)}
        for scenario in model.scenarios:
            graph = graph_by_name.get(scenario.graph)
            if graph is None:
                raise KeyError(
                    f"Scenario {model.name}.{scenario.id} references graph {scenario.graph!r}, "
                    f"but available graphs are {sorted(graph_by_name)}."
                )
            nodes = discover_nodes(graph)
            for tier in _expand_tiers(model.gate, model.tiers.enabled(), scenario):
                key = (model.name, tier, graph.name, scenario.id)
                if key in seen:
                    continue
                seen.add(key)
                cases.append(ParityCase(model=model, scenario=scenario, tier=tier, graph=graph, nodes=nodes))
    return tuple(cases)


def discover_graph_specs(model: ModelSpec) -> tuple[GraphSpec, ...]:
    """Discover graph files from the configured SeedOmni V2 config directory."""

    config_dir = model.v2_model.config_dir
    if not config_dir.exists():
        raise FileNotFoundError(f"V2 config_dir does not exist: {config_dir}")

    graphs: list[GraphSpec] = []
    train_path = config_dir / "graph_train.yaml"
    if train_path.exists():
        graphs.append(GraphSpec(name="train", path=train_path, domain="training"))
    for path in sorted(config_dir.glob("graph_infer_*.yaml")):
        graphs.append(GraphSpec(name=path.stem.removeprefix("graph_"), path=path, domain="inference"))

    include = set(model.graphs.include)
    if include:
        missing = include.difference(graph.name for graph in graphs)
        if missing:
            raise KeyError(f"Configured graph include list has unknown graphs: {sorted(missing)}")
        graphs = [graph for graph in graphs if graph.name in include]
    return tuple(graphs)


def discover_nodes(graph: GraphSpec) -> tuple[NodeSpec, ...]:
    """Build a node catalog for one graph without instantiating any model module."""

    data = load_yaml_file(graph.path)
    if graph.domain == "training":
        training_graph = TrainingGraph(data.get("training_graph", data))
        return tuple(
            NodeSpec(
                name=node,
                module=training_graph.module_of(node),
                method=training_graph.method_of(node),
                graph=graph.name,
                state="train",
            )
            for node in training_graph.execution_order
        )

    generation_graph = GenerationGraph(data.get("generation_graph", data))
    nodes: list[NodeSpec] = []
    seen: set[tuple[str, str]] = set()
    for state_name in generation_graph.state_names:
        for node_name in generation_graph.state_node_sequence(state_name):
            key = (state_name, node_name)
            if key in seen:
                continue
            seen.add(key)
            node = generation_graph._node_pool[node_name]
            nodes.append(
                NodeSpec(
                    name=node_name,
                    module=node.module,
                    method=node.method,
                    graph=graph.name,
                    state=state_name,
                )
            )
    return tuple(nodes)


def _expand_tiers(gate: GateSpec, enabled_tiers: tuple[str, ...], scenario: ScenarioSpec) -> tuple[str, ...]:
    del gate
    requested = scenario.tiers or enabled_tiers
    return tuple(tier for tier in requested if tier in enabled_tiers)
