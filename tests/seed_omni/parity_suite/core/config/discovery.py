"""Discovery of SeedOmni V2 parity cases from model contracts and graph YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from veomni.models.seed_omni.graphs.generation_graph import GenerationGraph
from veomni.models.seed_omni.graphs.training_graph import TrainingGraph

from .spec import (
    DEFAULT_GATE,
    GateSpec,
    ModelSpec,
    RecipeSpec,
    RunSpec,
    V2ModelTargetSpec,
    load_model_spec,
    load_yaml_file,
    select_v2_model_target,
)


# Discovered case contract -----------------------------------------------------


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
    recipe: RecipeSpec
    run: RunSpec
    graph: GraphSpec
    nodes: tuple[NodeSpec, ...]

    @property
    def node_id(self) -> str:
        return f"{self.model.name}.{self.recipe.id}.{self.tier}.{self.run.id}"

    @property
    def tier(self) -> str:
        return self.run.tier

    @property
    def v2_model(self) -> V2ModelTargetSpec:
        return _recipe_v2_model_target(self.model, self.recipe)

    @property
    def effective_gate(self) -> GateSpec:
        return DEFAULT_GATE.merge(self.model.gate).merge(self.recipe.gate).merge(self.run.gate)

    @property
    def requires_cuda(self) -> bool:
        return bool(self.effective_gate.requires_cuda)

    @property
    def min_cuda_devices(self) -> int:
        return self.effective_gate.min_cuda_devices

    def static_skip_reason(self) -> str | None:
        from .gate import case_skip_reason

        return case_skip_reason(self)


# Public discovery entrypoints -------------------------------------------------


def discover_cases(model_dirs: Iterable[str | Path]) -> tuple[ParityCase, ...]:
    """Discover all configured parity cases without executing reference or V2 models."""

    cases: list[ParityCase] = []
    seen: set[tuple[str, str, str, str]] = set()
    for model_dir in model_dirs:
        model = load_model_spec(model_dir)
        if not model.discover:
            continue
        for recipe in model.recipes:
            graph_by_name = {graph.name: graph for graph in discover_graph_specs(model, recipes=(recipe,))}
            graph = graph_by_name.get(recipe.graph)
            if graph is None:
                raise KeyError(
                    f"Recipe {model.name}.{recipe.id} references graph {recipe.graph!r}, "
                    f"but available graphs are {sorted(graph_by_name)}."
                )
            nodes = discover_nodes(graph)
            for run in _enabled_runs(model.tiers.enabled(), recipe):
                if run.tier == "module" and graph.domain != "inference":
                    raise ValueError(
                        f"{model.name}.{recipe.id}.{run.tier}.{run.id} selects graph {graph.name!r} "
                        f"with domain {graph.domain!r}; module tier is inference-only."
                    )
                key = (model.name, recipe.id, run.tier, run.id)
                if key in seen:
                    raise ValueError(
                        "Duplicate parity case id "
                        f"{model.name}.{recipe.id}.{run.tier}.{run.id}; "
                        "run.id must be unique within each recipe id and tier."
                    )
                seen.add(key)
                cases.append(ParityCase(model=model, recipe=recipe, run=run, graph=graph, nodes=nodes))
    return tuple(cases)


# Reference dispatch helpers ---------------------------------------------------


def effective_reference_kind(case: ParityCase) -> str:
    """Return the reference runner kind after applying graph-domain defaults."""

    kind = case.recipe.reference.get("kind")
    if kind is not None:
        return str(kind)
    if case.graph.domain == "training":
        return "train_forward_backward"
    return case.graph.name


# Graph and node discovery -----------------------------------------------------


def discover_graph_specs(
    model: ModelSpec,
    *,
    recipes: tuple[RecipeSpec, ...] | None = None,
) -> tuple[GraphSpec, ...]:
    """Discover graph files from the configured SeedOmni V2 config directory."""

    active_recipes = model.recipes if recipes is None else recipes
    graphs: list[GraphSpec] = []
    seen_graphs: set[tuple[str, Path]] = set()
    for recipe in active_recipes:
        if recipe.graph == "reference":
            continue
        target = _recipe_v2_model_target(model, recipe)
        for graph in _discover_v2_graphs(target.config_dir):
            key = (graph.name, graph.path)
            if key in seen_graphs:
                continue
            seen_graphs.add(key)
            graphs.append(graph)
    if any(recipe.graph == "reference" for recipe in active_recipes):
        graphs.append(GraphSpec(name="reference", path=model.root / "recipes" / "reference.yaml", domain="reference"))

    include = set(model.graphs.include)
    if include:
        graph_names = {graph.name for graph in graphs}
        referenced = {recipe.graph for recipe in active_recipes}
        missing = include.difference(graph_names)
        if missing:
            raise KeyError(f"Configured graph include list has unknown graphs: {sorted(missing)}")
        graphs = [graph for graph in graphs if graph.name in include or graph.name in referenced]
    return tuple(graphs)


def discover_nodes(graph: GraphSpec) -> tuple[NodeSpec, ...]:
    """Build a node catalog for one graph without instantiating any model module."""

    data = load_yaml_file(graph.path)
    if graph.domain == "reference":
        return ()
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


# Internal filters -------------------------------------------------------------


def _enabled_runs(enabled_tiers: tuple[str, ...], recipe: RecipeSpec) -> tuple[RunSpec, ...]:
    return tuple(run for run in recipe.runs if run.enable and run.tier in enabled_tiers)


def _recipe_v2_model_target(model: ModelSpec, recipe: RecipeSpec) -> V2ModelTargetSpec:
    oracle = str(recipe.reference.get("oracle", ""))
    return select_v2_model_target(
        model.v2_model,
        oracle,
        model_name=model.name,
        recipe_id=recipe.id,
    )


def _discover_v2_graphs(config_dir: Path) -> tuple[GraphSpec, ...]:
    if not config_dir.exists():
        raise FileNotFoundError(f"V2 config_dir does not exist: {config_dir}")

    graphs: list[GraphSpec] = []
    train_path = config_dir / "graph_train.yaml"
    if train_path.exists():
        graphs.append(GraphSpec(name="train", path=train_path, domain="training"))
    for path in sorted(config_dir.glob("graph_infer_*.yaml")):
        graphs.append(GraphSpec(name=path.stem.removeprefix("graph_"), path=path, domain="inference"))
    return tuple(graphs)
