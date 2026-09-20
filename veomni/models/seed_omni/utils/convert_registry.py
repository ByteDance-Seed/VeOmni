"""Registry for monolithic HF checkpoint → SeedOmni split checkpoints.

Each family registers a converter under its upstream HuggingFace
``model_type``. The converter **returns** split modules plus both graphs;
:func:`save_converted_omni` is the only writer (CLI:
``scripts/seed_omni/convert_model.py``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Callable

from torch import nn

from ....utils.registry import Registry  # VeOmni shared name→factory registry; not seed_omni-local.
from ..configuration_omni import DEFAULT_GENERATION_GRAPH_FILE, DEFAULT_TRAINING_GRAPH_FILE


OMNI_CONVERT_REGISTRY = Registry("OmniConvert")

ConvertFn = Callable[..., dict[str, Any]]


def save_converted_omni(
    output_dir: str,
    *,
    modules: Mapping[str, nn.Module],
    training_graph: list[dict[str, Any]],
    generation_graphs: dict[str, dict[str, Any]],
    infer_type: str | None = None,
    generation_kwargs: dict[str, Any] | None = None,
) -> None:
    """Write a split omni checkpoint: module subfolders + both graph sidecars.

    Validates the graphs before touching disk so a malformed family convert
    fails before it leaves a half-written directory.
    """
    from ..configuration_omni import OmniConfig
    from ..graphs.generation_graph import GenerationGraph
    from ..graphs.training_graph import TrainingGraph
    from ..modeling_omni import OmniModel

    TrainingGraph(training_graph)
    if not generation_graphs:
        raise ValueError("Omni convert must produce at least one generation-graph scenario under `generation_graphs`.")
    for spec in generation_graphs.values():
        GenerationGraph(spec)

    config = OmniConfig(
        modules={name: {"subfolder": name} for name in modules},
        training_graph=training_graph,
        generation_graphs=generation_graphs,
        infer_type=infer_type,
        generation_kwargs=generation_kwargs,
    )
    _assert_graph_modules(config, modules)
    OmniModel(config, modules).save_pretrained(output_dir)


def convert_checkpoint(
    model_path: str,
    output_dir: str,
    *,
    training_graph: str | None = None,
    generation_graph: str | None = None,
    **kwargs,
) -> None:
    """Run the registered family converter and write the split omni checkpoint.

    ``training_graph`` / ``generation_graph`` are YAML paths. When set they
    override whatever the family converter returned, so the CLI can bake a
    complete omni checkpoint (module subfolders + both graph sidecars) without
    the family hard-coding one DAG/FSM.
    """
    if training_graph is not None:
        kwargs.setdefault("train_graph", training_graph)
    if generation_graph is not None:
        kwargs.setdefault("generation_graph", generation_graph)
    converted = _run_converter(model_path, **kwargs)
    _apply_graph_files(converted, training_graph=training_graph, generation_graph=generation_graph)
    save_converted_omni(output_dir, **converted)
    _require_converted_graphs(output_dir)


def _apply_graph_files(
    converted: dict[str, Any],
    *,
    training_graph: str | None,
    generation_graph: str | None,
) -> None:
    """Replace converter graphs with YAML from disk when the caller supplied paths."""
    from ..configuration_omni import OmniConfig

    if training_graph is not None:
        converted["training_graph"] = OmniConfig._read_graph_file(str(training_graph), "training_graph")
    if generation_graph is not None:
        converted["generation_graphs"] = OmniConfig._read_generation_graphs(str(generation_graph))
        infer_type = converted.get("infer_type")
        graphs = converted["generation_graphs"]
        if infer_type is None or infer_type not in graphs:
            converted["infer_type"] = next(iter(graphs)) if graphs else None


def _run_converter(model_path: str, **kwargs) -> dict[str, Any]:
    """Dispatch to the registered converter; returns kwargs for :func:`save_converted_omni`.

    Lazy-imports ``modules`` to break the convert_registry ↔ family cycle:
    every family's ``convert_model`` (imported by ``modules/__init__``) imports
    this module.
    """
    from ..modules import read_hf_model_type

    model_type = read_hf_model_type(model_path)
    converter: ConvertFn = OMNI_CONVERT_REGISTRY[model_type]()
    return converter(model_path, **kwargs)


def _require_converted_graphs(output_dir: str) -> None:
    """Fail if ``output_dir`` is not a complete omni checkpoint with both graphs."""
    from ..configuration_omni import OmniConfig
    from ..graphs.generation_graph import GenerationGraph
    from ..graphs.training_graph import TrainingGraph

    root = str(output_dir)
    training_path = os.path.join(root, DEFAULT_TRAINING_GRAPH_FILE)
    generation_path = os.path.join(root, DEFAULT_GENERATION_GRAPH_FILE)
    missing = [path for path in (training_path, generation_path) if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Omni convert must write both graph sidecars "
            f"({DEFAULT_TRAINING_GRAPH_FILE} and {DEFAULT_GENERATION_GRAPH_FILE}) "
            f"under {root}. Missing: {', '.join(missing)}."
        )

    config = OmniConfig.from_pretrained(root)
    if not config.training_graph:
        raise ValueError(f"Omni convert wrote an empty `{DEFAULT_TRAINING_GRAPH_FILE}` under {root}.")
    if not config.generation_graphs:
        raise ValueError(
            f"Omni convert wrote no generation-graph scenarios in `{DEFAULT_GENERATION_GRAPH_FILE}` under {root}."
        )
    TrainingGraph(config.training_graph)
    for spec in config.generation_graphs.values():
        GenerationGraph(spec)
    _assert_graph_modules(config)


def _assert_graph_modules(config: Any, modules: Mapping[str, nn.Module] | None = None) -> None:
    """Every non-end graph endpoint must name a declared module and an existing method.

    Module-name checks run against ``config.modules``. Method checks need the
    live converted modules and therefore run only when ``modules`` is passed
    (convert write path). Bare training endpoints resolve to ``forward``; bare
    generation endpoints resolve to ``generate`` — the same defaults
    :class:`~veomni.models.seed_omni.modeling_omni.OmniModel` uses at
    train/generate time.
    """
    nodes = list(_iter_graph_nodes(config))
    declared = set(config.module_names)
    unknown = sorted({node.module for node in nodes} - declared)
    if unknown:
        raise ValueError(
            "Omni convert graphs reference modules that are not in `config.modules`: "
            f"{unknown}. Declared: {sorted(declared)}."
        )
    if modules is None:
        return

    missing: list[str] = []
    seen: set[tuple[str, str]] = set()
    for node in nodes:
        key = (node.module, node.method)
        if key in seen:
            continue
        seen.add(key)
        module = modules.get(node.module)
        if module is None:
            continue
        if getattr(module, node.method, None) is None:
            missing.append(f"{type(module).__name__}.{node.method}")
    if missing:
        raise ValueError(
            "Omni convert graphs reference methods that are not implemented on the "
            f"converted modules: {missing}. OmniModel would raise AttributeError at "
            "train or generate time."
        )


def _iter_graph_nodes(config: Any):
    for spec in config.training_graph:
        yield from _edge_nodes(spec, default_method="forward")
    for fsm in config.generation_graphs.values():
        for state in (fsm.get("states") or {}).values():
            for spec in state.get("body") or []:
                yield from _edge_nodes(spec, default_method="generate")


def _edge_nodes(spec: dict[str, Any], *, default_method: str):
    from ..graphs.base import EdgeDef

    edge = EdgeDef.parse(spec, default_method=default_method)
    yield edge.from_node
    if edge.to_node is not None:
        yield edge.to_node


__all__ = [
    "OMNI_CONVERT_REGISTRY",
    "convert_checkpoint",
    "save_converted_omni",
]
