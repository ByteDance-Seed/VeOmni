"""Mermaid export helpers for SeedOmni training / generation graphs."""

from __future__ import annotations

import os
import re
from typing import Protocol

from ..graphs.generation_graph import GenerationGraph
from ..graphs.training_graph import TrainingGraph


class GraphConfig(Protocol):
    """The graph surface these helpers need.

    Satisfied by both :class:`~veomni.models.seed_omni.configuration_omni.OmniConfig`
    and the launcher-side
    :class:`~veomni.arguments.omni_arguments_types.OmniModelRuntimeArguments`,
    so diagrams can be drawn without projecting the runtime view onto a checkpoint.
    """

    training_graph: list[dict]
    generation_graphs: dict

    @property
    def infer_types(self) -> list[str]: ...

    @property
    def generation_graph(self) -> dict: ...


GRAPH_VIS_SUBDIR = "graphs"
TRAINING_MMD_FILENAME = "training.mmd"

_SAFE_INFER_TYPE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def render_training_mermaid(config: GraphConfig, *, title: str = "Training graph") -> str:
    """Return Mermaid source for ``config.training_graph``."""
    return TrainingGraph(config.training_graph).to_mermaid(title=title)


def render_generation_mermaid(
    config: GraphConfig,
    *,
    title: str = "Generation graph",
    infer_type: str | None = None,
) -> str:
    """Return Mermaid source for one generation scenario (default: the active one)."""
    graph = config.generation_graphs[infer_type] if infer_type is not None else config.generation_graph
    return GenerationGraph(graph).to_mermaid(title=title)


def generation_mmd_filename(infer_type: str) -> str:
    """Sidecar diagram filename for scenario ``infer_type``.

    Scenario names come from launcher YAML and land in a path here, so anything
    that is not a single plain path segment is rejected rather than escaping the
    diagram directory.
    """
    if not _SAFE_INFER_TYPE.fullmatch(infer_type):
        raise ValueError(
            f"Invalid infer_type {infer_type!r}: scenario names must match {_SAFE_INFER_TYPE.pattern} "
            "so they can be used as diagram filenames."
        )
    return f"generation_{infer_type}.mmd"


def write_mermaid_file(path: str | os.PathLike, body: str) -> None:
    """Write raw Mermaid (``.mmd``) text to *path*."""
    path = str(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
        if not body.endswith("\n"):
            f.write("\n")


def save_graph_mermaid_diagrams(
    config: GraphConfig,
    save_directory: str | os.PathLike,
    *,
    training_title: str = "Training graph",
    generation_title: str = "Generation graph",
) -> list[str]:
    """Write ``graphs/training.mmd`` plus one ``graphs/generation_<infer_type>.mmd`` per scenario."""
    vis_dir = os.path.join(str(save_directory), GRAPH_VIS_SUBDIR)
    training_path = os.path.join(vis_dir, TRAINING_MMD_FILENAME)
    write_mermaid_file(training_path, render_training_mermaid(config, title=training_title))

    paths = [training_path]
    for infer_type in config.infer_types:
        path = os.path.join(vis_dir, generation_mmd_filename(infer_type))
        body = render_generation_mermaid(config, title=f"{generation_title} — {infer_type}", infer_type=infer_type)
        write_mermaid_file(path, body)
        paths.append(path)
    return paths


__all__ = [
    "GRAPH_VIS_SUBDIR",
    "TRAINING_MMD_FILENAME",
    "generation_mmd_filename",
    "render_generation_mermaid",
    "render_training_mermaid",
    "save_graph_mermaid_diagrams",
    "write_mermaid_file",
]
