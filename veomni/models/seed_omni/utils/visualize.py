"""Mermaid export for SeedOmni training / generation graphs."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

from ..graphs.generation_graph import GenerationGraph
from ..graphs.training_graph import TrainingGraph


if TYPE_CHECKING:
    from ..configuration_omni import OmniConfig


_SAFE_INFER_TYPE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _write_mermaid(path: str, body: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
        if not body.endswith("\n"):
            f.write("\n")


def save_graph_mermaid_diagrams(
    config: OmniConfig,
    save_directory: str | os.PathLike,
    *,
    training_title: str = "Training graph",
    generation_title: str = "Generation graph",
) -> list[str]:
    """Write ``graphs/training.mmd`` plus one ``graphs/generation_<infer_type>.mmd`` per scenario.

    A config with no ``training_graph`` gets no training diagram rather than an
    error: an inference-only checkpoint legitimately has none, and
    :class:`TrainingGraph` rejects an empty edge list. These files are
    diagnostics, so demanding a graph the config does not claim to have would
    fail a ``save_pretrained`` over a picture — after every real artifact,
    including each module's weights, is already on disk.
    """
    vis_dir = os.path.join(str(save_directory), "graphs")
    paths: list[str] = []

    if config.training_graph:
        training_path = os.path.join(vis_dir, "training.mmd")
        _write_mermaid(training_path, TrainingGraph(config.training_graph).to_mermaid(title=training_title))
        paths.append(training_path)

    for infer_type in config.infer_types:
        if not _SAFE_INFER_TYPE.fullmatch(infer_type):
            raise ValueError(
                f"Invalid infer_type {infer_type!r}: scenario names must match {_SAFE_INFER_TYPE.pattern} "
                "so they can be used as diagram filenames."
            )
        generation_path = os.path.join(vis_dir, f"generation_{infer_type}.mmd")
        _write_mermaid(
            generation_path,
            GenerationGraph(config.generation_graphs[infer_type]).to_mermaid(
                title=f"{generation_title} — {infer_type}"
            ),
        )
        paths.append(generation_path)

    return paths


__all__ = ["save_graph_mermaid_diagrams"]
