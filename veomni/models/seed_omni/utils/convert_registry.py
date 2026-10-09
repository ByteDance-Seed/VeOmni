"""Registry for monolithic HF checkpoint → SeedOmni split checkpoints.

Each family registers a converter under its upstream HuggingFace
``model_type``. The converter **returns** split modules and, when it has them,
graphs; :func:`convert_checkpoint` writes the split directory (CLI:
``scripts/seed_omni/convert_model.py``). Graphs are optional at convert time —
train / generate load sidecars from the checkpoint (or accept an override) and
error only if they still have nothing to run.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable

from ....utils.registry import Registry  # VeOmni shared name→factory registry; not seed_omni-local.


if TYPE_CHECKING:
    # Runtime import would close the convert_registry ↔ family cycle; see _run_converter.
    from ..modules.module_modeling_base import PretrainedOmniModule


OMNI_CONVERT_REGISTRY = Registry("OmniConvert")

ConvertFn = Callable[..., dict[str, Any]]


def _save_converted_omni(
    output_dir: str,
    *,
    modules: Mapping[str, PretrainedOmniModule],
    training_graphs: dict[str, list[dict[str, Any]]] | None = None,
    generation_graphs: dict[str, dict[str, Any]] | None = None,
    train_type: str | None = None,
    infer_type: str | None = None,
    generation_kwargs: dict[str, Any] | None = None,
) -> None:
    """Write a split omni checkpoint: module subfolders, plus graph sidecars when given.

    A family can split modules before its DAGs/FSMs exist. Graph endpoints are
    checked when the checkpoint is loaded, not here.
    """
    from ..configuration_omni import OmniConfig
    from ..modeling_omni import OmniModel

    config = OmniConfig(
        _module_entries={name: {"model_path": name} for name in modules},
        training_graphs=dict(training_graphs or {}),
        generation_graphs=dict(generation_graphs or {}),
        train_type=train_type,
        infer_type=infer_type,
        generation_kwargs=generation_kwargs,
    )
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
    override whatever the family converter returned. Convert does not require
    graphs: train / generate read sidecars from the checkpoint (or take an
    override) and fail only if they still have none.
    """
    if training_graph is not None:
        kwargs.setdefault("training_graph", training_graph)
    if generation_graph is not None:
        kwargs.setdefault("generation_graph", generation_graph)
    converted = _run_converter(model_path, **kwargs)
    _apply_graph_files(converted, training_graph=training_graph, generation_graph=generation_graph)
    _save_converted_omni(output_dir, **converted)


def _apply_graph_files(
    converted: dict[str, Any],
    *,
    training_graph: str | None,
    generation_graph: str | None,
) -> None:
    """Replace converter graphs with YAML from disk when the caller supplied paths."""
    from ..configuration_omni import OmniConfig

    if training_graph is not None:
        converted.pop("training_graph", None)
        converted["training_graphs"] = OmniConfig._read_graph_file(str(training_graph), list)
        graphs = converted["training_graphs"]
        train_type = converted.get("train_type")
        if train_type is None or train_type not in graphs:
            converted["train_type"] = next(iter(graphs)) if graphs else None
    if generation_graph is not None:
        converted["generation_graphs"] = OmniConfig._read_graph_file(str(generation_graph), dict)
        infer_type = converted.get("infer_type")
        graphs = converted["generation_graphs"]
        if infer_type is None or infer_type not in graphs:
            converted["infer_type"] = next(iter(graphs)) if graphs else None


def _run_converter(model_path: str, **kwargs) -> dict[str, Any]:
    """Dispatch to the registered converter; returns kwargs for :func:`_save_converted_omni`.

    Lazy-imports ``modules`` to break the convert_registry ↔ family cycle:
    every family's ``convert_model`` (imported by ``modules/__init__``) imports
    this module.
    """
    from ..modules import read_hf_model_type

    model_type = read_hf_model_type(model_path)
    converter: ConvertFn = OMNI_CONVERT_REGISTRY[model_type]()
    return converter(model_path, **kwargs)


__all__ = [
    "OMNI_CONVERT_REGISTRY",
    "convert_checkpoint",
]
