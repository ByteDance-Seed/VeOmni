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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ....utils.registry import Registry  # VeOmni shared name→factory registry; not seed_omni-local.


if TYPE_CHECKING:
    # Runtime import would close the convert_registry ↔ family cycle; see _run_converter.
    from ..modules.module_modeling_base import PretrainedOmniModule


OMNI_CONVERT_REGISTRY = Registry("OmniConvert")

ConvertFn = Callable[..., dict[str, Any]]

GraphFiles = str | Mapping[str, str]


def attach_module_assets(
    module: PretrainedOmniModule,
    *,
    tokenizer: Any = None,
    processor: Any = None,
    image_processor: Any = None,
    video_processor: Any = None,
) -> PretrainedOmniModule:
    """Hang HF assets on ``module`` so its ``save_pretrained`` writes them into the module subfolder.

    These are the attributes a module's preprocessor binds at load time, so a
    converted module saves exactly what the loaded one reads back.
    """
    for attr, asset in (
        ("_tokenizer", tokenizer),
        ("_processor", processor),
        ("_image_processor", image_processor),
        ("_video_processor", video_processor),
    ):
        if asset is not None:
            setattr(module, attr, asset)
    return module


def _repo_config_dir(relative: str | Path) -> Path:
    """Walk up from this file to ``<repo>/<relative>``."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / relative
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find {relative} above {__file__}. "
        "Pass training_graph= / generation_graph= explicitly, or run convert from a VeOmni checkout."
    )


def _read_graph_files(config_dir: Path, files: GraphFiles, value_type: type) -> dict[str, Any]:
    """Read one scenario-map file, or ``{scenario: file}`` where each file holds a single graph."""
    from ..configuration_omni import OmniConfig

    if isinstance(files, str):
        return OmniConfig._read_graph_file(str(config_dir / files), value_type)
    graphs: dict[str, Any] = {}
    for scenario, file in files.items():
        payload = OmniConfig._read_graph_file(str(config_dir / file), value_type)
        if len(payload) != 1:
            raise ValueError(f"{config_dir / file} holds {len(payload)} scenarios; expected one for {scenario!r}.")
        graphs[scenario] = next(iter(payload.values()))
    return graphs


def load_family_graphs(
    config_dir: str | Path,
    *,
    training: GraphFiles,
    generation: GraphFiles,
    training_graph: str | Path | None = None,
    generation_graph: str | Path | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Read a family's default training DAGs and generation FSMs from ``configs/``.

    ``config_dir`` is repo-relative (e.g. ``configs/seed_omni/Janus/janus_1.3b``);
    ``training`` / ``generation`` name files under it. ``training_graph`` /
    ``generation_graph`` are caller-supplied YAML paths that replace the
    defaults, so a converter still runs outside a checkout when both are given.
    """
    from ..configuration_omni import OmniConfig

    root: Path | None = None

    def configs() -> Path:
        nonlocal root
        if root is None:
            root = _repo_config_dir(config_dir)
        return root

    training_graphs = (
        OmniConfig._read_graph_file(str(training_graph), list)
        if training_graph is not None
        else _read_graph_files(configs(), training, list)
    )
    generation_graphs = (
        OmniConfig._read_graph_file(str(generation_graph), dict)
        if generation_graph is not None
        else _read_graph_files(configs(), generation, dict)
    )
    return training_graphs, generation_graphs


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
    "attach_module_assets",
    "convert_checkpoint",
    "load_family_graphs",
]
