"""Registry for monolithic HF checkpoint → SeedOmni V2 split checkpoints.

Each model family registers a converter under its upstream HuggingFace
``model_type`` (read from ``config.json`` at ``model_path``).  The unified
entry point is :func:`convert_checkpoint` (CLI: ``scripts/convert_model.py``).
"""

from __future__ import annotations

from typing import Callable

from ....utils.registry import Registry


OMNI_CONVERT_REGISTRY = Registry("OmniConvert")

ConvertFn = Callable[..., None]


def convert_checkpoint(model_path: str, output_dir: str, **kwargs) -> None:
    """Dispatch to the registered converter for ``model_path``'s ``model_type``."""
    # Lazy import to break the convert_registry ↔ modules cycle: every family's
    # ``convert_model`` (imported by ``modules/__init__``) imports this module.
    from ..modules import read_hf_model_type

    model_type = read_hf_model_type(model_path)
    converter: ConvertFn = OMNI_CONVERT_REGISTRY[model_type]()
    converter(model_path, output_dir, **kwargs)


__all__ = ["OMNI_CONVERT_REGISTRY", "convert_checkpoint"]
