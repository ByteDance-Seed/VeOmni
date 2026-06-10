"""Registry for monolithic HF checkpoint → SeedOmni V2 split checkpoints.

Each model family registers a converter under its upstream HuggingFace
``model_type`` (read from ``config.json`` at ``model_path``).  The unified
entry point is :func:`convert_checkpoint` (CLI: ``scripts/convert_model.py``).
"""

from __future__ import annotations

from typing import Callable

from transformers import PretrainedConfig

from ...utils.registry import Registry


OMNI_CONVERT_REGISTRY = Registry("OmniConvert")

ConvertFn = Callable[..., None]


def read_hf_model_type(model_path: str) -> str:
    """Read upstream ``model_type`` from a HuggingFace checkpoint directory."""
    config_dict, _ = PretrainedConfig.get_config_dict(model_path)
    model_type = config_dict.get("model_type")
    if not model_type and "BAGEL-7B-MoT" in config_dict.get("name", []):
        return "bagel"
    if not model_type:
        raise ValueError(f"Checkpoint at {model_path} has no `model_type` in config.json.")
    return model_type


def convert_checkpoint(model_path: str, output_dir: str, **kwargs) -> None:
    """Dispatch to the registered converter for ``model_path``'s ``model_type``."""
    # Side-effect: family packages register their converters on import.
    from .modules import bagel, janus  # noqa: F401

    model_type = read_hf_model_type(model_path)
    if model_type not in set(OMNI_CONVERT_REGISTRY.valid_keys()):
        known = sorted(OMNI_CONVERT_REGISTRY.valid_keys())
        raise KeyError(
            f"No SeedOmni convert handler for model_type={model_type!r} (from {model_path}). Known: {known}"
        )
    converter: ConvertFn = OMNI_CONVERT_REGISTRY[model_type]()
    converter(model_path, output_dir, **kwargs)


__all__ = ["OMNI_CONVERT_REGISTRY", "convert_checkpoint", "read_hf_model_type"]
