"""Helpers shared by SeedOmni module tests."""

from __future__ import annotations

from pathlib import Path

from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.modeling_omni import OmniModel


def save_as_omni(root: Path, name: str, module) -> Path:
    """Save ``module`` as the only module of an :class:`OmniModel` under ``root``; return its subfolder."""
    config = OmniConfig(_module_entries={name: {"model_path": name}}, training_graphs={}, generation_graphs={})
    OmniModel(config, {name: module}).save_pretrained(root)
    return Path(root) / name


def load_from_omni(root: Path, name: str):
    """Load module ``name`` back through :class:`OmniModel`; return ``(config, module)``."""
    model = OmniModel.from_pretrained(root)
    return model.config._module_configs[name], model.modules_dict[name]
