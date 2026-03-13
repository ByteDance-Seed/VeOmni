"""Toy config loading and validation for tests."""

import json
import os
from pathlib import Path


# Default toy config directory relative to VeOmni repo root
_DEFAULT_TOY_CONFIG_DIR = Path(__file__).parent.parent.parent / "tests" / "toy_config"


def get_toy_config_path(model_name: str, toy_config_dir: str | Path | None = None) -> str:
    """Return path to toy config for a model.

    Handles both directory-style (``llama31_toy/``) and file-style
    (``qwen3_5_toy/config.json``) configs.

    Parameters
    ----------
    model_name : str
        Model short name, e.g. ``"llama31"``, ``"qwen3_5"``.
    toy_config_dir : str or Path, optional
        Override for the toy config root directory.

    Returns
    -------
    str
        Absolute path to the config directory or ``config.json`` file.

    Raises
    ------
    FileNotFoundError
        If no toy config exists for the given model name.
    """
    base = Path(toy_config_dir) if toy_config_dir else _DEFAULT_TOY_CONFIG_DIR
    dir_path = base / f"{model_name}_toy"
    if dir_path.is_dir():
        config_json = dir_path / "config.json"
        if config_json.exists():
            return str(config_json)
        return str(dir_path)
    raise FileNotFoundError(f"No toy config found for {model_name} at {dir_path}")


def validate_toy_config(config_path: str) -> dict:
    """Load and validate a toy config.

    Parameters
    ----------
    config_path : str
        Path to a ``config.json`` or its parent directory.

    Returns
    -------
    dict
        Parsed config dictionary.

    Raises
    ------
    AssertionError
        If required keys are missing.
    """
    if os.path.isdir(config_path):
        config_path = os.path.join(config_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    assert "model_type" in config, f"Toy config missing 'model_type': {config_path}"
    return config
