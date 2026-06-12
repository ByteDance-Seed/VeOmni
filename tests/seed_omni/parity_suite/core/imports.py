"""Import helpers for model-local parity plugins."""

from __future__ import annotations

import importlib
from typing import Any


def import_object(path: str) -> Any:
    module_name, sep, object_name = path.partition(":")
    if not sep:
        raise ValueError(f"Expected import path in `module:object` form, got {path!r}")
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in object_name.split("."):
        obj = getattr(obj, part)
    return obj


def import_optional_module(module_name: str | None) -> Any | None:
    if not module_name:
        return None
    return importlib.import_module(module_name)
