"""Reference oracle loading helpers."""

from __future__ import annotations

import importlib
from typing import Any

from tests.seed_omni.parity_suite.core import ReferenceSpec


def load_transformers_reference_model(spec: ReferenceSpec, **kwargs: Any) -> Any:
    """Load a reference model through Transformers ``AutoModel``."""

    if spec.module is not None:
        _register_reference_model(spec.module)
    model_id = spec.checkpoint or spec.module
    if model_id is None:
        raise ValueError("Transformers reference loader requires reference.module or reference.checkpoint.")
    from transformers import AutoModel

    return AutoModel.from_pretrained(model_id, **kwargs)


def _register_reference_model(reference_module: str) -> None:
    if ":" not in reference_module:
        raise ValueError("reference.module must use 'module.path:ClassName' for Transformers reference registration.")
    module_name, class_name = reference_module.rsplit(":", 1)
    if not module_name or not class_name:
        raise ValueError("reference.module must use 'module.path:ClassName' for Transformers reference registration.")

    module = importlib.import_module(module_name)
    reference_class = getattr(module, class_name)
    register_auto_model = getattr(reference_class, "register_auto_model", None)
    if register_auto_model is not None:
        register_auto_model()


__all__ = ["load_transformers_reference_model"]
