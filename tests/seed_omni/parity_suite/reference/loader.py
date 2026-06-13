"""Reference oracle loading helpers."""

from __future__ import annotations

import importlib
from typing import Any

from tests.seed_omni.parity_suite.core import ReferenceSpec


def load_reference_model(spec: ReferenceSpec, **kwargs: Any) -> Any:
    """Load a reference model from a parsed reference spec."""

    if spec.loader == "vendored":
        return _load_vendored_reference(spec, **kwargs)
    if spec.loader == "transformers":
        return _load_transformers_reference(spec, **kwargs)
    raise ValueError(f"Unsupported reference loader: {spec.loader}")


def _load_vendored_reference(spec: ReferenceSpec, **kwargs: Any) -> Any:
    if not spec.module:
        raise ValueError("Vendored reference loader requires reference.module.")

    module = importlib.import_module(spec.module)
    if not hasattr(module, "load_vendored_model"):
        raise ValueError(f"Vendored reference module {spec.module!r} must expose load_vendored_model.")
    return module.load_vendored_model(spec.checkpoint, **kwargs)


def _load_transformers_reference(spec: ReferenceSpec, **kwargs: Any) -> Any:
    model_id = spec.checkpoint or spec.module
    if model_id is None:
        raise ValueError("Transformers reference loader requires reference.module or reference.checkpoint.")
    from transformers import AutoModel

    return AutoModel.from_pretrained(model_id, **kwargs)


__all__ = ["load_reference_model"]
