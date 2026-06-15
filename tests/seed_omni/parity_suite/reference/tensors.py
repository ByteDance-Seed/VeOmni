"""Tensor materialization helpers for reference-side capture.

Reference-side and V2-side capture share one materialization implementation: the
durable observer materializer in ``veomni.models.seed_omni.observer``. This module
is a thin reference-named wrapper so capture code keeps a stable public name.
"""

from __future__ import annotations

from typing import Any

from veomni.models.seed_omni.observer import (
    DEFAULT_MAX_CAPTURE_TENSOR_NUMEL,
    _materialize_observed_value,
)


def materialize_reference_value(
    value: Any,
    *,
    max_tensor_numel: int = DEFAULT_MAX_CAPTURE_TENSOR_NUMEL,
    field_path: str = "reference",
) -> Any:
    """Materialize a small reference value as CPU-owned data."""

    return _materialize_observed_value(value, max_tensor_numel=max_tensor_numel, field_path=field_path)


__all__ = ["DEFAULT_MAX_CAPTURE_TENSOR_NUMEL", "materialize_reference_value"]
