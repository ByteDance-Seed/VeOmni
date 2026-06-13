"""Tensor materialization helpers for reference-side capture."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


DEFAULT_MAX_CAPTURE_TENSOR_NUMEL = 1_000_000


def materialize_reference_value(
    value: Any,
    *,
    max_tensor_numel: int = DEFAULT_MAX_CAPTURE_TENSOR_NUMEL,
    field_path: str = "reference",
) -> Any:
    """Materialize a small reference value as CPU-owned data."""

    if isinstance(value, torch.Tensor):
        if value.numel() > max_tensor_numel:
            raise ValueError(
                f"Reference tap {field_path} has {value.numel()} elements, "
                f"exceeding the capture limit {max_tensor_numel}."
            )
        return value.detach().cpu().clone()

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Mapping):
        return {
            key: materialize_reference_value(
                child,
                max_tensor_numel=max_tensor_numel,
                field_path=f"{field_path}.{key}",
            )
            for key, child in value.items()
        }

    if isinstance(value, tuple):
        return tuple(
            materialize_reference_value(
                child,
                max_tensor_numel=max_tensor_numel,
                field_path=f"{field_path}[{idx}]",
            )
            for idx, child in enumerate(value)
        )

    if isinstance(value, list):
        return [
            materialize_reference_value(
                child,
                max_tensor_numel=max_tensor_numel,
                field_path=f"{field_path}[{idx}]",
            )
            for idx, child in enumerate(value)
        ]

    raise TypeError(f"Reference tap {field_path} has unsupported type {type(value).__name__}.")


__all__ = ["DEFAULT_MAX_CAPTURE_TENSOR_NUMEL", "materialize_reference_value"]
