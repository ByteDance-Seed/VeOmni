"""Shared BAGEL reference helper utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypeVar


T = TypeVar("T")


def normalize_reference_kwargs(
    value: Any,
    *,
    alias_fields: Mapping[str, tuple[Callable[[Any], Any], tuple[str, ...]]] | None = None,
    direct_fields: Mapping[str, Callable[[Any], Any]] | None = None,
    pair_fields: Mapping[str, Callable[[Any], Any]] | None = None,
    error_prefix: str = "BAGEL reference kwargs",
) -> dict[str, Any]:
    """Normalize authored kwargs into official BAGEL reference kwargs."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{error_prefix} must be a mapping.")
    kwargs: dict[str, Any] = {}

    for target_key, (caster, source_keys) in (alias_fields or {}).items():
        for source_key in source_keys:
            if source_key in value:
                kwargs[target_key] = caster(value[source_key])
                break
    for key, caster in (direct_fields or {}).items():
        if key in value:
            kwargs[key] = caster(value[key])
    for key, caster in (pair_fields or {}).items():
        if key in value:
            kwargs[key] = caster(value[key])
    return kwargs


def int_pair(value: Any, *, name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a length-2 sequence.")
    return int(value[0]), int(value[1])


def float_pair(value: Any, *, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a length-2 sequence.")
    return float(value[0]), float(value[1])


def first_output_of_type(outputs: Iterable[Any], expected_type: type[T]) -> T | None:
    return next((item for item in outputs if isinstance(item, expected_type)), None)


__all__ = [
    "first_output_of_type",
    "float_pair",
    "int_pair",
    "normalize_reference_kwargs",
]
