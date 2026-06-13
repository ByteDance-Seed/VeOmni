"""Recursive comparator for parity probe values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .metrics import MetricResult, Tolerance, compare_tensors


def compare_values(
    actual: Any,
    expected: Any,
    *,
    tolerance: Tolerance,
    path: str = "$",
    compare_steps: str = "last",
) -> MetricResult:
    """Compare nested values and return the first mismatch."""

    if _is_step_list(actual) and _is_step_list(expected) and compare_steps == "last":
        if not actual or not expected:
            return MetricResult(False, path, message="empty per-step list")
        return compare_values(
            actual[-1], expected[-1], tolerance=tolerance, path=f"{path}[-1]", compare_steps=compare_steps
        )

    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        return compare_tensors(actual, expected, tolerance=tolerance, path=path)

    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return _compare_mappings(actual, expected, tolerance=tolerance, path=path, compare_steps=compare_steps)

    if _is_sequence(actual) and _is_sequence(expected):
        return _compare_sequences(actual, expected, tolerance=tolerance, path=path, compare_steps=compare_steps)

    return _compare_scalars(actual, expected, tolerance=tolerance, path=path)


def _compare_mappings(
    actual: Mapping[Any, Any],
    expected: Mapping[Any, Any],
    *,
    tolerance: Tolerance,
    path: str,
    compare_steps: str,
) -> MetricResult:
    if set(actual) != set(expected):
        return MetricResult(
            False, path, message=f"mapping keys differ: actual={sorted(actual)} expected={sorted(expected)}"
        )
    for key in sorted(actual):
        result = compare_values(
            actual[key],
            expected[key],
            tolerance=tolerance,
            path=f"{path}.{key}",
            compare_steps=compare_steps,
        )
        if not result.passed:
            return result
    return MetricResult(True, path)


def _compare_sequences(
    actual: Sequence[Any],
    expected: Sequence[Any],
    *,
    tolerance: Tolerance,
    path: str,
    compare_steps: str,
) -> MetricResult:
    if len(actual) != len(expected):
        return MetricResult(
            False, path, message=f"sequence length differs: actual={len(actual)} expected={len(expected)}"
        )
    for idx, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
        result = compare_values(
            actual_item,
            expected_item,
            tolerance=tolerance,
            path=f"{path}[{idx}]",
            compare_steps=compare_steps,
        )
        if not result.passed:
            return result
    return MetricResult(True, path)


def _compare_scalars(actual: Any, expected: Any, *, tolerance: Tolerance, path: str) -> MetricResult:
    if tolerance.exact:
        passed = actual == expected
        return MetricResult(
            passed, path, message="" if passed else f"exact mismatch: actual={actual!r} expected={expected!r}"
        )

    try:
        actual_tensor = torch.as_tensor(actual)
        expected_tensor = torch.as_tensor(expected)
    except Exception:
        passed = actual == expected
        return MetricResult(
            passed, path, message="" if passed else f"value mismatch: actual={actual!r} expected={expected!r}"
        )
    return compare_tensors(actual_tensor, expected_tensor, tolerance=tolerance, path=path)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _is_step_list(value: Any) -> bool:
    return isinstance(value, list)


__all__ = ["compare_values"]
