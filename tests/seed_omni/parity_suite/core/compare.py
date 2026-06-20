"""Metric, comparison, and reporting helpers for SeedOmni V2 parity checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch


# Comparison contract ----------------------------------------------------------


@dataclass(frozen=True)
class Tolerance:
    name: str
    kind: str = "numeric"
    rtol: float = 0.0
    atol: float = 0.0

    @property
    def exact(self) -> bool:
        return self.kind == "exact"


@dataclass(frozen=True)
class MetricResult:
    passed: bool
    path: str
    message: str = ""
    max_abs_diff: float | None = None
    max_rel_diff: float | None = None


# Public comparison helpers ----------------------------------------------------


def tolerance_from_policy(name: str, policies: dict[str, Any]) -> Tolerance:
    raw = dict(policies.get(name, {}) or {})
    kind = str(raw.get("kind", "numeric"))
    return Tolerance(
        name=name,
        kind=kind,
        rtol=float(raw.get("rtol", 0.0) or 0.0),
        atol=float(raw.get("atol", 0.0) or 0.0),
    )


def compare_tensors(actual: torch.Tensor, expected: torch.Tensor, *, tolerance: Tolerance, path: str) -> MetricResult:
    if actual.shape != expected.shape:
        return MetricResult(
            False, path, message=f"shape mismatch: actual={tuple(actual.shape)} expected={tuple(expected.shape)}"
        )

    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    if tolerance.exact:
        passed = torch.equal(actual_cpu, expected_cpu)
    else:
        passed = torch.allclose(
            actual_cpu.to(torch.float64),
            expected_cpu.to(torch.float64),
            rtol=tolerance.rtol,
            atol=tolerance.atol,
        )

    diff = (actual_cpu.to(torch.float64) - expected_cpu.to(torch.float64)).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    denom = expected_cpu.to(torch.float64).abs().clamp_min(1e-12)
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    message = "" if passed else f"tensor mismatch: max_abs_diff={max_abs:.6g}, max_rel_diff={max_rel:.6g}"
    return MetricResult(passed, path, message=message, max_abs_diff=max_abs, max_rel_diff=max_rel)


def compare_values(
    actual: Any,
    expected: Any,
    *,
    tolerance: Tolerance,
    path: str = "$",
    compare_steps: str = "last",
) -> MetricResult:
    """Compare nested values and return the first mismatch."""

    return _compare_values(
        actual,
        expected,
        tolerance=tolerance,
        path=path,
        compare_steps=compare_steps,
        allow_step_list=True,
    )


# Internal recursive comparison ------------------------------------------------


def _compare_values(
    actual: Any,
    expected: Any,
    *,
    tolerance: Tolerance,
    path: str,
    compare_steps: str,
    allow_step_list: bool,
) -> MetricResult:
    """Compare nested values and return the first mismatch."""

    if allow_step_list and _is_step_list(actual) and _is_step_list(expected):
        if not actual or not expected:
            return MetricResult(False, path, message="empty per-step list")
        if compare_steps == "last":
            return _compare_values(
                actual[-1],
                expected[-1],
                tolerance=tolerance,
                path=f"{path}[-1]",
                compare_steps=compare_steps,
                allow_step_list=False,
            )
        if compare_steps != "all":
            return MetricResult(False, path, message=f"unsupported step policy: {compare_steps!r}")
        # compare_steps == "all": fall through to per-element (per-step) comparison below.

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
        result = _compare_values(
            actual[key],
            expected[key],
            tolerance=tolerance,
            path=f"{path}.{key}",
            compare_steps=compare_steps,
            allow_step_list=False,
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
        result = _compare_values(
            actual_item,
            expected_item,
            tolerance=tolerance,
            path=f"{path}[{idx}]",
            compare_steps=compare_steps,
            allow_step_list=False,
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


# Report contract --------------------------------------------------------------


@dataclass(frozen=True)
class ProbeReport:
    node: str
    probe: str
    passed: bool
    metric: MetricResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "probe": self.probe,
            "passed": self.passed,
            "path": self.metric.path,
            "message": self.metric.message,
            "max_abs_diff": self.metric.max_abs_diff,
            "max_rel_diff": self.metric.max_rel_diff,
        }


@dataclass(frozen=True)
class ParityReport:
    case_id: str
    probes: tuple[ProbeReport, ...]

    @property
    def all_pass(self) -> bool:
        return all(probe.passed for probe in self.probes)

    @property
    def first_failure(self) -> ProbeReport | None:
        for probe in self.probes:
            if not probe.passed:
                return probe
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "all_pass": self.all_pass,
            "first_failure": None if self.first_failure is None else self.first_failure.to_dict(),
            "probes": [probe.to_dict() for probe in self.probes],
        }


__all__ = [
    "MetricResult",
    "ParityReport",
    "ProbeReport",
    "Tolerance",
    "compare_tensors",
    "compare_values",
    "tolerance_from_policy",
]
