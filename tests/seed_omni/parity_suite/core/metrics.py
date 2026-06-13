"""Metric helpers for SeedOmni V2 parity comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


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
        passed = torch.allclose(actual_cpu, expected_cpu, rtol=tolerance.rtol, atol=tolerance.atol)

    diff = (actual_cpu.to(torch.float64) - expected_cpu.to(torch.float64)).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    denom = expected_cpu.to(torch.float64).abs().clamp_min(1e-12)
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    message = "" if passed else f"tensor mismatch: max_abs_diff={max_abs:.6g}, max_rel_diff={max_rel:.6g}"
    return MetricResult(passed, path, message=message, max_abs_diff=max_abs, max_rel_diff=max_rel)


__all__ = ["MetricResult", "Tolerance", "compare_tensors", "tolerance_from_policy"]
