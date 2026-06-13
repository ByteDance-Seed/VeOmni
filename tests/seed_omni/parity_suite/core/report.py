"""Structured reporting for SeedOmni V2 parity comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .metrics import MetricResult


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


__all__ = ["ParityReport", "ProbeReport"]
