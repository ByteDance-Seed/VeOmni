"""Default adapters for generated parity cases."""

from __future__ import annotations

from typing import Any

import torch

from tests.seed_omni.parity_suite.core.metrics import compare_tensor
from tests.seed_omni.parity_suite.core.probes import load_probe_bindings
from tests.seed_omni.parity_suite.core.report import ParityReport
from tests.seed_omni.parity_suite.core.spec import CaseSpec


class ReferenceV2ProbeAdapter:
    """Generic adapter for reference-capture versus V2-capture comparisons.

    Model-local code supplies captures/probes modules. The suite owns fixture
    loading, online reference capture, reference extraction, recursive compare,
    and report assembly.
    """

    def __init__(self, *, case: CaseSpec, probes_module: Any | None, captures_module: Any | None) -> None:
        self.case = case
        self.probes_module = probes_module
        self.captures_module = captures_module

    def run_case(self) -> ParityReport:
        if self.captures_module is None:
            raise ValueError(f"{self.case.node_id} requires a captures module.")
        if self.probes_module is None:
            raise ValueError(f"{self.case.node_id} requires a probes module.")

        bindings = load_probe_bindings(self.probes_module)
        capture_source = "fixture_cache" if self.case.fixture_path else "online"
        fixture = (
            self.captures_module.load_fixture(self.case.fixture_path)
            if self.case.fixture_path
            else self.captures_module.capture_case(self.case, bindings)
        )

        fixture_case_id = self.case.case.get("fixture_case_id") if self.case.fixture_path else self.case.id
        fixture_case_id = fixture_case_id or self.case.id
        actual_case_id = self.captures_module.fixture_case_id(fixture) if fixture else None
        case_id_passes = actual_case_id == fixture_case_id

        probe_reports = {
            probe: {
                "binding_declared": probe in bindings,
                "passes": probe in bindings,
            }
            for probe in self.case.probes
        }
        if fixture:
            available = self.captures_module.available_fixture_paths(fixture, self.case, bindings, self.case.probes)
            for probe, is_available in available.items():
                probe_reports[probe]["reference_capture_available"] = is_available
                probe_reports[probe]["passes"] = bool(probe_reports[probe]["binding_declared"] and is_available)

        v2_capture = None
        if hasattr(self.captures_module, "capture_v2_case"):
            v2_capture = self.captures_module.capture_v2_case(self.case, fixture, bindings)
            tolerance = self.captures_module.fixture_tolerance(fixture)
            for probe in self.case.probes:
                if probe not in bindings:
                    continue
                try:
                    reference_value = self.captures_module.extract_reference_value(fixture, self.case, probe)
                    v2_value = self.captures_module.extract_v2_value(v2_capture, bindings[probe])
                except KeyError as exc:
                    probe_reports[probe].update({"passes": False, "missing_value": str(exc)})
                    continue
                probe_reports[probe].update(compare_values(v2_value, reference_value, tolerance))
        else:
            for report in probe_reports.values():
                report["passes"] = False
                report["missing_v2_capture"] = True

        all_pass = bool(case_id_passes and all(item["passes"] for item in probe_reports.values()))
        return ParityReport(
            case_id=self.case.node_id,
            category=self.case.category,
            all_pass=all_pass,
            probes=probe_reports,
            metadata={
                "mode": "reference_v2_probe_comparison" if v2_capture is not None else "missing_v2_capture",
                "capture_source": capture_source,
                "graph": self.case.graph,
                "fixture_case_id": fixture_case_id,
                "actual_case_id": actual_case_id,
                "case_id_passes": case_id_passes,
                "archive_dependency": False,
            },
        )


def compare_values(actual: Any, expected: Any, tolerance: dict[str, float]) -> dict[str, Any]:
    if torch.is_tensor(actual) and torch.is_tensor(expected):
        return compare_tensor(actual.detach().cpu(), expected.detach().cpu(), tolerance)
    if isinstance(actual, dict) and isinstance(expected, dict):
        children: dict[str, Any] = {}
        keys_match = set(actual) == set(expected)
        for key in sorted(set(actual) | set(expected)):
            if key not in actual or key not in expected:
                children[key] = {"passes": False, "missing": "actual" if key not in actual else "expected"}
            else:
                children[key] = compare_values(actual[key], expected[key], tolerance)
        return {"passes": bool(keys_match and all(item["passes"] for item in children.values())), "items": children}
    if isinstance(actual, list) and isinstance(expected, list):
        children = {
            str(idx): compare_values(actual_item, expected_item, tolerance)
            for idx, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=False))
        }
        lengths_match = len(actual) == len(expected)
        return {"passes": bool(lengths_match and all(item["passes"] for item in children.values())), "items": children}
    equal = actual == expected
    return {"passes": bool(equal), "actual": actual, "expected": expected}
