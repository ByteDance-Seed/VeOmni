"""Tests for comparator, metrics, and report primitives."""

from __future__ import annotations

import torch

from tests.seed_omni.parity_suite.core import (
    ParityReport,
    ProbeReport,
    Tolerance,
    compare_values,
    tolerance_from_policy,
)


def test_exact_comparison_detects_token_mismatch() -> None:
    result = compare_values(torch.tensor([1, 2]), torch.tensor([1, 3]), tolerance=Tolerance("exact", kind="exact"))

    assert not result.passed
    assert result.path == "$"


def test_numeric_tolerance_allows_small_tensor_drift() -> None:
    tolerance = tolerance_from_policy("logits", {"logits": {"rtol": 0.0, "atol": 0.01}})

    result = compare_values(torch.tensor([1.005]), torch.tensor([1.0]), tolerance=tolerance)

    assert result.passed
    assert result.max_abs_diff is not None and result.max_abs_diff > 0


def test_nested_comparison_treats_inner_lists_as_structure() -> None:
    tolerance = Tolerance("hidden", rtol=0.0, atol=0.0)

    result = compare_values(
        [{"a": torch.tensor([1.0]), "b": [torch.tensor([2.0]), torch.tensor([4.0])]}],
        [{"a": torch.tensor([1.0]), "b": [torch.tensor([3.0]), torch.tensor([4.0])]}],
        tolerance=tolerance,
    )

    assert not result.passed
    assert result.path == "$[-1].b[0]"


def test_per_step_lists_compare_last_step_by_default() -> None:
    tolerance = Tolerance("logits", rtol=0.0, atol=0.0)

    result = compare_values(
        [torch.tensor([0.0]), torch.tensor([2.0])],
        [torch.tensor([1.0]), torch.tensor([2.0])],
        tolerance=tolerance,
    )

    assert result.passed
    assert result.path == "$[-1]"


def test_report_exposes_first_failure() -> None:
    passed_metric = compare_values(torch.tensor([1.0]), torch.tensor([1.0]), tolerance=Tolerance("hidden"))
    failed_metric = compare_values(torch.tensor([2.0]), torch.tensor([3.0]), tolerance=Tolerance("hidden"))
    report = ParityReport(
        case_id="toy.graph.case",
        probes=(
            ProbeReport(node="a.forward", probe="a", passed=passed_metric.passed, metric=passed_metric),
            ProbeReport(node="b.forward", probe="b", passed=failed_metric.passed, metric=failed_metric),
        ),
    )

    assert not report.all_pass
    assert report.first_failure is not None
    assert report.to_dict()["first_failure"]["probe"] == "b"
