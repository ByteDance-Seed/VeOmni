"""Reusable pytest entrypoint helpers for model-owned parity tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from .core import ParityCase, case_skip_reason, discover_cases
from .launcher import run_case_with_pytest_launcher, should_use_pytest_launcher
from .runner import run_parity_case


def make_parity_test(*model_dirs: str | Path) -> Any:
    """Return a parametrized pytest function for explicit model parity directories."""

    if not model_dirs:
        raise ValueError("make_parity_test requires at least one model parity directory.")
    cases = discover_cases(model_dirs)

    @pytest.mark.parametrize("case", cases, ids=_case_id)
    def test_func(case: ParityCase, request: pytest.FixtureRequest) -> None:
        run_pytest_parity_case(case, request)

    return test_func


def run_pytest_parity_case(case: ParityCase, request: pytest.FixtureRequest) -> None:
    """Run one discovered parity case under pytest."""

    if should_use_pytest_launcher(case, request):
        reason = case_skip_reason(case)
        if reason:
            pytest.skip(reason)
        result = run_case_with_pytest_launcher(case, request)
        if result.returncode != 0:
            pytest.fail(f"GPU launcher failed {case.node_id}. See log: {result.log_path}")
        return

    reason = case_skip_reason(case)
    if reason:
        pytest.skip(reason)

    report = run_parity_case(case)
    if not report.all_pass:
        pytest.fail(str(report.to_dict()))


def _case_id(case: ParityCase) -> str:
    return case.node_id


__all__ = ["make_parity_test", "run_pytest_parity_case"]
