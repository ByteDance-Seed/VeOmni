"""Pytest collection entrypoint for SeedOmni V2 parity cases."""

from __future__ import annotations

import pytest

from .core import ParityCase, case_skip_reason, discover_cases
from .launcher import run_case_with_pytest_launcher, should_use_pytest_launcher
from .runner import run_parity_case


CASES = discover_cases()


def _case_id(case: ParityCase) -> str:
    return case.node_id


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_seed_omni_v2_parity_case(case: ParityCase, request: pytest.FixtureRequest) -> None:
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
