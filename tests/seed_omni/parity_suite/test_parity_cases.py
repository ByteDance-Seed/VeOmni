"""Pytest collection entrypoint for SeedOmni V2 parity cases."""

from __future__ import annotations

import pytest

from .core import ParityCase, case_skip_reason, discover_cases
from .runner import run_parity_case


CASES = discover_cases()


def _case_id(case: ParityCase) -> str:
    return case.node_id


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_seed_omni_v2_parity_case(case: ParityCase) -> None:
    reason = case_skip_reason(case)
    if reason:
        pytest.skip(reason)

    report = run_parity_case(case)
    if not report.all_pass:
        pytest.fail(str(report.to_dict()))
