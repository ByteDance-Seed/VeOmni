"""Generated pytest entrypoint for SeedOmni model parity cases."""

from __future__ import annotations

import pytest
import torch

from tests.seed_omni.parity_suite.core.registry import discover_cases
from tests.seed_omni.parity_suite.core.runner import run_parity_case
from tests.seed_omni.parity_suite.core.spec import CaseSpec


CASES = discover_cases()


def _case_id(case: CaseSpec) -> str:
    return case.node_id


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_seed_omni_parity_case(case: CaseSpec) -> None:
    reason = case.static_skip_reason()
    if reason:
        pytest.skip(reason)
    if case.env.requires_cuda and not torch.cuda.is_available():
        pytest.skip(f"{case.node_id} requires CUDA.")
    if case.env.min_cuda_devices and torch.cuda.device_count() < case.env.min_cuda_devices:
        pytest.skip(f"{case.node_id} requires {case.env.min_cuda_devices} CUDA devices.")

    report = run_parity_case(case)
    assert report.all_pass, report.to_dict()
