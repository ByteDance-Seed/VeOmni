"""Pytest collection entrypoint for SeedOmni V2 parity cases."""

from __future__ import annotations

import pytest

from .core import ParityCase, discover_cases
from .runner import run_parity_case


CASES = discover_cases()


def _case_id(case: ParityCase) -> str:
    return case.node_id


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_seed_omni_v2_parity_case(case: ParityCase) -> None:
    reason = case.static_skip_reason()
    if reason:
        pytest.skip(reason)

    if case.requires_cuda:
        import torch

        if not torch.cuda.is_available():
            pytest.skip(f"{case.node_id} requires CUDA.")
        if case.min_cuda_devices and torch.cuda.device_count() < case.min_cuda_devices:
            pytest.skip(f"{case.node_id} requires {case.min_cuda_devices} CUDA devices.")

    report = run_parity_case(case)
    if not report.all_pass:
        pytest.fail(str(report.to_dict()))
