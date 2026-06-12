"""BAGEL model-local capture shim for parity_suite.

This module only loads BAGEL's reference backend. Generic capture/fixture
runtime helpers live in `tests.seed_omni.parity_suite.capture.runtime`.
"""

from __future__ import annotations

from typing import Any

import torch

from tests.seed_omni.bagel.transformers import BagelReferenceConfig, BagelReferenceForCausalLM
from tests.seed_omni.parity_suite.backends.transformers import TransformersBackend
from tests.seed_omni.parity_suite.capture.runtime import (
    available_fixture_paths,
    extract_reference_value,
    fixture_case_id,
    fixture_tolerance,
    get_path,
    load_fixture,
)
from tests.seed_omni.parity_suite.capture.runtime import capture_case as capture_reference_case
from tests.seed_omni.parity_suite.core.probes import ProbeBinding
from tests.seed_omni.parity_suite.core.spec import CaseSpec


__all__ = [
    "available_fixture_paths",
    "capture_case",
    "extract_reference_value",
    "fixture_case_id",
    "fixture_tolerance",
    "get_path",
    "load_fixture",
]


def capture_case(case: CaseSpec, probe_bindings: dict[str, ProbeBinding]) -> dict[str, Any]:
    """Online reference capture for a BAGEL parity case."""

    return capture_reference_case(case, probe_bindings, load_reference_model=_load_reference_model)


def _load_reference_model(case: CaseSpec) -> torch.nn.Module:
    if case.reference_backend.model:
        return TransformersBackend(case.reference_backend, case).load().model
    # Direct unit/smoke calls can exercise the capture executor without a real
    # checkpoint. Pytest parity cases are env-gated before reaching this path.
    return BagelReferenceForCausalLM(BagelReferenceConfig(vocab_size=32, hidden_size=16, intermediate_size=32)).eval()
