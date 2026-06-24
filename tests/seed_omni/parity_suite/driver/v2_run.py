"""Unified V2-side run context for parity driver hooks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch

from tests.seed_omni.parity_suite.core import ParityCase, RunCaptureOptions
from tests.seed_omni.parity_suite.reference.contract import ReferenceRunResult


@dataclass(frozen=True)
class V2RunContext:
    """Context shared by V2 request, execution, policy, and observation hooks."""

    case: ParityCase
    tier: str
    domain: str
    reference_output: Any
    canonical: Mapping[str, Any]
    whitelist: Mapping[tuple[str, str], frozenset[str]]
    device: torch.device
    dtype: torch.dtype
    capture_options: RunCaptureOptions
    purpose: str = "default"

    def with_purpose(self, purpose: str) -> V2RunContext:
        return replace(self, purpose=purpose)


def canonical_from_reference_output(reference_output: Any) -> dict[str, Any]:
    if reference_output is None:
        return {}
    if not isinstance(reference_output, ReferenceRunResult):
        raise TypeError(
            "V2 run context expects ReferenceRunResult from ReferenceOracle.capture; "
            f"got {type(reference_output).__name__}."
        )
    return dict(reference_output.canonical)


__all__ = ["V2RunContext", "canonical_from_reference_output"]
