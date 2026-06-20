"""Reference-side helpers and oracle backends for parity-suite execution."""

from .contract import (
    ReferenceCaptureResult,
    ReferenceOracle,
    ReferenceRunResult,
    ReferenceSubject,
    merge_reference_observations,
    normalize_reference_run_result,
)
from .oracles.factory import build_reference_oracle


__all__ = [
    "ReferenceCaptureResult",
    "ReferenceOracle",
    "ReferenceRunResult",
    "ReferenceSubject",
    "build_reference_oracle",
    "merge_reference_observations",
    "normalize_reference_run_result",
]
