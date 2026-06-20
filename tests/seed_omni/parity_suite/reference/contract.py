"""Reference oracle runtime contract for parity execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn

from tests.seed_omni.parity_suite.core.runtime import RunCaptureOptions
from tests.seed_omni.parity_suite.reference.capture.spec import ReferenceCapturePlan


# Runtime value contract -------------------------------------------------------


@dataclass(frozen=True)
class ReferenceRunResult:
    """Canonical reference output consumed by the shared parity runner."""

    canonical: Mapping[str, Any]
    observations: Mapping[str, list[Any]]
    raw_output: Any | None = None


@dataclass(frozen=True)
class ReferenceCaptureResult:
    """Captured reference observations and release diagnostics."""

    observations: dict[str, list[Any]]
    run_output: ReferenceRunResult
    memory_before_release: int = 0
    memory_after_release: int = 0


# Runtime protocol contract ----------------------------------------------------


class ReferenceOracle(Protocol):
    """Backend-independent reference execution protocol."""

    def capture(
        self,
        *,
        inputs: Mapping[str, Any],
        plan: ReferenceCapturePlan,
        device: torch.device,
        dtype: torch.dtype,
        capture_options: RunCaptureOptions,
    ) -> ReferenceCaptureResult:
        """Run reference execution and return canonical observations."""


class ReferenceSubject(Protocol):
    """Optional subject shape for references that wrap a larger vendor runtime."""

    @property
    def hook_root(self) -> nn.Module:
        """Root module used for suite-installed observation hooks."""


# Public normalization helpers -------------------------------------------------


def normalize_reference_run_result(value: Any) -> ReferenceRunResult:
    """Normalize a reference runner return value into ``ReferenceRunResult``."""

    if isinstance(value, ReferenceRunResult):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            f"Reference runner output must be ReferenceRunResult or a mapping; got {type(value).__name__}."
        )

    if "canonical" in value and "observations" in value:
        canonical = value["canonical"]
        observations = value["observations"]
        raw_output = value.get("raw_output")
        return ReferenceRunResult(
            canonical=_require_mapping(canonical, field="canonical"),
            observations=_normalize_observations(observations),
            raw_output=raw_output,
        )
    if "canonical" in value:
        raise TypeError('Reference runner mappings with "canonical" payloads must also declare "observations".')

    return ReferenceRunResult(canonical={}, observations=_normalize_observations(value), raw_output=value)


def merge_reference_observations(
    result: ReferenceRunResult,
    captured: Mapping[str, list[Any]],
) -> ReferenceRunResult:
    """Return ``result`` with captured observations appended by field name."""

    if not captured:
        return result
    observations = {name: list(values) for name, values in result.observations.items()}
    for name, values in captured.items():
        observations.setdefault(name, []).extend(values)
    return ReferenceRunResult(canonical=result.canonical, observations=observations, raw_output=result.raw_output)


# Internal normalization helpers ----------------------------------------------


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Reference runner output key {field!r} must be a mapping; got {type(value).__name__}.")
    return value


def _normalize_observations(value: Any) -> Mapping[str, list[Any]]:
    observations = _require_mapping(value, field="observations")
    normalized: dict[str, list[Any]] = {}
    for key, item in observations.items():
        if isinstance(item, list):
            normalized[str(key)] = item
        else:
            normalized[str(key)] = [item]
    return normalized


__all__ = [
    "ReferenceCaptureResult",
    "ReferenceOracle",
    "ReferenceRunResult",
    "ReferenceSubject",
    "merge_reference_observations",
    "normalize_reference_run_result",
]
