"""Reference handler output contract for the parity suite."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict


class ReferenceRunOutput(TypedDict):
    """Suite contract for normal reference handler return values."""

    canonical: Mapping[str, Any]
    reference: Mapping[str, Any]


def make_reference_run_output(
    canonical: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a plain dict shaped as ``{"canonical": ..., "reference": ...}``."""

    return {"canonical": canonical, "reference": reference}


def is_reference_run_output(value: Any) -> bool:
    """Return whether ``value`` matches the reference run output contract."""

    if not isinstance(value, Mapping):
        return False
    return "canonical" in value and "reference" in value


def canonical_from_reference_output(value: Any) -> Mapping[str, Any]:
    """Extract canonical payload from reference handler output for V2 request dispatch."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(
            "Reference handler output must be a mapping shaped as "
            '{"canonical": ..., "reference": ...}, '
            f"got {type(value).__name__}."
        )
    if "canonical" not in value:
        raise TypeError(
            'Reference handler output is missing required key "canonical". '
            'Expected mapping shaped as {"canonical": ..., "reference": ...}.'
        )
    if "reference" not in value:
        raise TypeError(
            'Reference handler output is missing required key "reference". '
            'Expected mapping shaped as {"canonical": ..., "reference": ...}.'
        )
    canonical = value["canonical"]
    reference = value["reference"]
    if not isinstance(canonical, Mapping):
        raise TypeError(f'Reference handler output key "canonical" must be a mapping. Got {type(canonical).__name__}.')
    if not isinstance(reference, Mapping):
        raise TypeError(f'Reference handler output key "reference" must be a mapping. Got {type(reference).__name__}.')
    return canonical


__all__ = [
    "ReferenceRunOutput",
    "canonical_from_reference_output",
    "is_reference_run_output",
    "make_reference_run_output",
]
