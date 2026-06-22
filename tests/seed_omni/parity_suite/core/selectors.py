"""Probe observation selection helpers."""

from __future__ import annotations

from typing import Any

import torch

from tests.seed_omni.parity_suite.core.config import ParityCase, ProbeMapping


# Public probe value selectors -------------------------------------------------


def reference_probe_values(taps: dict[str, list[Any]], mapping: ProbeMapping) -> list[Any]:
    """Read expected reference values for one resolved probe mapping."""

    try:
        return taps[mapping.ref_tap.field]
    except KeyError as exc:
        raise KeyError(f"Reference observation {mapping.ref_tap.field!r} was not captured.") from exc


def v2_probe_values(
    observations: dict[tuple[str, str], list[dict[str, Any]]],
    mapping: ProbeMapping,
    *,
    case: ParityCase,
) -> list[Any]:
    """Read observed V2 values for one resolved probe mapping."""

    values: list[Any] = []
    for node in case.nodes:
        if node.name != mapping.node or node.state is None:
            continue
        if mapping.state is not None and node.state != mapping.state:
            continue
        for record in observations.get((node.state, node.name), []):
            if mapping.v2_item_type is not None and record.get("_item_type") != mapping.v2_item_type:
                continue
            if mapping.v2_item_source is not None and record.get("_item_source") != mapping.v2_item_source:
                continue
            if mapping.v2_signal is not None and record.get("_fsm_signal") != mapping.v2_signal:
                continue
            if mapping.v2_field in record:
                values.append(record[mapping.v2_field])
    if not values:
        raise KeyError(f"V2 field {mapping.node}.{mapping.v2_field} was not observed.")
    return apply_v2_selector(values, mapping)


def apply_v2_selector(values: list[Any], mapping: ProbeMapping) -> list[Any]:
    """Apply the probe's V2 observation selector."""

    if mapping.v2_selector == "all":
        return values
    if mapping.v2_selector == "unique_consecutive":
        selected: list[Any] = []
        for value in values:
            if selected and _observed_values_equal(selected[-1], value):
                continue
            selected.append(value)
        return selected
    raise ValueError(f"Unsupported V2 selector {mapping.v2_selector!r} for probe {mapping.probe!r}.")


# Internal equality helpers ----------------------------------------------------


def _observed_values_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return bool(left.shape == right.shape and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(_observed_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(
            _observed_values_equal(left_item, right_item) for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


__all__ = [
    "apply_v2_selector",
    "reference_probe_values",
    "v2_probe_values",
]
