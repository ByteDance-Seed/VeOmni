"""Scoped observation support for SeedOmni V2 parity tests."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator


PARITY_ENABLE_ENV = "VEOMNI_V2_TEST_ENABLE_PARITY_CHECK"
DEFAULT_MAX_CAPTURE_TENSOR_NUMEL = 1_000_000
ObserverRecords = MutableMapping[tuple[str, str], list[dict[str, Any]]]


@dataclass
class _ObservationState:
    whitelist: dict[tuple[str, str], frozenset[str]]
    records: ObserverRecords
    max_tensor_numel: int


_OBSERVATION_STATE: ContextVar[_ObservationState | None] = ContextVar("seed_omni_v2_observation_state", default=None)


@contextmanager
def arm_observer(
    whitelist: Mapping[tuple[str, str], Iterable[str]],
    *,
    sink: ObserverRecords | None = None,
    max_tensor_numel: int = DEFAULT_MAX_CAPTURE_TENSOR_NUMEL,
) -> Iterator[ObserverRecords]:
    """Arm the generation observer under the parity-test gate."""

    if max_tensor_numel < 0:
        raise ValueError("max_tensor_numel must be non-negative.")
    normalized = _normalize_observation_whitelist(whitelist)
    records: ObserverRecords = sink if sink is not None else {}
    if os.environ.get(PARITY_ENABLE_ENV) != "1":
        yield records
        return

    token = _OBSERVATION_STATE.set(
        _ObservationState(whitelist=normalized, records=records, max_tensor_numel=max_tensor_numel)
    )
    try:
        yield records
    finally:
        _OBSERVATION_STATE.reset(token)


def observe_node_output(state: str, node: str, out: Mapping[str, Any]) -> None:
    """Record whitelisted node-return values for an armed observer."""

    observer = _OBSERVATION_STATE.get()
    if observer is None:
        return

    key = (state, node)
    fields = observer.whitelist.get(key)
    if fields is None:
        return

    record: dict[str, Any] = {}
    for field in fields:
        if field not in out:
            continue
        record[field] = _materialize_observed_value(
            out[field],
            max_tensor_numel=observer.max_tensor_numel,
            field_path=f"{state}:{node}.{field}",
        )
    observer.records.setdefault(key, []).append(record)


def _normalize_observation_whitelist(
    whitelist: Mapping[tuple[str, str], Iterable[str]],
) -> dict[tuple[str, str], frozenset[str]]:
    if not whitelist:
        raise ValueError("Observation whitelist is required and must not be empty.")

    normalized: dict[tuple[str, str], frozenset[str]] = {}
    for key, fields in whitelist.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError(f"Observation whitelist keys must be (state, node) tuples, got {key!r}.")
        state, node = key
        field_set = frozenset(str(field) for field in fields)
        if not field_set:
            raise ValueError(f"Observation whitelist for {(state, node)!r} must include at least one field.")
        normalized[(str(state), str(node))] = field_set
    return normalized


def _is_torch_tensor(value: Any) -> bool:
    import torch

    return isinstance(value, torch.Tensor)


def _materialize_observed_value(value: Any, *, max_tensor_numel: int, field_path: str) -> Any:
    if _is_torch_tensor(value):
        if value.numel() > max_tensor_numel:
            raise ValueError(
                f"Observed field {field_path} has {value.numel()} elements, "
                f"exceeding the capture limit {max_tensor_numel}."
            )
        return value.detach().cpu().clone()

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Mapping):
        return {
            key: _materialize_observed_value(
                child,
                max_tensor_numel=max_tensor_numel,
                field_path=f"{field_path}.{key}",
            )
            for key, child in value.items()
        }

    if isinstance(value, tuple):
        return tuple(
            _materialize_observed_value(
                child,
                max_tensor_numel=max_tensor_numel,
                field_path=f"{field_path}[{idx}]",
            )
            for idx, child in enumerate(value)
        )

    if isinstance(value, list):
        return [
            _materialize_observed_value(
                child,
                max_tensor_numel=max_tensor_numel,
                field_path=f"{field_path}[{idx}]",
            )
            for idx, child in enumerate(value)
        ]

    raise TypeError(f"Observed field {field_path} has unsupported type {type(value).__name__}.")


__all__ = [
    "DEFAULT_MAX_CAPTURE_TENSOR_NUMEL",
    "PARITY_ENABLE_ENV",
    "ObserverRecords",
    "arm_observer",
    "observe_node_output",
]
