"""Subject-owned reference observation capture helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


# Base capture -----------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceStopPolicy:
    """Stop reference execution once selected observation fields are captured."""

    fields: frozenset[str] = frozenset()
    mode: str = "all"

    def __post_init__(self) -> None:
        if self.mode not in {"all", "any"}:
            raise ValueError(f"ReferenceStopPolicy.mode must be 'all' or 'any', got {self.mode!r}.")


class ReferenceObservationSatisfied(Exception):
    """Raised when a reference capture has observed enough fields."""

    def __init__(self, observations: Mapping[str, list[Any]]) -> None:
        super().__init__("Reference observation stop policy was satisfied.")
        self.observations = {name: list(values) for name, values in observations.items()}


class ReferenceObservationCapture:
    """Base class for reference subjects that need capture around their run path."""

    def __init__(self, stop_policy: ReferenceStopPolicy | None = None) -> None:
        self._observations: dict[str, list[Any]] = {}
        self._stop_policy = stop_policy or ReferenceStopPolicy()

    @contextmanager
    def install(self, subject: Any) -> Iterator[ReferenceObservationCapture]:
        del subject
        yield self

    def ensure_field(self, name: str) -> None:
        self._observations.setdefault(name, [])

    def record(self, name: str, value: Any) -> None:
        self._observations.setdefault(name, []).append(value)
        if self._stop_policy_satisfied():
            raise ReferenceObservationSatisfied(self._observations)

    def observations(self) -> Mapping[str, list[Any]]:
        return self._observations

    @property
    def stop_policy(self) -> ReferenceStopPolicy:
        return self._stop_policy

    def _stop_policy_satisfied(self) -> bool:
        fields = self._stop_policy.fields
        if not fields:
            return False
        observed = {name for name in fields if self._observations.get(name)}
        if self._stop_policy.mode == "any":
            return bool(observed)
        return fields.issubset(observed)


@contextmanager
def reference_observation_stop_policy(
    capture: ReferenceObservationCapture,
    stop_policy: ReferenceStopPolicy,
) -> Iterator[ReferenceObservationCapture]:
    """Temporarily apply an early-stop policy to an existing capture object."""

    original = capture._stop_policy
    capture._stop_policy = stop_policy
    try:
        yield capture
    finally:
        capture._stop_policy = original


class NullReferenceObservationCapture(ReferenceObservationCapture):
    """No-op capture used by subjects that only return explicit observations."""


# Method wrapping --------------------------------------------------------------


class MethodPatchObservationCapture(ReferenceObservationCapture):
    """Reference capture helper that temporarily wraps subject methods."""

    def __init__(self, stop_policy: ReferenceStopPolicy | None = None) -> None:
        super().__init__(stop_policy=stop_policy)
        self._patches: list[tuple[Any, str, Any]] = []

    @contextmanager
    def install(self, subject: Any) -> Iterator[MethodPatchObservationCapture]:
        self._patches = []
        try:
            self.configure(subject)
            yield self
        finally:
            for owner, name, original in reversed(self._patches):
                setattr(owner, name, original)
            self._patches = []

    def configure(self, subject: Any) -> None:
        del subject

    def patch_method(self, owner: Any, name: str, wrapper_factory: Callable[[Any], Any]) -> None:
        original = getattr(owner, name)
        setattr(owner, name, wrapper_factory(original))
        self._patches.append((owner, name, original))


__all__ = [
    "MethodPatchObservationCapture",
    "NullReferenceObservationCapture",
    "ReferenceObservationSatisfied",
    "ReferenceObservationCapture",
    "ReferenceStopPolicy",
    "reference_observation_stop_policy",
]
