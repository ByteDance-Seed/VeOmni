"""Subject-owned reference observation adapters.

These classes are not the outer oracle capture pass. They are optional adapters
that a reference subject can install around its own methods to record values
that are easier to observe inside the official implementation than through a
generic field, hook, or extractor tap.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


# Base adapter -----------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceObservationAdapterStopPolicy:
    """Stop reference execution once selected observation fields are captured."""

    fields: frozenset[str] = frozenset()
    mode: str = "all"

    def __post_init__(self) -> None:
        if self.mode not in {"all", "any"}:
            raise ValueError(f"ReferenceObservationAdapterStopPolicy.mode must be 'all' or 'any', got {self.mode!r}.")


class ReferenceObservationAdapterSatisfied(Exception):
    """Raised when a reference observation adapter has observed enough fields."""

    def __init__(self, observations: Mapping[str, list[Any]]) -> None:
        super().__init__("Reference observation stop policy was satisfied.")
        self.observations = {name: list(values) for name, values in observations.items()}


class ReferenceObservationAdapter:
    """Subject-owned adapter that records observations while the reference runs."""

    def __init__(self, stop_policy: ReferenceObservationAdapterStopPolicy | None = None) -> None:
        self._observations: dict[str, list[Any]] = {}
        self._stop_policy = stop_policy or ReferenceObservationAdapterStopPolicy()

    @contextmanager
    def install(self, subject: Any) -> Iterator[ReferenceObservationAdapter]:
        del subject
        yield self

    def ensure_field(self, name: str) -> None:
        self._observations.setdefault(name, [])

    def record(self, name: str, value: Any) -> None:
        self._observations.setdefault(name, []).append(value)
        if self._stop_policy_satisfied():
            raise ReferenceObservationAdapterSatisfied(self._observations)

    def observations(self) -> Mapping[str, list[Any]]:
        return self._observations

    @property
    def stop_policy(self) -> ReferenceObservationAdapterStopPolicy:
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
def reference_observation_adapter_stop_policy(
    adapter: ReferenceObservationAdapter,
    stop_policy: ReferenceObservationAdapterStopPolicy,
) -> Iterator[ReferenceObservationAdapter]:
    """Temporarily apply an early-stop policy to an existing adapter."""

    original = adapter._stop_policy
    adapter._stop_policy = stop_policy
    try:
        yield adapter
    finally:
        adapter._stop_policy = original


class NullReferenceObservationAdapter(ReferenceObservationAdapter):
    """No-op adapter used by subjects that only return explicit observations."""


# Method wrapping --------------------------------------------------------------


class MethodPatchObservationAdapter(ReferenceObservationAdapter):
    """Observation adapter that temporarily wraps subject methods."""

    def __init__(self, stop_policy: ReferenceObservationAdapterStopPolicy | None = None) -> None:
        super().__init__(stop_policy=stop_policy)
        self._patches: list[tuple[Any, str, Any]] = []

    @contextmanager
    def install(self, subject: Any) -> Iterator[MethodPatchObservationAdapter]:
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
    "MethodPatchObservationAdapter",
    "NullReferenceObservationAdapter",
    "ReferenceObservationAdapterSatisfied",
    "ReferenceObservationAdapter",
    "ReferenceObservationAdapterStopPolicy",
    "reference_observation_adapter_stop_policy",
]
