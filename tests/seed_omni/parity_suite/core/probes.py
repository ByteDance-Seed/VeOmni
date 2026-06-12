"""Shared probe binding interface for parity_suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ProbeAnchor:
    """Location of a probe value in one execution backend."""

    path: str | None = None
    extractor: str | None = None
    node: str | None = None
    field: str | None = None

    @classmethod
    def parse(cls, value: str | Mapping[str, Any] | ProbeAnchor) -> ProbeAnchor:
        if isinstance(value, ProbeAnchor):
            return value
        if isinstance(value, str):
            if ":" in value:
                node, field = value.split(":", 1)
                return cls(path=value, node=node, field=field)
            return cls(path=value)
        return cls(
            path=value.get("path"),
            extractor=value.get("extractor"),
            node=value.get("node"),
            field=value.get("field"),
        )


@dataclass(frozen=True)
class ProbeBinding:
    """Semantic probe binding for a model-local V2 observation point."""

    v2: ProbeAnchor
    description: str = ""
    tolerance: str | None = None

    @classmethod
    def parse(cls, value: Mapping[str, Any] | ProbeBinding) -> ProbeBinding:
        if isinstance(value, ProbeBinding):
            return value
        return cls(
            v2=ProbeAnchor.parse(value["v2"]),
            description=str(value.get("description", "")),
            tolerance=value.get("tolerance"),
        )


def probe_binding(
    v2: str | Mapping[str, Any] | ProbeAnchor,
    description: str = "",
    *,
    tolerance: str | None = None,
) -> ProbeBinding:
    return ProbeBinding(
        v2=ProbeAnchor.parse(v2),
        description=description,
        tolerance=tolerance,
    )


def load_probe_bindings(probes_module: Any | None) -> dict[str, ProbeBinding]:
    if probes_module is None:
        return {}
    raw = getattr(probes_module, "PROBES", {})
    bindings: dict[str, ProbeBinding] = {}
    for name, value in raw.items():
        bindings[str(name)] = ProbeBinding.parse(value)
    return bindings


def missing_probe_bindings(probes: tuple[str, ...], bindings: Mapping[str, ProbeBinding]) -> list[str]:
    return [probe for probe in probes if probe not in bindings]
