"""Probe catalog schema and resolver for parity probes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml


if TYPE_CHECKING:
    from .discovery import NodeSpec


# Probe schema -----------------------------------------------------------------


@dataclass(frozen=True)
class RefTapSpec:
    kind: str
    target: str
    field: str

    @classmethod
    def from_ref_raw(cls, raw: Any, *, probe: str) -> RefTapSpec:
        if not isinstance(raw, dict):
            raise TypeError(f"Probe {probe!r} ref must be a mapping.")
        has_field = "field" in raw
        has_hook = "hook" in raw
        has_extractor = "extractor" in raw
        declared = int(has_field) + int(has_hook) + int(has_extractor)
        if declared != 1:
            raise ValueError(f"Probe {probe!r} ref must declare exactly one of field, hook, or extractor.")
        if has_field:
            field = str(raw["field"])
            return cls(kind="field", target=field, field=field)
        if has_hook:
            return cls(kind="hook", target=str(raw["hook"]), field=probe)
        return cls(kind="extractor", target=str(raw["extractor"]), field=probe)


@dataclass(frozen=True)
class V2GradSpec:
    module: str
    parameter: str

    @classmethod
    def from_raw(cls, raw: Any, *, probe: str, node: str) -> V2GradSpec | None:
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise TypeError(f"Probe {probe!r} v2.grad must be a mapping.")
        if "parameter" not in raw:
            raise ValueError(f"Probe {probe!r} v2.grad must declare parameter.")
        module = str(raw["module"]) if "module" in raw else node.split(".", 1)[0]
        return cls(
            module=module,
            parameter=str(raw["parameter"]),
        )


_STEP_POLICIES = frozenset({"last", "all"})
_V2_SELECTORS = frozenset({"all", "unique_consecutive"})


@dataclass(frozen=True)
class ProbeMapping:
    node: str
    probe: str
    v2_field: str
    ref_tap: RefTapSpec
    tol: str
    state: str | None = None
    v2_item_type: str | None = None
    v2_item_source: str | None = None
    v2_signal: str | None = None
    v2_grad: V2GradSpec | None = None
    v2_selector: str = "all"
    step: str = "last"

    @classmethod
    def from_raw(cls, probe: str, raw: Mapping[str, Any]) -> ProbeMapping:
        if "v2" not in raw:
            raise ValueError(f"Probe {probe!r} must declare v2.")
        v2_raw = raw["v2"]
        if not isinstance(v2_raw, dict):
            raise TypeError(f"Probe {probe!r} v2 must be a mapping.")
        if "node" not in v2_raw:
            raise ValueError(f"Probe {probe!r} v2 must declare node.")
        if "field" not in v2_raw:
            raise ValueError(f"Probe {probe!r} v2 must declare field.")
        if "ref" not in raw:
            raise ValueError(f"Probe {probe!r} must declare ref.")
        if "tol" not in raw:
            raise ValueError(f"Probe {probe!r} must declare tol.")
        step = str(raw.get("step", "last"))
        if step not in _STEP_POLICIES:
            raise ValueError(
                f"Probe {probe!r} has unsupported step policy {step!r}; expected one of {sorted(_STEP_POLICIES)}."
            )
        v2_selector = str(v2_raw.get("selector", "all"))
        if v2_selector not in _V2_SELECTORS:
            raise ValueError(
                f"Probe {probe!r} has unsupported v2.selector {v2_selector!r}; "
                f"expected one of {sorted(_V2_SELECTORS)}."
            )
        node = str(v2_raw["node"])
        v2_field = str(v2_raw["field"])
        if v2_field == "loss":
            from tests.seed_omni.parity_suite.v2.observation import LOSS_FIELD

            v2_field = LOSS_FIELD
        return cls(
            node=node,
            probe=probe,
            v2_field=v2_field,
            ref_tap=RefTapSpec.from_ref_raw(raw["ref"], probe=probe),
            tol=str(raw["tol"]),
            state=None if v2_raw.get("state") is None else str(v2_raw["state"]),
            v2_item_type=None if v2_raw.get("item_type") is None else str(v2_raw["item_type"]),
            v2_item_source=None if v2_raw.get("source") is None else str(v2_raw["source"]),
            v2_signal=None if v2_raw.get("signal") is None else str(v2_raw["signal"]),
            v2_grad=V2GradSpec.from_raw(v2_raw.get("grad"), probe=probe, node=node),
            v2_selector=v2_selector,
            step=step,
        )


# Probe catalog ----------------------------------------------------------------


@dataclass(frozen=True)
class ProbeCatalog:
    probes: tuple[ProbeMapping, ...] = ()

    def for_probe_names(self, probe_names: Iterable[str]) -> tuple[ProbeMapping, ...]:
        """Select probes by public probe name, preserving multi-node probes."""

        requested = tuple(probe_names)
        if not requested:
            return self.probes
        by_name = {(probe.node, probe.probe): probe for probe in self.probes}
        selected: list[ProbeMapping] = []
        for probe_name in requested:
            matches = [probe for (node, name), probe in by_name.items() if name == probe_name]
            if not matches:
                raise KeyError(f"Probe {probe_name!r} is not declared in probes.yaml.")
            selected.extend(matches)
        return tuple(selected)

    def by_node(self) -> dict[str, tuple[ProbeMapping, ...]]:
        grouped: dict[str, list[ProbeMapping]] = {}
        for probe in self.probes:
            grouped.setdefault(probe.node, []).append(probe)
        return {node: tuple(items) for node, items in grouped.items()}


# Resolved runtime plan --------------------------------------------------------


@dataclass(frozen=True)
class ResolvedProbes:
    probes: tuple[ProbeMapping, ...]
    ref_taps: tuple[tuple[str, RefTapSpec], ...]
    v2_whitelist: dict[tuple[str, str], frozenset[str]]


# Public loading and resolution ------------------------------------------------


def load_probe_catalog(path: Path) -> ProbeCatalog:
    """Load a probe-keyed probes.yaml file."""

    if not path.exists():
        return ProbeCatalog()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping.")
    if "input_boundary" in data:
        raise ValueError(f"{path} must not declare input_boundary; input shape is driver-owned.")
    if "nodes" in data:
        raise ValueError(f"{path} must be probe-keyed; declare each public probe name at the top level.")

    probes: list[ProbeMapping] = []
    for probe_name, raw_probe in data.items():
        if not isinstance(raw_probe, dict):
            raise TypeError(f"probes.yaml probe {probe_name!r} must be a mapping.")
        probes.append(ProbeMapping.from_raw(str(probe_name), raw_probe))
    return ProbeCatalog(probes=tuple(probes))


def resolve_probes(
    *,
    probes: Iterable[ProbeMapping],
    nodes: Iterable[NodeSpec],
) -> ResolvedProbes:
    """Resolve probe declarations to reference tap data and V2 observe inputs."""

    selected = tuple(probes)
    node_by_name: dict[str, list[NodeSpec]] = {}
    for node in nodes:
        node_by_name.setdefault(node.name, []).append(node)

    missing_nodes = sorted({probe.node for probe in selected if probe.node not in node_by_name})
    if missing_nodes:
        raise KeyError(f"Mapped node(s) are not present in discovered graph: {missing_nodes}")

    ref_taps: list[tuple[str, RefTapSpec]] = []
    seen_ref_taps: set[tuple[str, str, str]] = set()
    whitelist: dict[tuple[str, str], set[str]] = {}

    for probe in selected:
        node_states = {node.state for node in node_by_name[probe.node] if node.state is not None}
        if probe.state is None and len(node_states) > 1:
            raise ValueError(
                f"Probe {probe.probe!r} maps node {probe.node!r} which appears in multiple "
                f"states {sorted(node_states)}; declare an explicit v2.state."
            )
        _collect_ref_tap(probe, ref_taps=ref_taps, seen=seen_ref_taps)
        for node in node_by_name[probe.node]:
            if node.state is None:
                continue
            if probe.state is not None and node.state != probe.state:
                continue
            # The V2 observer is armed only for fields named by selected probes,
            # so unrelated large tensors never leave the model-side execution.
            whitelist.setdefault((node.state, node.name), set()).add(probe.v2_field)

    return ResolvedProbes(
        probes=selected,
        ref_taps=tuple(ref_taps),
        v2_whitelist={key: frozenset(fields) for key, fields in whitelist.items()},
    )


# Internal helpers -------------------------------------------------------------


def _collect_ref_tap(
    probe: ProbeMapping,
    *,
    ref_taps: list[tuple[str, RefTapSpec]],
    seen: set[tuple[str, str, str]],
) -> None:
    # Deduplicate identical registrations while preserving distinct probe names
    # that intentionally read the same module or output target.
    key = (probe.ref_tap.kind, probe.ref_tap.field, probe.ref_tap.target)
    if key in seen:
        return
    seen.add(key)
    ref_taps.append((probe.ref_tap.field, probe.ref_tap))


__all__ = [
    "ProbeCatalog",
    "ProbeMapping",
    "RefTapSpec",
    "ResolvedProbes",
    "V2GradSpec",
    "load_probe_catalog",
    "resolve_probes",
]
