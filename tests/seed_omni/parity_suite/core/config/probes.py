"""Probe catalog schema and resolver for parity probes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml


if TYPE_CHECKING:
    from .discovery import NodeSpec


@dataclass(frozen=True)
class RefTapSpec:
    kind: str
    target: str

    @classmethod
    def from_raw(cls, raw: Any, *, probe: str) -> RefTapSpec:
        if isinstance(raw, str):
            return cls(kind="hook", target=raw)
        if isinstance(raw, dict) and "extractor" in raw:
            return cls(kind="extractor", target=str(raw["extractor"]))
        if isinstance(raw, dict) and "output" in raw:
            return cls(kind="output", target=str(raw["output"]))
        raise TypeError(
            f"Probe {probe} ref_tap must be a hook path string, "
            "{extractor: entrypoint}, or {output: context.output.path} mapping."
        )


@dataclass(frozen=True)
class V2GradSpec:
    module: str
    parameter: str
    rows_from: str | None = None

    @classmethod
    def from_raw(cls, raw: Any, *, probe: str) -> V2GradSpec | None:
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise TypeError(f"Probe {probe} v2_grad must be a mapping.")
        if "module" not in raw or "parameter" not in raw:
            raise ValueError(f"Probe {probe} v2_grad must declare module and parameter.")
        rows_from = raw.get("rows_from")
        return cls(
            module=str(raw["module"]),
            parameter=str(raw["parameter"]),
            rows_from=None if rows_from is None else str(rows_from),
        )


_STEP_POLICIES = frozenset({"last", "all"})


@dataclass(frozen=True)
class ProbeMapping:
    node: str
    probe: str
    v2_field: str
    ref_tap: RefTapSpec
    tol: str
    state: str | None = None
    v2_grad: V2GradSpec | None = None
    step: str = "last"

    @classmethod
    def from_raw(cls, node: str, probe: str, raw: Mapping[str, Any]) -> ProbeMapping:
        if "v2_field" not in raw:
            raise ValueError(f"Probe {node}.{probe} must declare v2_field.")
        if "ref_tap" not in raw:
            raise ValueError(f"Probe {node}.{probe} must declare ref_tap.")
        if "tol" not in raw:
            raise ValueError(f"Probe {node}.{probe} must declare tol.")
        step = str(raw.get("step", "last"))
        if step not in _STEP_POLICIES:
            raise ValueError(
                f"Probe {node}.{probe} has unsupported step policy {step!r}; expected one of {sorted(_STEP_POLICIES)}."
            )
        return cls(
            node=node,
            probe=probe,
            v2_field=str(raw["v2_field"]),
            ref_tap=RefTapSpec.from_raw(raw["ref_tap"], probe=f"{node}.{probe}"),
            tol=str(raw["tol"]),
            state=None if raw.get("state") is None else str(raw["state"]),
            v2_grad=V2GradSpec.from_raw(raw.get("v2_grad"), probe=f"{node}.{probe}"),
            step=step,
        )


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


@dataclass(frozen=True)
class ResolvedProbes:
    probes: tuple[ProbeMapping, ...]
    ref_taps: tuple[tuple[str, RefTapSpec], ...]
    v2_whitelist: dict[tuple[str, str], frozenset[str]]


def load_probe_catalog(path: Path) -> ProbeCatalog:
    """Load a node-keyed probes.yaml file."""

    if not path.exists():
        return ProbeCatalog()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping.")
    if "input_boundary" in data:
        raise ValueError(f"{path} must not declare input_boundary; input shape is driver-owned.")
    nodes = data.get("nodes", {}) or {}
    if not isinstance(nodes, dict):
        raise TypeError("probes.yaml nodes must be a mapping.")

    probes: list[ProbeMapping] = []
    for node, raw_node in nodes.items():
        if not isinstance(raw_node, dict):
            raise TypeError(f"probes.yaml node {node!r} must map probes to specs.")
        for probe, raw_probe in raw_node.items():
            if not isinstance(raw_probe, dict):
                raise TypeError(f"probes.yaml probe {node}.{probe} must be a mapping.")
            probes.append(ProbeMapping.from_raw(str(node), str(probe), raw_probe))
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


def _collect_ref_tap(
    probe: ProbeMapping,
    *,
    ref_taps: list[tuple[str, RefTapSpec]],
    seen: set[tuple[str, str, str]],
) -> None:
    # Deduplicate identical registrations while preserving distinct probe names
    # that intentionally read the same module or output target.
    key = (probe.ref_tap.kind, probe.probe, probe.ref_tap.target)
    if key in seen:
        return
    seen.add(key)
    ref_taps.append((probe.probe, probe.ref_tap))


__all__ = [
    "ProbeCatalog",
    "ProbeMapping",
    "RefTapSpec",
    "ResolvedProbes",
    "V2GradSpec",
    "load_probe_catalog",
    "resolve_probes",
]
