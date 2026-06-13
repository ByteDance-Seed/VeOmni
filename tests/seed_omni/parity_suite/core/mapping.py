"""Mapping schema and resolver for parity probes."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import yaml

from tests.seed_omni.parity_suite.reference.capture import ExtractorTap, ReferenceCaptureContext, ReferenceCapturePlan
from tests.seed_omni.parity_suite.reference.hooks import HookTap


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
        raise TypeError(f"Probe {probe} ref_tap must be a hook path string or {{extractor: entrypoint}} mapping.")


@dataclass(frozen=True)
class ProbeMapping:
    node: str
    probe: str
    v2_field: str
    ref_tap: RefTapSpec
    tol: str

    @classmethod
    def from_raw(cls, node: str, probe: str, raw: Mapping[str, Any]) -> ProbeMapping:
        if "v2_field" not in raw:
            raise ValueError(f"Mapping {node}.{probe} must declare v2_field.")
        if "ref_tap" not in raw:
            raise ValueError(f"Mapping {node}.{probe} must declare ref_tap.")
        if "tol" not in raw:
            raise ValueError(f"Mapping {node}.{probe} must declare tol.")
        return cls(
            node=node,
            probe=probe,
            v2_field=str(raw["v2_field"]),
            ref_tap=RefTapSpec.from_raw(raw["ref_tap"], probe=f"{node}.{probe}"),
            tol=str(raw["tol"]),
        )


@dataclass(frozen=True)
class MappingSpec:
    probes: tuple[ProbeMapping, ...] = ()

    def for_probe_names(self, probe_names: Iterable[str]) -> tuple[ProbeMapping, ...]:
        requested = tuple(probe_names)
        if not requested:
            return self.probes
        by_name = {(probe.node, probe.probe): probe for probe in self.probes}
        selected: list[ProbeMapping] = []
        for probe_name in requested:
            matches = [probe for (node, name), probe in by_name.items() if name == probe_name]
            if not matches:
                raise KeyError(f"Probe {probe_name!r} is not declared in mapping.yaml.")
            selected.extend(matches)
        return tuple(selected)

    def by_node(self) -> dict[str, tuple[ProbeMapping, ...]]:
        grouped: dict[str, list[ProbeMapping]] = {}
        for probe in self.probes:
            grouped.setdefault(probe.node, []).append(probe)
        return {node: tuple(items) for node, items in grouped.items()}


@dataclass(frozen=True)
class ResolvedMapping:
    probes: tuple[ProbeMapping, ...]
    reference_plan: ReferenceCapturePlan
    v2_whitelist: dict[tuple[str, str], frozenset[str]]


def load_mapping_spec(path: Path) -> MappingSpec:
    """Load a node-keyed mapping.yaml file."""

    if not path.exists():
        return MappingSpec()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping.")
    if "input_boundary" in data:
        raise ValueError(f"{path} must not declare input_boundary; input shape is driver-owned.")
    nodes = data.get("nodes", {}) or {}
    if not isinstance(nodes, dict):
        raise TypeError("mapping.yaml nodes must be a mapping.")

    probes: list[ProbeMapping] = []
    for node, raw_node in nodes.items():
        if not isinstance(raw_node, dict):
            raise TypeError(f"mapping.yaml node {node!r} must map probes to specs.")
        for probe, raw_probe in raw_node.items():
            if not isinstance(raw_probe, dict):
                raise TypeError(f"mapping.yaml probe {node}.{probe} must be a mapping.")
            probes.append(ProbeMapping.from_raw(str(node), str(probe), raw_probe))
    return MappingSpec(probes=tuple(probes))


def resolve_mapping(
    *,
    mappings: Iterable[ProbeMapping],
    nodes: Iterable[NodeSpec],
) -> ResolvedMapping:
    """Resolve probe mappings to reference capture and V2 observe inputs."""

    selected = tuple(mappings)
    node_by_name: dict[str, list[NodeSpec]] = {}
    for node in nodes:
        node_by_name.setdefault(node.name, []).append(node)

    missing_nodes = sorted({mapping.node for mapping in selected if mapping.node not in node_by_name})
    if missing_nodes:
        raise KeyError(f"Mapped node(s) are not present in discovered graph: {missing_nodes}")

    hook_taps: list[HookTap] = []
    extractor_taps: list[ExtractorTap] = []
    seen_ref_taps: set[tuple[str, str, str]] = set()
    whitelist: dict[tuple[str, str], set[str]] = {}

    for mapping in selected:
        _append_ref_tap(mapping, hook_taps=hook_taps, extractor_taps=extractor_taps, seen=seen_ref_taps)
        for node in node_by_name[mapping.node]:
            if node.state is None:
                continue
            whitelist.setdefault((node.state, node.name), set()).add(mapping.v2_field)

    return ResolvedMapping(
        probes=selected,
        reference_plan=ReferenceCapturePlan(hook_taps=tuple(hook_taps), extractor_taps=tuple(extractor_taps)),
        v2_whitelist={key: frozenset(fields) for key, fields in whitelist.items()},
    )


def _append_ref_tap(
    mapping: ProbeMapping,
    *,
    hook_taps: list[HookTap],
    extractor_taps: list[ExtractorTap],
    seen: set[tuple[str, str, str]],
) -> None:
    key = (mapping.ref_tap.kind, mapping.probe, mapping.ref_tap.target)
    if key in seen:
        return
    seen.add(key)
    if mapping.ref_tap.kind == "hook":
        hook_taps.append(HookTap(name=mapping.probe, module_path=mapping.ref_tap.target))
        return
    if mapping.ref_tap.kind == "extractor":
        extractor_taps.append(ExtractorTap(name=mapping.probe, extractor=_load_extractor(mapping.ref_tap.target)))
        return
    raise ValueError(f"Unsupported ref_tap kind: {mapping.ref_tap.kind}")


def _load_extractor(entrypoint: str) -> Callable[[ReferenceCaptureContext], Any]:
    module_name, symbol_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


__all__ = [
    "MappingSpec",
    "ProbeMapping",
    "RefTapSpec",
    "ResolvedMapping",
    "load_mapping_spec",
    "resolve_mapping",
]
