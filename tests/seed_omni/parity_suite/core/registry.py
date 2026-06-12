"""Case discovery for model-local `cases.yaml` files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tests.seed_omni.parity_suite.core.imports import import_optional_module
from tests.seed_omni.parity_suite.core.probes import ProbeBinding, load_probe_bindings
from tests.seed_omni.parity_suite.core.spec import BackendSpec, CaseSpec, EnvSpec, V2ModelSpec
from tests.seed_omni.parity_suite.v2.planner import (
    NodeAnchor,
    build_node_catalog_for_graph,
    discover_graph_specs,
    graph_exists,
)


def seed_omni_tests_root() -> Path:
    return Path(__file__).resolve().parents[2]


def discover_case_files(root: Path | None = None) -> list[Path]:
    base = root or seed_omni_tests_root()
    files: list[Path] = []
    for path in sorted(base.glob("*/cases.yaml")):
        if "archive" in path.parts or path.parent.name == "parity_suite":
            continue
        files.append(path)
    return files


def discover_cases(root: Path | None = None) -> list[CaseSpec]:
    cases: list[CaseSpec] = []
    for path in discover_case_files(root):
        cases.extend(load_cases(path))
    return cases


def load_cases(path: Path) -> list[CaseSpec]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    model_name = str(data.get("model") or path.parent.name)
    backend = BackendSpec.from_dict(data.get("reference_backend"))
    v2_model = V2ModelSpec.from_dict(data.get("v2_model"))
    env = EnvSpec.from_dict(data.get("env"))
    adapter = str(data.get("adapter") or "")
    probes_module = data.get("probes")
    captures_module = data.get("captures")

    return _load_discovered_cases(
        data,
        model_name=model_name,
        source_path=path,
        backend=backend,
        v2_model=v2_model,
        env=env,
        adapter=adapter,
        probes_module=probes_module,
        captures_module=captures_module,
    )


def _make_case_spec(
    *,
    model_name: str,
    source_path: Path,
    backend: BackendSpec,
    v2_model: V2ModelSpec,
    env: EnvSpec,
    adapter: str,
    probes_module: str | None,
    captures_module: str | None,
    domain: str,
    level: str,
    item: dict[str, Any],
) -> CaseSpec:
    case_id = str(item["id"])
    return CaseSpec(
        model_name=model_name,
        id=case_id,
        domain=domain,
        level=level,
        case=dict(item),
        source_path=source_path,
        reference_backend=backend,
        v2_model=v2_model,
        env=env,
        adapter=adapter,
        probes_module=None if probes_module is None else str(probes_module),
        captures_module=None if captures_module is None else str(captures_module),
        probes=tuple(str(probe) for probe in item.get("probes", []) or []),
        graph=item.get("graph"),
        fixture=item.get("fixture"),
        capture=item.get("capture"),
        reference=item.get("reference"),
        category=str(item.get("category") or _default_category(domain, level, item)),
    )


def _load_discovered_cases(
    raw_config: dict[str, Any],
    *,
    model_name: str,
    source_path: Path,
    backend: BackendSpec,
    v2_model: V2ModelSpec,
    env: EnvSpec,
    adapter: str,
    probes_module: str | None,
    captures_module: str | None,
) -> list[CaseSpec]:
    manifest = raw_config.get("discovery")
    if manifest is None:
        raise ValueError(f"{source_path} must declare a `discovery` manifest.")
    if not isinstance(manifest, dict):
        raise TypeError(f"`discovery` must be a map, got {type(manifest).__name__}")
    config_dir = v2_model.config_dir
    if not config_dir:
        raise ValueError(f"`discovery` in {source_path} requires v2_model.config_dir.")

    scenarios = _load_scenarios(source_path)
    probe_bindings = load_probe_bindings(import_optional_module(None if probes_module is None else str(probes_module)))
    specs: list[CaseSpec] = []
    graph_items: list[tuple[str, str, dict[str, Any]]] = []
    graph_policy = dict(manifest.get("graphs") or manifest.get("graph") or {})
    graph_specs = {spec.name: spec for spec in discover_graph_specs(config_dir)}
    included_graphs = _included_graphs(graph_specs, graph_policy)
    for scenario_id, scenario in scenarios.items():
        graph = scenario.get("graph")
        if not graph:
            continue
        graph = str(graph)
        if graph not in included_graphs:
            continue
        if not graph_exists(config_dir, graph):
            raise ValueError(f"Scenario {scenario_id!r} references missing graph {graph!r}.")
        item = _scenario_case_item(scenario_id, scenario)
        item["graph"] = graph
        domain = _scenario_domain(scenario_id, scenario, graph_specs[graph].domain)
        specs.append(
            _make_case_spec(
                model_name=model_name,
                source_path=source_path,
                backend=backend,
                v2_model=v2_model,
                env=env,
                adapter=adapter,
                probes_module=probes_module,
                captures_module=captures_module,
                domain=domain,
                level="graph",
                item=item,
            )
        )
        graph_items.append((domain, graph, item))

    module_policy = dict(manifest.get("modules") or manifest.get("module") or {})
    if module_policy.get("enabled", False):
        for domain in ("training", "inference"):
            specs.extend(
                _generate_module_cases(
                    graph_items,
                    domain=domain,
                    config_dir=config_dir,
                    module_section=module_policy,
                    probe_bindings=probe_bindings,
                    model_name=model_name,
                    source_path=source_path,
                    backend=backend,
                    v2_model=v2_model,
                    env=env,
                    adapter=adapter,
                    probes_module=probes_module,
                    captures_module=captures_module,
                )
            )

    framework_policy = dict(manifest.get("framework") or {})
    if framework_policy.get("enabled", False):
        specs.extend(
            _generate_framework_cases(
                scenarios,
                framework_policy=framework_policy,
                model_name=model_name,
                source_path=source_path,
                backend=backend,
                v2_model=v2_model,
                env=env,
                adapter=adapter,
                probes_module=probes_module,
                captures_module=captures_module,
            )
        )
    return specs


def _load_scenarios(source_path: Path) -> dict[str, dict[str, Any]]:
    path = source_path.with_name("capture.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    scenarios = data.get("scenarios") or {}
    if not isinstance(scenarios, dict):
        raise TypeError(f"`scenarios` in {path} must be a map, got {type(scenarios).__name__}")
    return {str(name): dict(value) for name, value in scenarios.items() if isinstance(value, dict)}


def _included_graphs(graph_specs: dict[str, Any], graph_policy: dict[str, Any]) -> set[str]:
    available = set(graph_specs)
    include = graph_policy.get("include", "all")
    if include is None or include == "all":
        selected = set(available)
    else:
        selected = {str(item) for item in include}
    exclude = {str(item) for item in graph_policy.get("exclude", []) or []}
    return selected - exclude


def _scenario_domain(scenario_id: str, scenario: dict[str, Any], graph_domain: str) -> str:
    explicit = scenario.get("domain")
    if explicit is None:
        return graph_domain
    domain = str(explicit)
    if domain != graph_domain:
        raise ValueError(f"Scenario {scenario_id!r} declares domain {domain!r}, but its graph is {graph_domain!r}.")
    return domain


def _scenario_case_item(scenario_id: str, scenario: dict[str, Any]) -> dict[str, Any]:
    reference = dict(scenario.get("reference") or {})
    item: dict[str, Any] = {
        "id": str(scenario.get("id") or scenario_id),
        "capture": scenario_id,
        "dedupe_role": str(scenario.get("dedupe_role") or scenario.get("scenario_kind") or scenario_id),
        "generated": True,
        "probes": [str(probe) for probe in scenario.get("probes", []) or []],
    }
    for key in (
        "fixture",
        "fixture_case_id",
        "module_probes",
        "requires_reference_model",
        "requires_v2_model",
        "requires_fixture",
        "category",
    ):
        if key in scenario:
            item[key] = scenario[key]
    if "case_reference" in scenario:
        item["reference"] = scenario["case_reference"]
    if "fixture" not in item and "fixture" in reference:
        item["fixture"] = reference["fixture"]
    return item


def _generate_framework_cases(
    scenarios: dict[str, dict[str, Any]],
    *,
    framework_policy: dict[str, Any],
    model_name: str,
    source_path: Path,
    backend: BackendSpec,
    v2_model: V2ModelSpec,
    env: EnvSpec,
    adapter: str,
    probes_module: str | None,
    captures_module: str | None,
) -> list[CaseSpec]:
    include = framework_policy.get("include", "all")
    include_set = None if include is None or include == "all" else {str(item) for item in include}
    specs: list[CaseSpec] = []
    for scenario_id, scenario in scenarios.items():
        level = scenario.get("level")
        if level not in {"trainer", "fsdp", "launcher", "checkpoint"}:
            continue
        if include_set is not None and scenario_id not in include_set:
            continue
        item = _scenario_case_item(scenario_id, scenario)
        domain = str(scenario.get("domain") or "training")
        specs.append(
            _make_case_spec(
                model_name=model_name,
                source_path=source_path,
                backend=backend,
                v2_model=v2_model,
                env=env,
                adapter=adapter,
                probes_module=probes_module,
                captures_module=captures_module,
                domain=domain,
                level=str(level),
                item=item,
            )
        )
    return specs


def _generate_module_cases(
    generated_graph_items: list[tuple[str, str, dict[str, Any]]],
    *,
    domain: str,
    config_dir: str,
    module_section: dict[str, Any],
    probe_bindings: dict[str, ProbeBinding],
    model_name: str,
    source_path: Path,
    backend: BackendSpec,
    v2_model: V2ModelSpec,
    env: EnvSpec,
    adapter: str,
    probes_module: str | None,
    captures_module: str | None,
) -> list[CaseSpec]:
    specs: list[CaseSpec] = []
    seen: set[tuple[str, str, str]] = set()
    for graph_domain, graph, graph_item in generated_graph_items:
        if graph_domain != domain:
            continue
        dedupe_role = str(graph_item.get("dedupe_role") or graph_item["id"])
        catalog = build_node_catalog_for_graph(config_dir, graph)
        module_probes = dict(graph_item.get("module_probes") or {})
        for node in catalog.values():
            probes = _module_probes(node, graph_item, module_probes, probe_bindings)
            if not probes:
                continue
            dedupe_key = (node.module, node.method, dedupe_role)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            item = {
                "id": _module_case_id(graph_item["id"], node),
                "graph": graph,
                "capture": graph_item.get("capture"),
                "fixture": graph_item.get("fixture"),
                "fixture_case_id": graph_item.get("fixture_case_id") or graph_item["id"],
                "generated": True,
                "generated_from": graph_item["id"],
                "module": node.module,
                "method": node.method,
                "node": node.name,
                "dedupe_role": dedupe_role,
                "probes": probes,
            }
            for key in ("requires_reference_model", "requires_v2_model", "requires_fixture", "reference"):
                if key in graph_item:
                    item[key] = graph_item[key]
            specs.append(
                _make_case_spec(
                    model_name=model_name,
                    source_path=source_path,
                    backend=backend,
                    v2_model=v2_model,
                    env=env,
                    adapter=adapter,
                    probes_module=probes_module,
                    captures_module=captures_module,
                    domain=domain,
                    level="module",
                    item=item,
                )
            )
    return specs


def _module_probes(
    node: NodeAnchor,
    graph_item: dict[str, Any],
    module_probes: dict[str, Any],
    probe_bindings: dict[str, ProbeBinding],
) -> list[str]:
    for key in (node.name, f"{node.module}.{node.method}", node.module):
        if key in module_probes:
            return [str(probe) for probe in module_probes[key] or []]
    node_key = f"{node.module}.{node.method}"
    probes: list[str] = []
    for probe in graph_item.get("probes", []) or []:
        binding = probe_bindings.get(str(probe))
        if binding is None or binding.v2.node != node_key:
            continue
        probes.append(str(probe))
    return probes


def _module_case_id(graph_case_id: str, node: NodeAnchor) -> str:
    module = node.module.replace("_", "-")
    method = node.method.replace("_", "-")
    return f"{graph_case_id}.{module}.{method}"


def _default_category(domain: str, level: str, item: dict[str, Any]) -> str:
    if domain == "framework" or level in {"fsdp", "launcher", "checkpoint"}:
        return "framework_smoke"
    if item.get("reference") == "v2_graph":
        return "v2_consistency"
    return "reference_parity"
