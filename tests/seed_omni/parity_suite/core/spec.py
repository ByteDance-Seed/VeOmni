"""Typed specification loading for SeedOmni V2 parity cases."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .mapping import MappingSpec, load_mapping_spec


PARITY_ENABLE_ENV = "VEOMNI_V2_TEST_ENABLE_PARITY_CHECK"


def repository_root() -> Path:
    """Return the Open-VeOmni repository root."""

    return Path(__file__).resolve().parents[4]


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from ``path``."""

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping.")
    if "input_boundary" in data:
        raise ValueError(f"{path} must not declare input_boundary; input shape is driver-owned.")
    return data


def _expand_env(value: str | None) -> str | None:
    if value is None:
        return None
    return os.path.expandvars(value)


def _resolve_repo_path(value: str | None, *, repo_root: Path) -> Path | None:
    expanded = _expand_env(value)
    if expanded is None:
        return None
    path = Path(expanded)
    return path if path.is_absolute() else repo_root / path


def _string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list of strings.")
    return tuple(str(item) for item in value)


def _node_pairs(value: Any, *, field_name: str) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list of [state, node] pairs.")
    pairs: list[tuple[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            if "state" not in item or "node" not in item:
                raise TypeError(f"{field_name} dict entries must contain state and node.")
            pairs.append((str(item["state"]), str(item["node"])))
            continue
        if isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append((str(item[0]), str(item[1])))
            continue
        raise TypeError(f"{field_name} entries must be [state, node] pairs or mappings.")
    return tuple(pairs)


@dataclass(frozen=True)
class ReferenceSpec:
    loader: str
    module: str | None = None
    checkpoint: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, *, repo_root: Path) -> ReferenceSpec:
        values = dict(data or {})
        known = {
            "loader": str(values.pop("loader", "transformers")),
            "module": values.pop("module", None),
            "checkpoint": _resolve_repo_path(values.pop("checkpoint", None), repo_root=repo_root),
        }
        return cls(**known, extra=values)


@dataclass(frozen=True)
class V2ModelSpec:
    model_root: Path | None
    config_dir: Path
    module_dirs: str = "auto"
    dtype_overrides: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, *, repo_root: Path) -> V2ModelSpec:
        values = dict(data or {})
        config_dir = _resolve_repo_path(values.pop("config_dir", None), repo_root=repo_root)
        if config_dir is None:
            raise ValueError("v2_model.config_dir is required.")
        dtype_overrides = values.pop("dtype_overrides", {}) or {}
        if not isinstance(dtype_overrides, dict):
            raise TypeError("v2_model.dtype_overrides must be a mapping.")
        known = {
            "model_root": _resolve_repo_path(values.pop("model_root", None), repo_root=repo_root),
            "config_dir": config_dir,
            "module_dirs": str(values.pop("module_dirs", "auto")),
            "dtype_overrides": {str(key): str(val) for key, val in dtype_overrides.items()},
        }
        return cls(**known, extra=values)


@dataclass(frozen=True)
class GraphSelection:
    include: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GraphSelection:
        values = dict(data or {})
        return cls(include=_string_tuple(values.get("include"), field_name="graphs.include"))


@dataclass(frozen=True)
class TierSelection:
    graph: bool = True
    module: bool = False
    framework: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TierSelection:
        values = dict(data or {})
        return cls(
            graph=bool(values.get("graph", True)),
            module=bool(values.get("module", False)),
            framework=bool(values.get("framework", False)),
        )

    def enabled(self) -> tuple[str, ...]:
        tiers: list[str] = []
        if self.graph:
            tiers.append("graph")
        if self.module:
            tiers.append("module")
        if self.framework:
            tiers.append("framework")
        return tuple(tiers)


@dataclass(frozen=True)
class GateSpec:
    requires_cuda: bool = False
    min_cuda_devices: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GateSpec:
        values = dict(data or {})
        return cls(
            requires_cuda=bool(values.get("requires_cuda", False)),
            min_cuda_devices=int(values.get("min_cuda_devices", 0) or 0),
        )


@dataclass(frozen=True)
class ModulePolicySpec:
    required_nodes: tuple[tuple[str, str], ...] = ()
    max_steps: int | None = None
    allow_finalize: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, *, field_name: str) -> ModulePolicySpec:
        values = dict(data or {})
        max_steps = values.get("max_steps")
        return cls(
            required_nodes=_node_pairs(values.get("required_nodes"), field_name=f"{field_name}.required_nodes"),
            max_steps=None if max_steps is None else int(max_steps),
            allow_finalize=bool(values.get("allow_finalize", False)),
        )


@dataclass(frozen=True)
class RunSpec:
    id: str
    tier: str
    kind: str
    probes: tuple[str, ...] = ()
    gate: GateSpec = field(default_factory=GateSpec)
    policy: ModulePolicySpec = field(default_factory=ModulePolicySpec)
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, *, scenario_id: str, tier: str, index: int, data: dict[str, Any]) -> RunSpec:
        if not isinstance(data, dict):
            raise TypeError(f"Scenario {scenario_id} runs.{tier}[{index}] must be a mapping.")
        values = dict(data)
        run_id = str(values.pop("id", f"{tier}_{index}"))
        kind = str(values.pop("kind", _default_run_kind(tier)))
        probes = _string_tuple(
            values.pop("probes", None), field_name=f"scenarios.{scenario_id}.runs.{tier}.{run_id}.probes"
        )
        gate = GateSpec.from_dict(values.pop("gate", None))
        policy = ModulePolicySpec.from_dict(
            values.pop("policy", None),
            field_name=f"scenarios.{scenario_id}.runs.{tier}.{run_id}.policy",
        )
        return cls(
            id=run_id,
            tier=tier,
            kind=kind,
            probes=probes,
            gate=gate,
            policy=policy,
            options=values,
        )


@dataclass(frozen=True)
class ScenarioSpec:
    id: str
    graph: str
    driver_case: str
    stimulus: dict[str, Any] = field(default_factory=dict)
    gate: GateSpec = field(default_factory=GateSpec)
    runs: tuple[RunSpec, ...] = ()

    @classmethod
    def from_dict(cls, scenario_id: str, data: dict[str, Any]) -> ScenarioSpec:
        if "input_boundary" in data:
            raise ValueError(f"Scenario {scenario_id} must not declare input_boundary; input shape is driver-owned.")
        if "graph" not in data:
            raise ValueError(f"Scenario {scenario_id} must declare graph.")
        if "driver_case" not in data:
            raise ValueError(f"Scenario {scenario_id} must declare driver_case.")
        stimulus = data.get("stimulus", {}) or {}
        if not isinstance(stimulus, dict):
            raise TypeError(f"Scenario {scenario_id} stimulus must be a mapping.")
        return cls(
            id=scenario_id,
            graph=str(data["graph"]),
            driver_case=str(data["driver_case"]),
            stimulus=stimulus,
            gate=GateSpec.from_dict(data.get("gate")),
            runs=_run_specs(scenario_id, data.get("runs")),
        )


def _default_run_kind(tier: str) -> str:
    if tier == "framework":
        return "forward_backward"
    return tier


def _run_specs(scenario_id: str, raw_runs: Any) -> tuple[RunSpec, ...]:
    if raw_runs is None:
        raise ValueError(f"Scenario {scenario_id} must declare runs.")
    if not isinstance(raw_runs, dict):
        raise TypeError(f"Scenario {scenario_id} runs must be a mapping from tier to run list.")
    runs: list[RunSpec] = []
    for tier, raw_tier_runs in raw_runs.items():
        if tier not in {"graph", "module", "framework"}:
            raise ValueError(f"Scenario {scenario_id} has unsupported run tier {tier!r}.")
        if not isinstance(raw_tier_runs, list):
            raise TypeError(f"Scenario {scenario_id} runs.{tier} must be a list.")
        for index, raw_run in enumerate(raw_tier_runs):
            runs.append(RunSpec.from_dict(scenario_id=scenario_id, tier=str(tier), index=index, data=raw_run or {}))
    if not runs:
        raise ValueError(f"Scenario {scenario_id} must declare at least one run.")
    return tuple(runs)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    root: Path
    reference: ReferenceSpec
    v2_model: V2ModelSpec
    graphs: GraphSelection
    tiers: TierSelection
    seed: int
    tolerance: dict[str, Any]
    gate: GateSpec
    scenarios: tuple[ScenarioSpec, ...]
    mapping: MappingSpec


def load_model_spec(model_dir: str | Path) -> ModelSpec:
    """Load ``base.yaml`` and ``scenarios.yaml`` for one parity consumer."""

    root = Path(model_dir)
    repo_root = repository_root()
    base = load_yaml_file(root / "base.yaml")
    scenarios_doc = load_yaml_file(root / "scenarios.yaml")
    raw_scenarios = scenarios_doc.get("scenarios", {})
    if not isinstance(raw_scenarios, dict):
        raise TypeError("scenarios.yaml must contain a scenarios mapping.")

    return ModelSpec(
        name=str(base.get("name") or root.name),
        root=root,
        reference=ReferenceSpec.from_dict(base.get("reference"), repo_root=repo_root),
        v2_model=V2ModelSpec.from_dict(base.get("v2_model"), repo_root=repo_root),
        graphs=GraphSelection.from_dict(base.get("graphs")),
        tiers=TierSelection.from_dict(base.get("tiers")),
        seed=int(base.get("seed", 1234)),
        tolerance=dict(base.get("tolerance", {}) or {}),
        gate=GateSpec.from_dict(base.get("gate")),
        scenarios=tuple(ScenarioSpec.from_dict(str(name), data or {}) for name, data in raw_scenarios.items()),
        mapping=load_mapping_spec(root / "mapping.yaml"),
    )
