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
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, *, repo_root: Path) -> V2ModelSpec:
        values = dict(data or {})
        config_dir = _resolve_repo_path(values.pop("config_dir", None), repo_root=repo_root)
        if config_dir is None:
            raise ValueError("v2_model.config_dir is required.")
        known = {
            "model_root": _resolve_repo_path(values.pop("model_root", None), repo_root=repo_root),
            "config_dir": config_dir,
            "module_dirs": str(values.pop("module_dirs", "auto")),
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
    reference: bool = True
    graph: bool = True
    module: bool = False
    framework: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TierSelection:
        values = dict(data or {})
        return cls(
            reference=bool(values.get("reference", True)),
            graph=bool(values.get("graph", True)),
            module=bool(values.get("module", False)),
            framework=bool(values.get("framework", False)),
        )

    def enabled(self) -> tuple[str, ...]:
        tiers: list[str] = []
        if self.reference:
            tiers.append("reference")
        if self.graph:
            tiers.append("graph")
        if self.module:
            tiers.append("module")
        if self.framework:
            tiers.append("framework")
        return tuple(tiers)


@dataclass(frozen=True)
class GateSpec:
    requires_parity_env: bool | None = None
    requires_cuda: bool | None = None
    requires_reference_checkpoint: bool | None = None
    requires_v2_model: bool | None = None
    min_cuda_devices: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GateSpec:
        values = dict(data or {})
        return cls(
            requires_parity_env=_optional_bool(values.get("requires_parity_env")),
            requires_cuda=_optional_bool(values.get("requires_cuda")),
            requires_reference_checkpoint=_optional_bool(values.get("requires_reference_checkpoint")),
            requires_v2_model=_optional_bool(values.get("requires_v2_model")),
            min_cuda_devices=int(values.get("min_cuda_devices", 0) or 0),
        )

    def merge(self, other: GateSpec) -> GateSpec:
        return GateSpec(
            requires_parity_env=_coalesce(other.requires_parity_env, self.requires_parity_env),
            requires_cuda=_coalesce(other.requires_cuda, self.requires_cuda),
            requires_reference_checkpoint=_coalesce(
                other.requires_reference_checkpoint,
                self.requires_reference_checkpoint,
            ),
            requires_v2_model=_coalesce(other.requires_v2_model, self.requires_v2_model),
            min_cuda_devices=max(self.min_cuda_devices, other.min_cuda_devices),
        )


DEFAULT_GATE = GateSpec(
    requires_parity_env=True,
    requires_cuda=False,
    requires_reference_checkpoint=True,
    requires_v2_model=True,
)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _coalesce(value: bool | None, fallback: bool | None) -> bool | None:
    return fallback if value is None else value


@dataclass(frozen=True)
class LauncherSpec:
    enable_parallel: bool = False
    max_cuda_devices: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> LauncherSpec:
        values = dict(data or {})
        max_cuda_devices = values.pop("max_cuda_devices", None)
        return cls(
            enable_parallel=bool(values.pop("enable_parallel", False)),
            max_cuda_devices=None if max_cuda_devices is None else int(max_cuda_devices),
        )


@dataclass(frozen=True)
class RunSpec:
    id: str
    tier: str
    kind: str
    probes: tuple[str, ...] = ()
    gate: GateSpec = field(default_factory=GateSpec)
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, *, recipe_id: str, tier: str, index: int, data: dict[str, Any]) -> RunSpec:
        if not isinstance(data, dict):
            raise TypeError(f"Recipe {recipe_id} runs.{tier}[{index}] must be a mapping.")
        values = dict(data)
        run_id = str(values.pop("id", f"{tier}_{index}"))
        kind = str(values.pop("kind", _default_run_kind(tier)))
        probes = _string_tuple(
            values.pop("probes", None), field_name=f"recipes.{recipe_id}.runs.{tier}.{run_id}.probes"
        )
        gate = GateSpec.from_dict(values.pop("gate", None))
        options = values.pop("options", {}) or {}
        if not isinstance(options, dict):
            raise TypeError(f"Recipe {recipe_id} runs.{tier}.{run_id}.options must be a mapping.")
        if values:
            unknown = ", ".join(sorted(values))
            raise ValueError(
                f"Recipe {recipe_id} runs.{tier}.{run_id} has unknown field(s): {unknown}. "
                "Put tier- or kind-specific settings under options."
            )
        return cls(
            id=run_id,
            tier=tier,
            kind=kind,
            probes=probes,
            gate=gate,
            options=options,
        )


@dataclass(frozen=True)
class RecipeSpec:
    id: str
    graph: str
    stimulus: dict[str, Any] = field(default_factory=dict)
    gate: GateSpec = field(default_factory=GateSpec)
    runs: tuple[RunSpec, ...] = ()

    @classmethod
    def from_dict(
        cls,
        recipe_id: str,
        data: dict[str, Any],
        *,
        default_graph: str | None = None,
        variant_index: int | None = None,
    ) -> RecipeSpec:
        if not isinstance(data, dict):
            raise TypeError(f"Recipe {recipe_id} variant must be a mapping.")
        recipe_label = f"recipes.{recipe_id}" if variant_index is None else f"recipes.{recipe_id}[{variant_index}]"
        if "input_boundary" in data:
            raise ValueError(f"{recipe_label} must not declare input_boundary; input shape is driver-owned.")
        graph = data.get("graph", default_graph)
        if graph is None:
            raise ValueError(f"{recipe_label} must declare graph.")
        if default_graph is not None and "graph" in data and str(data["graph"]) != default_graph:
            raise ValueError(f"{recipe_label} declares graph {data['graph']!r}, expected {default_graph!r}.")
        stimulus = data.get("stimulus", {}) or {}
        if not isinstance(stimulus, dict):
            raise TypeError(f"{recipe_label} stimulus must be a mapping.")
        return cls(
            id=recipe_id,
            graph=str(graph),
            stimulus=stimulus,
            gate=GateSpec.from_dict(data.get("gate")),
            runs=_run_specs(recipe_label, data.get("runs")),
        )


def _default_run_kind(tier: str) -> str:
    if tier == "framework":
        return "forward_backward"
    return tier


def _run_specs(recipe_id: str, raw_runs: Any) -> tuple[RunSpec, ...]:
    if raw_runs is None:
        raise ValueError(f"Recipe {recipe_id} must declare runs.")
    if not isinstance(raw_runs, dict):
        raise TypeError(f"Recipe {recipe_id} runs must be a mapping from tier to run list.")
    runs: list[RunSpec] = []
    for tier, raw_tier_runs in raw_runs.items():
        if tier not in {"reference", "graph", "module", "framework"}:
            raise ValueError(f"Recipe {recipe_id} has unsupported run tier {tier!r}.")
        if not isinstance(raw_tier_runs, list):
            raise TypeError(f"Recipe {recipe_id} runs.{tier} must be a list.")
        for index, raw_run in enumerate(raw_tier_runs):
            runs.append(RunSpec.from_dict(recipe_id=recipe_id, tier=str(tier), index=index, data=raw_run or {}))
    if not runs:
        raise ValueError(f"Recipe {recipe_id} must declare at least one run.")
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
    launcher: LauncherSpec
    recipes: tuple[RecipeSpec, ...]
    mapping: MappingSpec


def load_model_spec(model_dir: str | Path) -> ModelSpec:
    """Load ``base.yaml`` and recipe specs for one parity consumer."""

    root = Path(model_dir)
    repo_root = repository_root()
    base = load_yaml_file(root / "base.yaml")
    recipes = _load_recipe_specs(root)

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
        launcher=LauncherSpec.from_dict(base.get("launcher")),
        recipes=recipes,
        mapping=load_mapping_spec(root / "mapping.yaml"),
    )


def _load_recipe_specs(root: Path) -> tuple[RecipeSpec, ...]:
    recipe_docs: list[tuple[Path, dict[str, Any], str | None]] = []
    single_file = root / "recipes.yaml"
    if single_file.exists():
        recipe_docs.append((single_file, load_yaml_file(single_file), None))

    recipes_dir = root / "recipes"
    if recipes_dir.exists():
        for path in sorted(recipes_dir.glob("*.yaml")):
            recipe_docs.append((path, load_yaml_file(path), path.stem))

    if not recipe_docs:
        raise FileNotFoundError(f"No recipes.yaml or recipes/*.yaml found under {root}.")

    recipes: list[RecipeSpec] = []
    seen: set[str] = set()
    for path, raw_recipes, default_graph in recipe_docs:
        for recipe_id, raw_variants in raw_recipes.items():
            recipe_key = str(recipe_id)
            if recipe_key in seen:
                raise ValueError(f"Duplicate recipe id {recipe_key!r} while loading {path}.")
            seen.add(recipe_key)
            if not isinstance(raw_variants, list):
                raise TypeError(f"Recipe {recipe_key!r} in {path} must be a list of variants.")
            if not raw_variants:
                raise ValueError(f"Recipe {recipe_key!r} in {path} must declare at least one variant.")
            for index, raw_recipe in enumerate(raw_variants):
                recipes.append(
                    RecipeSpec.from_dict(
                        recipe_key,
                        raw_recipe or {},
                        default_graph=default_graph,
                        variant_index=index,
                    )
                )
    return tuple(recipes)
