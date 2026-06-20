"""Typed specification loading for SeedOmni V2 parity cases."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tests.seed_omni.parity_suite.core.stimulus import validate_stimulus

from .probes import ProbeCatalog, load_probe_catalog


PARITY_ENABLE_ENV = "VEOMNI_V2_TEST_ENABLE_PARITY_CHECK"


# Repository IO ----------------------------------------------------------------


def repository_root() -> Path:
    """Return the Open-VeOmni repository root."""

    return Path(__file__).resolve().parents[5]


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


# Reference config -------------------------------------------------------------


@dataclass(frozen=True)
class HfModelReferenceSpec:
    module: str | None = None
    checkpoint: Path | None = None
    load_kwargs: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, *, repo_root: Path) -> HfModelReferenceSpec | None:
        if data is None:
            return None
        values = dict(data or {})
        load_kwargs = values.pop("load_kwargs", {}) or {}
        options = values.pop("options", {}) or {}
        if not isinstance(load_kwargs, dict):
            raise TypeError("reference.hf_model.load_kwargs must be a mapping.")
        if not isinstance(options, dict):
            raise TypeError("reference.hf_model.options must be a mapping.")
        if values.keys() - {"module", "checkpoint"}:
            unknown = ", ".join(sorted(values.keys() - {"module", "checkpoint"}))
            raise ValueError(f"reference.hf_model has unknown field(s): {unknown}.")
        return cls(
            module=None if values.get("module") is None else str(values["module"]),
            checkpoint=_resolve_repo_path(values.get("checkpoint"), repo_root=repo_root),
            load_kwargs=load_kwargs,
            options=options,
        )


@dataclass(frozen=True)
class ReferenceSpec:
    hf_model: HfModelReferenceSpec | None = None
    hf_module: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, *, repo_root: Path) -> ReferenceSpec:
        values = dict(data or {})
        if "module" in values or "checkpoint" in values:
            raise ValueError("reference must configure reference.hf_model or reference.hf_module.")
        hf_model = HfModelReferenceSpec.from_dict(values.pop("hf_model", None), repo_root=repo_root)
        raw_hf_module = values.pop("hf_module", None)
        hf_module = _string_tuple(raw_hf_module, field_name="reference.hf_module")
        if hf_module and hf_model is None:
            raise ValueError("reference.hf_module requires reference.hf_model; module references reuse its subject.")
        if values:
            unknown = ", ".join(sorted(values))
            raise ValueError(f"reference has unknown field(s): {unknown}.")
        return cls(hf_model=hf_model, hf_module=hf_module)


# V2 model config --------------------------------------------------------------


@dataclass(frozen=True)
class V2ModelTargetSpec:
    model_root: Path | None
    config_dir: Path
    module_dirs: str = "auto"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        *,
        repo_root: Path,
        field_name: str,
        inherited_model_root: Path | None = None,
    ) -> V2ModelTargetSpec:
        values = dict(data or {})
        config_dir = _resolve_repo_path(values.pop("config_dir", None), repo_root=repo_root)
        if config_dir is None:
            raise ValueError(f"{field_name}.config_dir is required.")
        target_model_root = _resolve_repo_path(values.pop("model_root", None), repo_root=repo_root)
        known = {
            "model_root": inherited_model_root if target_model_root is None else target_model_root,
            "config_dir": config_dir,
            "module_dirs": str(values.pop("module_dirs", "auto")),
        }
        return cls(**known, extra=values)


@dataclass(frozen=True)
class V2ModelSpec:
    model_root: Path | None = None
    hf_model: V2ModelTargetSpec | None = None
    hf_module: dict[str, V2ModelTargetSpec] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, *, repo_root: Path) -> V2ModelSpec:
        values = dict(data or {})
        if {"config_dir", "module_dirs"}.intersection(values):
            raise ValueError(
                "v2_model must configure v2_model.hf_model and/or v2_model.hf_module targets; "
                "flat v2_model.config_dir is not supported."
            )
        model_root = _resolve_repo_path(values.pop("model_root", None), repo_root=repo_root)

        hf_model = None
        if "hf_model" in values:
            hf_model = V2ModelTargetSpec.from_dict(
                values.pop("hf_model"),
                repo_root=repo_root,
                field_name="v2_model.hf_model",
                inherited_model_root=model_root,
            )

        raw_hf_module = values.pop("hf_module", {}) or {}
        if not isinstance(raw_hf_module, dict):
            raise TypeError("v2_model.hf_module must be a mapping.")
        hf_module = {
            str(name): V2ModelTargetSpec.from_dict(
                spec or {},
                repo_root=repo_root,
                field_name=f"v2_model.hf_module.{name}",
                inherited_model_root=model_root,
            )
            for name, spec in raw_hf_module.items()
        }

        if values:
            unknown = ", ".join(sorted(values))
            raise ValueError(f"v2_model has unknown field(s): {unknown}.")
        return cls(model_root=model_root, hf_model=hf_model, hf_module=hf_module)


def select_v2_model_target(
    v2_model: V2ModelSpec,
    oracle: str,
    *,
    model_name: str,
    recipe_id: str,
) -> V2ModelTargetSpec:
    if oracle == "hf_model":
        if v2_model.hf_model is None:
            raise KeyError(
                f"Recipe {model_name}.{recipe_id} selects hf_model, but v2_model.hf_model is not configured."
            )
        return v2_model.hf_model

    prefix = "hf_module."
    if oracle.startswith(prefix):
        module_name = oracle.removeprefix(prefix)
        if not module_name:
            raise ValueError(f"Recipe {model_name}.{recipe_id} selects empty hf_module oracle.")
        try:
            return v2_model.hf_module[module_name]
        except KeyError as exc:
            available = sorted(v2_model.hf_module)
            raise KeyError(
                f"Recipe {model_name}.{recipe_id} selects {oracle!r}, but v2_model.hf_module.{module_name} "
                f"is not configured. Available module targets: {available}."
            ) from exc

    raise ValueError(f"Recipe {model_name}.{recipe_id} has unsupported reference.oracle {oracle!r}.")


# Graph, tier, gate, and launcher config ---------------------------------------


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
            reference=_bool_field(values.get("reference", True), field_name="tiers.reference"),
            graph=_bool_field(values.get("graph", True), field_name="tiers.graph"),
            module=_bool_field(values.get("module", False), field_name="tiers.module"),
            framework=_bool_field(values.get("framework", False), field_name="tiers.framework"),
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
    discover: bool | None = None
    requires_parity_env: bool | None = None
    requires_cuda: bool | None = None
    requires_reference_capture: bool | None = None
    requires_reference_checkpoint: bool | None = None
    requires_v2_model: bool | None = None
    min_cuda_devices: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GateSpec:
        values = dict(data or {})
        return cls(
            discover=_optional_bool(values.get("discover"), field_name="gate.discover"),
            requires_parity_env=_optional_bool(
                values.get("requires_parity_env"), field_name="gate.requires_parity_env"
            ),
            requires_cuda=_optional_bool(values.get("requires_cuda"), field_name="gate.requires_cuda"),
            requires_reference_capture=_optional_bool(
                values.get("requires_reference_capture"),
                field_name="gate.requires_reference_capture",
            ),
            requires_reference_checkpoint=_optional_bool(
                values.get("requires_reference_checkpoint"),
                field_name="gate.requires_reference_checkpoint",
            ),
            requires_v2_model=_optional_bool(values.get("requires_v2_model"), field_name="gate.requires_v2_model"),
            min_cuda_devices=int(values.get("min_cuda_devices", 0) or 0),
        )

    def merge(self, other: GateSpec) -> GateSpec:
        """Merge a narrower gate over this one.

        Boolean fields use ``other`` when explicitly set. Device count is a
        floor, so nested gates can only raise the CUDA requirement.
        """

        return GateSpec(
            discover=_coalesce(other.discover, self.discover),
            requires_parity_env=_coalesce(other.requires_parity_env, self.requires_parity_env),
            requires_cuda=_coalesce(other.requires_cuda, self.requires_cuda),
            requires_reference_capture=_coalesce(
                other.requires_reference_capture,
                self.requires_reference_capture,
            ),
            requires_reference_checkpoint=_coalesce(
                other.requires_reference_checkpoint,
                self.requires_reference_checkpoint,
            ),
            requires_v2_model=_coalesce(other.requires_v2_model, self.requires_v2_model),
            min_cuda_devices=max(self.min_cuda_devices, other.min_cuda_devices),
        )


DEFAULT_GATE = GateSpec(
    discover=True,
    requires_parity_env=True,
    requires_cuda=False,
    requires_reference_capture=True,
    requires_reference_checkpoint=True,
    requires_v2_model=True,
)


def _optional_bool(value: Any, *, field_name: str) -> bool | None:
    if value is None:
        return None
    return _bool_field(value, field_name=field_name)


def _bool_field(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a YAML bool.")
    return value


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
            enable_parallel=_bool_field(values.pop("enable_parallel", False), field_name="launcher.enable_parallel"),
            max_cuda_devices=None if max_cuda_devices is None else int(max_cuda_devices),
        )


# Recipe config ----------------------------------------------------------------


@dataclass(frozen=True)
class RunSpec:
    id: str
    tier: str
    kind: str
    enable: bool = True
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
        enable = values.pop("enable", True)
        if not isinstance(enable, bool):
            raise TypeError(f"Recipe {recipe_id} runs.{tier}.{run_id}.enable must be a bool.")
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
            enable=enable,
            probes=probes,
            gate=gate,
            options=options,
        )


@dataclass(frozen=True)
class RecipeV2ModelSpec:
    module_overrides: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, *, recipe_label: str) -> RecipeV2ModelSpec:
        if data is not None and not isinstance(data, dict):
            raise TypeError(f"{recipe_label} v2_model must be a mapping.")
        values = dict(data or {})
        module_overrides = values.pop("module_overrides", {}) or {}
        if not isinstance(module_overrides, dict):
            raise TypeError(f"{recipe_label} v2_model.module_overrides must be a mapping.")
        if values:
            unknown = ", ".join(sorted(values))
            raise ValueError(
                f"{recipe_label} v2_model has unknown field(s): {unknown}. "
                "Recipes may only declare v2_model.module_overrides; configure V2 model targets in base.yaml."
            )
        return cls(module_overrides=module_overrides)


@dataclass(frozen=True)
class RecipeSpec:
    id: str
    graph: str
    stimulus: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] | None = None
    reference: dict[str, Any] = field(default_factory=dict)
    gate: GateSpec = field(default_factory=GateSpec)
    v2_model: RecipeV2ModelSpec = field(default_factory=RecipeV2ModelSpec)
    runs: tuple[RunSpec, ...] = ()

    @classmethod
    def from_dict(
        cls,
        recipe_id: str,
        data: dict[str, Any],
        *,
        default_graph: str | None = None,
        repo_root: Path | None = None,
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
        has_stimulus = "stimulus" in data
        has_data = "data" in data
        # Normal parity recipes are either driver-owned synthetic stimuli or
        # data-backed framework smokes. Reference recipes may be pure policies.
        if str(graph) != "reference" and has_stimulus == has_data:
            raise ValueError(f"{recipe_label} must declare exactly one of stimulus or data.")
        stimulus = data.get("stimulus", {}) or {}
        if not isinstance(stimulus, dict):
            raise TypeError(f"{recipe_label} stimulus must be a mapping.")
        validate_stimulus(recipe_label, stimulus)
        recipe_data = data.get("data")
        if recipe_data is not None and not isinstance(recipe_data, dict):
            raise TypeError(f"{recipe_label} data must be a mapping.")
        reference = data.get("reference", {}) or {}
        if not isinstance(reference, dict):
            raise TypeError(f"{recipe_label} reference must be a mapping.")
        if str(graph) != "reference" and "oracle" not in reference:
            raise ValueError(f"{recipe_label} must declare reference.oracle.")
        return cls(
            id=recipe_id,
            graph=str(graph),
            stimulus=stimulus,
            data=recipe_data,
            reference=dict(reference),
            gate=GateSpec.from_dict(data.get("gate")),
            v2_model=RecipeV2ModelSpec.from_dict(data.get("v2_model"), recipe_label=recipe_label),
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


# Model config loading ---------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    name: str
    root: Path
    discover: bool
    reference: ReferenceSpec
    v2_model: V2ModelSpec
    graphs: GraphSelection
    tiers: TierSelection
    seed: int
    tolerance: dict[str, Any]
    gate: GateSpec
    launcher: LauncherSpec
    recipes: tuple[RecipeSpec, ...]
    probes: ProbeCatalog


def load_model_spec(model_dir: str | Path) -> ModelSpec:
    """Load ``base.yaml`` and recipe specs for one parity consumer."""

    root = Path(model_dir)
    repo_root = repository_root()
    base = load_yaml_file(root / "base.yaml")
    recipes = _load_recipe_specs(root, repo_root=repo_root)

    return ModelSpec(
        name=str(base.get("name") or root.name),
        root=root,
        discover=_discover_enabled(base),
        reference=ReferenceSpec.from_dict(base.get("reference"), repo_root=repo_root),
        v2_model=V2ModelSpec.from_dict(base.get("v2_model"), repo_root=repo_root),
        graphs=GraphSelection.from_dict(base.get("graphs")),
        tiers=TierSelection.from_dict(base.get("tiers")),
        seed=int(base.get("seed", 1234)),
        tolerance=dict(base.get("tolerance", {}) or {}),
        gate=GateSpec.from_dict(base.get("gate")),
        launcher=LauncherSpec.from_dict(base.get("launcher")),
        recipes=recipes,
        probes=load_probe_catalog(root / "probes.yaml"),
    )


def _load_recipe_specs(root: Path, *, repo_root: Path) -> tuple[RecipeSpec, ...]:
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
                        repo_root=repo_root,
                        variant_index=index,
                    )
                )
    return tuple(recipes)


def _discover_enabled(base: dict[str, Any]) -> bool:
    gate = base.get("gate") or {}
    if isinstance(gate, dict) and "discover" in gate:
        return _bool_field(gate["discover"], field_name="gate.discover")
    return _bool_field(base.get("discover", True), field_name="discover")
