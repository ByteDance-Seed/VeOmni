"""Shared V2 model behavior helpers for parity drivers."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from tests.seed_omni.parity_suite.core.runtime import resolve_torch_dtype
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY, read_model_type


def graph_active_module_names(case: Any) -> frozenset[str]:
    """Return every module referenced by the case's selected graph."""

    return frozenset(str(node.module) for node in case.nodes)


def load_graph_active_omni_config(case: Any, module_names: Iterable[str] | None = None) -> OmniConfig:
    """Load an OmniConfig narrowed to the modules used by ``case.graph``."""

    names = graph_active_module_names(case) if module_names is None else frozenset(module_names)
    graph = None if case.graph.domain == "training" else case.graph.name
    v2_model = _case_v2_model(case)
    return load_omni_config_from_dir(
        v2_model.config_dir,
        graph=graph,
        graph_domain=case.graph.domain,
        module_names=names,
    )


def load_graph_active_omni_model(
    case: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> OmniModel:
    """Load the default graph-active V2 model for a parity case."""

    config = load_graph_active_omni_config(case)
    modules = load_graph_active_omni_modules(case, config.module_names, device=device, dtype=dtype)
    return OmniModel(config, modules).eval()


def load_graph_active_omni_modules(
    case: Any,
    module_names: Iterable[str],
    *,
    device: torch.device,
    dtype: torch.dtype,
    module_overrides: Any | None = None,
) -> dict[str, nn.Module]:
    module_names = tuple(module_names)
    if module_overrides is None:
        module_overrides = _case_recipe_module_overrides(case)
    overrides = v2_module_override_map(module_overrides)
    unknown = sorted(set(overrides).difference(module_names))
    if unknown:
        raise KeyError(f"v2_model.module_overrides references inactive module(s): {unknown}")
    _validate_v2_module_override_scope(case, module_names, overrides)

    v2_model = _case_v2_model(case)
    model_root = v2_model.model_root
    if model_root is None:
        config = load_graph_active_omni_config(case, module_names)
        modules: dict[str, nn.Module] = {}
        for name in module_names:
            module_device, module_dtype = v2_module_target(
                overrides.get(name),
                default_device=device,
                default_dtype=dtype,
            )
            modules[name] = load_omni_module_from_parity_config(
                name,
                config.modules[name],
                seed=int(case.model.seed),
                device=module_device,
                dtype=module_dtype,
            ).eval()
        return modules

    modules = {}
    for name in module_names:
        module_device, module_dtype = v2_module_target(
            overrides.get(name),
            default_device=device,
            default_dtype=dtype,
        )
        modules[name] = load_omni_module_from_pretrained(
            model_root / name,
            device=module_device,
            dtype=module_dtype,
        ).eval()
    return modules


def v2_module_override_map(overrides: Any | None) -> Mapping[str, Any]:
    if overrides is None:
        return {}
    if not isinstance(overrides, Mapping):
        raise TypeError("v2_model.module_overrides must be a mapping.")
    return overrides


def v2_module_target(
    override: Any,
    *,
    default_device: torch.device,
    default_dtype: torch.dtype,
) -> tuple[torch.device, torch.dtype]:
    if override is None:
        return default_device, default_dtype
    if not isinstance(override, Mapping):
        raise TypeError("v2_model.module_overrides.<module> must be a mapping.")
    raw_device = override.get("device", default_device)
    raw_dtype = override.get("dtype", default_dtype)
    target_device = raw_device if isinstance(raw_device, torch.device) else torch.device(str(raw_device))
    return target_device, resolve_torch_dtype(raw_dtype)


def load_omni_config_from_dir(
    config_dir: str | Path,
    *,
    graph: str | None = None,
    graph_domain: str | None = None,
    module_names: Iterable[str] | None = None,
) -> OmniConfig:
    root = Path(config_dir)
    modules = _load_yaml(root / "modules_train.yaml")
    if module_names is not None:
        modules = _filter_modules(modules, module_names)
    data: dict[str, Any] = {"modules": modules}
    if graph_domain != "inference":
        data["training_graph"] = _load_yaml(root / "graph_train.yaml")["training_graph"]
    if graph is not None:
        graph_name = graph if graph.endswith(".yaml") else f"graph_{graph}.yaml"
        infer = _load_yaml(root / graph_name)
        data["generation_graph"] = infer["generation_graph"]
        data["generation_kwargs"] = infer.get("generation_kwargs", {})
    return OmniConfig.from_dict(data)


def load_omni_module_from_pretrained(
    module_dir: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    model_type = read_model_type(str(module_dir))
    model_cls = OMNI_MODEL_REGISTRY[model_type]()
    module = model_cls.from_pretrained(module_dir, torch_dtype=dtype)
    return module.to(device=device)


def load_omni_module_from_parity_config(
    module_name: str,
    module_config: Mapping[str, Any],
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    parity = module_config.get("parity")
    if not isinstance(parity, Mapping):
        raise ValueError(f"{module_name} has no pretrained model_root and no modules_train.yaml parity config.")
    model_type = str(parity["model_type"])
    cfg_cls = OMNI_CONFIG_REGISTRY[model_type]()
    model_cls = OMNI_MODEL_REGISTRY[model_type]()
    config_values = dict(parity.get("config") or {})
    torch.manual_seed(seed)
    module = model_cls(cfg_cls(**config_values))
    setup = parity.get("setup")
    if setup is not None:
        _load_callable(str(setup))(module, seed=seed)
    return module.to(device=device, dtype=dtype)


def _load_callable(entrypoint: str):
    module_name, symbol_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _case_v2_model(case: Any) -> Any:
    v2_model = getattr(case, "v2_model", None)
    return v2_model if v2_model is not None else case.model.v2_model


def _case_recipe_module_overrides(case: Any) -> Any | None:
    recipe = getattr(case, "recipe", None)
    recipe_v2_model = getattr(recipe, "v2_model", None)
    if recipe_v2_model is None:
        return None
    return recipe_v2_model.module_overrides


def _validate_v2_module_override_scope(
    case: Any,
    module_names: Iterable[str],
    overrides: Mapping[str, Any],
) -> None:
    if not overrides:
        return
    tier = _case_tier(case)
    if tier in {"graph", "framework"}:
        return
    if tier == "module":
        inactive = sorted(set(overrides).difference(module_names))
        if inactive:
            raise KeyError(
                "v2_model.module_overrides in module tier may only reference module(s) "
                f"loaded by the current module target; inactive module(s): {inactive}"
            )
        return
    raise ValueError(
        "v2_model.module_overrides is only supported for graph/framework tiers, "
        "or for module tier when the override targets the current module."
    )


def _case_tier(case: Any) -> str | None:
    tier = getattr(case, "tier", None)
    if tier is not None:
        return str(tier)
    run = getattr(case, "run", None)
    run_tier = getattr(run, "tier", None)
    if run_tier is not None:
        return str(run_tier)
    return None


def _filter_modules(modules: Mapping[str, Any], module_names: Iterable[str]) -> dict[str, Any]:
    selected = frozenset(module_names)
    missing = sorted(selected.difference(modules))
    if missing:
        raise KeyError(f"Graph references module(s) missing from modules_train.yaml: {missing}")
    return {name: modules[name] for name in modules if name in selected}


__all__ = [
    "graph_active_module_names",
    "load_graph_active_omni_config",
    "load_graph_active_omni_model",
    "load_graph_active_omni_modules",
    "load_omni_config_from_dir",
    "load_omni_module_from_parity_config",
    "load_omni_module_from_pretrained",
    "v2_module_override_map",
    "v2_module_target",
]
