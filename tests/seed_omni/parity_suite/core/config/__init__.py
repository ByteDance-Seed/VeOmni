"""Configuration, discovery, gating, and probe helpers for the parity suite."""

from .discovery import (
    GraphSpec,
    NodeSpec,
    ParityCase,
    discover_cases,
    discover_graph_specs,
    discover_nodes,
    effective_reference_kind,
)
from .gate import case_skip_reason
from .probes import (
    ProbeCatalog,
    ProbeMapping,
    RefTapSpec,
    ResolvedProbes,
    V2GradSpec,
    load_probe_catalog,
    resolve_probes,
)
from .spec import (
    DEFAULT_GATE,
    PARITY_ENABLE_ENV,
    GateSpec,
    HfModelReferenceSpec,
    LauncherSpec,
    ModelSpec,
    RecipeSpec,
    RecipeV2ModelSpec,
    ReferenceSpec,
    RunSpec,
    V2ModelSpec,
    V2ModelTargetSpec,
    load_model_spec,
    select_v2_model_target,
)


__all__ = [
    # Spec contract
    "DEFAULT_GATE",
    "GateSpec",
    "HfModelReferenceSpec",
    "LauncherSpec",
    "ModelSpec",
    "PARITY_ENABLE_ENV",
    "RecipeSpec",
    "RecipeV2ModelSpec",
    "ReferenceSpec",
    "RunSpec",
    "V2ModelSpec",
    "V2ModelTargetSpec",
    "load_model_spec",
    "select_v2_model_target",
    # Discovery contract
    "GraphSpec",
    "NodeSpec",
    "ParityCase",
    "discover_cases",
    "discover_graph_specs",
    "discover_nodes",
    "effective_reference_kind",
    # Probe contract
    "ProbeCatalog",
    "ProbeMapping",
    "RefTapSpec",
    "ResolvedProbes",
    "V2GradSpec",
    "load_probe_catalog",
    "resolve_probes",
    # Gate evaluation
    "case_skip_reason",
]
