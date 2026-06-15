"""Core model-contract and discovery helpers for the parity suite."""

from .comparator import compare_values
from .discovery import GraphSpec, NodeSpec, ParityCase, discover_cases, discover_graph_specs, discover_nodes
from .gate import case_skip_reason
from .mapping import (
    MappingSpec,
    ProbeMapping,
    RefTapSpec,
    ResolvedMapping,
    V2GradSpec,
    load_mapping_spec,
    resolve_mapping,
)
from .metrics import MetricResult, Tolerance, compare_tensors, tolerance_from_policy
from .report import ParityReport, ProbeReport
from .spec import (
    DEFAULT_GATE,
    PARITY_ENABLE_ENV,
    GateSpec,
    LauncherSpec,
    ModelSpec,
    RecipeSpec,
    ReferenceSpec,
    RunSpec,
    load_model_spec,
)
from .utilities import (
    autocast_for_dtype,
    configure_torch_determinism,
    patched_randn_like,
    sample_grad,
    sample_named_grad,
    sum_losses,
    to_cpu,
    to_device,
    zero_module_grads,
)


__all__ = [
    "DEFAULT_GATE",
    "GateSpec",
    "GraphSpec",
    "LauncherSpec",
    "MappingSpec",
    "ModelSpec",
    "NodeSpec",
    "PARITY_ENABLE_ENV",
    "ParityCase",
    "ParityReport",
    "MetricResult",
    "ProbeReport",
    "ProbeMapping",
    "RecipeSpec",
    "ReferenceSpec",
    "RefTapSpec",
    "ResolvedMapping",
    "RunSpec",
    "Tolerance",
    "V2GradSpec",
    "autocast_for_dtype",
    "case_skip_reason",
    "configure_torch_determinism",
    "compare_tensors",
    "compare_values",
    "discover_cases",
    "discover_graph_specs",
    "discover_nodes",
    "load_mapping_spec",
    "load_model_spec",
    "patched_randn_like",
    "sample_grad",
    "sample_named_grad",
    "resolve_mapping",
    "sum_losses",
    "to_cpu",
    "to_device",
    "tolerance_from_policy",
    "zero_module_grads",
]
