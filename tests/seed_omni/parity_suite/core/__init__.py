"""Core case schema, discovery, and runner primitives for parity_suite."""

from tests.seed_omni.parity_suite.core.probes import ProbeAnchor, ProbeBinding, probe_binding
from tests.seed_omni.parity_suite.core.registry import discover_cases
from tests.seed_omni.parity_suite.core.spec import BackendSpec, CaseSpec, EnvSpec, V2ModelSpec


__all__ = [
    "BackendSpec",
    "CaseSpec",
    "EnvSpec",
    "ProbeAnchor",
    "ProbeBinding",
    "V2ModelSpec",
    "discover_cases",
    "probe_binding",
]
