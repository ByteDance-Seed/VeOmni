"""SeedOmni V2 parity suite test harness.

The package exposes the shared building blocks used by
model-owned pytest entrypoints and parity drivers:

- ``core`` discovers YAML contracts, resolves graph nodes, applies gates, and
  compares mapped probes.
- ``reference`` loads reference oracles and captures configured tap values.
- ``v2`` runs the SeedOmni V2 graph/module/framework tiers and records
  observations for comparison.
- ``launcher`` optionally schedules pytest subprocesses across visible CUDA
  devices for large suites.
"""

from .core import PARITY_ENABLE_ENV, LauncherSpec, ModelSpec, ParityCase, RecipeSpec, discover_cases, load_model_spec
from .driver import ParityDriver
from .runner import run_parity_case


__all__ = [
    "ModelSpec",
    "PARITY_ENABLE_ENV",
    "LauncherSpec",
    "ParityCase",
    "ParityDriver",
    "RecipeSpec",
    "discover_cases",
    "load_model_spec",
    "run_parity_case",
]
