"""Shared runtime helpers for fixture-backed and online reference capture."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from tests.seed_omni.parity_suite.capture.plan import capture_output, execute_capture_plan, load_capture_plan
from tests.seed_omni.parity_suite.core.probes import ProbeBinding
from tests.seed_omni.parity_suite.core.spec import CaseSpec


ReferenceModelLoader = Callable[[CaseSpec], torch.nn.Module]


def load_fixture(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def fixture_case_id(fixture: dict[str, Any]) -> str | None:
    metadata = fixture.get("metadata", {})
    value = metadata.get("case_id")
    return value if isinstance(value, str) else None


def capture_case(
    case: CaseSpec,
    probe_bindings: dict[str, ProbeBinding],
    *,
    load_reference_model: ReferenceModelLoader,
) -> dict[str, Any]:
    plan = load_capture_plan(case)
    model = load_reference_model(case)
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError(f"{case.node_id} reference model does not expose a config.")
    return execute_capture_plan(case, plan, probe_bindings, model=model, config=config)


def get_path(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(path)
        value = value[part]
    return value


def available_fixture_paths(
    fixture: dict[str, Any], case: CaseSpec, probe_bindings: dict[str, ProbeBinding], probes: tuple[str, ...]
) -> dict[str, bool]:
    result: dict[str, bool] = {}
    plan = load_capture_plan(case)
    for probe in probes:
        if probe not in probe_bindings:
            result[probe] = False
            continue
        try:
            output = capture_output(plan, probe)
            get_path(fixture, output.path)
            result[probe] = True
        except KeyError:
            result[probe] = False
    return result


def fixture_tolerance(fixture: dict[str, Any]) -> dict[str, float]:
    metadata = fixture.get("metadata", {})
    dtype = str(metadata.get("dtype", "fp32"))
    tolerances = fixture.get("tolerances", {})
    return dict(tolerances.get(dtype) or next(iter(tolerances.values()), {}))


def extract_reference_value(fixture: dict[str, Any], case: CaseSpec, probe: str) -> Any:
    output = capture_output(load_capture_plan(case), probe)
    return get_path(fixture, output.path)
