"""End-to-end parity execution for discovered cases."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from tests.seed_omni.parity_suite.core import (
    ParityCase,
    ParityReport,
    ProbeMapping,
    ProbeReport,
    compare_values,
    resolve_mapping,
    tolerance_from_policy,
)
from tests.seed_omni.parity_suite.driver import ParityDriver
from tests.seed_omni.parity_suite.reference.capture import capture_reference_taps


def run_parity_case(case: ParityCase) -> ParityReport:
    """Run one discovered graph case against its online reference oracle."""

    if case.tier not in {"graph", "module", "framework"}:
        raise NotImplementedError(f"Unsupported parity tier for execution: {case.tier!r}")
    selected = case.model.mapping.for_probe_names(case.scenario.probes)
    resolved = resolve_mapping(mappings=selected, nodes=case.nodes)
    driver = _load_driver(case)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = driver.dtype()

    driver.configure_determinism(case.model.seed)
    reference = capture_reference_taps(
        reference_factory=lambda: driver.load_reference(device=device, dtype=dtype),
        driver=driver,
        inputs=driver.reference_inputs(),
        plan=resolved.reference_plan,
    )

    driver.configure_determinism(case.model.seed)
    v2_result = _run_v2(driver, case, reference.run_output, resolved.v2_whitelist, device=device, dtype=dtype)

    reports: list[ProbeReport] = []
    for mapping in resolved.probes:
        actual = _v2_probe_values(v2_result["observations"], mapping, case=case)
        expected = _reference_probe_values(reference.taps, mapping)
        metric = compare_values(
            actual,
            expected,
            tolerance=tolerance_from_policy(mapping.tol, case.model.tolerance),
            path=mapping.probe,
        )
        reports.append(ProbeReport(node=mapping.node, probe=mapping.probe, passed=metric.passed, metric=metric))
    return ParityReport(case_id=case.node_id, probes=tuple(reports))


def _load_driver(case: ParityCase) -> ParityDriver:
    module = importlib.import_module(f"tests.seed_omni.{case.model.name}.driver")
    driver = module.create_driver(case)
    if not isinstance(driver, ParityDriver):
        raise TypeError(f"{module.__name__}.create_driver must return a ParityDriver, got {type(driver).__name__}.")
    return driver


def _run_v2(
    driver: ParityDriver,
    case: ParityCase,
    reference_output: Any,
    v2_whitelist: dict[tuple[str, str], frozenset[str]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    if case.graph.domain == "training":
        if case.tier == "graph":
            return driver.run_v2_train_graph(reference_output, v2_whitelist, device=device, dtype=dtype)
        if case.tier == "module":
            return driver.run_v2_train_module(reference_output, v2_whitelist, device=device, dtype=dtype)
        if case.tier == "framework":
            return driver.run_v2_train_framework(reference_output, v2_whitelist, device=device, dtype=dtype)
    if case.graph.domain == "inference":
        if case.tier == "graph":
            return driver.run_v2_infer_graph(reference_output, v2_whitelist, device=device, dtype=dtype)
        if case.tier == "module":
            return driver.run_v2_infer_module(reference_output, v2_whitelist, device=device, dtype=dtype)
        if case.tier == "framework":
            return driver.run_v2_infer_framework(reference_output, v2_whitelist, device=device, dtype=dtype)
    raise NotImplementedError(f"Unsupported V2 dispatch for domain={case.graph.domain!r}, tier={case.tier!r}.")


def _reference_probe_values(taps: dict[str, list[Any]], mapping: ProbeMapping) -> list[Any]:
    try:
        return taps[mapping.probe]
    except KeyError as exc:
        raise KeyError(f"Reference tap {mapping.probe!r} was not captured.") from exc


def _v2_probe_values(
    observations: dict[tuple[str, str], list[dict[str, Any]]],
    mapping: ProbeMapping,
    *,
    case: ParityCase,
) -> list[Any]:
    values: list[Any] = []
    for node in case.nodes:
        if node.name != mapping.node or node.state is None:
            continue
        for record in observations.get((node.state, node.name), []):
            if mapping.v2_field in record:
                values.append(record[mapping.v2_field])
    if not values:
        raise KeyError(f"V2 field {mapping.node}.{mapping.v2_field} was not observed.")
    return values


__all__ = ["run_parity_case"]
