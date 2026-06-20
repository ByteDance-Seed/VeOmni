"""End-to-end parity execution for discovered cases."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

from tests.seed_omni.parity_suite.core import (
    ParityCase,
    ParityReport,
    ProbeReport,
    compare_values,
    reference_probe_values,
    resolve_probes,
    tolerance_from_policy,
    v2_probe_values,
)
from tests.seed_omni.parity_suite.driver import ParityDriver
from tests.seed_omni.parity_suite.reference.capture import build_reference_capture_plan


_V2_DISPATCH: dict[tuple[str, str], str] = {
    ("training", "graph"): "run_v2_train_graph_recipe",
    ("training", "framework"): "run_v2_train_framework_recipe",
    ("inference", "graph"): "run_v2_infer_graph_recipe",
    ("inference", "module"): "run_v2_infer_module_recipe",
    ("inference", "framework"): "run_v2_infer_framework_recipe",
}


# Public runner entrypoint -----------------------------------------------------


def run_parity_case(case: ParityCase) -> ParityReport:
    """Run one discovered graph case against its online reference oracle."""

    driver = _load_driver(case)
    if case.tier == "reference":
        return driver.run_reference_only_recipe()
    if case.tier not in {"graph", "module", "framework"}:
        raise NotImplementedError(f"Unsupported parity tier for execution: {case.tier!r}")
    # Non-forward framework policies produce their own reports instead of
    # comparing node-level reference observations.
    selected = () if _is_framework_policy_run(case) else case.model.probes.for_probe_names(case.run.probes)
    resolved = resolve_probes(probes=selected, nodes=case.nodes)
    reference_plan = build_reference_capture_plan(resolved.ref_taps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = driver.dtype()

    reference = None
    reference_output = None
    if case.effective_gate.requires_reference_capture:
        driver.configure_determinism(case.model.seed)
        with _reference_grad_context(case):
            reference = driver.reference_oracle().capture(
                inputs=driver.reference_inputs(),
                plan=reference_plan,
                device=device,
                dtype=dtype,
            )
        reference_output = reference.run_output

    driver.configure_determinism(case.model.seed)
    v2_result = _run_v2(driver, case, reference_output, resolved.v2_whitelist, device=device, dtype=dtype)
    if isinstance(v2_result, ParityReport):
        return v2_result
    if reference is None:
        raise RuntimeError(f"{case.node_id} disabled reference capture but did not return a policy report.")

    reports: list[ProbeReport] = []
    for mapping in resolved.probes:
        actual = v2_probe_values(v2_result["observations"], mapping, case=case)
        expected = reference_probe_values(reference.observations, mapping)
        metric = compare_values(
            actual,
            expected,
            tolerance=tolerance_from_policy(mapping.tol, case.model.tolerance),
            path=mapping.probe,
            compare_steps=mapping.step,
        )
        reports.append(ProbeReport(node=mapping.node, probe=mapping.probe, passed=metric.passed, metric=metric))
    return ParityReport(case_id=case.node_id, probes=tuple(reports))


# Internal dispatch helpers ----------------------------------------------------


def _is_framework_policy_run(case: ParityCase) -> bool:
    return case.tier == "framework" and case.run.kind != "forward_backward"


@contextmanager
def _reference_grad_context(case: ParityCase) -> Iterator[None]:
    if case.graph.domain == "training":
        with torch.enable_grad():
            yield
        return
    with torch.no_grad():
        yield


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
) -> dict[str, Any] | ParityReport:
    method_name = _V2_DISPATCH.get((case.graph.domain, case.tier))
    if method_name is None:
        raise NotImplementedError(f"Unsupported V2 dispatch for domain={case.graph.domain!r}, tier={case.tier!r}.")
    return getattr(driver, method_name)(reference_output, v2_whitelist, device=device, dtype=dtype)


__all__ = ["run_parity_case"]
