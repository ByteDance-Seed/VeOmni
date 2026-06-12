"""Runtime dispatcher for generated parity cases."""

from __future__ import annotations

from tests.seed_omni.parity_suite.core.imports import import_object, import_optional_module
from tests.seed_omni.parity_suite.core.probes import load_probe_bindings, missing_probe_bindings
from tests.seed_omni.parity_suite.core.report import ParityReport
from tests.seed_omni.parity_suite.core.spec import CaseSpec


class CaseRunner:
    """Small facade that delegates model semantics to the configured adapter."""

    def __init__(self, case: CaseSpec) -> None:
        if not case.adapter:
            raise ValueError(f"{case.node_id} does not declare an adapter.")
        adapter_cls = import_object(case.adapter)
        self.case = case
        self.probes_module = import_optional_module(case.probes_module)
        self.captures_module = import_optional_module(case.captures_module)
        self.adapter = adapter_cls(
            case=case,
            probes_module=self.probes_module,
            captures_module=self.captures_module,
        )

    def run(self) -> ParityReport:
        if hasattr(self.adapter, "run_case"):
            report = self.adapter.run_case()
            if isinstance(report, ParityReport):
                return report
            if isinstance(report, dict):
                return ParityReport(
                    case_id=self.case.node_id,
                    category=self.case.category,
                    all_pass=bool(report.get("all_pass")),
                    probes=dict(report.get("probes", {})),
                    metadata={key: value for key, value in report.items() if key not in {"all_pass", "probes"}},
                )
            raise TypeError(f"{self.case.adapter}.run_case returned unsupported type: {type(report).__name__}")
        return self._declaration_report()

    def _declaration_report(self) -> ParityReport:
        probes = {}
        bindings = load_probe_bindings(self.probes_module)
        missing = missing_probe_bindings(self.case.probes, bindings)
        for probe in self.case.probes:
            probes[probe] = {"declared": probe in bindings, "passes": probe in bindings}
        return ParityReport(
            case_id=self.case.node_id,
            category=self.case.category,
            all_pass=not missing,
            probes=probes,
            metadata={
                "mode": "declaration",
                "missing_probe_bindings": missing,
            },
        )


def run_parity_case(case: CaseSpec) -> ParityReport:
    return CaseRunner(case).run()
