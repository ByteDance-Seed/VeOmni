"""Pytest integration for the parity-suite GPU launcher."""

from __future__ import annotations

import os
import threading
from typing import Any

from tests.seed_omni.parity_suite.core import ParityCase, case_skip_reason

from .gpu import LAUNCHER_CHILD_ENV, LauncherResult, run_cases


_LAUNCHER_SESSIONS: dict[tuple[str, ...], _PytestLauncherSession] = {}


class _PytestLauncherSession:
    def __init__(self, cases: tuple[ParityCase, ...]) -> None:
        self.cases = cases
        self._condition = threading.Condition()
        self._results: dict[str, LauncherResult] = {}
        self._error: BaseException | None = None
        self._done = False
        self._started = False

    def wait_for(self, case: ParityCase) -> LauncherResult:
        self._start()
        with self._condition:
            while case.node_id not in self._results and not self._done and self._error is None:
                self._condition.wait()
            if self._error is not None:
                raise RuntimeError("Parity GPU launcher failed.") from self._error
            result = self._results.get(case.node_id)
            if result is None:
                raise RuntimeError(f"Parity GPU launcher did not report a result for {case.node_id}.")
            return result

    def _start(self) -> None:
        with self._condition:
            if self._started:
                return
            self._started = True
        thread = threading.Thread(target=self._run, name="parity-gpu-launcher", daemon=True)
        thread.start()

    def _run(self) -> None:
        try:
            run_cases(self.cases, on_result=self._record_result)
        except BaseException as exc:  # noqa: BLE001 - propagate worker failures to pytest items.
            with self._condition:
                self._error = exc
                self._done = True
                self._condition.notify_all()
        else:
            with self._condition:
                self._done = True
                self._condition.notify_all()

    def _record_result(self, result: LauncherResult) -> None:
        with self._condition:
            self._results[result.case_id] = result
            self._condition.notify_all()


def is_launcher_child() -> bool:
    return os.environ.get(LAUNCHER_CHILD_ENV) == "1"


def should_use_pytest_launcher(case: ParityCase, request: Any) -> bool:
    return (
        case.model.launcher.enable_parallel
        and not is_launcher_child()
        and not _is_direct_case_selection(case, request)
    )


def run_case_with_pytest_launcher(case: ParityCase, request: Any) -> LauncherResult:
    return _launcher_session(request).wait_for(case)


def _is_direct_case_selection(case: ParityCase, request: Any) -> bool:
    selected_id = f"[{case.node_id}]"
    return any(selected_id in arg for arg in request.config.args)


def _launcher_session(request: Any) -> _PytestLauncherSession:
    cases = _selected_runnable_launcher_cases(request)
    key = tuple(case.node_id for case in cases)
    session = _LAUNCHER_SESSIONS.get(key)
    if session is None:
        session = _PytestLauncherSession(cases)
        _LAUNCHER_SESSIONS[key] = session
    return session


def _selected_runnable_launcher_cases(request: Any) -> tuple[ParityCase, ...]:
    cases: list[ParityCase] = []
    for item in request.session.items:
        callspec = getattr(item, "callspec", None)
        params = getattr(callspec, "params", {})
        case = params.get("case") if isinstance(params, dict) else None
        if isinstance(case, ParityCase) and case.model.launcher.enable_parallel and case_skip_reason(case) is None:
            cases.append(case)
    return tuple(cases)
