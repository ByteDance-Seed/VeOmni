"""Minimal GPU-pool launcher for SeedOmni V2 parity cases."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.seed_omni.parity_suite.core import PARITY_ENABLE_ENV, ParityCase


LAUNCHER_CHILD_ENV = "VEOMNI_PARITY_GPU_LAUNCHER_CHILD"
TEST_ENTRYPOINT = "tests/seed_omni/parity_suite/test_parity_cases.py::test_seed_omni_v2_parity_case"


@dataclass(frozen=True)
class LauncherResult:
    case_id: str
    returncode: int
    log_path: Path


@dataclass
class _RunningCase:
    case: ParityCase
    process: subprocess.Popen[bytes]
    log_file: Any
    log_path: Path
    cuda_devices: tuple[str, ...]


def required_cuda_devices(case: ParityCase) -> int:
    """Return the number of CUDA devices this case should reserve."""

    if not case.requires_cuda:
        return 0
    return max(1, case.min_cuda_devices)


def configured_cuda_devices(cases: Sequence[ParityCase]) -> tuple[str, ...]:
    """Return visible CUDA devices, capped by selected model launcher config."""

    devices = _visible_cuda_devices()
    configured_max = max(
        (case.model.launcher.max_cuda_devices or 0 for case in cases),
        default=0,
    )
    if configured_max:
        devices = devices[:configured_max]
    return devices


def build_pytest_command(case_id: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        f"{TEST_ENTRYPOINT}[{case_id}]",
    ]


def build_case_env(
    *,
    cuda_devices: Sequence[str],
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env[PARITY_ENABLE_ENV] = "1"
    env[LAUNCHER_CHILD_ENV] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_devices)
    return env


def run_cases(
    cases: Sequence[ParityCase],
    *,
    cuda_devices: Sequence[str] | None = None,
    output_dir: Path | None = None,
    parallel: bool | None = None,
    on_result: Callable[[LauncherResult], None] | None = None,
) -> list[LauncherResult]:
    """Run parity cases through pytest subprocesses with GPU pool scheduling."""

    selected_devices = tuple(configured_cuda_devices(cases) if cuda_devices is None else cuda_devices)
    output_root = output_dir or Path("outputs") / "parity_suite" / "launcher"
    output_root.mkdir(parents=True, exist_ok=True)
    enable_parallel = _enable_parallel(cases) if parallel is None else parallel
    if not enable_parallel or not selected_devices:
        return _run_serial(cases, cuda_devices=selected_devices, output_dir=output_root, on_result=on_result)
    return _run_parallel(cases, cuda_devices=selected_devices, output_dir=output_root, on_result=on_result)


def _run_serial(
    cases: Sequence[ParityCase],
    *,
    cuda_devices: tuple[str, ...],
    output_dir: Path,
    on_result: Callable[[LauncherResult], None] | None,
) -> list[LauncherResult]:
    results: list[LauncherResult] = []
    for case in cases:
        devices = _case_cuda_devices(case, cuda_devices)
        running = _start_case(case, cuda_devices=devices, output_dir=output_dir)
        result = _wait_case(running)
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results


def _run_parallel(
    cases: Sequence[ParityCase],
    *,
    cuda_devices: tuple[str, ...],
    output_dir: Path,
    on_result: Callable[[LauncherResult], None] | None,
) -> list[LauncherResult]:
    pending = list(cases)
    available = list(cuda_devices)
    running: list[_RunningCase] = []
    results: list[LauncherResult] = []

    while pending or running:
        launched = False
        for case in tuple(pending):
            needed = _case_slots(case, total_cuda_devices=len(cuda_devices))
            if needed > len(available):
                continue
            devices = tuple(available[:needed])
            del available[:needed]
            pending.remove(case)
            running.append(_start_case(case, cuda_devices=devices, output_dir=output_dir))
            launched = True

        completed = _collect_completed(running)
        if completed:
            for finished in completed:
                available.extend(finished.cuda_devices)
                result = _finish_case(finished)
                results.append(result)
                if on_result is not None:
                    on_result(result)
            continue
        if not launched:
            time.sleep(0.5)

    return results


def _start_case(case: ParityCase, *, cuda_devices: tuple[str, ...], output_dir: Path) -> _RunningCase:
    log_path = output_dir / f"{_safe_case_id(case.node_id)}.log"
    log_file = log_path.open("wb")
    command = build_pytest_command(case.node_id)
    print(f"START {case.node_id} cuda={','.join(cuda_devices) or '<none>'} log={log_path}")
    process = subprocess.Popen(
        command, env=build_case_env(cuda_devices=cuda_devices), stdout=log_file, stderr=subprocess.STDOUT
    )
    return _RunningCase(case=case, process=process, log_file=log_file, log_path=log_path, cuda_devices=cuda_devices)


def _wait_case(running: _RunningCase) -> LauncherResult:
    running.process.wait()
    return _finish_case(running)


def _finish_case(running: _RunningCase) -> LauncherResult:
    running.log_file.close()
    result = LauncherResult(
        case_id=running.case.node_id,
        returncode=running.process.returncode,
        log_path=running.log_path,
    )
    status = "PASS" if result.returncode == 0 else "FAIL"
    print(f"{status} {result.case_id} log={result.log_path}")
    return result


def _collect_completed(running: list[_RunningCase]) -> list[_RunningCase]:
    completed = [item for item in running if item.process.poll() is not None]
    for item in completed:
        running.remove(item)
    return completed


def _case_cuda_devices(case: ParityCase, cuda_devices: tuple[str, ...]) -> tuple[str, ...]:
    needed = _case_slots(case, total_cuda_devices=len(cuda_devices))
    return cuda_devices[:needed]


def _case_slots(case: ParityCase, *, total_cuda_devices: int) -> int:
    if not case.model.launcher.enable_parallel:
        return total_cuda_devices
    needed = required_cuda_devices(case)
    if needed == 0:
        return 0
    if total_cuda_devices == 0:
        return 0
    return min(needed, total_cuda_devices)


def _enable_parallel(cases: Sequence[ParityCase]) -> bool:
    return any(case.model.launcher.enable_parallel for case in cases)


def _visible_cuda_devices() -> tuple[str, ...]:
    raw_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw_visible is not None:
        return tuple(device.strip() for device in raw_visible.split(",") if device.strip())

    try:
        import torch
    except ImportError:
        return ()
    if not torch.cuda.is_available():
        return ()
    return tuple(str(index) for index in range(torch.cuda.device_count()))


def _safe_case_id(case_id: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in case_id)
