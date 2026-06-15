"""GPU-aware launcher for SeedOmni V2 parity cases."""

from .gpu import (
    LAUNCHER_CHILD_ENV,
    TEST_ENTRYPOINT,
    LauncherResult,
    build_case_env,
    build_pytest_command,
    configured_cuda_devices,
    required_cuda_devices,
    run_cases,
)
from .pytest import run_case_with_pytest_launcher, should_use_pytest_launcher


__all__ = [
    "LauncherResult",
    "LAUNCHER_CHILD_ENV",
    "TEST_ENTRYPOINT",
    "build_case_env",
    "build_pytest_command",
    "configured_cuda_devices",
    "required_cuda_devices",
    "run_case_with_pytest_launcher",
    "run_cases",
    "should_use_pytest_launcher",
]
