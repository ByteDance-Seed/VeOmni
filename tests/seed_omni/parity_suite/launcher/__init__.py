"""GPU-aware launcher for SeedOmni V2 parity cases."""

# GPU subprocess launcher ------------------------------------------------------

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

# Pytest integration -----------------------------------------------------------
from .pytest import run_case_with_pytest_launcher, should_use_pytest_launcher


__all__ = [
    # GPU subprocess launcher
    "LauncherResult",
    "LAUNCHER_CHILD_ENV",
    "TEST_ENTRYPOINT",
    "build_case_env",
    "build_pytest_command",
    "configured_cuda_devices",
    "required_cuda_devices",
    "run_cases",
    # Pytest integration
    "run_case_with_pytest_launcher",
    "should_use_pytest_launcher",
]
