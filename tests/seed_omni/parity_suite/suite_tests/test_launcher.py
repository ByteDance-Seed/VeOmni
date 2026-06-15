"""Tests for the parity-suite GPU launcher planning helpers."""

from __future__ import annotations

from types import SimpleNamespace

from tests.seed_omni.parity_suite.core import PARITY_ENABLE_ENV, LauncherSpec
from tests.seed_omni.parity_suite.launcher import (
    LAUNCHER_CHILD_ENV,
    build_case_env,
    build_pytest_command,
    required_cuda_devices,
)
from tests.seed_omni.parity_suite.launcher.gpu import _case_slots


def _case(
    *,
    requires_cuda: bool,
    min_cuda_devices: int = 0,
    enable_parallel: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        requires_cuda=requires_cuda,
        min_cuda_devices=min_cuda_devices,
        model=SimpleNamespace(launcher=LauncherSpec(enable_parallel=enable_parallel)),
    )


def test_required_cuda_devices_defaults_cuda_case_to_one_device() -> None:
    assert required_cuda_devices(_case(requires_cuda=False)) == 0
    assert required_cuda_devices(_case(requires_cuda=True)) == 1
    assert required_cuda_devices(_case(requires_cuda=True, min_cuda_devices=2)) == 2


def test_serial_model_case_reserves_all_visible_devices_in_parallel_pool() -> None:
    case = _case(requires_cuda=False, enable_parallel=False)

    assert _case_slots(case, total_cuda_devices=8) == 8


def test_build_case_env_sets_parity_gate_and_cuda_visibility() -> None:
    env = build_case_env(cuda_devices=("2", "3"), base_env={"PATH": "/bin"})

    assert env[PARITY_ENABLE_ENV] == "1"
    assert env[LAUNCHER_CHILD_ENV] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert env["PATH"] == "/bin"


def test_build_pytest_command_targets_single_parametrized_case() -> None:
    command = build_pytest_command("bagel.image_gen.graph.base_one_step")

    assert command[-1].endswith(
        "test_parity_cases.py::test_seed_omni_v2_parity_case[bagel.image_gen.graph.base_one_step]"
    )
