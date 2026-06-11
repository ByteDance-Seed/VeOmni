"""Durable BAGEL module-level training official parity tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from tests.seed_omni.fixtures.bagel.compare_gradient_graph import compare_gradient_graph
from tests.seed_omni.fixtures.bagel.compare_gradient_module import compare_gradient_module


_ENV_PREFIX = "VEOMNI_V2_TEST_BAGEL_"


def _env_name(suffix: str) -> str:
    return f"{_ENV_PREFIX}{suffix}"


def _env_value(suffix: str) -> str | None:
    return os.environ.get(_env_name(suffix))


def _env_flag(suffix: str) -> bool:
    value = _env_value(suffix)
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


pytestmark = pytest.mark.skipif(
    not _env_flag("ENABLE_PARITY_CHECK"),
    reason=f"Set {_env_name('ENABLE_PARITY_CHECK')}=1 to run BAGEL official training parity checks.",
)


def _assert_module_gradient_parity(fixture_suffix: str, description: str) -> None:
    fixture_path = _env_value(fixture_suffix)
    model_root = _env_value("SPLIT_MODEL_ROOT")
    if not fixture_path or not model_root:
        pytest.skip(
            f"Set {_env_name(fixture_suffix)} and {_env_name('SPLIT_MODEL_ROOT')} "
            f"to run BAGEL {description} module-level backward parity."
        )
    if not torch.cuda.is_available():
        pytest.skip(f"BAGEL {description} module-level backward parity requires CUDA efficient attention.")

    report = compare_gradient_module(
        Path(fixture_path),
        Path(model_root),
    )
    assert report["all_pass"], report


def _bagel_cfg_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "seed_omni" / "Bagel" / "bagel_7b_mot"


def _assert_graph_gradient_parity(fixture_suffix: str, description: str) -> None:
    fixture_path = _env_value(fixture_suffix)
    model_root = _env_value("SPLIT_MODEL_ROOT")
    if not fixture_path or not model_root:
        pytest.skip(
            f"Set {_env_name(fixture_suffix)} and {_env_name('SPLIT_MODEL_ROOT')} "
            f"to run BAGEL {description} graph-level backward parity."
        )
    if not torch.cuda.is_available():
        pytest.skip(f"BAGEL {description} graph-level backward parity requires CUDA efficient attention.")

    report = compare_gradient_graph(
        Path(fixture_path),
        Path(model_root),
        config_dir=_bagel_cfg_dir(),
    )
    assert report["all_pass"], report


def test_bagel_gradient_ce_only_module_backward_matches_official_fixture() -> None:
    _assert_module_gradient_parity("GRADIENT_CE_PARITY_FIXTURE", "CE-only")


def test_bagel_gradient_text_image_ce_module_backward_matches_official_fixture() -> None:
    _assert_module_gradient_parity("GRADIENT_TEXT_IMAGE_CE_PARITY_FIXTURE", "text+image CE")


def test_bagel_gradient_mse_only_module_backward_matches_official_fixture() -> None:
    _assert_module_gradient_parity("GRADIENT_MSE_PARITY_FIXTURE", "MSE-only")


def test_bagel_gradient_ce_mse_module_backward_matches_official_fixture() -> None:
    _assert_module_gradient_parity("GRADIENT_CE_MSE_PARITY_FIXTURE", "CE+MSE")


def test_bagel_gradient_ce_only_graph_backward_matches_official_fixture() -> None:
    _assert_graph_gradient_parity("GRADIENT_CE_PARITY_FIXTURE", "CE-only")


def test_bagel_gradient_text_image_ce_graph_backward_matches_official_fixture() -> None:
    _assert_graph_gradient_parity("GRADIENT_TEXT_IMAGE_CE_PARITY_FIXTURE", "text+image CE")


def test_bagel_gradient_mse_only_graph_backward_matches_official_fixture() -> None:
    _assert_graph_gradient_parity("GRADIENT_MSE_PARITY_FIXTURE", "MSE-only")


def test_bagel_gradient_ce_mse_graph_backward_matches_official_fixture() -> None:
    _assert_graph_gradient_parity("GRADIENT_CE_MSE_PARITY_FIXTURE", "CE+MSE")
