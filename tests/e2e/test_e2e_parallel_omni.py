"""E2E parallel-alignment tests for omni-modal models (text + vision + audio).

Covers the Qwen2.5-Omni and Qwen3-Omni-MoE families. Add new omni
entries to the appropriate `*_test_cases` list below.
"""

import pytest

from ._harness import DEFAULT_ATOL, DEFAULT_RTOL, main, v4_only, v5_only


pytestmark = pytest.mark.e2e


qwen2omni_test_cases = [
    pytest.param(
        "qwen25_omni",
        "./tests/toy_config/qwen25omni_toy",
        False,
        DEFAULT_RTOL,
        DEFAULT_ATOL,
        marks=v4_only,
    ),
]


qwen3omni_test_cases = [
    pytest.param(
        "qwen3_omni_moe",
        "./tests/toy_config/qwen3omni_toy",
        True,
        DEFAULT_RTOL,
        DEFAULT_ATOL,
        marks=v4_only,
    ),
    pytest.param(
        "qwen3_omni_moe",
        "./tests/toy_config/qwen3omni_toy",
        True,
        DEFAULT_RTOL,
        DEFAULT_ATOL,
        marks=v5_only,
    ),
]


@pytest.mark.parametrize("model_name, config_path, is_moe, rtol, atol", qwen2omni_test_cases)
def test_qwen2omni_parallel_align(
    model_name: str, config_path: str, is_moe: bool, rtol: float, atol: float, dummy_qwen2omni_dataset
):
    main(
        task_name="train_vlm_test",
        model_name=model_name,
        config_path=config_path,
        is_moe=is_moe,
        rtol=rtol,
        atol=atol,
        train_path=dummy_qwen2omni_dataset,
    )


@pytest.mark.parametrize("model_name, config_path, is_moe, rtol, atol", qwen3omni_test_cases)
def test_qwen3omni_parallel_align(
    model_name: str, config_path: str, is_moe: bool, rtol: float, atol: float, dummy_qwen3omni_dataset
):
    main(
        task_name="train_vlm_test",
        model_name=model_name,
        config_path=config_path,
        is_moe=is_moe,
        rtol=rtol,
        atol=atol,
        train_path=dummy_qwen3omni_dataset,
    )
