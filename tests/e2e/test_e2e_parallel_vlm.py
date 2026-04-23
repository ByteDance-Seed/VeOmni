"""E2E parallel-alignment tests for vision-language models (VLMs).

Covers the Qwen2-VL and Qwen3-VL families, including the Qwen3.5 text
and MoE variants that share the same VL-style dataset fixture. Add new
VLM entries to the appropriate `*_test_cases` list below.
"""

import pytest

from ._harness import DEFAULT_ATOL, DEFAULT_RTOL, main, v4_only, v5_only


pytestmark = pytest.mark.e2e


qwen2vl_test_cases = [
    pytest.param(
        "qwen2vl",
        "./tests/toy_config/qwen2vl_toy",
        False,
        DEFAULT_RTOL,
        DEFAULT_ATOL,
        marks=v4_only,
    ),
    pytest.param(
        "qwen2vl",
        "./tests/toy_config/qwen2vl_toy",
        False,
        DEFAULT_RTOL,
        DEFAULT_ATOL,
        marks=v5_only,
    ),
    pytest.param(
        "qwen25vl",
        "./tests/toy_config/qwen25vl_toy",
        False,
        DEFAULT_RTOL,
        DEFAULT_ATOL,
        marks=v4_only,
    ),
    pytest.param(
        "qwen25vl",
        "./tests/toy_config/qwen25vl_toy",
        False,
        DEFAULT_RTOL,
        DEFAULT_ATOL,
        marks=v5_only,
    ),
]


qwen3vl_test_cases = [
    pytest.param(
        "qwen3vl",
        "./tests/toy_config/qwen3vl_toy",
        False,
        DEFAULT_RTOL,
        DEFAULT_ATOL,
        None,  # max_sp_size
        marks=v4_only,
    ),
    pytest.param(
        "qwen3vl",
        "./tests/toy_config/qwen3vl_toy",
        False,
        DEFAULT_RTOL,
        DEFAULT_ATOL,
        None,  # max_sp_size
        marks=v5_only,
    ),
    pytest.param(
        "qwen3vlmoe",
        "./tests/toy_config/qwen3vlmoe_toy",
        True,
        DEFAULT_RTOL,
        DEFAULT_ATOL,
        None,  # max_sp_size
        marks=v4_only,
    ),
    pytest.param(
        "qwen3vlmoe",
        "./tests/toy_config/qwen3vlmoe_toy",
        True,
        DEFAULT_RTOL,
        DEFAULT_ATOL,
        None,  # max_sp_size
        marks=v5_only,
    ),
    pytest.param(
        "qwen3_5_moe",
        "./tests/toy_config/qwen3_5_moe_toy/config.json",
        True,  # is_moe
        DEFAULT_RTOL,
        DEFAULT_ATOL,
        None,  # max_sp_size
        marks=v5_only,
    ),
    pytest.param(
        "qwen3_5",
        "./tests/toy_config/qwen3_5_toy/config.json",
        False,  # is_moe
        DEFAULT_RTOL,
        DEFAULT_ATOL,
        None,  # max_sp_size
        marks=v5_only,
    ),
]


@pytest.mark.parametrize("model_name, config_path, is_moe, rtol, atol", qwen2vl_test_cases)
def test_qwen2vl_parallel_align(
    model_name: str, config_path: str, is_moe: bool, rtol: float, atol: float, dummy_qwen2vl_dataset
):
    main(
        task_name="train_vlm_test",
        model_name=model_name,
        config_path=config_path,
        is_moe=is_moe,
        rtol=rtol,
        atol=atol,
        train_path=dummy_qwen2vl_dataset,
    )


@pytest.mark.parametrize("model_name, config_path, is_moe, rtol, atol, max_sp_size", qwen3vl_test_cases)
def test_qwen3vl_parallel_align(
    model_name: str,
    config_path: str,
    is_moe: bool,
    rtol: float,
    atol: float,
    max_sp_size: int | None,
    dummy_qwen3vl_dataset,
):
    main(
        task_name="train_vlm_test",
        model_name=model_name,
        config_path=config_path,
        is_moe=is_moe,
        rtol=rtol,
        atol=atol,
        max_sp_size=max_sp_size,
        train_path=dummy_qwen3vl_dataset,
    )
