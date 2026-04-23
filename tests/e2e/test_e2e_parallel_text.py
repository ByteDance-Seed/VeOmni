"""E2E parallel-alignment tests for pure-text LLMs.

All cases here exercise the FSDP2 + SP + EP axes for decoder-only text
models. Add a new entry to :data:`text_test_cases` when onboarding a
text-only model. Multimodal models live in the sibling
`test_e2e_parallel_vlm.py` / `test_e2e_parallel_omni.py` files.
"""

import pytest

from ._harness import DEFAULT_ATOL, DEFAULT_RTOL, main, v4_only, v5_only


pytestmark = pytest.mark.e2e


text_test_cases = [
    pytest.param(
        "llama3.1",
        "./tests/fixtures/toy_config/llama31_toy",
        False,  # is_moe
        DEFAULT_RTOL,
        DEFAULT_ATOL,
        None,  # max_sp_size
        marks=v4_only,
    ),
    pytest.param(
        "qwen2",
        "./tests/fixtures/toy_config/qwen2_toy/config.json",
        False,  # is_moe
        DEFAULT_RTOL,
        DEFAULT_ATOL,
        None,  # max_sp_size
        marks=v5_only,
    ),
    pytest.param(
        "qwen2.5",
        "./tests/fixtures/toy_config/qwen25_toy",
        False,  # is_moe
        DEFAULT_RTOL,
        DEFAULT_ATOL,
        None,  # max_sp_size
        marks=v4_only,
    ),
    pytest.param(
        "qwen3",
        "./tests/fixtures/toy_config/qwen3_toy",
        False,  # is_moe
        DEFAULT_RTOL,
        DEFAULT_ATOL,
        None,  # max_sp_size
        marks=v4_only,
    ),
    pytest.param(
        "qwen3_moe",
        "./tests/fixtures/toy_config/qwen3_moe_toy",
        True,  # is_moe
        DEFAULT_RTOL,
        DEFAULT_ATOL,
        None,  # max_sp_size
        marks=v4_only,
    ),
    pytest.param(
        "qwen3_moe",
        "./tests/fixtures/toy_config/qwen3_moe_toy",
        True,  # is_moe
        DEFAULT_RTOL,
        DEFAULT_ATOL,
        None,  # max_sp_size
        marks=v5_only,
        id="qwen3_moe_v5",
    ),
    pytest.param(
        "seed_oss",
        "./tests/fixtures/toy_config/seed_oss_toy",
        False,  # is_moe
        DEFAULT_RTOL,
        DEFAULT_ATOL,
        None,  # max_sp_size
        marks=v4_only,
    ),
    pytest.param(
        "deepseek_v3",
        "./tests/fixtures/toy_config/deepseek_v3_toy",
        True,  # is_moe
        DEFAULT_RTOL,
        DEFAULT_ATOL,
        None,  # max_sp_size
        marks=v4_only,
    ),
]


@pytest.mark.parametrize("model_name, config_path, is_moe, rtol, atol, max_sp_size", text_test_cases)
def test_text_parallel_align(
    model_name: str,
    config_path: str,
    is_moe: bool,
    rtol: float,
    atol: float,
    max_sp_size: int | None,
    dummy_text_dataset,
):
    main(
        task_name="train_text_test",
        model_name=model_name,
        config_path=config_path,
        is_moe=is_moe,
        rtol=rtol,
        atol=atol,
        train_path=dummy_text_dataset,
        max_sp_size=max_sp_size,
    )
