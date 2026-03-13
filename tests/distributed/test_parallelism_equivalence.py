"""L3: Parallelism combination equivalence tests.

Validates that different parallelism combinations (FSDP, FSDP+SP, FSDP+EP,
FSDP+SP+EP) produce consistent training metrics. This catches bugs in:
- Sequence parallelism (Ulysses) gradient aggregation
- Expert parallelism routing/communication
- Interactions between SP and EP when combined

Each test:
1. Materializes a toy model with random weights
2. Runs training with pure FSDP2 (baseline)
3. Runs training with FSDP2 + SP, FSDP2 + EP, FSDP2 + SP + EP
4. Compares per-step loss and grad_norm across all configs

Requires: 4+ GPUs (torchrun-based subprocess tests).
"""

import os
import shutil

import pytest

from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to

from .utils import ParallelConfig


_is_transformers_v5 = is_transformers_version_greater_or_equal_to("5.0.0")
_v4_only = pytest.mark.skipif(_is_transformers_v5, reason="Not compatible with transformers >= 5.0.0")
_v5_only = pytest.mark.skipif(not _is_transformers_v5, reason="Requires transformers >= 5.0.0")

_DEFAULT_RTOL = 1e-1
_DEFAULT_ATOL = 1e-1

_TEXT_TRAIN_SCRIPT = "tests/e2e/train_text_test.py"
_VLM_TRAIN_SCRIPT = "tests/e2e/train_vlm_test.py"


def _run_parallelism_equivalence(
    model_name: str,
    config_path: str,
    is_moe: bool,
    rtol: float,
    atol: float,
    max_sp_size: int | None = None,
    dataset_type: str = "text",
    train_script: str = _TEXT_TRAIN_SCRIPT,
):
    """Run parallelism combination comparison for a model.

    Compares pure FSDP2 baseline against SP/EP/SP+EP variants.
    """
    from tests.e2e.utils import DummyDataset

    from ._training_core import materialize_weights, run_and_compare

    test_dir = f"./_test_par_equiv_{model_name}"
    os.makedirs(test_dir, exist_ok=True)

    save_original_format = model_name != "qwen3_5_moe"
    materialize_weights(config_path, test_dir, save_original_format=save_original_format)

    dummy_dataset = DummyDataset(seq_len=2048, dataset_type=dataset_type)
    train_path = dummy_dataset.save_path

    try:
        # Build parallelism configs based on model type
        configs = _build_parallel_configs(is_moe, max_sp_size)

        run_and_compare(
            script=train_script,
            config_path=config_path,
            model_path=test_dir,
            train_path=train_path,
            output_dir=test_dir,
            configs=configs,
            model_name=model_name,
            rtol=rtol,
            atol=atol,
        )
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        del dummy_dataset


def _build_parallel_configs(is_moe: bool, max_sp_size: int | None = None) -> list[ParallelConfig]:
    """Build the set of parallelism configs to compare.

    For dense models: FSDP (sp=1) vs FSDP+SP (sp=2)
    For MoE models: FSDP (sp=1, ep=1) vs FSDP+SP vs FSDP+EP vs FSDP+SP+EP
    """
    sp_sizes = [1, 2]
    if max_sp_size is not None:
        sp_sizes = [s for s in sp_sizes if s <= max_sp_size]

    ep_sizes = [1, 2] if is_moe else [1]

    configs = []
    for sp in sp_sizes:
        for ep in ep_sizes:
            configs.append(ParallelConfig(sp_size=sp, ep_size=ep))

    return configs


# --- Text model test cases ---

text_test_cases = [
    pytest.param(
        "llama3.1",
        "./tests/toy_config/llama31_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        None,  # max_sp_size
        marks=_v4_only,
    ),
    pytest.param(
        "qwen2.5",
        "./tests/toy_config/qwen25_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        None,
        marks=_v4_only,
    ),
    pytest.param(
        "qwen3",
        "./tests/toy_config/qwen3_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        None,
        marks=_v4_only,
    ),
    pytest.param(
        "qwen3_moe",
        "./tests/toy_config/qwen3_moe_toy",
        True,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        None,
        marks=_v4_only,
    ),
    pytest.param(
        "seed_oss",
        "./tests/toy_config/seed_oss_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        None,
        marks=_v4_only,
    ),
    pytest.param(
        "deepseek_v3",
        "./tests/toy_config/deepseek_v3_toy",
        True,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        None,
        marks=_v4_only,
    ),
    pytest.param(
        "qwen3_5",
        "./tests/toy_config/qwen3_5_toy/config.json",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        None,
        marks=_v5_only,
    ),
    pytest.param(
        "qwen3_5_moe",
        "./tests/toy_config/qwen3_5_moe_toy/config.json",
        False,  # EP not yet supported for stacked weights
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        None,
        marks=_v5_only,
    ),
]


@pytest.mark.L3
@pytest.mark.multi_gpu
@pytest.mark.parametrize("model_name, config_path, is_moe, rtol, atol, max_sp_size", text_test_cases)
def test_text_parallelism_equivalence(
    model_name: str,
    config_path: str,
    is_moe: bool,
    rtol: float,
    atol: float,
    max_sp_size: int | None,
):
    """Verify FSDP vs FSDP+SP vs FSDP+EP produce equivalent metrics for text models."""
    _run_parallelism_equivalence(
        model_name=model_name,
        config_path=config_path,
        is_moe=is_moe,
        rtol=rtol,
        atol=atol,
        max_sp_size=max_sp_size,
        dataset_type="text",
        train_script=_TEXT_TRAIN_SCRIPT,
    )


# --- VLM model test cases ---

qwen2vl_test_cases = [
    pytest.param(
        "qwen2vl",
        "./tests/toy_config/qwen2vl_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        marks=_v4_only,
    ),
    pytest.param(
        "qwen25vl",
        "./tests/toy_config/qwen25vl_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        marks=_v4_only,
    ),
]

qwen3vl_test_cases = [
    pytest.param(
        "qwen3vl",
        "./tests/toy_config/qwen3vl_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        marks=_v4_only,
    ),
    pytest.param(
        "qwen3vlmoe",
        "./tests/toy_config/qwen3vlmoe_toy",
        True,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        marks=_v4_only,
    ),
]

qwen2omni_test_cases = [
    pytest.param(
        "qwen25_omni",
        "./tests/toy_config/qwen25omni_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        marks=_v4_only,
    ),
]

qwen3omni_test_cases = [
    pytest.param(
        "qwen3_omni_moe",
        "./tests/toy_config/qwen3omni_toy",
        True,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        marks=_v4_only,
    ),
]


@pytest.mark.L3
@pytest.mark.multi_gpu
@pytest.mark.parametrize("model_name, config_path, is_moe, rtol, atol", qwen2vl_test_cases)
def test_qwen2vl_parallelism_equivalence(
    model_name: str,
    config_path: str,
    is_moe: bool,
    rtol: float,
    atol: float,
):
    """Verify parallelism equivalence for Qwen2-VL models."""
    _run_parallelism_equivalence(
        model_name=model_name,
        config_path=config_path,
        is_moe=is_moe,
        rtol=rtol,
        atol=atol,
        dataset_type="qwen2vl",
        train_script=_VLM_TRAIN_SCRIPT,
    )


@pytest.mark.L3
@pytest.mark.multi_gpu
@pytest.mark.parametrize("model_name, config_path, is_moe, rtol, atol", qwen3vl_test_cases)
def test_qwen3vl_parallelism_equivalence(
    model_name: str,
    config_path: str,
    is_moe: bool,
    rtol: float,
    atol: float,
):
    """Verify parallelism equivalence for Qwen3-VL models."""
    _run_parallelism_equivalence(
        model_name=model_name,
        config_path=config_path,
        is_moe=is_moe,
        rtol=rtol,
        atol=atol,
        dataset_type="qwen3vl",
        train_script=_VLM_TRAIN_SCRIPT,
    )


@pytest.mark.L3
@pytest.mark.multi_gpu
@pytest.mark.parametrize("model_name, config_path, is_moe, rtol, atol", qwen2omni_test_cases)
def test_qwen2omni_parallelism_equivalence(
    model_name: str,
    config_path: str,
    is_moe: bool,
    rtol: float,
    atol: float,
):
    """Verify parallelism equivalence for Qwen2-Omni models."""
    _run_parallelism_equivalence(
        model_name=model_name,
        config_path=config_path,
        is_moe=is_moe,
        rtol=rtol,
        atol=atol,
        dataset_type="qwen2omni",
        train_script=_VLM_TRAIN_SCRIPT,
    )


@pytest.mark.L3
@pytest.mark.multi_gpu
@pytest.mark.parametrize("model_name, config_path, is_moe, rtol, atol", qwen3omni_test_cases)
def test_qwen3omni_parallelism_equivalence(
    model_name: str,
    config_path: str,
    is_moe: bool,
    rtol: float,
    atol: float,
):
    """Verify parallelism equivalence for Qwen3-Omni models."""
    _run_parallelism_equivalence(
        model_name=model_name,
        config_path=config_path,
        is_moe=is_moe,
        rtol=rtol,
        atol=atol,
        dataset_type="qwen3omni",
        train_script=_VLM_TRAIN_SCRIPT,
    )
