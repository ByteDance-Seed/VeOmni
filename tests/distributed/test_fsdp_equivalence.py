"""L2: Single-GPU vs FSDP equivalence tests.

Validates that training with FSDP2 produces the same loss and gradient norms
as single-GPU training (no FSDP wrapping). This catches FSDP-induced numerical
differences such as:
- Incorrect gradient reduction
- Mixed-precision casting issues in FSDP communication
- Parameter sharding/gathering affecting computation order

Each test:
1. Materializes a toy model with random weights
2. Runs single-GPU training (1 process, no FSDP)
3. Runs multi-GPU training with FSDP2
4. Compares per-step loss and grad_norm within tolerance

Requires: 2+ GPUs (torchrun-based subprocess tests).
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

# Test script paths (relative to VeOmni repo root)
_TEXT_TRAIN_SCRIPT = "tests/e2e/train_text_test.py"
_VLM_TRAIN_SCRIPT = "tests/e2e/train_vlm_test.py"


def _setup_model_and_data(model_name, config_path, dataset_type="text"):
    """Materialize model weights and create dummy dataset."""
    from tests.e2e.utils import DummyDataset

    from ._training_core import materialize_weights

    test_dir = f"./_test_fsdp_equiv_{model_name}"
    os.makedirs(test_dir, exist_ok=True)

    # Qwen3_5Moe uses stacked 3D expert params; disable save_original_format
    save_original_format = model_name != "qwen3_5_moe"
    materialize_weights(config_path, test_dir, save_original_format=save_original_format)

    dummy_dataset = DummyDataset(seq_len=2048, dataset_type=dataset_type)
    train_path = dummy_dataset.save_path

    return test_dir, train_path, dummy_dataset


def _run_fsdp_equivalence(
    model_name: str,
    config_path: str,
    is_moe: bool,
    rtol: float,
    atol: float,
    dataset_type: str = "text",
    train_script: str = _TEXT_TRAIN_SCRIPT,
):
    """Run single-GPU vs FSDP2 comparison for a model."""

    test_dir, train_path, dummy_dataset = _setup_model_and_data(model_name, config_path, dataset_type)

    try:
        # Single-GPU baseline: sp=1, ep=1, nproc=1
        single_gpu = ParallelConfig(sp_size=1, ep_size=1, fsdp_mode="fsdp2")
        # FSDP2 with 2 GPUs: sp=1, ep=1
        fsdp2_config = ParallelConfig(sp_size=1, ep_size=1, fsdp_mode="fsdp2")

        configs_with_nproc = [
            (single_gpu, 1, f"{model_name}_single_gpu"),
            (fsdp2_config, 2, f"{model_name}_fsdp2_2gpu"),
        ]

        # Run each config separately with different nproc
        results = {}
        from ._training_core import run_training_config

        for config, nproc, task_name in configs_with_nproc:
            output = run_training_config(
                script=train_script,
                config_path=config_path,
                model_path=test_dir,
                train_path=train_path,
                output_dir=test_dir,
                parallel_config=config,
                task_name=task_name,
                nproc=nproc,
                extra_args=["--train.accelerator.ulysses_size=1", "--train.accelerator.ep_size=1"],
            )
            results[task_name] = output

        # Compare
        from .utils import compare_metrics, print_comparison_table

        for key in list(results[next(iter(results))].keys()):
            print_comparison_table(results, key, title=f"{model_name} FSDP Equivalence")

        compare_metrics(results, rtol=rtol, atol=atol)

    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        del dummy_dataset


# --- Text model test cases ---

_text_test_cases_v4 = [
    pytest.param(
        "qwen3",
        "./tests/toy_config/qwen3_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen3",
        marks=_v4_only,
    ),
    pytest.param(
        "qwen3_moe",
        "./tests/toy_config/qwen3_moe_toy",
        True,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen3_moe",
        marks=_v4_only,
    ),
    pytest.param(
        "llama3.1",
        "./tests/toy_config/llama31_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="llama3.1",
        marks=_v4_only,
    ),
]

_text_test_cases_v5 = [
    pytest.param(
        "qwen3_5",
        "./tests/toy_config/qwen3_5_toy/config.json",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen3_5",
        marks=_v5_only,
    ),
    pytest.param(
        "qwen3_5_moe",
        "./tests/toy_config/qwen3_5_moe_toy/config.json",
        False,  # EP not yet supported for stacked weights
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen3_5_moe",
        marks=_v5_only,
    ),
]

_text_test_cases = _text_test_cases_v4 + _text_test_cases_v5


@pytest.mark.L2
@pytest.mark.multi_gpu
@pytest.mark.parametrize("model_name, config_path, is_moe, rtol, atol", _text_test_cases)
def test_text_fsdp_equivalence(
    model_name: str,
    config_path: str,
    is_moe: bool,
    rtol: float,
    atol: float,
):
    """Verify single-GPU vs FSDP2 produce equivalent loss/grad_norm for text models."""
    _run_fsdp_equivalence(
        model_name=model_name,
        config_path=config_path,
        is_moe=is_moe,
        rtol=rtol,
        atol=atol,
        dataset_type="text",
        train_script=_TEXT_TRAIN_SCRIPT,
    )


# --- VLM model test cases ---

_vlm_test_cases = [
    pytest.param(
        "qwen3vl",
        "./tests/toy_config/qwen3vl_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        "qwen3vl",
        id="qwen3vl",
        marks=_v4_only,
    ),
]


@pytest.mark.L2
@pytest.mark.multi_gpu
@pytest.mark.parametrize("model_name, config_path, is_moe, rtol, atol, dataset_type", _vlm_test_cases)
def test_vlm_fsdp_equivalence(
    model_name: str,
    config_path: str,
    is_moe: bool,
    rtol: float,
    atol: float,
    dataset_type: str,
):
    """Verify single-GPU vs FSDP2 produce equivalent loss/grad_norm for VLM models."""
    _run_fsdp_equivalence(
        model_name=model_name,
        config_path=config_path,
        is_moe=is_moe,
        rtol=rtol,
        atol=atol,
        dataset_type=dataset_type,
        train_script=_VLM_TRAIN_SCRIPT,
    )
