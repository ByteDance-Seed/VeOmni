"""Single-GPU vs FSDP2 equivalence tests.

Validates that training with FSDP2 produces the same gradient norms
as single-GPU training (no FSDP wrapping). This catches FSDP-induced numerical
differences such as:
- Incorrect gradient reduction
- Mixed-precision casting issues in FSDP communication
- Parameter sharding/gathering affecting computation order

Approach:
  Both runs use the SAME VeOmni trainer (train_text_test.py) to ensure identical
  data pipeline, loss normalization (per-token via mean_global_loss), and batch
  invariant mode. The only difference is FSDP wrapping:
  - Baseline: nproc=1, init_device=device (no FSDP, full gradient accumulation)
  - FSDP:    nproc=2, init_device=meta   (FSDP2 sharded)

  We compare **grad_norm** as the primary correctness signal. With per-token loss
  normalization and the fsdp_size multiplier, the effective gradient is mathematically
  identical between the two configurations. The logged loss values differ because
  rank 0 only accumulates its own micro-batches (with fsdp_size scaling), whereas
  the single-GPU run accumulates all micro-batches (with fsdp_size=1).

Requires: 2+ GPUs.
"""

import os
import shutil

import pytest

from veomni.utils.device import get_device_type
from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to

from ..tools import ParallelConfig


_is_transformers_v5 = is_transformers_version_greater_or_equal_to("5.0.0")
_v4_only = pytest.mark.skipif(_is_transformers_v5, reason="Not compatible with transformers >= 5.0.0")
_v5_only = pytest.mark.skipif(not _is_transformers_v5, reason="Requires transformers >= 5.0.0")

_DEFAULT_RTOL = 1e-1
_DEFAULT_ATOL = 1e-1

_TEXT_TRAIN_SCRIPT = "tests/train_scripts/train_text_test.py"


def _setup_model_and_data(model_name, config_path, dataset_type="text"):
    """Materialize model weights and create dummy dataset."""
    from ..tools import DummyDataset, materialize_weights

    test_dir = f"./_test_fsdp_equiv_{model_name}"
    os.makedirs(test_dir, exist_ok=True)

    save_original_format = model_name != "qwen3_5_moe"
    materialize_weights(config_path, test_dir, save_original_format=save_original_format)

    dummy_dataset = DummyDataset(seq_len=2048, dataset_type=dataset_type)
    train_path = dummy_dataset.save_path

    return test_dir, train_path, dummy_dataset


def _run_single_gpu_training(config_path, model_path, train_path, output_dir):
    """Run plain single-GPU training (nproc=1, no parallelism).

    Uses the same trainer script as the FSDP run, but with nproc=1 and no
    parallel config. This mimics a HuggingFace-style single-GPU training
    baseline with no FSDP wrapping, no sequence parallelism, and no expert
    parallelism.
    """
    from ..tools import run_training_config

    return run_training_config(
        script=_TEXT_TRAIN_SCRIPT,
        config_path=config_path,
        model_path=model_path,
        train_path=train_path,
        output_dir=output_dir,
        task_name="single_gpu",
        nproc=1,
        init_device=get_device_type(),
        extra_args=["--train.accelerator.fsdp_config.mixed_precision.enable=False"],
    )


def _get_nproc():
    """Return the number of available GPUs/NPUs, requiring at least 2."""
    from veomni.utils.device import get_torch_device

    torch_device = get_torch_device()
    count = torch_device.device_count() if torch_device.is_available() else 0
    if count < 2:
        pytest.skip(f"Requires at least 2 devices, found {count}")
    return count


def _run_fsdp2_training(config_path, model_path, train_path, output_dir, nproc=None):
    """Run FSDP2 distributed training and return metrics."""
    from ..tools import run_training_config

    if nproc is None:
        nproc = _get_nproc()

    config = ParallelConfig(sp_size=1, ep_size=1, fsdp_mode="fsdp2")
    return run_training_config(
        script=_TEXT_TRAIN_SCRIPT,
        config_path=config_path,
        model_path=model_path,
        train_path=train_path,
        output_dir=output_dir,
        parallel_config=config,
        task_name="fsdp2",
        nproc=nproc,
        extra_args=[
            "--train.accelerator.ulysses_size=1",
            "--train.accelerator.ep_size=1",
            "--train.accelerator.fsdp_config.mixed_precision.enable=False",
        ],
    )


def _run_fsdp_equivalence(
    model_name: str,
    config_path: str,
    is_moe: bool,
    rtol: float,
    atol: float,
    dataset_type: str = "text",
):
    """Run single-GPU vs FSDP2 comparison for a model.

    Compares grad_norm across both runs. Loss is printed for visual inspection
    but NOT asserted, because the logged loss on rank 0 of the FSDP run only
    covers that rank's micro-batches (with fsdp_size scaling), while the
    single-GPU run covers all micro-batches (with fsdp_size=1).
    """
    from ..tools import compare_metrics, print_comparison_table

    test_dir, train_path, dummy_dataset = _setup_model_and_data(model_name, config_path, dataset_type)

    try:
        # 1. Run single-GPU baseline (nproc=1, no FSDP)
        baseline_results = _run_single_gpu_training(
            config_path=config_path,
            model_path=test_dir,
            train_path=train_path,
            output_dir=test_dir,
        )

        # 2. Run FSDP2 with all available GPUs
        fsdp2_results = _run_fsdp2_training(
            config_path=config_path,
            model_path=test_dir,
            train_path=train_path,
            output_dir=test_dir,
        )

        # 3. Compare grad_norm (primary correctness signal)
        results = {
            f"{model_name}_single_gpu": baseline_results,
            f"{model_name}_fsdp2_2gpu": fsdp2_results,
        }

        # Print all metrics for visual inspection
        for key in baseline_results:
            if key in fsdp2_results:
                print_comparison_table(results, key, title=f"{model_name} FSDP Equivalence")

        # Assert only grad_norm matches (see module docstring for why loss differs)
        compare_metrics(results, rtol=rtol, atol=atol, keys=["grad_norm"])

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
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen3_5_moe",
        marks=_v5_only,
    ),
]

_text_test_cases = _text_test_cases_v4 + _text_test_cases_v5


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
    )
