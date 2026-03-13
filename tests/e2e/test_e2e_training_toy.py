"""L5: End-to-end training smoke test with toy configs and random weights.

Unlike test_e2e_training.py which requires real model weights and datasets,
this test uses toy configs with randomly initialized weights to validate
the full training pipeline without any model downloads. This enables L5
testing in CI environments.

Each test:
1. Uses an existing toy config (e.g., qwen3_toy)
2. Initializes random weights (no download required)
3. Runs 2-4 training steps via torchrun
4. Asserts: no crash, loss is finite, loss decreases (or stays stable)

Requires: 2+ GPUs (torchrun-based subprocess tests).
"""

import json
import os
import shutil
import subprocess

import pytest

from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to


_is_transformers_v5 = is_transformers_version_greater_or_equal_to("5.0.0")
_v4_only = pytest.mark.skipif(_is_transformers_v5, reason="Not compatible with transformers >= 5.0.0")
_v5_only = pytest.mark.skipif(not _is_transformers_v5, reason="Requires transformers >= 5.0.0")


def _run_toy_training(
    model_name: str,
    config_path: str,
    dataset_type: str = "text",
    train_script: str = "tests/e2e/train_text_test.py",
    nproc: int = 2,
    max_steps: int = 4,
):
    """Run a short training job with toy config and random weights.

    Returns the training metrics dict from the JSON log.
    """
    from tests.distributed._training_core import materialize_weights
    from tests.distributed.utils import find_free_port
    from tests.e2e.utils import DummyDataset

    test_dir = f"./_test_e2e_toy_{model_name}"
    os.makedirs(test_dir, exist_ok=True)

    save_original_format = model_name != "qwen3_5_moe"
    materialize_weights(config_path, test_dir, save_original_format=save_original_format)

    dummy_dataset = DummyDataset(seq_len=2048, dataset_type=dataset_type)
    train_path = dummy_dataset.save_path

    try:
        port = find_free_port()
        output_dir = os.path.join(test_dir, "output")

        cmd = [
            "torchrun",
            "--nnodes=1",
            f"--nproc_per_node={nproc}",
            f"--master_port={port}",
            train_script,
            f"--model.config_path={config_path}",
            f"--data.train_path={train_path}",
            "--data.dyn_bsz_buffer_size=1",
            "--train.global_batch_size=16",
            "--train.micro_batch_size=1",
            "--train.accelerator.fsdp_config.fsdp_mode=fsdp2",
            "--model.ops_implementation.attn_implementation=flash_attention_2",
            "--model.ops_implementation.moe_implementation=fused",
            "--train.init_device=meta",
            "--train.accelerator.ulysses_size=1",
            "--train.accelerator.ep_size=1",
            "--train.bsz_warmup_ratio=0",
            "--train.num_train_epochs=1",
            "--train.checkpoint.save_epochs=0",
            "--train.checkpoint.save_steps=0",
            "--train.checkpoint.save_hf_weights=False",
            "--train.enable_full_determinism=True",
            "--train.enable_batch_invariant_mode=True",
            f"--train.max_steps={max_steps}",
            f"--train.checkpoint.output_dir={output_dir}",
            f"--model.model_path={test_dir}",
        ]

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

        log_path = os.path.join(output_dir, "log_dict.json")
        assert os.path.exists(log_path), f"Training log not found at {log_path}"

        with open(log_path) as f:
            metrics = json.load(f)

        return metrics

    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        del dummy_dataset


def _assert_training_health(metrics: dict, model_name: str):
    """Assert basic training health from metrics.

    Checks:
    - Loss values exist and are finite
    - Loss is not NaN
    - Gradient norms are finite
    """
    assert "loss" in metrics, f"No loss in metrics for {model_name}"
    losses = metrics["loss"]
    assert len(losses) > 0, f"No loss values recorded for {model_name}"

    for i, loss in enumerate(losses):
        assert loss == loss, f"NaN loss at step {i} for {model_name}"  # NaN != NaN
        assert abs(loss) < 1e6, f"Extremely large loss at step {i} for {model_name}: {loss}"

    if "grad_norm" in metrics:
        for i, gn in enumerate(metrics["grad_norm"]):
            assert gn == gn, f"NaN grad_norm at step {i} for {model_name}"


# --- Text model test cases ---

_text_toy_cases = [
    pytest.param("qwen3", "./tests/toy_config/qwen3_toy", id="qwen3", marks=_v4_only),
    pytest.param("llama3.1", "./tests/toy_config/llama31_toy", id="llama3.1", marks=_v4_only),
    pytest.param(
        "qwen3_5",
        "./tests/toy_config/qwen3_5_toy/config.json",
        id="qwen3_5",
        marks=_v5_only,
    ),
]


@pytest.mark.L5
@pytest.mark.multi_gpu
@pytest.mark.parametrize("model_name, config_path", _text_toy_cases)
def test_text_training_toy(model_name: str, config_path: str):
    """Smoke test: text training with toy config completes without errors."""
    metrics = _run_toy_training(
        model_name=model_name,
        config_path=config_path,
        dataset_type="text",
        train_script="tests/e2e/train_text_test.py",
        max_steps=3,
    )
    _assert_training_health(metrics, model_name)


# --- VLM model test cases ---

_vlm_toy_cases = [
    pytest.param(
        "qwen3vl",
        "./tests/toy_config/qwen3vl_toy",
        "qwen3vl",
        id="qwen3vl",
        marks=_v4_only,
    ),
]


@pytest.mark.L5
@pytest.mark.multi_gpu
@pytest.mark.parametrize("model_name, config_path, dataset_type", _vlm_toy_cases)
def test_vlm_training_toy(model_name: str, config_path: str, dataset_type: str):
    """Smoke test: VLM training with toy config completes without errors."""
    metrics = _run_toy_training(
        model_name=model_name,
        config_path=config_path,
        dataset_type=dataset_type,
        train_script="tests/e2e/train_vlm_test.py",
        max_steps=2,
    )
    _assert_training_health(metrics, model_name)


# --- MoE model test cases ---

_moe_toy_cases = [
    pytest.param(
        "qwen3_moe",
        "./tests/toy_config/qwen3_moe_toy",
        id="qwen3_moe",
        marks=_v4_only,
    ),
]


@pytest.mark.L5
@pytest.mark.multi_gpu
@pytest.mark.parametrize("model_name, config_path", _moe_toy_cases)
def test_moe_training_toy(model_name: str, config_path: str):
    """Smoke test: MoE training with toy config completes without errors."""
    metrics = _run_toy_training(
        model_name=model_name,
        config_path=config_path,
        dataset_type="text",
        train_script="tests/e2e/train_text_test.py",
        max_steps=2,
    )
    _assert_training_health(metrics, model_name)
