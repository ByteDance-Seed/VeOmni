"""E2E test for FSDP2 equivalence across parallelism configurations.

Validates that FSDP2 with different Ulysses SP sizes produces numerically
equivalent training results (loss, grad_norm) using the TestTextTrainer
infrastructure and DummyDataset.

Usage:
    pytest tests/e2e/test_e2e_fsdp2.py -v -s
"""

import json
import os
import random
import subprocess

import pytest

from tests.e2e.utils import DummyDataset, compare_multi_items, print_all_values
from veomni.models.auto import build_foundation_model
from veomni.utils.device import get_device_type, get_torch_device

NPROC_PER_NODE = 8
_DEFAULT_RTOL = 1e-2
_DEFAULT_ATOL = 1e-2


def _materialize_weights_dir(config_path: str, output_path: str) -> None:
    """Materialize random model weights to disk for reproducible initialization."""
    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype="float32",
        attn_implementation="eager",
        init_device=get_device_type(),
    )
    model.save_pretrained(output_path)


def _build_fsdp2_commands(
    config_path: str,
    model_path: str,
    train_path: str,
    output_dir: str,
    sp_sizes: list[int],
) -> list[tuple[str, list[str]]]:
    """Build torchrun commands for each FSDP2 + Ulysses SP configuration.

    Args:
        config_path: Path to model config directory.
        model_path: Path to materialized model weights.
        train_path: Path to training data directory.
        output_dir: Base directory for test outputs.
        sp_sizes: List of Ulysses SP sizes to test.

    Returns:
        List of (task_name, command) tuples.
    """
    command_list = []
    for sp_size in sp_sizes:
        port = 12345 + random.randint(0, 100)
        task_name = f"fsdp2_sp{sp_size}"
        command = [
            "torchrun",
            "--nnodes=1",
            f"--nproc_per_node={NPROC_PER_NODE}",
            f"--master_port={port}",
            "tests/e2e/train_text_test.py",
            f"--model.config_path={config_path}",
            f"--model.model_path={model_path}",
            f"--data.train_path={train_path}",
            "--data.dyn_bsz_buffer_size=1",
            "--train.data_parallel_mode=fsdp2",
            "--model.attn_implementation=flash_attention_2",
            "--train.init_device=meta",
            "--train.global_batch_size=16",
            "--train.micro_batch_size=1",
            f"--train.ulysses_parallel_size={sp_size}",
            "--train.bsz_warmup_ratio=0",
            "--train.num_train_epochs=1",
            "--train.max_steps=5",
            "--train.enable_full_determinism=true",
            "--train.save_epochs=0",
            "--train.save_steps=0",
            "--train.save_hf_weights=False",
            "--train.use_wandb=false",
            f"--train.output_dir={os.path.join(output_dir, task_name)}",
        ]
        command_list.append((task_name, command))
    return command_list


fsdp2_test_cases = [
    pytest.param(
        "qwen3",
        "./tests/toy_config/qwen3_toy",
        [1, 4],
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
    ),
]


@pytest.fixture(scope="session")
def dummy_text_dataset():
    """Create a shared dummy text dataset for all FSDP2 tests."""
    dummy_dataset = DummyDataset(dataset_type="text")
    train_path = dummy_dataset.save_path
    yield train_path


@pytest.mark.parametrize("model_name, config_path, sp_sizes, rtol, atol", fsdp2_test_cases)
def test_fsdp2_equivalence(
    model_name: str,
    config_path: str,
    sp_sizes: list[int],
    rtol: float,
    atol: float,
    dummy_text_dataset: str,
):
    """Test that FSDP2 with different Ulysses SP sizes produces equivalent results."""
    gpu_count = get_torch_device().device_count()
    if gpu_count < NPROC_PER_NODE:
        pytest.skip(f"Requires {NPROC_PER_NODE} GPUs, found {gpu_count}")

    test_path = f"./{model_name}_fsdp2_test"
    os.makedirs(test_path, exist_ok=True)

    _materialize_weights_dir(config_path, test_path)

    command_list = _build_fsdp2_commands(
        config_path=config_path,
        model_path=test_path,
        train_path=dummy_text_dataset,
        output_dir=test_path,
        sp_sizes=sp_sizes,
    )

    res = {}
    log_keys = []
    for task_name, cmd in command_list:
        print(f"\n{'-' * 10} {task_name} {'-' * 10}")
        subprocess.run(cmd, check=True)
        log_path = os.path.join(test_path, task_name, "log_dict.json")
        with open(log_path) as f:
            output = json.load(f)
        if not log_keys:
            log_keys = set(output.keys())
        else:
            assert log_keys == set(output.keys()), f"Key mismatch: {log_keys} vs {set(output.keys())}"
        res[task_name] = output

    for key in log_keys:
        print_all_values(res, key, model_type=model_name)
    compare_multi_items(model_name, res, rtol=rtol, atol=atol)
