import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.utils import DummyDataset, parse_training_log, prepare_exec_cmd
from tests.models.utils import compare_multi_items
from veomni.models.auto import build_foundation_model
from veomni.utils.device import get_device_type


def _materialize_weights_dir(config_path: str, output_path: str) -> Path:
    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype="float32",
        attn_implementation="eager",
        moe_implementation="eager",
        init_device=get_device_type(),
    )
    model.save_pretrained(output_path)


def _run_and_parse(cmd: str) -> Any:
    res = subprocess.run(cmd, check=True)
    df = parse_training_log(res.stdout)
    return df


_DEFAULT_RTOL = 1e-2
_DEFAULT_ATOL = 1e-2

text_test_cases = [
    pytest.param(
        "llama3.1",
        "./tests/toy_config/llama31_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
    ),
    pytest.param(
        "qwen2.5",
        "./tests/toy_config/qwen25_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
    ),
    pytest.param(
        "qwen3",
        "./tests/toy_config/qwen3_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
    ),
    pytest.param(
        "qwen3_moe",
        "./tests/toy_config/qwen3_moe_toy",
        True,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
    ),
    pytest.param(
        "seed_oss",
        "./tests/toy_config/seed_oss_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
    ),
    pytest.param(
        "deepseek_v3",
        "./tests/toy_config/deepseek_v3_toy",
        True,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
    ),
]


@pytest.fixture(scope="session")
def dummy_text_dataset():
    dummy_dataset = DummyDataset(dataset_type="text")
    train_path = dummy_dataset.save_path
    yield train_path
    # del dummy_dataset


@pytest.mark.parametrize("model_name, config_path, is_moe, rtol, atol", text_test_cases)
def test_text_parallel_align(
    model_name: str, config_path: str, is_moe: bool, rtol: float, atol: float, dummy_text_dataset
):
    test_path = f"./{model_name}"
    os.makedirs(test_path, exist_ok=True)

    _materialize_weights_dir(config_path, test_path)

    test_tasks = ["train_text_test"]
    command_list = prepare_exec_cmd(
        test_tasks,
        model_name,
        config_path,
        model_path=test_path,
        train_path=dummy_text_dataset,
        output_dir=test_path,
        is_moe=is_moe,
    )

    res = {}
    for task_name, cmd in command_list:
        print(f"{'-' * 10} {task_name} {'-' * 10}")

        df = _run_and_parse(cmd)
        res[task_name] = df
    compare_multi_items(res, rtol=rtol, atol=atol)
