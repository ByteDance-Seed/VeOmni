import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from utils import DummyDataset, compare_multi_items, prepare_exec_cmd, print_all_values

from veomni.models.auto import build_foundation_model
from veomni.utils.device import get_device_type
from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to


# See
_is_transformers_v5 = is_transformers_version_greater_or_equal_to("5.0.0")
_v4_only = pytest.mark.skipif(_is_transformers_v5, reason="Not compatible with transformers >= 5.0.0")
_v5_only = pytest.mark.skipif(not _is_transformers_v5, reason="Requires transformers >= 5.0.0")


def _materialize_weights_dir(config_path: str, output_path: str, save_original_format: bool = True) -> Path:
    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype="float32",
        attn_implementation="eager",
        moe_implementation="eager",
        init_device=get_device_type(),
    )
    model.save_pretrained(output_path, save_original_format=save_original_format)


def main(
    task_name: str,
    model_name: str,
    config_path: str,
    is_moe: bool,
    rtol: float,
    atol: float,
    train_path: str,
    max_sp_size: int | None = None,
):
    test_path = f"./{model_name}"
    os.makedirs(test_path, exist_ok=True)

    # Qwen3_5Moe uses stacked 3D expert params (gate_up_proj [E, 2*I, H], down_proj [E, H, I])
    # which is also the native HF safetensor format for this model. HF's save_pretrained() with
    # save_original_format=True calls revert_weight_conversion() that splits them into per-expert
    # keys (experts.*.gate_proj.weight, etc.), but VeOmni's load_model_weights doesn't apply the
    # reverse merge. Disable save_original_format for qwen3_5_moe to save in native stacked format.
    save_original_format = model_name != "qwen3_5_moe"
    _materialize_weights_dir(config_path, test_path, save_original_format=save_original_format)

    test_tasks = [task_name]
    command_list = prepare_exec_cmd(
        test_tasks,
        model_name,
        config_path,
        model_path=test_path,
        train_path=train_path,
        output_dir=test_path,
        is_moe=is_moe,
        max_sp_size=max_sp_size,
    )
    res = {}
    log_keys = []
    for task_name, cmd in command_list:
        print(f"{'-' * 10} {task_name} {'-' * 10}")
        subprocess.run(cmd, check=True)
        with open(os.path.join(test_path, f"{task_name}/log_dict.json")) as f:
            output = json.load(f)
        if not log_keys:
            log_keys = set(output.keys())
        else:
            assert log_keys == set(output.keys())
        res[task_name] = output

    for key in log_keys:
        print_all_values(res, key, model_type=model_name)
    compare_multi_items(model_name, res, rtol=rtol, atol=atol)

    shutil.rmtree(test_path)


_DEFAULT_RTOL = 1e-1
_DEFAULT_ATOL = 1e-1


text_test_cases = [
    pytest.param(
        "llama3.1",
        "./tests/toy_config/llama31_toy",
        False,  # is_moe
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        None,  # max_sp_size
        marks=_v4_only,
    ),
    pytest.param(
        "qwen2.5",
        "./tests/toy_config/qwen25_toy",
        False,  # is_moe
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        None,  # max_sp_size
        marks=_v4_only,
    ),
    pytest.param(
        "qwen3",
        "./tests/toy_config/qwen3_toy",
        False,  # is_moe
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        None,  # max_sp_size
        marks=_v4_only,
    ),
    pytest.param(
        "qwen3_moe",
        "./tests/toy_config/qwen3_moe_toy",
        True,  # is_moe
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        None,  # max_sp_size
        marks=_v4_only,
    ),
    pytest.param(
        "seed_oss",
        "./tests/toy_config/seed_oss_toy",
        False,  # is_moe
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        None,  # max_sp_size
        marks=_v4_only,
    ),
    pytest.param(
        "deepseek_v3",
        "./tests/toy_config/deepseek_v3_toy",
        True,  # is_moe
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        None,  # max_sp_size
        marks=_v4_only,
    ),
    # TODO: we only support text input now. move this to VLM test once vision input is supported.
    pytest.param(
        "qwen3_5_moe",
        "./tests/toy_config/qwen3_5_moe_toy/config.json",
        # TODO: Test with EP once Merged fc1_1_2_weight is supported with expert parallelism.
        False,  # is_moe
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        None,  # max_sp_size
        marks=_v5_only,
    ),
]

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
        None,  # max_sp_size
        marks=_v4_only,
    ),
    pytest.param(
        "qwen3vlmoe",
        "./tests/toy_config/qwen3vlmoe_toy",
        True,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        None,  # max_sp_size
        marks=_v4_only,
    ),
    pytest.param(
        "qwen3_5",
        "./tests/toy_config/qwen3_5_toy/config.json",
        False,  # is_moe
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        None,  # max_sp_size
        marks=_v5_only,
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


@pytest.fixture(scope="session")
def dummy_text_dataset():
    dummy_dataset = DummyDataset(seq_len=2048, dataset_type="text")
    train_path = dummy_dataset.save_path
    yield train_path
    del dummy_dataset


@pytest.fixture(scope="session")
def dummy_qwen2vl_dataset():
    dummy_dataset = DummyDataset(seq_len=2048, dataset_type="qwen2vl")
    train_path = dummy_dataset.save_path
    yield train_path
    del dummy_dataset


@pytest.fixture(scope="session")
def dummy_qwen3vl_dataset():
    dummy_dataset = DummyDataset(seq_len=2048, dataset_type="qwen3vl")
    train_path = dummy_dataset.save_path
    yield train_path
    del dummy_dataset


@pytest.fixture(scope="session")
def dummy_qwen2omni_dataset():
    dummy_dataset = DummyDataset(seq_len=2048, dataset_type="qwen2omni")
    train_path = dummy_dataset.save_path
    yield train_path
    del dummy_dataset


@pytest.fixture(scope="session")
def dummy_qwen3omni_dataset():
    dummy_dataset = DummyDataset(seq_len=2048, dataset_type="qwen3omni")
    train_path = dummy_dataset.save_path
    yield train_path
    del dummy_dataset


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
