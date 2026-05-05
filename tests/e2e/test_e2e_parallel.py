import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from veomni.models.auto import build_foundation_model
from veomni.utils.import_utils import is_diffusers_available, is_transformers_version_greater_or_equal_to

from ..tools import DummyDataset, build_torchrun_cmd, compare_metrics, print_comparison_table
from .utils import prepare_exec_cmd


# See
_is_transformers_v5 = is_transformers_version_greater_or_equal_to("5.0.0")
_v4_only = pytest.mark.skipif(_is_transformers_v5, reason="Not compatible with transformers >= 5.0.0")
_v5_only = pytest.mark.skipif(not _is_transformers_v5, reason="Requires transformers >= 5.0.0")
_dit_only = pytest.mark.skipif(not is_diffusers_available(), reason="Requires diffusers")


def _materialize_weights_dir(config_path: str, output_path: str, save_original_format: bool = True) -> Path:
    # Seed CPU RNG and init on CPU so the materialized checkpoint is bit-identical
    # across pytest invocations *and* across GPU architectures (L20 in CI vs
    # H100/A100 locally). Without this, the four sub-runs (sp/ep grid) would
    # share weights within one pytest run but differ between runs.
    #
    # Cross-EP structural spread on H100. EP=1 vs EP=2 paths are structurally
    # different — different per-rank token composition after `all_to_all`,
    # and the world is reorganized as `ep × ep_fsdp` so FSDP gradient
    # reduction occurs over a different rank set in bf16. On 8×H100 we ran
    # 100 in-process reps under all three determinism flags
    # (`enable_full_determinism`, `enable_high_precision_for_bf16`,
    # `enable_batch_invariant_mode`) and observed bit-identical results
    # across reps with cross-EP step-2 grad_norm spread = 0.0547 — well
    # inside the 0.1 envelope below.
    #
    # CI L20 outlier (unexplained). We have seen one CI run on L20 where the
    # step-2 cross-EP spread reached ~0.87, which is 16× the deterministic
    # H100 baseline. The 100-rep H100 result rules out generic bf16 EP-path
    # noise as the cause: that noise produces a stable 0.055 spread and is
    # not stochastic across reps. Whatever drives the L20 0.87 number is
    # therefore something the H100 environment does not see — most likely
    # CI machine instability (driver / NCCL build, kernel autotune cache,
    # Ada-vs-Hopper kernel selection, transient host load) rather than a
    # model-side bug. We do not have L20 access to run the 100-rep
    # robustness check, so this remains a hypothesis.
    #
    # Keep the strict tolerance. Relaxing it globally to absorb a single
    # unexplained L20 outlier would mask real EP-path regressions on the
    # hardware where the test is robust. If CI flakes on this assertion,
    # retry first; if it persists, capture the failing logs and investigate
    # L20-specific causes (driver, NCCL build, kernel autotune cache) before
    # widening the bound.
    torch.manual_seed(0)
    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype="float32",
        attn_implementation="eager",
        init_device="cpu",
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

    # Models with stacked 3D expert params (gate_up_proj [E, 2*I, H], down_proj [E, H, I]):
    #
    # - qwen3_5_moe: native HF safetensor format is already stacked. HF's save_pretrained() with
    #   save_original_format=True calls revert_weight_conversion() that splits them into per-expert
    #   keys (experts.*.gate_proj.weight, etc.), but VeOmni has no runtime converter for this model.
    #   Disable save_original_format to save in native stacked format.
    #
    # - qwen3_moe (v5): VeOmni registers a runtime CheckpointTensorConverter that merges per-expert
    #   HF keys back to fused format at load time, so save_original_format=True works correctly.
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
    for task_name, cmd_kwargs in command_list:
        print(f"{'-' * 10} {task_name} {'-' * 10}")
        cmd = build_torchrun_cmd(**cmd_kwargs)
        subprocess.run(cmd, check=True)
        with open(os.path.join(test_path, f"{task_name}/log_dict.json")) as f:
            output = json.load(f)
        if not log_keys:
            log_keys = set(output.keys())
        else:
            assert log_keys == set(output.keys())
        res[task_name] = output

    for key in log_keys:
        print_comparison_table(res, key, title=model_name)
    compare_metrics(res, rtol=rtol, atol=atol)

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
        "qwen2",
        "./tests/toy_config/qwen2_toy/config.json",
        False,  # is_moe
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        None,  # max_sp_size
        marks=_v5_only,
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
        "qwen3_moe",
        "./tests/toy_config/qwen3_moe_toy",
        True,  # is_moe
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        None,  # max_sp_size
        marks=_v5_only,
        id="qwen3_moe_v5",
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
        "qwen2vl",
        "./tests/toy_config/qwen2vl_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        marks=_v5_only,
    ),
    pytest.param(
        "qwen25vl",
        "./tests/toy_config/qwen25vl_toy",
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
        marks=_v5_only,
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
        "qwen3vl",
        "./tests/toy_config/qwen3vl_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        None,  # max_sp_size
        marks=_v5_only,
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
        "qwen3vlmoe",
        "./tests/toy_config/qwen3vlmoe_toy",
        True,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        None,  # max_sp_size
        marks=_v5_only,
    ),
    pytest.param(
        "qwen3_5_moe",
        "./tests/toy_config/qwen3_5_moe_toy/config.json",
        True,  # is_moe
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        None,  # max_sp_size
        marks=_v5_only,
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
    pytest.param(
        "qwen3_omni_moe",
        "./tests/toy_config/qwen3omni_toy",
        True,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        marks=_v5_only,
    ),
]

wan_dit_test_cases = [
    pytest.param(
        "wan_t2v",
        "./tests/toy_config/wan_t2v_toy",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        marks=_dit_only,
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


@pytest.fixture(scope="session")
def dummy_wan_t2v_dataset():
    dummy_dataset = DummyDataset(seq_len=2048, dataset_type="wan_t2v")
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


@pytest.mark.parametrize("model_name, config_path, is_moe, rtol, atol", wan_dit_test_cases)
def test_wan_dit_parallel_align(
    model_name: str, config_path: str, is_moe: bool, rtol: float, atol: float, dummy_wan_t2v_dataset
):
    """Validate that WanTransformer3DModel loss and grad_norm are identical with
    and without Ulysses sequence-parallelism at equal DP sizes.
    """
    main(
        task_name="train_dit_test",
        model_name=model_name,
        config_path=config_path,
        is_moe=is_moe,
        rtol=rtol,
        atol=atol,
        train_path=dummy_wan_t2v_dataset,
    )
