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
    # H100/A100 locally). Without this, the four sub-runs of the (sp, ep) grid
    # would still see the same weights within one pytest run, but successive
    # runs would each materialize a different random init, giving every CI
    # attempt a different cross-config divergence to chew on.
    #
    # Note: a seeded init does NOT guarantee that EP=1 and EP=2 produce
    # close grad_norms — see `_assert_parallel_alignment` for why cross-EP
    # divergence is intrinsic to MoE training and how the test handles it.
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
    _assert_parallel_alignment(res, is_moe=is_moe, rtol=rtol, atol=atol)

    shutil.rmtree(test_path)


# Cross-EP grad_norm tolerance for MoE models. Loss stays close (router output
# is the same averaged signal) but grad_norm picks up the routing-flip cascade
# documented in `_assert_parallel_alignment`. 0.5 rtol + 1.0 atol absorbs the
# observed CI range (≤0.9 absolute spread on toy configs) while still catching
# order-of-magnitude regressions in the EP path.
_MOE_CROSS_EP_GRAD_NORM_RTOL = 0.5
_MOE_CROSS_EP_GRAD_NORM_ATOL = 1.0


def _assert_parallel_alignment(
    res: dict,
    *,
    is_moe: bool,
    rtol: float,
    atol: float,
) -> None:
    """Compare metrics across distributed configs with MoE-aware tolerances.

    For dense (non-MoE) models, every parallel config (varying only SP) must
    match within (rtol, atol).

    For MoE models, the configs span both SP and EP. EP=1 vs EP=2 cannot be
    bitwise-aligned because:
      - the EP path runs experts on a different per-rank token composition
        (after all_to_all) than the non-EP path (only local tokens), and
      - the world is reorganized as ep × ep_fsdp, so FSDP gradient
        reduction occurs over a different rank set in bf16.
    Even with deterministic kernels and stable argsort, the residual bf16
    noise in the router input flips top-k decisions for a small fraction of
    tokens (~1% by layer 1, ~5% by layer 3, growing each step). On toy
    configs this can amplify the cross-EP grad_norm spread above the strict
    tolerance on certain GPUs.

    Strategy:
      - same-EP runs (only SP varies): strict tolerance — SP must not change
        forward/backward semantics.
      - cross-EP runs: loss is checked at strict tolerance (it's stable);
        grad_norm uses a relaxed tolerance to absorb routing-flip noise.
    """
    if not is_moe:
        compare_metrics(res, rtol=rtol, atol=atol)
        return

    # Group runs by EP size parsed from the trailing `_ep<N>` segment of the
    # task name (set by `prepare_exec_cmd`). `rpartition` plus integer parse
    # avoids the substring trap of `"_ep1" in k` (which would also match
    # `_ep10`) and naturally extends to any EP grid larger than {1, 2}.
    runs_by_ep: dict[int, dict] = {}
    for k, v in res.items():
        head, sep, tail = k.rpartition("_ep")
        if not sep or not tail.isdigit():
            raise AssertionError(f"MoE run key missing _ep<N> suffix: {k!r}")
        runs_by_ep.setdefault(int(tail), {})[k] = v

    if len(runs_by_ep) < 2:
        # Single-EP slice (e.g. max_sp_size filtered everything down). Fall
        # back to the dense comparison.
        compare_metrics(res, rtol=rtol, atol=atol)
        return

    # Same-EP runs (only SP varies) must match within strict tolerance.
    for ep_runs in runs_by_ep.values():
        if len(ep_runs) >= 2:
            compare_metrics(ep_runs, rtol=rtol, atol=atol)

    # Cross-EP: pick the smallest EP as the baseline and compare each other
    # EP value against it. Loss stays close (router output is the same
    # averaged signal) so it uses the strict tolerance; grad_norm uses the
    # relaxed tolerance to absorb routing-flip noise.
    ep_values = sorted(runs_by_ep.keys())
    base_ep = ep_values[0]
    base_name = next(iter(runs_by_ep[base_ep]))
    base_run = runs_by_ep[base_ep][base_name]

    metric_keys = list(base_run.keys())
    grad_keys = [k for k in metric_keys if "grad_norm" in k]
    other_keys = [k for k in metric_keys if k not in grad_keys]

    for other_ep in ep_values[1:]:
        other_name = next(iter(runs_by_ep[other_ep]))
        cross = {base_name: base_run, other_name: runs_by_ep[other_ep][other_name]}
        if other_keys:
            compare_metrics(cross, rtol=rtol, atol=atol, keys=other_keys)
        if grad_keys:
            compare_metrics(
                cross,
                rtol=_MOE_CROSS_EP_GRAD_NORM_RTOL,
                atol=_MOE_CROSS_EP_GRAD_NORM_ATOL,
                keys=grad_keys,
            )


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
