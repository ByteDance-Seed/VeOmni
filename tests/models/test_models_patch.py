import copy
import gc

import pytest
import torch
from transformers import AutoConfig

from veomni import _safe_apply_patches
from veomni.utils.device import empty_cache, synchronize

from ..tools.common_utils import print_device_mem_info
from .utils import (
    build_base_model_optim,
    compare_multi_items,
    prepare_data,
    prepare_model_modes,
    print_all_values,
    set_environ_param,
    train_one_step,
)
from .weight_sync_adapters import get_sync_weight_func


def _release_device_memory():
    synchronize()
    gc.collect()
    empty_cache()


# Test case: (config_path, is_moe, rtol, atol). id= must match weight_sync_adapters key if the model needs custom sync.
# rtol/atol: tolerances for compare_multi_items; can be set per case.
_DEFAULT_RTOL = 1e-2
_DEFAULT_ATOL = 1e-2

test_cases = [
    pytest.param(
        "./tests/toy_config/llama31_toy/config.json",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="llama3.1",
    ),
    pytest.param(
        "./tests/toy_config/qwen25_toy/config.json",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen2.5",
    ),
    pytest.param(
        "./tests/toy_config/qwen3_toy/config.json",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen3",
    ),
    pytest.param(
        "./tests/toy_config/qwen3_moe_toy/config.json",
        True,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen3_moe",
    ),
    pytest.param(
        "./tests/toy_config/seed_oss_toy/config.json",
        False,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="seed_oss",
    ),
    pytest.param(
        "./tests/toy_config/deepseek_v3_toy/config.json",
        True,
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="deepseek_v3",
    ),
]


@pytest.mark.parametrize("config_path, is_moe, rtol, atol", test_cases)
def test_models_patch_fwd_bwd(
    request: pytest.FixtureRequest,
    config_path: str,
    is_moe: bool,
    rtol: float,
    atol: float,
):
    case_id = request.node.callspec.id
    sync_weight_func = get_sync_weight_func(case_id)
    hf_model_modes, veomni_model_modes = prepare_model_modes(is_moe=is_moe, sync_weight_func=sync_weight_func)
    dummy_data = prepare_data(bsz=2, max_seq_len=1024, seq_lens=torch.tensor([1024, 1024]))

    config = AutoConfig.from_pretrained(config_path)
    print_device_mem_info("[Memory Info] start train_compare_models:")

    set_environ_param(hf_model_modes[0])
    _safe_apply_patches()
    model_base, optim_base = build_base_model_optim(
        config_path,
        attn_implementation=hf_model_modes[0].attn_implementation,
        moe_implementation=hf_model_modes[0].moe_implementation,
    )

    state_dict = copy.deepcopy(model_base.state_dict())
    del model_base, optim_base
    _release_device_memory()
    print_device_mem_info("[Memory Info] after building the base model and optimizer:")

    res = {}

    def run_step(idx, model_mode):
        print(f"{'-' * 10} {config.model_type}_{model_mode} {'-' * 10}")

        set_environ_param(model_mode)
        _safe_apply_patches()

        model_cur, optim_cur = build_base_model_optim(
            config_path,
            attn_implementation=model_mode.attn_implementation,
            moe_implementation=model_mode.moe_implementation,
        )
        print_device_mem_info(f"[Memory Info] after building model {idx}:")

        # Sync weights
        if model_mode.sync_weight_func is None:
            model_cur.load_state_dict(state_dict)
        else:
            model_mode.sync_weight_func(config, state_dict, model_cur)

        loss, gnorm = train_one_step(model_cur, optim_cur, dummy_data[model_mode.attn_case])

        result_metrics = {
            "loss": loss.item(),
            "gnorm": gnorm.item(),
        }

        del model_cur, optim_cur, loss, gnorm
        _release_device_memory()
        print_device_mem_info(f"[Memory Info] after model {idx} train_one_step:")

        return result_metrics

    # delete flash_attention_3 mode for hf deepseek_v3.
    # TODO: transformers v5 fixed this, remove this after veomni support transformers v5.
    if case_id == "deepseek_v3":
        hf_model_modes = [mode for mode in hf_model_modes if mode.attn_implementation != "flash_attention_3"]

    # Train HF backend models
    for idx, mode in enumerate(hf_model_modes):
        res[mode] = run_step(idx, mode)
    # Train VeOmni backend models
    for idx, mode in enumerate(veomni_model_modes):
        res[mode] = run_step(idx, mode)

    assert len(res) == len(hf_model_modes) + len(veomni_model_modes)
    print_all_values(res, "loss", config.model_type)
    print_all_values(res, "gnorm", config.model_type)
    compare_multi_items(res, rtol=rtol, atol=atol)

    _release_device_memory()
    print_device_mem_info("[Memory Info] after running train_compare_models:")
