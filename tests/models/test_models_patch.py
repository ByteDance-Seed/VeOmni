import copy
import gc

import pytest
import torch
from transformers import AutoConfig

from veomni import _safe_apply_patches
from veomni.utils.device import get_torch_device

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


# Model configs for testing - add new models here
test_cases = [
    pytest.param("./tests/models/toy_config/llama31_toy/config.json", prepare_model_modes(), id="llama3.1"),
    pytest.param("./tests/models/toy_config/qwen25_toy/config.json", prepare_model_modes(), id="qwen2.5"),
    pytest.param("./tests/models/toy_config/qwen3_toy/config.json", prepare_model_modes(), id="qwen3"),
]


@pytest.mark.parametrize("config_path, model_modes", test_cases)
def test_models_patch_fwd_bwd(config_path, model_modes, rtol=1e-3, atol=1e-5):
    hf_model_modes, veomni_model_modes = model_modes
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

        print_device_mem_info(f"[Memory Info] after model {idx} train_one_step:")
        del model_cur, optim_cur, loss, gnorm

        return result_metrics

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

    gc.collect()
    get_torch_device().empty_cache()

    print_device_mem_info("[Memory Info] after running train_compare_models:")
