import copy
import gc

import pytest
import torch
from transformers import AutoConfig

from veomni.utils.device import get_torch_device

from ..tools.common_utils import print_device_mem_info
from .utils import (
    build_base_model_optim,
    compare_multi_items,
    prepare_data,
    prepare_models_modes,
    print_all_values,
    train_one_step,
)


test_cases = [
    pytest.param("./tests/models/toy_config/qwen25_toy.json", prepare_models_modes()),
    pytest.param("./tests/models/toy_config/qwen3_toy.json", prepare_models_modes()),
]


@pytest.mark.parametrize("config_path, model_modes", test_cases)
def test_models_patch_fwd_bwd(config_path, model_modes, rtol=1e-3, atol=1e-5):
    dummy_data = prepare_data(bsz=2, max_seq_len=1024, seq_lens=torch.tensor([1024, 1024]))
    assert len(model_modes) >= 2
    config = AutoConfig.from_pretrained(config_path)
    print_device_mem_info("[Memory Info] start train_compare_models:")

    model_base, optim_base = build_base_model_optim(
        config_path,
        force_use_huggingface=model_modes[0].force_use_huggingface,
        attn_implementation=model_modes[0].attn_implementation,
        moe_implementation=model_modes[0].moe_implementation,
    )

    state_dict = copy.deepcopy(model_base.state_dict())
    del model_base, optim_base
    print_device_mem_info("[Memory Info] after building the base model and optimizer:")

    res = {}
    # train and compare models
    for idx, model_mode_cur in enumerate(model_modes):
        model_source = "hf" if model_mode_cur.force_use_huggingface else "veomni"
        running_id = f"[{config.model_type}_{model_source}]-[attn-{model_mode_cur.attn_implementation}]_[moe-{model_mode_cur.moe_implementation}]_[{model_mode_cur.attn_case}]"
        print(f"{'-' * 10} {running_id=} {'-' * 10}")

        model_cur, optim_cur = build_base_model_optim(
            config_path,
            force_use_huggingface=model_mode_cur.force_use_huggingface,
            attn_implementation=model_mode_cur.attn_implementation,
            moe_implementation=model_mode_cur.moe_implementation,
        )

        print_device_mem_info(f"[Memory Info] after building the model and optimizer {idx}:")

        # apply sync weight so that the weight init is the same between models
        if model_mode_cur.sync_weight_func is None:
            model_cur.load_state_dict(state_dict)
        else:
            model_mode_cur.sync_weight_func(config, state_dict, model_cur)

        loss, gnorm = train_one_step(model_cur, optim_cur, dummy_data[model_mode_cur.attn_case])
        res[running_id] = {
            "loss": loss.item(),
            "gnorm": gnorm.item(),
        }
        print_device_mem_info(f"[Memory Info] after model {idx} train_one_step:")

        del model_cur, optim_cur, loss, gnorm

    assert len(res) == len(model_modes)
    print_all_values(res, "loss")
    print_all_values(res, "gnorm")
    compare_multi_items(res, rtol=rtol, atol=atol)

    gc.collect()
    get_torch_device().empty_cache()

    print_device_mem_info("[Memory Info] after running train_compare_models:")
    return res
