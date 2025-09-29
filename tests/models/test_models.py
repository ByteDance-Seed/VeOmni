import copy
import gc
from functools import partial
from typing import Dict

from veomni.utils.device import empty_cache, get_torch_device

from .utils import build_base_model_optim, compare_multi_items, get_dummy_data, print_all_values, train_one_step


def print_device_mem_info(prefix_info=""):
    current_memory_allocated = get_torch_device().memory_allocated() / (1024**2)
    max_memory_allocated = get_torch_device().max_memory_allocated() / (1024**2)

    print(f"{prefix_info}current_memory:{current_memory_allocated:.2f} MB | max_memory:{max_memory_allocated:.2f} MB")


def prepare_data(bsz, seqlen):
    dummy_data = {
        "cu_seqlens": get_dummy_data(data_type="cu_seqlens", bsz=bsz, seqlen=seqlen, seed=42),
        "position_ids": get_dummy_data(data_type="position_ids", bsz=bsz, seqlen=seqlen, seed=42),
        "padded_bsh": get_dummy_data(data_type="padded_bsh", bsz=bsz, seqlen=seqlen, seed=42),
    }
    print(dummy_data)

    return dummy_data


def model_fwd_bwd(config_path, inputs: Dict, force_use_huggingface: bool, attn_implementation: str = "eager"):
    model, optim = build_base_model_optim(
        config_path, force_use_huggingface=force_use_huggingface, attn_implementation=attn_implementation
    )
    loss, gnorm = train_one_step(model, optim, inputs)

    ret = {
        "loss": loss.item(),
        "gnorm": gnorm.item(),
    }

    del model, optim, loss, gnorm
    gc.collect()
    empty_cache()

    return ret


def hf_models_fwd_bwd(config_path, eager_dummy_data, fa_dummy_data, attn_case, run_hf_fa=True):
    # eager
    eager_outputs_dict = (
        {
            f"veomni_eager_{attn_case}_out": model_fwd_bwd(
                config_path, eager_dummy_data[attn_case], force_use_huggingface=False, attn_implementation="eager"
            ),
            f"hf_eager_{attn_case}_out": model_fwd_bwd(
                config_path, eager_dummy_data[attn_case], force_use_huggingface=True, attn_implementation="eager"
            ),
        }
        if attn_case == "padded_bsh"
        else {}
    )

    # fa
    fa_outputs_dict = {
        f"veomni_fa_{attn_case}_out": model_fwd_bwd(
            config_path, fa_dummy_data[attn_case], force_use_huggingface=False, attn_implementation="flash_attention_2"
        ),
    }
    # In some cases like Qwen2.5VL, the current HF FA version implementation on NPU is buggy, so
    # we add an option to skip it.
    if run_hf_fa:
        fa_outputs_dict[f"hf_fa_{attn_case}_out"] = model_fwd_bwd(
            config_path, fa_dummy_data[attn_case], force_use_huggingface=True, attn_implementation="flash_attention_2"
        )

    return {**eager_outputs_dict, **fa_outputs_dict}


def seed_models_fwd_bwd(config_path, eager_dummy_data, fa_dummy_data, attn_case):
    # eager
    eager_outputs_dict = (
        {
            f"veomni_eager_{attn_case}_out": model_fwd_bwd(
                config_path, eager_dummy_data[attn_case], force_use_huggingface=False, attn_implementation="eager"
            ),
        }
        if attn_case == "padded_bsh"
        else {}
    )

    # fa
    fa_outputs_dict = {
        f"veomni_fa_{attn_case}_out": model_fwd_bwd(
            config_path, fa_dummy_data[attn_case], force_use_huggingface=False, attn_implementation="flash_attention_2"
        ),
    }

    return {**eager_outputs_dict, **fa_outputs_dict}


def train_step_hf_models(config_path, bsz, seqlen, rtol=1e-3, atol=1e-5, run_hf_fa=True):
    """
    Compare opensource models implementation between veomni and huggingface in fa and eager.
    Args:
        config_path (str): model config path.
        bsz (int): batch size.
        seqlen (int): sequence length.
    """
    eager_dummy_data = prepare_data(bsz=bsz, seqlen=seqlen)
    fa_dummy_data = copy.deepcopy(eager_dummy_data)

    # padded_bsh
    padded_bsh_out = hf_models_fwd_bwd(
        config_path, eager_dummy_data, fa_dummy_data, attn_case="padded_bsh", run_hf_fa=run_hf_fa
    )
    print_device_mem_info("[Memory Info] after padded_bsh case:")

    # position_ids
    position_ids_out = hf_models_fwd_bwd(
        config_path, eager_dummy_data, fa_dummy_data, attn_case="position_ids", run_hf_fa=run_hf_fa
    )
    print_device_mem_info("[Memory Info] after position_ids case:")

    # compare
    all_out = {
        **padded_bsh_out,
        **position_ids_out,
    }
    compare_multi_items(all_out, rtol=rtol, atol=atol)

    print_all_values(all_out, "loss")
    print_all_values(all_out, "gnorm")


def train_step_seed_models(config_path, bsz, seqlen):
    """
    Compare seed models implementation between fa and eager.
    Args:
        config_path (str): model config path.
        bsz (int): batch size.
        seqlen (int): sequence length.
    """
    eager_dummy_data = prepare_data(bsz=bsz, seqlen=seqlen)
    fa_dummy_data = copy.deepcopy(eager_dummy_data)

    # padded_bsh
    padded_bsh_out = seed_models_fwd_bwd(config_path, eager_dummy_data, fa_dummy_data, attn_case="padded_bsh")
    print_device_mem_info("[Memory Info] after padded_bsh case:")

    # position_ids
    position_ids_out = seed_models_fwd_bwd(config_path, eager_dummy_data, fa_dummy_data, attn_case="position_ids")
    print_device_mem_info("[Memory Info] after position_ids case:")

    # cu_seqlens
    cu_seqlens_out = seed_models_fwd_bwd(config_path, eager_dummy_data, fa_dummy_data, attn_case="cu_seqlens")
    print_device_mem_info("[Memory Info] after cu_seqlens case:")

    # compare
    all_out = {
        **padded_bsh_out,
        **position_ids_out,
        **cu_seqlens_out,
    }
    compare_multi_items(all_out)

    print_all_values(all_out, "loss")
    print_all_values(all_out, "gnorm")


# hf models
test_qwen3_train = partial(train_step_hf_models, "./tests/model_config/qwen3_toy.json", 2, 2048)
test_qwen25_train = partial(train_step_hf_models, "./tests/model_config/qwen25_toy.json", 2, 2048)

# qwen25vl_toy.json is modified from https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/config.json:
# 1. Changed num_hidden_layers and max_window_layers to 2.
# 2. Changed vision_config.depth to 4 and updated fullatt_block_indexes accordingly.
# TODO: Add tests for dummy vision inputs as well.
test_qwen25vl_train = partial(
    train_step_hf_models,
    "./tests/model_config/qwen25vl_toy.json",
    2,
    2048,
    # In NPU, veomni_fa_position_ids_out.gnorm is 3.59375 vs 3.609375 others.
    rtol=0.01,
    # TODO: Run HF's NPU FA implementation after the bug is fixed in >=4.53.
    # See https://github.com/huggingface/transformers/pull/37575
    run_hf_fa=False,
)
