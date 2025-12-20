import argparse
import copy
import gc
import json
import os
import subprocess
import tempfile

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

    # 1. build base model once
    model_base, optim_base = build_base_model_optim(
        config_path,
        attn_implementation=model_modes[0].attn_implementation,
        moe_implementation=model_modes[0].moe_implementation,
    )

    state_dict = copy.deepcopy(model_base.state_dict())
    del model_base, optim_base
    print_device_mem_info("[Memory Info] after building the base model and optimizer:")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "state_dict.pt")
        data_path = os.path.join(tmpdir, "dummy_data.pt")
        torch.save(state_dict, model_path)
        torch.save(dummy_data, data_path)

        outputs = {}

        for idx, model_mode in enumerate(model_modes):
            output_path = os.path.join(tmpdir, f"result_{idx}.json")

            running_id = (
                f"[{config.model_type}_"
                f"{model_mode.modeling_backend}]"
                f"-[attn-{model_mode.attn_implementation}]"
                f"_[moe-{model_mode.moe_implementation}]"
                f"_[{model_mode.attn_case}]"
            )
            print(f"{'-' * 10} {running_id=} {'-' * 10}")
            env = os.environ.copy()
            env["MODELING_BACKEND"] = "hf" if model_mode.modeling_backend == "hf" else "veomni"

            cmd = [
                "python",
                "-m",
                "tests.models.test_models_patch",
                "--config_path",
                config_path,
                "--model_path",
                model_path,
                "--data_path",
                data_path,
                "--attn_impl",
                model_mode.attn_implementation,
                "--moe_impl",
                model_mode.moe_implementation,
                "--attn_case",
                model_mode.attn_case,
                "--output_path",
                output_path,
            ]

            proc = subprocess.Popen(cmd, env=env)

            ret = proc.wait()
            assert ret == 0, f"{running_id} failed"

            with open(output_path) as f:
                outputs[running_id] = json.load(f)

    # 3. compare
    print_all_values(outputs, "loss")
    print_all_values(outputs, "gnorm")
    compare_multi_items(outputs, rtol=rtol, atol=atol)

    gc.collect()
    get_torch_device().empty_cache()

    print_device_mem_info("[Memory Info] after running train_compare_models:")


def main(args):
    config_path = args.config_path
    model_path = args.model_path
    data_path = args.data_path
    attn_impl = args.attn_impl
    moe_impl = args.moe_impl
    attn_case = args.attn_case
    output_path = args.output_path
    model_cur, optim_cur = build_base_model_optim(
        config_path,
        attn_implementation=attn_impl,
        moe_implementation=moe_impl,
    )

    state_dict = torch.load(model_path)
    dummy_data = torch.load(data_path)
    model_cur.load_state_dict(state_dict)

    loss, gnorm = train_one_step(model_cur, optim_cur, dummy_data[attn_case])
    res = {
        "loss": loss.item(),
        "gnorm": gnorm.item(),
    }

    json.dump(res, open(output_path, "w"), indent=4)

    del model_cur, optim_cur, loss, gnorm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--attn_impl", type=str)
    parser.add_argument("--moe_impl", type=str)
    parser.add_argument("--attn_case", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    main(args)
