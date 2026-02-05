# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Checkpoint trainer save/load test scripts (exec_scripts style).
# One base_config; per-model only config_path/tokenizer_path; each model tests 3 EP cases.

import os

from ..tools.launch_utils import find_free_port


MODEL_CONFIGS = {
    "qwen3_moe": {
        "config_path": "tests/toy_config/qwen3_moe_toy/config.json",
        "tokenizer_path": "Qwen/Qwen3-30B-A3B",
    },
    "deepseek_v3": {
        "config_path": "tests/toy_config/deepseek_v3_toy/config.json",
        "tokenizer_path": "deepseek-ai/DeepSeek-V3",
    },
}


# Get some dir functions
def get_output_dir(model_name, ep_size):
    return f"./test_trainer_saveload_{model_name}_{ep_size}"


def get_checkpoint_dir(model_name, ep_size):
    return os.path.join(get_output_dir(model_name, ep_size), "checkpoints", "global_step_5")


def get_hf_output_dir(model_name, ep_size):
    return os.path.join(get_output_dir(model_name, ep_size), "hf_ckpt")


def get_model_assets_dir(model_name, ep_size):
    return os.path.join(get_output_dir(model_name, ep_size), "model_assets")


# running command functions
def get_checkpoint_test_command(
    model_name,
    ep_size,
):
    config_path = MODEL_CONFIGS[model_name]["config_path"]
    tokenizer_path = MODEL_CONFIGS[model_name]["tokenizer_path"]
    output_dir = get_output_dir(model_name, ep_size)
    port = find_free_port()

    params = [
        f"torchrun --nnodes=1 --nproc_per_node=8 --master-port={port}",
        "tests/checkpoints/test_trainer_saveload.py",
        f"--model.config_path {config_path}",
        f"--model.tokenizer_path {tokenizer_path}",
        "--model.moe_implementation fused",
        "--model.attn_implementation flash_attention_2",
        "--data.train_path dummy",
        "--data.max_seq_len 128",
        f"--train.output_dir {output_dir}",
        "--train.data_parallel_mode fsdp2",
        "--train.init_device meta",
        f"--train.expert_parallel_size {ep_size}",
        "--train.global_batch_size 8",
        "--train.micro_batch_size 1",
        "--train.rmpad false",
        "--train.rmpad_with_pos_ids true",
        "--train.dyn_bsz_margin 0",
        "--train.lr 3.0e-4",
        "--train.lr_warmup_ratio 0.007",
        "--train.lr_decay_style constant",
        "--train.lr_decay_ratio 1.0",
        "--train.weight_decay 0.01",
        "--train.max_grad_norm 1.0",
        "--train.max_steps 5",
        "--train.ckpt_manager dcp",
        "--train.save_async True",
    ]

    exec_script = " \\\n".join(params)

    return exec_script


def get_merge_dcp_to_hf_command(
    model_name,
    ep_size,
):
    checkpoint_dir = get_checkpoint_dir(model_name, ep_size)
    hf_output_dir = get_hf_output_dir(model_name, ep_size)
    model_assets_dir = get_model_assets_dir(model_name, ep_size)

    params = [
        "python",
        "scripts/merge_dcp_to_hf.py",
        f"--load-dir {checkpoint_dir}",
        f"--save-dir {hf_output_dir}",
        f"--model-assets-dir {model_assets_dir}",
    ]

    merge_script = " \\\n".join(params)

    return merge_script
