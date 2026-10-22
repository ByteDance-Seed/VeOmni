# Checkpoint trainer save/load test scripts (exec_scripts style).
# One base_config; per-model only config_path/tokenizer_path; each model tests 3 EP cases.

import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import hf_local_or_remote, resolve_ops_overrides
from tools.launch_utils import find_free_port

from veomni.checkpoint.layout import weights_dir


def get_checkpoint_test_nproc() -> int:
    """World size for trainer saveload torchrun jobs.

    Defaults to 8 to match the L20-8 CI runners. Override with
    ``VEOMNI_CHECKPOINT_TEST_NPROC`` for local machines with fewer GPUs.
    """
    return int(os.environ.get("VEOMNI_CHECKPOINT_TEST_NPROC", "8"))


MODEL_CONFIGS = {
    "qwen3_moe": {
        "config_path": "tests/toy_config/qwen3_moe_toy/config.json",
        "tokenizer_path": "Qwen/Qwen3-30B-A3B",
    },
    "deepseek_v3": {
        "config_path": "tests/toy_config/deepseek_v3_toy/config.json",
        "tokenizer_path": "deepseek-ai/DeepSeek-V3",
    },
    # Reuse the DeepSeek-V3 tokenizer assets; the saveload trainer only needs a
    # tokenizer object for model-assets bookkeeping and trains on dummy text.
    "deepseek_v4": {
        "config_path": "tests/toy_config/deepseek_v4_toy/config.json",
        "tokenizer_path": "deepseek-ai/DeepSeek-V3",
    },
}


# Get some dir functions.
# ``dp_replicate_size`` and ``nproc`` are threaded into the dir name so HSDP
# runs and different world sizes don't collide.
def get_output_dir(model_name, ep_size, dp_replicate_size=None, nproc=None):
    suffix = f"_dpr{dp_replicate_size}" if dp_replicate_size else ""
    if nproc is not None:
        suffix += f"_n{nproc}"
    return f"./test_trainer_saveload_{model_name}_{ep_size}{suffix}"


def get_checkpoint_dir(model_name, ep_size, dp_replicate_size=None, nproc=None, save_step=1):
    """Step directory. Resume takes this; it is not itself a DCP directory."""
    return os.path.join(
        get_output_dir(model_name, ep_size, dp_replicate_size, nproc=nproc),
        "checkpoints",
        f"global_step_{save_step}",
    )


def get_dcp_weights_dir(model_name, ep_size, dp_replicate_size=None, nproc=None, save_step=1):
    """DCP directory holding the weights, which is what a reader of shards wants."""
    return weights_dir(get_checkpoint_dir(model_name, ep_size, dp_replicate_size, nproc=nproc, save_step=save_step))


def get_hf_output_dir(model_name, ep_size, dp_replicate_size=None, nproc=None):
    return os.path.join(get_output_dir(model_name, ep_size, dp_replicate_size, nproc=nproc), "hf_ckpt")


def get_model_assets_dir(model_name, ep_size, dp_replicate_size=None, nproc=None):
    return os.path.join(get_output_dir(model_name, ep_size, dp_replicate_size, nproc=nproc), "model_assets")


# running command functions
def get_checkpoint_test_command(
    model_name,
    ep_size,
    save_hf_weights=False,
    dp_replicate_size=None,
    nproc=None,
    max_steps=2,
):
    config_path = MODEL_CONFIGS[model_name]["config_path"]
    tokenizer_path = hf_local_or_remote(MODEL_CONFIGS[model_name]["tokenizer_path"])
    output_dir = get_output_dir(model_name, ep_size, dp_replicate_size, nproc=nproc)
    port = find_free_port()
    if nproc is None:
        nproc = get_checkpoint_test_nproc()
    if ep_size > nproc:
        raise ValueError(f"ep_size={ep_size} exceeds checkpoint-test world size nproc={nproc}")
    # Keep global batch a multiple of micro_batch * dp_size. With micro_batch=1
    # and dp_size = nproc // ep_size, nproc itself is always valid.
    global_batch_size = nproc

    params = [
        f"{sys.executable} -m torch.distributed.run --nnodes=1 --nproc_per_node={nproc} --master-port={port}",
        "tests/checkpoints/test_trainer_saveload.py",
        f"--model.config_path {config_path}",
        f"--model.tokenizer_path {tokenizer_path}",
        # Hardware-aware ops_implementation overrides. ``resolve_ops_overrides``
        # emits NPU-supported per-op backends on NPU (with eager fallback for
        # ops without an NPU kernel) and the GPU-optimal baseline on GPU.
        *resolve_ops_overrides(model_name),
        "--data.train_path dummy",
        "--data.max_seq_len 128",
        f"--train.checkpoint.output_dir {output_dir}",
        "--model.accelerator.fsdp_config.fsdp_mode fsdp2",
        "--model.accelerator.init_device meta",
        f"--model.accelerator.ep_size {ep_size}",
        f"--train.global_batch_size {global_batch_size}",
        "--train.micro_batch_size 1",
        "--model.optimizer.lr 1e-4",
        "--model.optimizer.lr_warmup_ratio 0.0",
        "--model.optimizer.lr_decay_style constant",
        "--model.optimizer.lr_decay_ratio 1.0",
        "--model.optimizer.weight_decay 0.01",
        "--model.optimizer.max_grad_norm 1.0",
        f"--train.max_steps {max_steps}",
        "--train.checkpoint.manager dcp",
        "--train.checkpoint.save_async True",
        f"--train.checkpoint.save_hf_weights {save_hf_weights}",
    ]
    # HSDP: split the FSDP dim into (dp_replicate, dp_shard). dp_shard is
    # inferred as dp_size // dp_replicate_size by the argument resolver, and the
    # expert mesh becomes 3D (ep_replicate, ep_fsdp, ep), exercising the
    # 3-placement save/load path in the DCP checkpointer.
    if dp_replicate_size is not None:
        params.append(f"--model.accelerator.dp_replicate_size {dp_replicate_size}")

    exec_script = " \\\n".join(params)

    return exec_script


def get_merge_dcp_to_hf_command(
    model_name,
    ep_size,
    dp_replicate_size=None,
    nproc=None,
    save_step=1,
):
    checkpoint_dir = get_checkpoint_dir(model_name, ep_size, dp_replicate_size, nproc=nproc, save_step=save_step)
    hf_output_dir = get_hf_output_dir(model_name, ep_size, dp_replicate_size, nproc=nproc)
    model_assets_dir = get_model_assets_dir(model_name, ep_size, dp_replicate_size, nproc=nproc)

    params = [
        "python",
        "scripts/merge_dcp_to_hf.py",
        f"--load-dir {checkpoint_dir}",
        f"--save-dir {hf_output_dir}",
        f"--model-assets-dir {model_assets_dir}",
    ]

    merge_script = " \\\n".join(params)

    return merge_script
