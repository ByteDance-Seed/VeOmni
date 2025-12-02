import json
import os
import random
import subprocess
from dataclasses import asdict, dataclass, field

import pytest
import torch
import torch.distributed as dist

from veomni.checkpoint import build_checkpointer
from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_tokenizer
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device
from veomni.utils.import_utils import is_torch_npu_available


logger = helper.create_logger(__name__)


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


"""
torchrun --nnodes=1 --nproc-per-node=8 --master-port=4321 tests/utils/test_checkpointer.py \
    --model.model_path qwen2-1_5b-instruct \
    --data.train_path None \
    --train.global_batch_size 16 \
    --train.micro_batch_size 2 \
    --train.data_parallel_mode fsdp1 \
    --train.output_dir "ckpt_test" \
    --train.rmpad False \
    --train.rmpad_with_pos_ids False \
    --train.ckpt_manager "omnistore" \
    --train.max_steps 10 \
"""


def run_checkpointer_test():
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    dist.init_process_group(backend=get_dist_comm_backend())

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    args.train.compute_train_steps()
    logger.info_rank0("Prepare model")
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
        force_use_huggingface=args.model.force_use_huggingface,
    )

    model = build_parallelize_model(
        model,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        init_device=args.train.init_device,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=model._no_split_modules + args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
    )

    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
    )

    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=args.train.train_steps * args.train.num_train_epochs,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    # prepare data
    tokenizer = build_tokenizer(args.model.tokenizer_path)
    raw_text = "Hello, how are you?, I am fine, thank you."
    input_ids = tokenizer.encode(raw_text, return_tensors="pt")
    micro_batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids,
    }
    micro_batch = {
        k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in micro_batch.items()
    }

    helper.print_example(micro_batch, rank=args.train.local_rank)

    loss: "torch.Tensor" = model(**micro_batch, use_cache=False).loss

    logger.info(f"rank {args.train.local_rank} loss: {loss}")

    # save checkpoint
    global_step = 1
    state = {
        "model": model,
        "optimizer": optimizer,
        "extra_state": {
            "global_step": global_step,
            "lr_scheduler": lr_scheduler.state_dict(),
        },
    }

    logger.info_rank0(f"Distributed checkpoint saving... global_step {global_step}")
    Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
    logger.info_rank0("Distributed checkpoint saved successfully!")

    # load checkpoint
    state = {"model": model, "optimizer": optimizer, "extra_state": {}}
    load_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
    Checkpointer.load(load_checkpoint_path, state)
    lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
    global_step = state["extra_state"]["global_step"]
    global_step += 1
    logger.info_rank0("load checkpoint successfully!")

    # for dropout, reset seed
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    helper.print_example(micro_batch, rank=args.train.local_rank)
    resume_loss: "torch.Tensor" = model(**micro_batch, use_cache=False).loss.mean()

    logger.info(f"rank {args.train.local_rank} loss: {loss}, resume_loss: {resume_loss}")
    assert torch.allclose(loss, resume_loss), (
        f"rank {args.train.local_rank} loss is not equal, loss: {loss}, resume_loss: {resume_loss}"
    )

    logger.info(f"[rank{args.train.local_rank}] finish!!!!!")

    dist.destroy_process_group()


@pytest.mark.skipif(is_torch_npu_available(), reason="npu skip omnistore")
def test_omnistore_checkpointer():
    port = 12345 + random.randint(0, 100)

    command = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=8",
        f"--master_port={port}",
        "tests/utils/test_checkpointer.py",
        "--model.model_path=qwen2-1_5b-instruct",
        "--data.train_path=None",
        "--train.global_batch_size=16",
        "--train.micro_batch_size=2",
        "--train.data_parallel_mode=fsdp1",
        "--train.output_dir=omnistore_test",
        "--train.rmpad=False",
        "--train.rmpad_with_pos_ids=False",
        "--train.ckpt_manager=omnistore",
        "--train.max_steps=10",
    ]

    result = subprocess.run(command, check=True)
    assert result.returncode == 0


def test_dcp_checkpointer():
    port = 12345 + random.randint(0, 100)

    command = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=8",
        f"--master_port={port}",
        "tests/utils/test_checkpointer.py",
        "--model.model_path=qwen2-1_5b-instruct",
        "--data.train_path=None",
        "--train.global_batch_size=16",
        "--train.micro_batch_size=2",
        "--train.data_parallel_mode=fsdp1",
        "--train.output_dir=dcp_test",
        "--train.rmpad=False",
        "--train.rmpad_with_pos_ids=False",
        "--train.ckpt_manager=dcp",
        "--train.max_steps=10",
        f"--train.init_device={get_device_type()}",
    ]

    result = subprocess.run(command, check=True)
    assert result.returncode == 0


if __name__ == "__main__":
    run_checkpointer_test()
