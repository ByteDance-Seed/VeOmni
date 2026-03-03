import os
import random
import subprocess
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from transformers import Qwen2Config

from veomni.arguments import DataArguments, ModelArguments, TrainingArguments, VeOmniArguments, parse_args
from veomni.distributed.parallel_state import init_parallel_state
from veomni.utils import helper
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


logger = helper.create_logger(__name__)


@dataclass
class Arguments(VeOmniArguments):
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


def run_environ_meter(args):
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    dist.init_process_group(backend=get_dist_comm_backend(), world_size=world_size, rank=rank)

    config = Qwen2Config()
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
        async_enabled=args.train.async_enabled,
    )

    # Test update()
    micro_batch = {"attention_mask": torch.ones((1, 512), dtype=torch.int64)}

    cu_seqlens = torch.tensor([0, 67, 275, 382, 512])
    seqlens = cu_seqlens.diff()
    position_ids = torch.cat(
        [torch.arange(length, dtype=torch.long, device=cu_seqlens.device) for length in seqlens]
    ).unsqueeze(0)
    micro_batch["cu_seqlens"] = cu_seqlens
    micro_batch["position_ids"] = position_ids

    train_meter = helper.EnvironMeter(
        config=config,
        global_batch_size=args.train.global_batch_size,
    )

    micro_batches = [micro_batch] * 10

    for micro_batch in micro_batches:
        train_meter.add(micro_batch)

    delta_time = 0.1
    train_metrics = train_meter.step(delta_time, global_step=1)
    print(train_metrics)


def test_environ_meter():
    port = 12345 + random.randint(0, 100)

    command = [
        "torchrun",
        "--nproc_per_node=8",
        f"--master_port={port}",
        "tests/utils/test_helper.py",
        "--model.config_path=test",
        "--data.train_path=tests",
        "--train.output_dir=.tests/cache",
    ]

    result = subprocess.run(command, check=True)
    assert result.returncode == 0


if __name__ == "__main__":
    args = parse_args(Arguments)
    run_environ_meter(args)
