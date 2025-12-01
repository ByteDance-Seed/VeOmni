import os
import random
import subprocess
from dataclasses import dataclass, field

import pytest
import torch.distributed as dist

from veomni.distributed.parallel_state import init_parallel_state
from veomni.models import build_foundation_model
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


logger = helper.create_logger(__name__)


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


"""
torchrun --nnodes=1 --nproc-per-node=8 --master-port=4321 tests/utils/test_helper.py \
    --model.config_path test \
    --data.train_path tests \
    --train.rmpad True \
    --train.output_dir .tests/cache \
"""


def run_environ_meter(args: Arguments):
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    dist.init_process_group(backend=get_dist_comm_backend(), world_size=world_size, rank=rank)

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

    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        init_device=args.train.init_device,
        force_use_huggingface=args.model.force_use_huggingface,
    )
    print(f"Model Class: {type(model)}")


@pytest.mark.parametrize(
    "model_path",
    [
        "qwen2vl-7b-instruct",
        "llama3_2-3b-instruct",
    ],
)
def test_model_loader(model_path):
    port = 12345 + random.randint(0, 100)

    command = [
        "torchrun",
        "--nproc_per_node=4",
        f"--master_port={port}",
        "tests/utils/test_model_loader.py",
        f"--model.config_path={model_path}",
        "--data.train_path=tests",
        "--train.output_dir=.tests/cache",
        f"--train.init_device={get_device_type()}",
    ]

    result = subprocess.run(command, check=True)
    assert result.returncode == 0


if __name__ == "__main__":
    args = parse_args(Arguments)
    run_environ_meter(args)
