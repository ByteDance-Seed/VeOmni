import os
import random
import subprocess
from dataclasses import dataclass, field

import pytest
import torch.distributed as dist

from veomni.arguments import DataArguments, ModelArguments, TrainingArguments, VeOmniArguments, parse_args
from veomni.distributed.parallel_state import init_parallel_state
from veomni.models import build_foundation_model
from veomni.utils import helper
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


logger = helper.create_logger(__name__)


@dataclass
class Arguments(VeOmniArguments):
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


"""
torchrun --nnodes=1 --nproc-per-node=8 --master-port=4321 tests/utils/test_helper.py \
    --model.config_path test \
    --data.train_path tests \
    --train.checkpoint.output_dir .tests/cache \
"""


def run_environ_meter(args: Arguments):
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    dist.init_process_group(backend=get_dist_comm_backend(), world_size=world_size, rank=rank)

    init_parallel_state(
        dp_size=args.train.accelerator.dp_size,
        dp_replicate_size=args.train.accelerator.dp_replicate_size,
        dp_shard_size=args.train.accelerator.dp_shard_size,
        tp_size=args.train.accelerator.tp_size,
        pp_size=args.train.accelerator.pp_size,
        cp_size=args.train.accelerator.cp_size,
        ulysses_size=args.train.accelerator.ulysses_size,
        extra_parallel_sizes=args.train.accelerator.extra_parallel_sizes,
        extra_parallel_placement_innermost=args.train.accelerator.extra_parallel_placement_innermost,
        extra_parallel_names=args.train.accelerator.extra_parallel_names,
        dp_mode=args.train.accelerator.fsdp_config.fsdp_mode,
    )

    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        init_device=args.train.init_device,
        ops_implementation=args.model.ops_implementation,
    )
    print(f"Model Class: {type(model)}")


_MODEL_NAME_FOR_OVERRIDES = {
    # Maps the parametrize model_path to the short name used by
    # _NPU_PER_MODEL_OVERRIDES — only matters on NPU.
    "qwen2vl-7b-instruct": "qwen2vl",
    "llama3_2-3b-instruct": None,
}


@pytest.mark.parametrize(
    "model_path",
    list(_MODEL_NAME_FOR_OVERRIDES.keys()),
)
def test_model_loader(model_path):
    from tests.tools.training_utils import resolve_ops_overrides

    port = 12345 + random.randint(0, 100)

    command = [
        "torchrun",
        "--nproc_per_node=4",
        f"--master_port={port}",
        "tests/utils/test_model_loader.py",
        f"--model.config_path={model_path}",
        "--data.train_path=tests",
        "--train.checkpoint.output_dir=.tests/cache",
        f"--train.init_device={get_device_type()}",
        # On NPU the dataclass defaults raise at parse time; pin to the
        # NPU-supported backend per op. No-op on GPU.
        *resolve_ops_overrides(_MODEL_NAME_FOR_OVERRIDES[model_path]),
    ]

    result = subprocess.run(command, check=True)
    assert result.returncode == 0


if __name__ == "__main__":
    args = parse_args(Arguments)
    run_environ_meter(args)
