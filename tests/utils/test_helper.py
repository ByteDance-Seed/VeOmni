import os
import random
import subprocess
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from transformers import Qwen2Config

from veomni.arguments import DataArguments, ModelArguments, TrainingArguments, VeOmniArguments, parse_args
from veomni.distributed.parallel_state import init_parallel_state
from veomni.lora import VeOmniLoraConfig
from veomni.utils import helper
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


logger = helper.create_logger(__name__)


def test_environ_meter_passes_supported_lora_config(monkeypatch):
    calls = []

    class FakeFlopsCounter:
        supports_lora = True

        def __init__(self, config):
            self.config = config

        def estimate_flops(self, batch_seqlens, delta_time, **kwargs):
            calls.append((batch_seqlens, delta_time, kwargs))
            return 1.0, 2.0

    fake_device = SimpleNamespace(
        max_memory_allocated=lambda: 0,
        max_memory_reserved=lambda: 0,
        memory_stats=lambda: {"num_alloc_retries": 0},
    )
    fake_memory = SimpleNamespace(used=0, available=0, percent=0)
    monkeypatch.setattr(helper, "VeomniFlopsCounter", FakeFlopsCounter)
    monkeypatch.setattr(helper.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(helper, "all_reduce", lambda values, **kwargs: values)
    monkeypatch.setattr(helper, "get_torch_device", lambda: fake_device)
    monkeypatch.setattr(helper.psutil, "virtual_memory", lambda: fake_memory)

    meter = helper.EnvironMeter(
        config=SimpleNamespace(model_type="qwen3"),
        global_batch_size=1,
        empty_cache_steps=0,
        parallel_state=SimpleNamespace(dp_group=None),
    )
    meter.batch_seqlens = [12, 5]
    meter.images_seqlens = [16]
    lora_config = VeOmniLoraConfig(r=8, target_modules=["q_proj"])

    meter.step(
        delta_time=0.5,
        global_step=1,
        lora_config=lora_config,
        freeze_vit=True,
    )

    assert calls == [
        (
            [12, 5],
            0.5,
            {
                "images_seqlens": [16],
                "lora_config": lora_config,
                "freeze_vit": True,
            },
        )
    ]

    calls.clear()
    meter.supports_lora_flops = False
    meter.batch_seqlens = [7]
    with patch("veomni.utils.helper.logger.warning_rank0") as warning:
        metrics = meter.step(delta_time=0.25, global_step=2, lora_config=lora_config, freeze_vit=True)

        meter.batch_seqlens = [8]
        meter.step(delta_time=0.25, global_step=3, lora_config=lora_config)

    assert calls == [([7], 0.25, {}), ([8], 0.25, {})]
    assert metrics["flops_achieved(T)"] == 0
    assert metrics["mfu"] == 0
    warning.assert_called_once()


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
        async_enabled=args.train.accelerator.enable_async,
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
        "--train.checkpoint.output_dir=.tests/cache",
    ]

    result = subprocess.run(command, check=True)
    assert result.returncode == 0


if __name__ == "__main__":
    args = parse_args(Arguments)
    run_environ_meter(args)
