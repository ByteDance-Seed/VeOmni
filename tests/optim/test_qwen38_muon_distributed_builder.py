# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.optim import optimizer as factory


def _worker(rank, endpoint):
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        model = torch.nn.Module()
        model.o_proj = torch.nn.Linear(4, 4, bias=False)
        model.ngram_embedding = torch.nn.Embedding(8, 4)
        factory.get_parallel_state = lambda: SimpleNamespace(
            dp_mode="ddp", any_extra_parallel_enabled=False, device_mesh=None
        )
        pattern = "missing" if rank else "ngram_embedding"
        with pytest.raises((RuntimeError, ValueError), match="building Qwen3.8|match no trainable"):
            factory.build_optimizer(
                model, optimizer_type="qwen38_muon", weight_decay=0.0, cpu_offload_param_patterns=[pattern]
            )
        done = torch.tensor(1)
        dist.all_reduce(done)
        assert done.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_one_rank_qwen_builder_failure_stops_all_ranks():
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_worker, args=(), nprocs=2)
