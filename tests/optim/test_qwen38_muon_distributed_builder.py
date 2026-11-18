# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.optim import optimizer as factory


def _worker(rank, mode, endpoint):
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        model = torch.nn.Module()
        model.o_proj = torch.nn.Linear(4, 4, bias=False)
        model.ngram_embedding = torch.nn.Embedding(8, 4)
        factory.get_parallel_state = lambda: SimpleNamespace(
            dp_mode="ddp", any_extra_parallel_enabled=False, device_mesh=None
        )
        if rank == 1:

            def fail_locally(*_args, **_kwargs):
                raise OSError("injected local optimizer failure")

            if mode == "builder":
                factory.select_cpu_offload_groups = fail_locally
            else:
                factory._qwen38_cuda_domain = fail_locally
        with pytest.raises((RuntimeError, OSError), match="Qwen3.8|optimizer failure"):
            factory.build_optimizer(
                model, optimizer_type="qwen38_muon", weight_decay=0.0, cpu_offload_param_patterns=["ngram_embedding"]
            )
        done = torch.tensor(1)
        dist.all_reduce(done)
        assert done.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("mode", ["builder", "domain"])
def test_one_rank_qwen_builder_failure_stops_all_ranks(mode):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_worker, args=(mode,), nprocs=2)
