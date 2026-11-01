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


def _type_mismatch_worker(rank, endpoint):
    import contextlib

    import veomni.models.model_runtime as runtime_module

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:

        class TestRuntime(runtime_module.VeOmniModelRuntime):
            def setup(self):
                pass

            def _build_model(self):
                self.model = torch.nn.Linear(4, 4)

            def _freeze_model_module(self):
                pass

            def _build_parallelized_model(self):
                pass

        runtime_module.use_parallel_state = lambda *_: contextlib.nullcontext()
        args = SimpleNamespace(optimizer=SimpleNamespace(type="qwen38_muon" if rank else "adamw"))
        train = SimpleNamespace(checkpoint=SimpleNamespace(extra_parallel_canonical_layout=False))
        with pytest.raises(RuntimeError, match="optimizer type differs across ranks"):
            TestRuntime(args, train=train)
        done = torch.tensor(1)
        dist.all_reduce(done)
        assert done.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_optimizer_type_mismatch_stops_all_ranks():
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_type_mismatch_worker, args=(), nprocs=2)
