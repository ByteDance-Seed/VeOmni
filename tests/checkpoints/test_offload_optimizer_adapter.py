# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""Precision and failure propagation checks for registered offload optimizers."""

from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.optim.optimizer import MultiOptimizer


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _dtype(optimizer, param, field):
    return torch.float32 if field == "step" else optimizer._state_dtype(param, field)


def _offloaded(optimizer, param):
    return id(param) in optimizer._offloaded


@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize("kind", ["master", "offload"])
def test_bf16_moments_and_fp32_step_resume_exactly(tmp_path, initialized, kind):
    from veomni.optim.cpu_offload_adamw import CPUOffloadAdamW
    from veomni.optim.master_fp32_adamw import MasterFP32AdamW

    def build():
        torch.manual_seed(117)
        model = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
        if kind == "master":
            leaf = MasterFP32AdamW([model.weight], lr=0.01)
        else:
            leaf = CPUOffloadAdamW(
                [{"params": [model.weight], "cpu_offload": True}],
                lr=0.01,
                cpu_moment_dtype=torch.bfloat16,
            )
        return model, MultiOptimizer(model, {"leaf": leaf}, ["leaf"])

    source, optimizer = build()
    if initialized:
        source.weight.grad = torch.ones_like(source.weight)
        optimizer.step()
        optimizer.zero_grad()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": source, "optimizer": optimizer}, parallel_state=parallel)
    target, restored = build()
    DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)
    source.weight.grad = torch.full_like(source.weight, 0.125)
    target.weight.grad = source.weight.grad.clone()
    optimizer.step()
    restored.step()
    torch.testing.assert_close(target.weight, source.weight, rtol=0, atol=0)
    left = optimizer.optimizers_dict["leaf"].state[source.weight]
    right = restored.optimizers_dict["leaf"].state[target.weight]
    for key in ("step", "master_weight", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    assert left["step"].dtype == torch.float32
    assert left["master_weight"].dtype == torch.float32
    assert left["exp_avg"].dtype == left["exp_avg_sq"].dtype == torch.bfloat16


@pytest.mark.parametrize("initialize_weight", [False, True])
def test_registered_sparse_adamw_adapter_preserves_lazy_state(tmp_path, initialize_weight):
    from contextlib import contextmanager

    from veomni.optim.checkpoint_adapter import register_rank_local_adamw
    from veomni.optim.cpu_offload_adamw import CPUOffloadAdamW

    class CustomOptimizer(CPUOffloadAdamW):
        def state_dict(self):
            if not getattr(self, "active", False):
                raise RuntimeError("custom raw format must not be used")
            return super().state_dict()

        def _prepare_load_state_dict(self, state_dict):
            if not getattr(self, "active", False):
                raise RuntimeError("custom raw restore must not be used")
            return state_dict

    @contextmanager
    def sparse_context(optimizer):
        optimizer.active = True
        try:
            yield
        finally:
            del optimizer.active

    register_rank_local_adamw(CustomOptimizer, sparse_context, lambda optimizer: ("custom-v1",), _dtype, _offloaded)

    def build():
        torch.manual_seed(31)
        model = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
        leaf = CustomOptimizer([{"params": list(model.parameters()), "cpu_offload": True}], lr=0.01)
        return model, MultiOptimizer(model, {"leaf": leaf}, ["leaf"])

    source, optimizer = build()
    if initialize_weight:
        source.weight.grad = torch.ones_like(source.weight)
        optimizer.step()
        optimizer.zero_grad()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": source, "optimizer": optimizer}, parallel_state=parallel)
    target, restored = build()
    DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)
    assert target.bias not in restored.optimizers_dict["leaf"].state
    for left, right in zip(source.parameters(), target.parameters(), strict=True):
        left.grad = torch.full_like(left, 0.125)
        right.grad = left.grad.clone()
    optimizer.step()
    restored.step()
    for left, right in zip(source.parameters(), target.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_registered_adapter_rejects_changed_algorithm_identity_before_loading(tmp_path):
    from contextlib import nullcontext

    from veomni.optim.checkpoint_adapter import register_rank_local_adamw
    from veomni.optim.cpu_offload_adamw import CPUOffloadAdamW

    class ScaledOptimizer(CPUOffloadAdamW):
        def __init__(self, params, factor):
            super().__init__(params)
            self.factor = factor

    register_rank_local_adamw(
        ScaledOptimizer, lambda _optimizer: nullcontext(), lambda optimizer: (optimizer.factor,), _dtype, _offloaded
    )

    def build(factor):
        model = torch.nn.Linear(2, 2)
        leaf = ScaledOptimizer([model.weight], factor)
        return model, MultiOptimizer(model, {"leaf": leaf}, ["leaf"])

    source, optimizer = build(1.0)
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": source, "optimizer": optimizer}, parallel_state=parallel)
    target, restored = build(2.0)
    before = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="identity|layout|checkpoint"):
        DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)
    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)


def _adapter_load_failure_worker(rank, checkpoint, endpoint):
    from contextlib import nullcontext
    from datetime import timedelta

    import torch.distributed as dist

    from veomni.optim.checkpoint_adapter import register_rank_local_adamw
    from veomni.optim.cpu_offload_adamw import CPUOffloadAdamW

    class FirstOptimizer(CPUOffloadAdamW):
        def _prepare_load_state_dict(self, state_dict):
            if self.fail_load:
                raise RuntimeError("injected first-leaf load failure")
            return state_dict

    register_rank_local_adamw(
        FirstOptimizer, lambda _opt: nullcontext(), lambda _opt: ("test-v1",), _dtype, _offloaded
    )
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:

        def build(fail):
            torch.manual_seed(41)
            model = torch.nn.Linear(2, 2)
            first = FirstOptimizer([model.weight])
            first.fail_load = fail
            second = torch.optim.AdamW([model.bias])
            return model, MultiOptimizer(model, {"first": first, "second": second}, ["first", "second"])

        source, optimizer = build(False)
        source.weight.grad = torch.ones_like(source.weight)
        source.bias.grad = torch.ones_like(source.bias)
        optimizer.step()
        optimizer.zero_grad()
        parallel = SimpleNamespace(dp_mode="ddp")
        DistributedCheckpointer.save(checkpoint, {"model": source, "optimizer": optimizer}, parallel_state=parallel)
        target, restored = build(rank == 1)
        with pytest.raises(RuntimeError, match="first-leaf|loading optimizer leaf"):
            DistributedCheckpointer.load(checkpoint, {"model": target, "optimizer": restored}, parallel_state=parallel)
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_registered_adapter_one_rank_load_failure_stops_all_ranks(tmp_path):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_adapter_load_failure_worker, args=(str(tmp_path),), nprocs=2)
