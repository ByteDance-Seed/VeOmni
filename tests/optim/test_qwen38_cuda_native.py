# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""Native multi-GPU IEEE and unsafe precision guard for Qwen Muon."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_capabilities


def _cuda_policy_worker(rank, rendezvous, output_dir):
    from veomni.optim.optimizer import MultiOptimizer
    from veomni.optim.qwen38_muon import Qwen38MuonAdamW, add_qwen38_plans

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        torch.set_float32_matmul_precision("highest")
        model = torch.nn.Module()
        model.down_proj = torch.nn.Linear(4, 4, bias=False, device=f"cuda:{rank}", dtype=torch.float32)
        initial = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4) / 20
        gradient = torch.tensor(
            [
                [0.25, -0.5, 0.125, 0.0625],
                [0.5, 0.125, -0.25, 0.375],
                [0.125, 0.375, 0.5, -0.25],
                [0.5, -0.125, 0.25, 0.125],
            ],
            dtype=torch.float32,
        )
        with torch.no_grad():
            model.down_proj.weight.copy_(initial.to(device=torch.device("cuda", rank)))
        param = model.down_proj.weight
        plan = add_qwen38_plans(model, [{"params": [param]}])
        leaf = Qwen38MuonAdamW(plan, lr=1e-3, cuda_precision_group=dist.group.WORLD)
        optimizer = MultiOptimizer(
            model, {"muon": leaf}, ["muon"], qwen38_cuda_guard=True, qwen38_cuda_group=dist.group.WORLD
        )
        reference = torch.nn.Module()
        reference.down_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.float32)
        with torch.no_grad():
            reference.down_proj.weight.copy_(initial)
        ref_param = reference.down_proj.weight
        ref = Qwen38MuonAdamW(add_qwen38_plans(reference, [{"params": [ref_param]}]), lr=1e-3)
        param.grad = gradient.to(device=torch.device("cuda", rank))
        ref_param.grad = gradient
        optimizer.step()
        ref.step()
        torch.testing.assert_close(
            leaf.state[param]["master_weight"].cpu(), ref.state[ref_param]["master_weight"], rtol=1e-5, atol=1e-7
        )
        torch.testing.assert_close(
            leaf.state[param]["exp_avg"].cpu(), ref.state[ref_param]["exp_avg"], rtol=1e-6, atol=1e-8
        )
        before_param = param.detach().clone()
        before_state = {key: value.detach().clone() for key, value in leaf.state[param].items()}
        if rank == 1:
            torch.set_float32_matmul_precision("medium")
        with pytest.raises(RuntimeError):
            optimizer.step()
        assert torch.equal(param, before_param)
        assert before_state.keys() == leaf.state[param].keys()
        for key, before in before_state.items():
            assert torch.equal(leaf.state[param][key], before)
        torch.save({"passed": True}, Path(output_dir) / f"cuda-rank{rank}.pt")
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2))
def test_native_cuda_ieee_update_and_cross_rank_unsafe_policy_fails_before_update(tmp_path):
    capabilities = runner_capabilities()
    if capabilities.backend == "cpu":
        pytest.skip("Requires 2 native CUDA devices")
    assert capabilities.backend == "cuda", "This test belongs to CUDA CI only"
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, "Assigned CUDA devices are unavailable"
    mp.spawn(_cuda_policy_worker, args=(str(tmp_path / "rdzv"), str(tmp_path)), nprocs=2, join=True)
    assert [torch.load(tmp_path / f"cuda-rank{rank}.pt", weights_only=True) for rank in range(2)] == [
        {"passed": True}
    ] * 2
