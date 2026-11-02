# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""Qwen Muon dormant state and distributed checkpoint recovery."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer, OptimizerState
from veomni.checkpoint.layout import optimizer_dir
from veomni.optim.optimizer import MultiOptimizer
from veomni.optim.qwen38_muon import Qwen38MuonAdamW, add_qwen38_plans


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _make_model_and_optimizer():
    model = torch.nn.Module()
    for name in ("first", "second"):
        block = torch.nn.Module()
        block.down_proj = torch.nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
        setattr(model, name, block)
    children = {}
    for name in ("first", "second"):
        param = getattr(model, name).down_proj.weight
        groups = add_qwen38_plans(model, [{"params": [param], "cpu_offload": False}])
        children[name] = Qwen38MuonAdamW(groups, lr=0.001)
    return model, MultiOptimizer(model, children, list(children))


def test_dcp_preserves_dormant_muon_recipe_without_creating_state(tmp_path):
    torch.manual_seed(123)
    model, optimizer = _make_model_and_optimizer()
    model.first.down_proj.weight.grad = torch.ones_like(model.first.down_proj.weight)
    optimizer.step()
    optimizer.zero_grad()
    dormant = optimizer.optimizers_dict["second"]
    ps = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=ps)
    metadata = dcp.FileSystemReader(optimizer_dir(str(tmp_path))).read_metadata().state_dict_metadata
    assert "optimizer.param_groups.second.down_proj.weight.qwen38_plan" in metadata
    assert not dormant.state
    target, restored = _make_model_and_optimizer()
    DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=ps)
    assert not restored.optimizers_dict["second"].state
    for source_param, target_param in zip(model.parameters(), target.parameters(), strict=True):
        source_param.grad = torch.ones_like(source_param)
        target_param.grad = source_param.grad.clone()
    optimizer.step()
    restored.step()
    for name in ("first", "second"):
        left = getattr(model, name).down_proj.weight
        right = getattr(target, name).down_proj.weight
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        source_state = optimizer.optimizers_dict[name].state[left]
        target_state = restored.optimizers_dict[name].state[right]
        assert source_state.keys() == target_state.keys()
        for key in source_state:
            torch.testing.assert_close(source_state[key], target_state[key], rtol=0, atol=0)


def test_frozen_muon_parameter_history_is_saved_and_recovered(tmp_path):
    model, optimizer = _make_model_and_optimizer()
    weight = model.first.down_proj.weight
    weight.grad = torch.ones_like(weight)
    optimizer.step()
    optimizer.zero_grad()
    weight.requires_grad_(False)
    ps = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=ps)
    target, restored = _make_model_and_optimizer()
    target_weight = target.first.down_proj.weight
    target_weight.requires_grad_(False)
    DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=ps)
    assert not target_weight.requires_grad
    for key, value in optimizer.optimizers_dict["first"].state[weight].items():
        torch.testing.assert_close(restored.optimizers_dict["first"].state[target_weight][key], value, rtol=0, atol=0)
    weight.requires_grad_(True)
    target_weight.requires_grad_(True)
    weight.grad = target_weight.grad = torch.ones_like(weight)
    optimizer.step()
    restored.step()
    torch.testing.assert_close(weight, target_weight, rtol=0, atol=0)


def test_frozen_muon_restore_target_mismatch_is_rejected(tmp_path):
    model, optimizer = _make_model_and_optimizer()
    model.first.down_proj.weight.grad = torch.ones_like(model.first.down_proj.weight)
    optimizer.step()
    optimizer.zero_grad()
    ps = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=ps)
    target, restored = _make_model_and_optimizer()
    target.first.down_proj.weight.requires_grad_(False)
    before = target.first.down_proj.weight.detach().clone()
    with pytest.raises(RuntimeError, match="layout|generation|identity"):
        DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=ps)
    torch.testing.assert_close(target.first.down_proj.weight, before, rtol=0, atol=0)
    assert not restored.optimizers_dict["first"].state


def _global_marker_worker(rank, rendezvous, checkpoint_dir, all_dormant):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        torch.manual_seed(13)
        model, optimizer = _make_model_and_optimizer()
        if not all_dormant:
            model.first.down_proj.weight.grad = torch.ones_like(model.first.down_proj.weight)
            optimizer.step()
            optimizer.zero_grad()
        ps = SimpleNamespace(dp_mode="ddp")
        DistributedCheckpointer.save(checkpoint_dir, {"model": model, "optimizer": optimizer}, parallel_state=ps)
        target, restored = _make_model_and_optimizer()
        DistributedCheckpointer.load(checkpoint_dir, {"model": target, "optimizer": restored}, parallel_state=ps)
        assert not restored.optimizers_dict["second"].state
        assert bool(restored.optimizers_dict["first"].state) is (not all_dormant)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_zero_optimizer_marker_uses_all_ranks(tmp_path):
    for all_dormant in (False, True):
        mp.start_processes(
            _global_marker_worker,
            args=(str(tmp_path / f"rdzv-{all_dormant}"), str(tmp_path / f"ckpt-{all_dormant}"), all_dormant),
            nprocs=2,
            join=True,
            start_method="fork",
        )


def _ordinary_adam_hook_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        calls = []
        hook = optimizer.register_state_dict_pre_hook(lambda _: calls.append(True))
        wrapper = OptimizerState(model, optimizer, parallel_state=SimpleNamespace(dp_mode="ddp"))
        wrapper.state_dict()
        assert calls == [True]
        hook.remove()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_ordinary_adam_state_dict_hooks_are_preserved(tmp_path):
    mp.start_processes(
        _ordinary_adam_hook_worker, args=(str(tmp_path / "rdzv-hook"),), nprocs=2, join=True, start_method="fork"
    )


def _empty_ple_owner_worker(rank, rendezvous, checkpoint_dir):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    from veomni.distributed.parallel_plan import SpecInfo

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        mesh = init_device_mesh("cpu", (2, 1), mesh_dim_names=("ple_fsdp", "ple"))

        def make():
            model = torch.nn.Module()
            model.ngram_embedding = torch.nn.Embedding(8, 8, dtype=torch.bfloat16)
            model.ngram_embedding.weight = torch.nn.Parameter(
                distribute_tensor(torch.ones(8, 8, dtype=torch.bfloat16), mesh, [Shard(1), Shard(0)])
            )
            model._fqn2spec_info = {
                "ngram_embedding.weight": SpecInfo("ple", Shard(0), "ngram_embedding.weight", mesh, 1, 1, (8, 8))
            }
            groups = add_qwen38_plans(model, [{"params": [model.ngram_embedding.weight], "weight_decay": 0.0}])
            leaf = Qwen38MuonAdamW(groups, lr=0.001)
            return model, leaf, MultiOptimizer(model, {"ple": leaf}, ["ple"])

        source, leaf, optimizer = make()
        ps = SimpleNamespace(
            dp_mode="fsdp2", extra_parallel_names=["ple"], extra_parallel_fsdp_device_mesh={"ple": mesh}
        )
        DistributedCheckpointer.save(checkpoint_dir, {"model": source, "optimizer": optimizer}, parallel_state=ps)
        assert not leaf.state
        target, restored_leaf, restored = make()
        DistributedCheckpointer.load(checkpoint_dir, {"model": target, "optimizer": restored}, parallel_state=ps)
        assert not restored_leaf.state
        p = source.ngram_embedding.weight
        q = target.ngram_embedding.weight
        p.grad = torch.ones_like(p)
        q.grad = torch.ones_like(q)
        optimizer.step()
        restored.step()
        torch.testing.assert_close(p.to_local(), q.to_local(), rtol=0, atol=0)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_empty_ple_owner_checkpoint_keeps_optimizer_lazy(tmp_path):
    mp.start_processes(
        _empty_ple_owner_worker,
        args=(str(tmp_path / "rdzv-ple"), str(tmp_path / "ckpt-ple")),
        nprocs=2,
        join=True,
        start_method="fork",
    )
