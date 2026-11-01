# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.checkpoint.rank_local_optimizer import (
    local_optimizer_state,
    recover_frozen_optimizer_ownership,
    restore_optimizer_state,
    verify_replicated_model_shards,
)
from veomni.distributed.parallel_plan import SpecInfo
from veomni.optim.cpu_offload_adamw import CPUOffloadAdamW
from veomni.optim.muon import DistributedMuon
from veomni.optim.optimizer import MultiOptimizer


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _roundtrip_worker(rank, root, inactive_owner):
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/init", rank=rank, world_size=4, timeout=timedelta(seconds=30)
    )
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("fsdp", "ep"))
        fsdp, owner = mesh["fsdp"], mesh.get_local_rank("ep")
        local = distribute_tensor(torch.arange(16, dtype=torch.float32).reshape(4, 4) + 100 * owner, fsdp, [Shard(0)])
        source = {"param_groups.expert.lr": 0.005 + 0.001 * owner}
        if not inactive_owner or owner == 0:
            source.update(
                {
                    "state.expert.step": torch.tensor(7 if owner == 0 else 3),
                    "state.expert.exp_avg": local,
                    "state.expert.exp_avg_sq": local + 0.5,
                }
            )
        dcp.save({"optimizer": local_optimizer_state(source, rank)}, checkpoint_id=f"{root}/checkpoint")
        target = {"param_groups.expert.lr": 0.0}
        if not inactive_owner or owner == 0:
            target.update(
                {
                    "state.expert.step": torch.tensor(0),
                    "state.expert.exp_avg": torch.zeros_like(local),
                    "state.expert.exp_avg_sq": torch.zeros_like(local),
                }
            )
        checkpoint = {"optimizer": local_optimizer_state(target, rank)}
        dcp.load(checkpoint, checkpoint_id=f"{root}/checkpoint")
        restored = restore_optimizer_state(checkpoint["optimizer"], target, rank)
        assert set(restored) == set(source)
        for name in source:
            if torch.is_tensor(source[name]):
                actual = restored[name].to_local() if hasattr(restored[name], "to_local") else restored[name]
                expected = source[name].to_local() if hasattr(source[name], "to_local") else source[name]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                assert restored[name] == source[name]
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=4, device="cpu"))
@pytest.mark.parametrize("inactive_owner", [False, True])
def test_rank_local_state_keeps_owner_steps_and_absence(tmp_path, inactive_owner):
    mp.spawn(_roundtrip_worker, args=(str(tmp_path), inactive_owner), nprocs=4, join=True)


def _optimizer_resume_worker(rank, root, inactive_owner):
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/optimizer-init", rank=rank, world_size=4, timeout=timedelta(seconds=40)
    )
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ep_fsdp", "ep"))
        fsdp, owner = mesh["ep_fsdp"], mesh.get_local_rank("ep")
        parallel = SimpleNamespace(
            dp_mode="fsdp2", extra_parallel_names=("ep",), extra_parallel_fsdp_device_mesh={"ep": mesh}
        )

        def build():
            model = torch.nn.Module()
            values = torch.arange(16, dtype=torch.float32).reshape(4, 4) + 100 * owner
            model.register_parameter("weight", torch.nn.Parameter(distribute_tensor(values, fsdp, [Shard(1)])))
            model._fqn2spec_info = {"weight": SpecInfo("ep", Shard(0), "weight", mesh, 1)}
            leaf = torch.optim.AdamW([model.weight], lr=0.01)
            return model, MultiOptimizer(model, {"expert": leaf}, ["expert"])

        source, optimizer = build()
        if not inactive_owner or owner == 0:
            source.weight.grad = torch.ones_like(source.weight)
            optimizer.step()
            optimizer.zero_grad()
            if owner == 0:
                source.weight.grad = torch.full_like(source.weight, 0.5)
                optimizer.step()
                optimizer.zero_grad()
        DistributedCheckpointer.save(
            f"{root}/training", {"model": source, "optimizer": optimizer}, parallel_state=parallel
        )
        target, restored = build()
        DistributedCheckpointer.load(
            f"{root}/training", {"model": target, "optimizer": restored}, parallel_state=parallel
        )
        torch.testing.assert_close(target.weight.to_local(), source.weight.to_local(), rtol=0, atol=0)
        source_state = optimizer.optimizers_dict["expert"].state.get(source.weight, {})
        target_state = restored.optimizers_dict["expert"].state.get(target.weight, {})
        assert set(source_state) == set(target_state)
        for key, value in source_state.items():
            expected = value.to_local() if hasattr(value, "to_local") else value
            actual = target_state[key].to_local() if hasattr(target_state[key], "to_local") else target_state[key]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        source.weight.grad = torch.full_like(source.weight, 0.2)
        target.weight.grad = torch.full_like(target.weight, 0.2)
        optimizer.step()
        restored.step()
        torch.testing.assert_close(target.weight.to_local(), source.weight.to_local(), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=4, device="cpu"))
@pytest.mark.parametrize("inactive_owner", [False, True])
def test_high_level_rank_owned_optimizer_cold_resume(tmp_path, inactive_owner):
    mp.spawn(_optimizer_resume_worker, args=(str(tmp_path), inactive_owner), nprocs=4, join=True)


def _cold_optimizer_worker(rank, root, inactive_owner, phase, kind, canonical, same_axis):
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/{phase}-init", rank=rank, world_size=4, timeout=timedelta(seconds=50)
    )
    try:
        if kind == "muon" and same_axis:
            # Gloo lacks list all_to_all; emulate the same rank-to-rank tensor
            # exchange with its supported all_to_all_single for this CPU test.
            def gloo_all_to_all(outputs, inputs, group=None, async_op=False):
                assert not async_op
                send_sizes = [tensor.numel() for tensor in inputs]
                recv_sizes = [tensor.numel() for tensor in outputs]
                send = torch.cat([tensor.reshape(-1) for tensor in inputs])
                recv = torch.empty(sum(recv_sizes), dtype=send.dtype, device=send.device)
                dist.all_to_all_single(recv, send, recv_sizes, send_sizes, group=group)
                offset = 0
                for tensor, size in zip(outputs, recv_sizes, strict=True):
                    tensor.copy_(recv[offset : offset + size].reshape(tensor.shape))
                    offset += size

            dist.all_to_all = gloo_all_to_all
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ep_fsdp", "ep"))
        fsdp, owner = mesh["ep_fsdp"], mesh.get_local_rank("ep")
        parallel = SimpleNamespace(
            dp_mode="fsdp2", extra_parallel_names=("ep",), extra_parallel_fsdp_device_mesh={"ep": mesh}
        )
        model = torch.nn.Module()
        values = torch.arange(16, dtype=torch.float32).reshape(4, 4) + 100 * owner
        if kind == "offload":
            values = values.to(torch.bfloat16)
        fsdp_axis = 0 if same_axis else 1
        model.register_parameter("weight", torch.nn.Parameter(distribute_tensor(values, fsdp, [Shard(fsdp_axis)])))
        model._fqn2spec_info = {"weight": SpecInfo("ep", Shard(0), "weight", mesh, fsdp_axis, None, (8, 4), canonical)}
        if kind == "offload":
            leaf = CPUOffloadAdamW([{"params": [model.weight], "cpu_offload": True}], lr=0.01)
        elif kind == "muon":
            leaf = DistributedMuon([model.weight], lr=0.01, ns_implementation="std")
        else:
            leaf = torch.optim.AdamW([model.weight], lr=0.01)
        leaves = {"expert": leaf}
        if kind == "muon":
            model.register_parameter("bias", torch.nn.Parameter(torch.ones(2)))
            model._fqn2spec_info["bias"] = SpecInfo("ep", Replicate(), "bias", mesh)
            leaves["fallback"] = torch.optim.AdamW([model.bias], lr=0.001)
        optimizer = MultiOptimizer(model, leaves, list(leaves))

        def snapshot():
            state = optimizer.optimizers_dict["expert"].state.get(model.weight, {})
            result = {
                "weight": model.weight.to_local().detach().clone(),
                "state": {
                    key: (value.to_local() if hasattr(value, "to_local") else value).detach().clone()
                    for key, value in state.items()
                },
            }
            if kind == "muon":
                result["bias"] = model.bias.detach().clone()
                bias_state = optimizer.optimizers_dict["fallback"].state.get(model.bias, {})
                result["bias_state"] = {key: value.detach().clone() for key, value in bias_state.items()}
            return result

        def advance(gradient, *, include_bias=True):
            model.weight.grad = torch.full_like(model.weight, gradient)
            if kind == "muon" and include_bias:
                model.bias.grad = torch.full_like(model.bias, 0.03)
            optimizer.step()
            optimizer.zero_grad()

        if phase == "save":
            for _ in range(2 if owner == 0 else (0 if inactive_owner else 1)):
                advance(1.0, include_bias=False)
            if kind == "muon":
                model.bias.grad = torch.full_like(model.bias, 0.03)
                optimizer.step()
                optimizer.zero_grad()
            DistributedCheckpointer.save(
                f"{root}/cold",
                {"model": model, "optimizer": optimizer},
                parallel_state=parallel,
            )
            before = snapshot()
            advance(0.2)
            torch.save({"before": before, "after": snapshot()}, f"{root}/reference-{rank}.pt")
        else:
            DistributedCheckpointer.load(
                f"{root}/cold", {"model": model, "optimizer": optimizer}, parallel_state=parallel
            )
            reference = torch.load(f"{root}/reference-{rank}.pt", weights_only=True)
            torch.testing.assert_close(snapshot(), reference["before"], rtol=0, atol=0)
            advance(0.2)
            torch.testing.assert_close(snapshot(), reference["after"], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=4, device="cpu"))
@pytest.mark.parametrize("inactive_owner", [False, True])
@pytest.mark.parametrize("kind", ["adam", "offload", "muon"])
@pytest.mark.parametrize("canonical", [False, True])
def test_rank_owned_optimizer_fresh_process_resume(tmp_path, inactive_owner, kind, canonical):
    mp.spawn(
        _cold_optimizer_worker,
        args=(str(tmp_path), inactive_owner, "save", kind, canonical, False),
        nprocs=4,
        join=True,
    )
    mp.spawn(
        _cold_optimizer_worker,
        args=(str(tmp_path), inactive_owner, "load", kind, canonical, False),
        nprocs=4,
        join=True,
    )


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=4, device="cpu"))
def test_canonical_same_axis_muon_fresh_process_resume(tmp_path):
    mp.spawn(_cold_optimizer_worker, args=(str(tmp_path), False, "save", "muon", True, True), nprocs=4, join=True)
    mp.spawn(_cold_optimizer_worker, args=(str(tmp_path), False, "load", "muon", True, True), nprocs=4, join=True)


def _replica_check_worker(rank, root):
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/replica-init", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:
        with pytest.raises(RuntimeError, match="replicas differ"):
            verify_replicated_model_shards({"bias": torch.tensor([float(rank)])})
        with pytest.raises(RuntimeError, match="replicas differ"):
            verify_replicated_model_shards({"extra": {"scale": torch.tensor(float(rank))}})
        with pytest.raises(RuntimeError, match="replicas differ"):
            verify_replicated_model_shards({"extra": {"scale": float(rank)}})
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_divergent_replicas_rejected_before_checkpoint_deduplication(tmp_path):
    mp.spawn(_replica_check_worker, args=(str(tmp_path),), nprocs=2, join=True)


def _global_tensor_metadata_mismatch_worker(rank, root, mismatch):
    from torch.distributed.tensor import DTensor

    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/global-metadata-init",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        mesh = init_device_mesh("cpu", (2,))
        dtype = torch.bfloat16 if mismatch == "dtype" and rank else torch.float32
        global_size = 6 if mismatch == "shape" and rank else 4
        local = torch.ones(global_size // 2, dtype=dtype)
        tensor = DTensor.from_local(local, mesh, [Shard(0)], shape=torch.Size([global_size]), stride=(1,))
        with pytest.raises(RuntimeError, match="global dtype or shape"):
            verify_replicated_model_shards({"expert": tensor})
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("mismatch", ["dtype", "shape"])
def test_checkpoint_global_tensor_metadata_mismatch_fails_all_ranks(tmp_path, mismatch):
    mp.spawn(_global_tensor_metadata_mismatch_worker, args=(str(tmp_path), mismatch), nprocs=2, join=True)


def _overlapping_tensor_shards_worker(rank, root, same_offset):
    from torch.distributed.tensor import DTensor

    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/overlap-init",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        mesh = init_device_mesh("cpu", (2,))
        shard = rank == (0 if same_offset else 1)
        placement = Shard(0) if shard else Replicate()
        tensor = DTensor.from_local(
            torch.ones(2 if shard else 4), mesh, [placement], shape=torch.Size([4]), stride=(1,), run_check=False
        )
        with pytest.raises(RuntimeError, match="overlapping nonidentical shards"):
            verify_replicated_model_shards({"expert": tensor})
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("same_offset", [False, True])
def test_checkpoint_overlapping_shard_geometries_fail_all_ranks(tmp_path, same_offset):
    mp.spawn(_overlapping_tensor_shards_worker, args=(str(tmp_path), same_offset), nprocs=2, join=True)


def test_checkpoint_rejects_empty_shard_with_same_offset_as_nonempty_shard():
    from veomni.checkpoint.rank_local_optimizer import _reject_overlapping_geometries

    with pytest.raises(RuntimeError, match="overlapping nonidentical shards"):
        _reject_overlapping_geometries("expert", [((0,), (0,)), ((0,), (2,))])


def _owner_trainability_worker(rank, root, within_owner_mismatch):
    from veomni.checkpoint.optimizer_resume_layout import coordinated_multi_layout

    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/owner-trainability-init",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ple_fsdp", "ple"))
        model = torch.nn.Module()
        values = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        model.weight = torch.nn.Parameter(distribute_tensor(values, mesh, [Shard(1), Shard(0)]))
        model._fqn2spec_info = {"weight": SpecInfo("ple", Shard(0), "weight", mesh, 1, 1, (8, 8))}
        optimizer = MultiOptimizer(model, {"leaf": torch.optim.AdamW([model.weight])}, ["leaf"])
        owner = mesh.get_local_rank("ple")
        replica = mesh.get_local_rank("ple_fsdp")
        if owner == 1 and (not within_owner_mismatch or replica == 1):
            model.weight.requires_grad_(False)
        if within_owner_mismatch:
            with pytest.raises(RuntimeError, match="trainability differs"):
                coordinated_multi_layout(model, optimizer, writing=True)
        else:
            coordinated_multi_layout(model, optimizer, writing=True)
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 4
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=4, device="cpu"))
@pytest.mark.parametrize("within_owner_mismatch", [False, True])
def test_trainability_may_differ_between_owners_but_not_replicas(tmp_path, within_owner_mismatch):
    mp.spawn(_owner_trainability_worker, args=(str(tmp_path), within_owner_mismatch), nprocs=4, join=True)


def _frozen_resume_worker(rank, root, phase):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/frozen-{phase}-init",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=40),
    )
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ep_fsdp", "ep"))
        fsdp, owner = mesh["ep_fsdp"], mesh.get_local_rank("ep")
        parallel = SimpleNamespace(
            dp_mode="fsdp2", extra_parallel_names=("ep",), extra_parallel_fsdp_device_mesh={"ep": mesh}
        )
        model = torch.nn.Module()
        values = torch.arange(16, dtype=torch.float32).reshape(4, 4) + 100 * owner
        model.register_parameter("weight", torch.nn.Parameter(distribute_tensor(values, fsdp, [Shard(1)])))
        model.register_parameter("bias", torch.nn.Parameter(torch.ones(2)))
        model._fqn2spec_info = {"weight": SpecInfo("ep", Shard(0), "weight", mesh, 1)}

        def build():
            leaf = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
            return MultiOptimizer(model, {"expert": leaf}, ["expert"])

        if phase == "save":
            optimizer = build()
            model.weight.grad = torch.ones_like(model.weight)
            model.bias.grad = torch.full_like(model.bias, 0.1)
            optimizer.step()
            optimizer.zero_grad()
            model.weight.requires_grad_(False)
            DistributedCheckpointer.save(
                f"{root}/frozen", {"model": model, "optimizer": optimizer}, parallel_state=parallel
            )
            model.weight.requires_grad_(True)
            model.weight.grad = torch.full_like(model.weight, 0.2)
            optimizer.step()
            torch.save(model.weight.to_local().detach().clone(), f"{root}/frozen-ref-{rank}.pt")
        else:
            model.weight.requires_grad_(False)
            with recover_frozen_optimizer_ownership(model, f"{root}/frozen"):
                assert model.weight.requires_grad
                optimizer = build()
            assert not model.weight.requires_grad
            DistributedCheckpointer.load(
                f"{root}/frozen", {"model": model, "optimizer": optimizer}, parallel_state=parallel
            )
            model.weight.requires_grad_(True)
            model.weight.grad = torch.full_like(model.weight, 0.2)
            optimizer.step()
            expected = torch.load(f"{root}/frozen-ref-{rank}.pt", weights_only=True)
            torch.testing.assert_close(model.weight.to_local(), expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=4, device="cpu"))
def test_frozen_owned_parameter_rebuilt_on_fresh_process(tmp_path):
    mp.spawn(_frozen_resume_worker, args=(str(tmp_path), "save"), nprocs=4, join=True)
    mp.spawn(_frozen_resume_worker, args=(str(tmp_path), "load"), nprocs=4, join=True)


def _missing_rank_payload_worker(rank, root):
    from pathlib import Path

    from veomni.checkpoint.layout import optimizer_dir
    from veomni.checkpoint.rank_local_optimizer import rank_scalar_key

    dist.init_process_group(
        "gloo", init_method=f"file://{root}/corrupt-init", rank=rank, world_size=2, timeout=timedelta(seconds=25)
    )
    try:
        mesh = init_device_mesh("cpu", (1, 2), mesh_dim_names=("ep_fsdp", "ep"))
        owner = mesh.get_local_rank("ep")
        parallel = SimpleNamespace(
            dp_mode="fsdp2", extra_parallel_names=("ep",), extra_parallel_fsdp_device_mesh={"ep": mesh}
        )

        def build():
            model = torch.nn.Module()
            tensor = torch.full((2, 2), float(owner))
            model.register_parameter(
                "weight", torch.nn.Parameter(distribute_tensor(tensor, mesh["ep_fsdp"], [Shard(1)]))
            )
            model._fqn2spec_info = {"weight": SpecInfo("ep", Shard(0), "weight", mesh, 1)}
            leaf = torch.optim.AdamW([model.weight], lr=0.01)
            return model, MultiOptimizer(model, {"expert": leaf}, ["expert"])

        source, optimizer = build()
        source.weight.grad = torch.ones_like(source.weight)
        optimizer.step()
        optimizer.zero_grad()
        checkpoint = f"{root}/corrupt"
        DistributedCheckpointer.save(checkpoint, {"model": source, "optimizer": optimizer}, parallel_state=parallel)
        if rank == 0:
            opt_path = optimizer_dir(checkpoint, "")
            metadata = dcp.FileSystemReader(opt_path).read_metadata()
            storage = next(
                item for index, item in metadata.storage_data.items() if index.fqn == f"optimizer.{rank_scalar_key(0)}"
            )
            Path(opt_path, storage.relative_path).unlink()
        dist.barrier()
        target, restored = build()
        before = target.weight.to_local().detach().clone()
        from torch.distributed.checkpoint.api import CheckpointException

        with pytest.raises((CheckpointException, RuntimeError)):
            DistributedCheckpointer.load(checkpoint, {"model": target, "optimizer": restored}, parallel_state=parallel)
        torch.testing.assert_close(target.weight.to_local(), before, rtol=0, atol=0)
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_one_rank_missing_optimizer_payload_fails_all_before_model_copy(tmp_path):
    mp.spawn(_missing_rank_payload_worker, args=(str(tmp_path),), nprocs=2, join=True)
