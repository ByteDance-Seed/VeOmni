# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""CPU regressions for proven MultiOptimizer checkpoint recovery."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.optim.lr_scheduler import build_lr_scheduler
from veomni.optim.muon import DistributedMuon
from veomni.optim.optimizer import AnyPrecisionAdamW, MultiOptimizer


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _model_and_optimizer(*, split=True):
    model = torch.nn.Linear(2, 2)
    if split:
        weight = torch.optim.AdamW([model.weight], lr=0.01, weight_decay=0.0)
        bias = torch.optim.AdamW([model.bias], lr=0.02, weight_decay=0.0)
        return model, MultiOptimizer(model, {"weight": weight, "bias": bias}, ["weight", "bias"])
    merged = torch.optim.AdamW([model.weight, model.bias], lr=0.5, weight_decay=0.0)
    return model, MultiOptimizer(model, {"merged": merged}, ["merged"])


def _load_preflight_failure_worker(rank, checkpoint, failure, endpoint):
    from datetime import timedelta
    from unittest.mock import patch

    import torch.distributed as dist

    import veomni.checkpoint.dcp_checkpointer as checkpoint_module

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(27)
        source, optimizer = _model_and_optimizer()
        for param in source.parameters():
            param.grad = torch.ones_like(param)
        optimizer.step()
        optimizer.zero_grad()
        parallel = SimpleNamespace(dp_mode="ddp")
        DistributedCheckpointer.save(checkpoint, {"model": source, "optimizer": optimizer}, parallel_state=parallel)
        target, restored = _model_and_optimizer()
        original_model = checkpoint_module._ProvenModelState
        original_optimizer = checkpoint_module._ProvenMultiOptimizerState
        original_reader = DistributedCheckpointer._create_storage_reader
        original_needs_marker = checkpoint_module._needs_extra_parallel_layout_marker
        injected = False

        def model_state(*args, **kwargs):
            nonlocal injected
            if rank == 1 and failure == "model_state":
                injected = True
                raise OSError("injected model state failure")
            return original_model(*args, **kwargs)

        def optimizer_state(*args, **kwargs):
            nonlocal injected
            if rank == 1 and failure == "optimizer_state":
                injected = True
                raise OSError("injected optimizer state failure")
            return original_optimizer(*args, **kwargs)

        def reader(path):
            import inspect

            nonlocal injected
            callers = {frame.function for frame in inspect.stack()}
            direct = not callers.intersection(
                {"_preflight_multi_optimizer_layout", "_prepare_multi_optimizer_saved_states"}
            )
            kind = "optimizer_reader" if str(path).endswith("optimizer") else "model_reader"
            if rank == 1 and direct and failure == kind:
                injected = True
                raise OSError(f"injected {kind} failure")
            return original_reader(path)

        def needs_marker(specs):
            nonlocal injected
            if failure == "layout_policy":
                import inspect

                callers = {frame.function for frame in inspect.stack() if frame.filename == checkpoint_module.__file__}
                if "_reject_unproven_extra_parallel_optimizer" in callers:
                    return False
                if rank == 1:
                    injected = True
                return rank == 1
            return original_needs_marker(specs)

        with (
            patch.object(checkpoint_module, "_ProvenModelState", side_effect=model_state),
            patch.object(checkpoint_module, "_ProvenMultiOptimizerState", side_effect=optimizer_state),
            patch.object(checkpoint_module, "_needs_extra_parallel_layout_marker", side_effect=needs_marker),
            patch.object(DistributedCheckpointer, "_create_storage_reader", side_effect=reader),
            pytest.raises((OSError, RuntimeError), match="injected|failed on another rank|marker policy differs"),
        ):
            DistributedCheckpointer.load(checkpoint, {"model": target, "optimizer": restored}, parallel_state=parallel)
        if rank == 1:
            assert injected, f"the {failure} fault did not reach its intended load stage"
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize(
    "failure", ["model_reader", "optimizer_reader", "model_state", "optimizer_state", "layout_policy"]
)
def test_proven_load_preflight_failure_stops_all_ranks(tmp_path, failure):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_load_preflight_failure_worker, args=(str(tmp_path), failure), nprocs=2)


def test_multi_optimizer_async_checkpoint_rejected_before_writes(tmp_path):
    model, optimizer = _model_and_optimizer()
    with pytest.raises(RuntimeError, match="require synchronous DCP save"):
        DistributedCheckpointer.save(
            str(tmp_path),
            {"model": model, "optimizer": optimizer},
            save_async=True,
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
    assert not list(tmp_path.iterdir())


def test_failed_final_load_barrier_does_not_certify_multi_optimizer():
    from unittest.mock import MagicMock, patch

    from veomni.models.checkpoint.manager import ModelCheckpointManager, _scheduler_recipe

    model, optimizer = _model_and_optimizer()
    leaf = next(iter(optimizer.optimizers_dict.values()))
    leaf.checkpoint_abort_file_backed_state = MagicMock()
    manager = object.__new__(ModelCheckpointManager)
    manager.runtime = SimpleNamespace(
        model=model, optimizer=optimizer, lr_scheduler=None, load_extra_state=lambda _: None
    )
    manager.load_dir = lambda: "/checkpoint/global_step_2"
    manager.wait_for_pending_save = lambda: None
    manager._checkpointer_kwargs = lambda: {}

    def load(_path, state, **_kwargs):
        state["_veomni_optimizer_identity_verified"] = True
        state["_veomni_optimizer_checkpoint_generation"] = "generation"
        state["extra_state"].update(
            _veomni_optimizer_global_step=2,
            _veomni_scheduler_recipe={"status": "supported", "value": _scheduler_recipe(None, optimizer)},
        )

    manager.checkpointer = SimpleNamespace(load=load)
    with patch("torch.distributed.barrier", side_effect=RuntimeError("injected final barrier failure")):
        with pytest.raises(RuntimeError, match="final barrier failure"):
            manager.load()
    assert manager._loaded_optimizer_global_step is None
    assert manager._loaded_optimizer_generation is None
    leaf.checkpoint_abort_file_backed_state.assert_called_once()


@pytest.mark.parametrize("mode", ["betas", "list", "nested"])
def test_shared_mutable_group_options_rejected_before_save(tmp_path, mode):
    model = torch.nn.Linear(2, 2)
    shared = {"values": [0.9, 0.95]} if mode == "nested" else [0.9, 0.95]
    groups = [
        {"params": [model.weight], "betas": (0.9, 0.95), "custom_recipe": {"shared": shared}},
        {"params": [model.bias], "betas": (0.9, 0.95), "custom_recipe": {"shared": shared}},
    ]
    if mode == "betas":
        for group in groups:
            group["betas"] = shared
            group.pop("custom_recipe")
    leaf = torch.optim.AdamW(groups, lr=0.01)
    optimizer = MultiOptimizer(model, {"leaf": leaf}, ["leaf"])
    with pytest.raises(RuntimeError, match="Shared mutable optimizer option"):
        DistributedCheckpointer.save(
            str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=SimpleNamespace(dp_mode="ddp")
        )
    assert not list(tmp_path.iterdir())


def test_tensor_lr_aliased_to_optimizer_defaults_rejected_before_save(tmp_path):
    model = torch.nn.Linear(2, 2)
    leaf = torch.optim.AdamW([model.weight], lr=torch.tensor(0.01))
    optimizer = MultiOptimizer(model, {"leaf": leaf}, ["leaf"])
    with pytest.raises(RuntimeError, match="Shared mutable optimizer tensor option"):
        DistributedCheckpointer.save(
            str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=SimpleNamespace(dp_mode="ddp")
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("singleton", [False, True])
def test_noncontiguous_adamw_parameter_recovers_and_updates(tmp_path, singleton):
    def build():
        model = torch.nn.Module()
        values = torch.arange(3, dtype=torch.float32)[::3] if singleton else torch.arange(12).float().reshape(3, 4).T
        model.register_parameter("weight", torch.nn.Parameter(values))
        leaf = torch.optim.AdamW([model.weight], lr=0.01)
        return model, MultiOptimizer(model, {"leaf": leaf}, ["leaf"])

    source, optimizer = build()
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tensor_byte_chunks_preserve_logical_order(dtype):
    from veomni.checkpoint.tensor_bytes import tensor_byte_chunks

    tensor = torch.arange(60, dtype=dtype).reshape(3, 4, 5).permute(2, 0, 1)
    chunks = list(tensor_byte_chunks(tensor, chunk_bytes=8))
    assert b"".join(chunks) == tensor.contiguous().view(torch.uint8).numpy().tobytes()
    assert max(map(len, chunks)) <= 8


def test_tensor_byte_chunks_bound_wide_index_iteration():
    from veomni.checkpoint.tensor_bytes import tensor_byte_chunks

    # A narrow transposed tensor must not allocate an index pool per row or
    # emit one tiny hash update per row.
    tensor = torch.arange(2_000_000, dtype=torch.float32).reshape(2, -1).T
    chunks = list(tensor_byte_chunks(tensor, chunk_bytes=4 << 20))
    assert len(chunks) == 2
    assert all(len(chunk) <= 4 << 20 for chunk in chunks)
    assert b"".join(chunks) == tensor.contiguous().view(torch.uint8).numpy().tobytes()


@pytest.mark.parametrize("chunk_bytes", [4, 8])
def test_tensor_byte_chunks_accept_singleton_strided_views(chunk_bytes):
    from veomni.checkpoint.tensor_bytes import tensor_byte_chunks

    for tensor in (torch.arange(3, dtype=torch.float32)[::3], torch.arange(6, dtype=torch.float32).reshape(2, 3).T):
        actual = b"".join(tensor_byte_chunks(tensor, chunk_bytes=chunk_bytes))
        expected = torch.tensor(tensor.tolist()).view(torch.uint8).numpy().tobytes()
        assert actual == expected


def _leaf_order_mismatch_worker(rank, mode, endpoint):
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        model = torch.nn.Linear(2, 2)
        if rank == 0:
            leaves = {"weight": torch.optim.AdamW([model.weight]), "bias": torch.optim.AdamW([model.bias])}
        else:
            leaves = {"both": torch.optim.AdamW(list(model.parameters()))}
        optimizer = MultiOptimizer(model, leaves, list(leaves))
        with pytest.raises(RuntimeError, match="optimizer leaf order differs across ranks"):
            if mode == "save":
                optimizer._sparse_state_dict(coordinate=True)
            else:
                optimizer._dcp_load_state_dict({})
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("mode", ["save", "load"])
def test_optimizer_leaf_order_mismatch_stops_all_ranks(mode):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_leaf_order_mismatch_worker, args=(mode,), nprocs=2)


def _high_level_leaf_mismatch_worker(rank, checkpoint, mismatch, mode, endpoint):
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:

        def build(mismatched):
            torch.manual_seed(45)
            model = torch.nn.Linear(2, 2)
            leaves = {"weight": torch.optim.AdamW([model.weight]), "bias": torch.optim.AdamW([model.bias])}
            keys = list(leaves)
            if mismatched and rank == 1:
                if mismatch == "count":
                    leaves.pop("bias")
                    keys.pop()
                else:
                    keys.reverse()
            return model, MultiOptimizer(model, leaves, keys)

        parallel = SimpleNamespace(dp_mode="ddp")
        if mode == "load":
            source, optimizer = build(False)
            DistributedCheckpointer.save(
                checkpoint, {"model": source, "optimizer": optimizer}, parallel_state=parallel
            )
        model, optimizer = build(True)
        before = model.weight.detach().clone()
        method = DistributedCheckpointer.save if mode == "save" else DistributedCheckpointer.load
        with pytest.raises(RuntimeError, match="optimizer leaf order differs across ranks"):
            method(checkpoint, {"model": model, "optimizer": optimizer}, parallel_state=parallel)
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("mode", ["save", "load"])
@pytest.mark.parametrize("mismatch", ["count", "order"])
def test_high_level_leaf_mismatch_rejected_before_model_mutation(tmp_path, mismatch, mode):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_high_level_leaf_mismatch_worker, args=(str(tmp_path), mismatch, mode), nprocs=2)


def test_staged_checkpoint_refuses_uncleared_previous_sidecar(tmp_path, monkeypatch):
    import veomni.checkpoint.dcp_checkpointer as checkpoint_module

    checkpoint = tmp_path / "final"
    stage = tmp_path / "stage"
    stale = stage / checkpoint_module._STAGE_ROOT / checkpoint_module._stage_key(str(checkpoint))
    stale_sidecar = stale / "extra_state" / "extra_state_rank_0.pt"
    stale_sidecar.parent.mkdir(parents=True)
    torch.save({"cursor": "stale"}, stale_sidecar)
    original = checkpoint_module.shutil.rmtree

    def fail_to_remove(path, *args, **kwargs):
        if str(path) == str(stale):
            return
        return original(path, *args, **kwargs)

    monkeypatch.setattr(checkpoint_module.shutil, "rmtree", fail_to_remove)
    model, optimizer = _model_and_optimizer()
    with pytest.raises(FileExistsError):
        DistributedCheckpointer.save(
            str(checkpoint),
            {"model": model, "optimizer": optimizer},
            stage_dir=str(stage),
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
    assert stale_sidecar.exists()
    assert not checkpoint.exists()


def _trainable_only_mismatch_worker(rank, checkpoint, mode, endpoint):
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(8)
        model, optimizer = _model_and_optimizer()
        parallel = SimpleNamespace(dp_mode="ddp")
        if mode == "load":
            DistributedCheckpointer.save(checkpoint, {"model": model, "optimizer": optimizer}, parallel_state=parallel)
        method = DistributedCheckpointer.save if mode == "save" else DistributedCheckpointer.load
        with pytest.raises(RuntimeError, match="trainable-only"):
            method(
                checkpoint,
                {"model": model, "optimizer": optimizer},
                parallel_state=parallel,
                trainable_only=bool(rank),
            )
    finally:
        dist.destroy_process_group()


def _proven_staged_resume_worker(rank, checkpoint, stage, endpoint):
    import os
    from datetime import timedelta

    import torch.distributed as dist

    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(9)
        model, optimizer = _model_and_optimizer()
        model.weight.grad = torch.full_like(model.weight, 0.25)
        optimizer.step()
        optimizer.zero_grad()
        parallel = SimpleNamespace(dp_mode="ddp")
        DistributedCheckpointer.save(
            checkpoint, {"model": model, "optimizer": optimizer}, parallel_state=parallel, stage_dir=stage
        )
        target, restored = _model_and_optimizer()
        DistributedCheckpointer.load(checkpoint, {"model": target, "optimizer": restored}, parallel_state=parallel)
        for source, destination in zip(model.parameters(), target.parameters(), strict=True):
            torch.testing.assert_close(source, destination, rtol=0, atol=0)
            source.grad = torch.full_like(source, 0.5)
            destination.grad = torch.full_like(destination, 0.5)
        optimizer.step()
        restored.step()
        for source, destination in zip(model.parameters(), target.parameters(), strict=True):
            torch.testing.assert_close(source, destination, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_proven_staged_checkpoint_restores_and_updates_on_two_ranks(tmp_path):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(
        _proven_staged_resume_worker, args=(str(tmp_path / "checkpoint"), str(tmp_path / "stage")), nprocs=2
    )


def _writer_error_worker(rank, checkpoint, target, endpoint):
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    original = DistributedCheckpointer._create_storage_writer
    try:
        torch.manual_seed(12)
        model, optimizer = _model_and_optimizer()
        parallel = SimpleNamespace(dp_mode="ddp")

        def make_writer(cls, path):
            if rank == 1 and Path(path).name == target:
                raise RuntimeError("injected writer factory failure")
            return original(path)

        DistributedCheckpointer._create_storage_writer = classmethod(make_writer)
        with pytest.raises(RuntimeError, match="writer"):
            DistributedCheckpointer.save(checkpoint, {"model": model, "optimizer": optimizer}, parallel_state=parallel)
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("target", ["ckpt", "optimizer"])
def test_writer_factory_failure_propagates_before_dcp_collective(tmp_path, target):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_writer_error_worker, args=(str(tmp_path), target), nprocs=2)


def _inventory_reader_failure_worker(rank, checkpoint, rank_local, endpoint):
    from datetime import timedelta

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    from veomni.distributed.parallel_plan import SpecInfo

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    original = DistributedCheckpointer._create_storage_reader
    try:

        def build():
            model = torch.nn.Module()
            values = torch.arange(16, dtype=torch.float32).reshape(4, 4)
            if rank_local:
                mesh = init_device_mesh("cpu", (1, 2), mesh_dim_names=("ple_fsdp", "ple"))
                model.weight = torch.nn.Parameter(distribute_tensor(values, mesh, [Shard(1), Shard(0)]))
                model._fqn2spec_info = {"weight": SpecInfo("ple", Shard(0), "weight", mesh, 1, 1)}
                parallel = SimpleNamespace(
                    dp_mode="fsdp2", extra_parallel_names=("ple",), extra_parallel_fsdp_device_mesh={"ple": mesh}
                )
            else:
                model.weight = torch.nn.Parameter(values)
                parallel = SimpleNamespace(dp_mode="ddp")
            leaf = torch.optim.AdamW([model.weight])
            return model, MultiOptimizer(model, {"leaf": leaf}, ["leaf"]), parallel

        source, optimizer, parallel = build()
        DistributedCheckpointer.save(checkpoint, {"model": source, "optimizer": optimizer}, parallel_state=parallel)
        target, restored, parallel = build()
        attempts = 0

        def reader(cls, path):
            nonlocal attempts
            if Path(path).name == "optimizer":
                attempts += 1
                if rank == 1 and attempts == 2:
                    raise OSError("injected inventory reader failure")
            return original(path)

        DistributedCheckpointer._create_storage_reader = classmethod(reader)
        with pytest.raises((RuntimeError, OSError), match="inventory reader"):
            DistributedCheckpointer.load(checkpoint, {"model": target, "optimizer": restored}, parallel_state=parallel)
        assert attempts == 2
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("rank_local", [False, True])
def test_inventory_reader_failure_propagates_before_collective(tmp_path, rank_local):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_inventory_reader_failure_worker, args=(str(tmp_path), rank_local), nprocs=2)


def _checkpoint_cache_failure_worker(rank, fail_at, endpoint):
    from datetime import timedelta

    import torch.distributed as dist

    from veomni.models.checkpoint import manager as manager_module

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    original = manager_module.helper.empty_cache
    try:
        calls = 0

        def clear_cache():
            nonlocal calls
            calls += 1
            if rank == 1 and calls == fail_at:
                raise RuntimeError("injected cache failure")

        manager_module.helper.empty_cache = clear_cache
        manager = object.__new__(manager_module.ModelCheckpointManager)
        manager.config = SimpleNamespace(
            save_path="/unused",
            save_async=False,
            dcp_save_to_lowest_rank=False,
            stage_dir=None,
            save_timeout_seconds=60,
        )
        manager.runtime = SimpleNamespace(
            model=torch.nn.Linear(2, 2), optimizer=SimpleNamespace(_is_multi_optimizer=True), lr_scheduler=None
        )
        manager._extra_state = lambda _state: {"_veomni_scheduler_recipe": None}
        manager._checkpointer_kwargs = lambda: {}
        manager.checkpointer = SimpleNamespace(save=lambda *args, **kwargs: "new-generation")
        manager._last_saved_step = 3
        manager._last_saved_optimizer_generation = "old-generation"
        with pytest.raises(RuntimeError, match="clearing cache|injected cache failure"):
            manager.save_dcp(SimpleNamespace(global_step=3))
        assert manager._last_saved_optimizer_generation is None
        assert manager._last_saved_step == -1
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        manager_module.helper.empty_cache = original
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("fail_at", [1, 2])
def test_checkpoint_cache_failure_propagates_before_next_collective(fail_at):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_checkpoint_cache_failure_worker, args=(fail_at,), nprocs=2)


def _inventory_hash_failure_worker(rank, endpoint):
    from datetime import timedelta

    import torch.distributed as dist

    from veomni.checkpoint import multi_optimizer_checkpoint as implementation

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    original = implementation.hashlib.sha256
    try:
        if rank == 1:

            def fail_hash(*_args, **_kwargs):
                raise MemoryError("injected inventory hash failure")

            implementation.hashlib.sha256 = fail_hash
        with pytest.raises((RuntimeError, MemoryError), match="hashing optimizer checkpoint inventory|inventory hash"):
            implementation._inventory_digests(["optimizer.state.weight.step"])
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        implementation.hashlib.sha256 = original
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_inventory_hash_failure_propagates_before_collective():
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_inventory_hash_failure_worker, args=(), nprocs=2)


def _trainability_mismatch_worker(rank, checkpoint, endpoint):
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        model, optimizer = _model_and_optimizer()
        if rank == 1:
            model.weight.requires_grad_(False)
        with pytest.raises(RuntimeError, match="trainability differs"):
            DistributedCheckpointer.save(
                checkpoint, {"model": model, "optimizer": optimizer}, parallel_state=SimpleNamespace(dp_mode="ddp")
            )
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_rank_asymmetric_trainability_is_rejected_before_save(tmp_path):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_trainability_mismatch_worker, args=(str(tmp_path),), nprocs=2)


def _tied_parameter_order_worker(rank, endpoint):
    from datetime import timedelta

    import torch.distributed as dist

    from veomni.checkpoint.optimizer_resume_layout import coordinated_multi_layout

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        model = torch.nn.Module()
        shared = torch.nn.Parameter(torch.ones(2, 2))
        for name in ("a", "b") if rank == 0 else ("b", "a"):
            model.register_parameter(name, shared)
        optimizer = MultiOptimizer(model, {"leaf": torch.optim.AdamW([shared])}, ["leaf"])
        coordinated_multi_layout(model, optimizer, writing=True)
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_tied_parameter_registration_order_does_not_change_trainability():
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_tied_parameter_order_worker, args=(), nprocs=2)


def _save_config_mismatch_worker(rank, checkpoint, mismatch, endpoint):
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(22)
        model, optimizer = _model_and_optimizer()
        state = {"model": model, "optimizer": optimizer}
        kwargs = {"parallel_state": SimpleNamespace(dp_mode="ddp")}
        if mismatch == "model" and rank == 1:
            state.pop("model")
        elif mismatch == "extra" and rank == 1:
            state["extra_state"] = {}
        elif mismatch == "stage" and rank == 1:
            kwargs["stage_dir"] = checkpoint + "-stage"
        elif mismatch == "stage_path":
            kwargs["stage_dir"] = checkpoint + f"-stage-{rank}"
        elif mismatch == "module":
            kwargs["module"] = f"model-{rank}"
        elif mismatch == "async" and rank == 1:
            kwargs["save_async"] = True
            kwargs["stage_dir"] = checkpoint + "-stage"
        with pytest.raises(RuntimeError, match="differs across ranks"):
            DistributedCheckpointer.save(checkpoint, state, **kwargs)
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("mismatch", ["model", "extra", "stage", "stage_path", "module", "async"])
def test_checkpoint_save_rank_config_mismatch_fails_all_ranks(tmp_path, mismatch):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_save_config_mismatch_worker, args=(str(tmp_path), mismatch), nprocs=2)


def _finalize_error_worker(rank, checkpoint, endpoint):
    from datetime import timedelta

    import torch.distributed as dist

    import veomni.checkpoint.dcp_checkpointer as checkpoint_module

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(27)
        model, optimizer = _model_and_optimizer()

        def synchronize():
            if rank == 1:
                raise RuntimeError("injected final synchronize failure")

        checkpoint_module.synchronize = synchronize
        with pytest.raises(RuntimeError, match="synchronize|finalizing"):
            DistributedCheckpointer.save(
                checkpoint, {"model": model, "optimizer": optimizer}, parallel_state=SimpleNamespace(dp_mode="ddp")
            )
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_save_finalize_error_propagates_to_all_ranks(tmp_path):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_finalize_error_worker, args=(str(tmp_path),), nprocs=2)


def _runtime_plan_mismatch_worker(rank, mode, endpoint):
    import contextlib
    from datetime import timedelta

    import torch.distributed as dist

    import veomni.models.model_runtime as runtime_module

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:

        class TestRuntime(runtime_module.VeOmniModelRuntime):
            def setup(self):
                pass

            def _build_model(self):
                self.model = torch.nn.Linear(2, 2)
                if mode != "missing" and (mode != "plan" or rank == 0):
                    self.model._fqn2spec_info = {}

            def _freeze_model_module(self):
                pass

            def _build_parallelized_model(self):
                pass

        runtime_module.use_parallel_state = lambda *_args, **_kwargs: contextlib.nullcontext()
        checkpoint = SimpleNamespace(
            extra_parallel_canonical_layout=(mode == "missing" or (mode == "flag" and rank == 1))
        )
        args = SimpleNamespace(optimizer=SimpleNamespace(type="adamw"))
        with pytest.raises(RuntimeError, match="differs across ranks|requires an ExtraParallel model plan"):
            TestRuntime(args, train=SimpleNamespace(checkpoint=checkpoint))
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("mode", ["plan", "flag", "missing"])
def test_runtime_checkpoint_plan_mismatch_fails_all_ranks(mode):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_runtime_plan_mismatch_worker, args=(mode,), nprocs=2)


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("mode", ["save", "load"])
def test_trainable_only_mismatch_fails_all_ranks(tmp_path, mode):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_trainable_only_mismatch_worker, args=(str(tmp_path), mode), nprocs=2)


@pytest.mark.parametrize("same_base", [True, False])
def test_trainable_only_resume_proves_frozen_base_before_loading(tmp_path, same_base):
    def make_model(base):
        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.fill_(base)
            model.bias.fill_(0.5)
        model.weight.requires_grad_(False)
        leaf = torch.optim.AdamW([model.bias], lr=0.02)
        return model, MultiOptimizer(model, {"bias": leaf}, ["bias"])

    source, optimizer = make_model(1.0)
    source.bias.grad = torch.ones_like(source.bias)
    optimizer.step()
    optimizer.zero_grad()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path), {"model": source, "optimizer": optimizer}, parallel_state=parallel, trainable_only=True
    )
    target, restored = make_model(1.0 if same_base else 2.0)
    before = target.bias.detach().clone()
    if same_base:
        DistributedCheckpointer.load(
            str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel, trainable_only=True
        )
        torch.testing.assert_close(target.bias, source.bias, rtol=0, atol=0)
    else:
        with pytest.raises(RuntimeError, match="identity|layout|checkpoint"):
            DistributedCheckpointer.load(
                str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel, trainable_only=True
            )
        torch.testing.assert_close(target.bias, before, rtol=0, atol=0)


def test_multi_optimizer_rejects_changed_group_identity(tmp_path):
    model, optimizer = _model_and_optimizer()
    model.weight.grad = torch.ones_like(model.weight)
    model.bias.grad = torch.ones_like(model.bias)
    optimizer.step()
    optimizer.zero_grad()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)

    target, changed = _model_and_optimizer(split=False)
    with pytest.raises(RuntimeError, match="(?i)optimizer.*(group|layout|identity)"):
        DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": changed}, parallel_state=parallel)


def test_unversioned_multi_optimizer_checkpoint_fails_closed(tmp_path):
    import torch.distributed.checkpoint as dcp

    from veomni.checkpoint.layout import optimizer_dir

    model, optimizer = _model_and_optimizer()
    model.weight.grad = torch.ones_like(model.weight)
    model.bias.grad = torch.ones_like(model.bias)
    optimizer.step()
    # Historical MultiOptimizer format had flat entries but no group identity.
    dcp.save({"optimizer": optimizer._sparse_state_dict()}, checkpoint_id=optimizer_dir(str(tmp_path)))
    target, restored = _model_and_optimizer()
    with pytest.raises(RuntimeError, match="(?i)group layout is unproven"):
        DistributedCheckpointer.load(
            str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=SimpleNamespace(dp_mode="ddp")
        )


def test_optimizer_serialization_hooks_fail_before_overwriting_checkpoint(tmp_path):
    model, optimizer = _model_and_optimizer()
    model.weight.grad = torch.ones_like(model.weight)
    model.bias.grad = torch.ones_like(model.bias)
    optimizer.step()
    optimizer.zero_grad()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    original = model.weight.detach().clone()

    def change_groups(_optimizer, state):
        state["param_groups"][0]["lr"] = 0.5
        return state

    handle = optimizer.optimizers_dict["weight"].register_state_dict_post_hook(change_groups)
    with pytest.raises(RuntimeError, match="serialization hooks"):
        DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    handle.remove()

    target, restored = _model_and_optimizer()
    DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)
    torch.testing.assert_close(target.weight, original, rtol=0, atol=0)


def test_nonfresh_target_is_rejected_before_any_checkpoint_load(tmp_path):
    model, optimizer = _model_and_optimizer()
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    optimizer.zero_grad()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)

    target, restored = _model_and_optimizer()
    restored.optimizers_dict["weight"].state[target.weight] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.zeros_like(target.weight, dtype=torch.bfloat16),
        "exp_avg_sq": torch.zeros_like(target.weight, dtype=torch.bfloat16),
    }
    original = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="fresh optimizer"):
        DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)
    torch.testing.assert_close(target.weight, original, rtol=0, atol=0)


def test_anyprecision_fp32_moments_with_bf16_parameter_fail_closed(tmp_path):
    def build():
        model = torch.nn.Linear(2, 2, bias=False).to(torch.bfloat16)
        leaf = AnyPrecisionAdamW(
            model.parameters(),
            momentum_dtype=torch.float32,
            variance_dtype=torch.float32,
            compensation_buffer_dtype=torch.float32,
        )
        return model, MultiOptimizer(model, {"weight": leaf}, ["weight"])

    model, optimizer = build()
    model.weight.grad = torch.full_like(model.weight, 0.1)
    optimizer.step()
    optimizer.zero_grad()
    parallel = SimpleNamespace(dp_mode="ddp")
    with pytest.raises(RuntimeError, match="state dtype cannot be safely restored"):
        DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    assert not list(tmp_path.iterdir())


def test_anyprecision_uninitialized_leaf_restores_and_updates(tmp_path):
    def build():
        model = torch.nn.Linear(2, 2)
        weight = AnyPrecisionAdamW([model.weight])
        bias = torch.optim.AdamW([model.bias], foreach=False)
        return model, MultiOptimizer(model, {"weight": weight, "bias": bias}, ["weight", "bias"])

    model, optimizer = build()
    model.bias.grad = torch.ones_like(model.bias)
    optimizer.step()
    optimizer.zero_grad()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    target, restored = build()
    DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)
    assert target.weight not in restored.optimizers_dict["weight"].state
    model.weight.grad = torch.ones_like(model.weight)
    target.weight.grad = torch.ones_like(target.weight)
    optimizer.step()
    restored.step()
    torch.testing.assert_close(model.weight, target.weight, rtol=0, atol=0)
    for name in ("exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            optimizer.optimizers_dict["weight"].state[model.weight][name],
            restored.optimizers_dict["weight"].state[target.weight][name],
            rtol=0,
            atol=0,
        )


def test_trainable_only_rejects_trained_then_frozen_optimizer_parameter(tmp_path):
    model, optimizer = _model_and_optimizer()
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    optimizer.zero_grad()
    model.weight.requires_grad_(False)
    with pytest.raises(RuntimeError, match="trainable-only save would omit a parameter with optimizer history"):
        DistributedCheckpointer.save(
            str(tmp_path),
            {"model": model, "optimizer": optimizer},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
            trainable_only=True,
        )


def test_trainable_only_resume_rejects_frozen_parameter_with_saved_history(tmp_path):
    model, optimizer = _model_and_optimizer()
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    optimizer.zero_grad()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    target, restored = _model_and_optimizer()
    target.weight.requires_grad_(False)
    original = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="trainable-only resume cannot prove"):
        DistributedCheckpointer.load(
            str(tmp_path),
            {"model": target, "optimizer": restored},
            parallel_state=parallel,
            trainable_only=True,
        )
    torch.testing.assert_close(target.weight, original, rtol=0, atol=0)


@pytest.mark.parametrize("mutation", ["freeze", "remove"])
def test_trainable_only_save_rejects_model_hook_hiding_trained_parameter(tmp_path, mutation):
    model, optimizer = _model_and_optimizer()
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    optimizer.zero_grad()

    def hide_weight(module, state_dict, prefix, local_metadata):
        if mutation == "freeze":
            module.weight.requires_grad_(False)
        else:
            state_dict.pop(prefix + "weight")

    model.register_state_dict_post_hook(hide_weight)
    with pytest.raises(RuntimeError, match="Cannot prove model serialization hook"):
        DistributedCheckpointer.save(
            str(tmp_path),
            {"model": model, "optimizer": optimizer},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
            trainable_only=True,
        )


def test_multi_optimizer_uninitialized_leaf_uses_its_first_real_step(tmp_path):
    model, optimizer = _model_and_optimizer()
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    optimizer.zero_grad()
    bias_before = model.bias.detach().clone()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    torch.testing.assert_close(model.bias, bias_before, rtol=0, atol=0)
    assert model.bias not in optimizer.optimizers_dict["bias"].state

    target, restored = _model_and_optimizer()
    DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)
    assert target.bias not in restored.optimizers_dict["bias"].state
    model.bias.grad = torch.full_like(model.bias, 0.4)
    target.bias.grad = torch.full_like(target.bias, 0.4)
    optimizer.step()
    restored.step()
    torch.testing.assert_close(target.bias, model.bias, rtol=0, atol=0)


def test_direct_multi_optimizer_without_provenance_is_rejected_before_mutation():
    model, optimizer = _model_and_optimizer()
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    saved = optimizer._sparse_state_dict()
    target, restored = _model_and_optimizer()
    before = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="Unverified MultiOptimizer state"):
        restored.load_state_dict(saved)
    assert torch.equal(before, target.weight)
    assert all(not leaf.state for leaf in restored.optimizers_dict.values())


def test_raw_dcp_multi_without_provenance_is_rejected_before_synthetic_step():
    from veomni.checkpoint.dcp_checkpointer import OptimizerState

    target, restored = _model_and_optimizer()
    wrapper = OptimizerState(target, restored, parallel_state=SimpleNamespace(dp_mode="ddp"), load=True)
    before = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="Raw DCP MultiOptimizer does not prove"):
        wrapper.state_dict()
    snapshot = restored.state_dict()
    assert snapshot
    assert torch.equal(before, target.weight)
    assert all(not leaf.state for leaf in restored.optimizers_dict.values())


def test_raw_dcp_load_rejection_preserves_existing_optimizer_state(tmp_path):
    import torch.distributed.checkpoint as dcp

    model, optimizer = _model_and_optimizer()
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    optimizer.zero_grad()
    dcp.save({"optimizer": optimizer}, checkpoint_id=tmp_path)
    target, restored = _model_and_optimizer()
    target.weight.grad = torch.full_like(target.weight, 0.4)
    restored.step()
    restored.zero_grad()
    before = {key: value.clone() for key, value in restored.optimizers_dict["weight"].state[target.weight].items()}
    with pytest.raises(RuntimeError, match="Unverified MultiOptimizer state"):
        dcp.load({"optimizer": restored}, checkpoint_id=tmp_path)
    for key, value in before.items():
        torch.testing.assert_close(restored.optimizers_dict["weight"].state[target.weight][key], value, rtol=0, atol=0)


def test_raw_snapshot_rejects_live_group_reordering_hook():
    model, optimizer = _model_and_optimizer()
    leaf = optimizer.optimizers_dict["weight"]
    leaf.register_state_dict_post_hook(lambda opt, saved: (opt.param_groups.clear(), saved)[1])
    with pytest.raises(RuntimeError, match="safe raw MultiOptimizer snapshot"):
        optimizer.state_dict()
    assert leaf.param_groups[0]["params"][0] is model.weight


def test_multi_model_generation_keeps_standard_torch_export_strictly_loadable(tmp_path):
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    model, optimizer = _model_and_optimizer()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    exported = tmp_path / "export.pt"
    dcp_to_torch_save(str(tmp_path / "model" / "ckpt"), str(exported))
    state = torch.load(exported, weights_only=False)
    target = torch.nn.Linear(2, 2)
    target.load_state_dict(state["model"], strict=True)
    for actual, expected in zip(target.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_multi_optimizer_rejects_changed_scheduler_before_loading_weights(tmp_path):
    model, optimizer = _model_and_optimizer()
    scheduler = build_lr_scheduler(optimizer, train_steps=100, lr_decay_style="linear", lr_warmup_ratio=0.1)
    model.weight.grad = torch.ones_like(model.weight)
    model.bias.grad = torch.ones_like(model.bias)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {
            "model": model,
            "optimizer": optimizer,
            "extra_state": {"lr_scheduler": scheduler.state_dict()},
            "_veomni_scheduler_object": scheduler,
        },
        parallel_state=parallel,
    )

    target, restored = _model_and_optimizer()
    different = build_lr_scheduler(restored, train_steps=200, lr_decay_style="linear", lr_warmup_ratio=0.1)
    original = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="(?i)optimizer.*scheduler.*layout"):
        DistributedCheckpointer.load(
            str(tmp_path),
            {"model": target, "optimizer": restored, "extra_state": {}, "_veomni_scheduler_object": different},
            parallel_state=parallel,
        )
    torch.testing.assert_close(target.weight, original, rtol=0, atol=0)
    assert different is not None


def test_multi_lr_scheduler_rejects_missing_leaf():
    _, optimizer = _model_and_optimizer()
    scheduler = build_lr_scheduler(optimizer, train_steps=100)
    state = scheduler.state_dict()
    state.pop("bias")
    with pytest.raises(RuntimeError, match="leaves"):
        scheduler.load_state_dict(state)


@pytest.mark.parametrize("override_container", [False, True])
@pytest.mark.parametrize("method", ["load_state_dict", "step", "get_last_lr"])
def test_multi_optimizer_rejects_overridden_scheduler_load(tmp_path, override_container, method):
    model, optimizer = _model_and_optimizer()
    scheduler = build_lr_scheduler(optimizer, train_steps=100, lr_decay_style="linear")
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {
            "model": model,
            "optimizer": optimizer,
            "extra_state": {"lr_scheduler": scheduler.state_dict()},
            "_veomni_scheduler_object": scheduler,
        },
        parallel_state=parallel,
    )
    target, restored = _model_and_optimizer()
    restored_scheduler = build_lr_scheduler(restored, train_steps=100, lr_decay_style="linear")
    owner = restored_scheduler if override_container else restored_scheduler["weight"]
    setattr(owner, method, lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="overridden scheduler"):
        DistributedCheckpointer.load(
            str(tmp_path),
            {
                "model": target,
                "optimizer": restored,
                "extra_state": {},
                "_veomni_scheduler_object": restored_scheduler,
            },
            parallel_state=parallel,
        )


def test_multi_optimizer_rejects_snapshot_from_an_old_scheduler_binding(tmp_path):
    model, optimizer = _model_and_optimizer()
    original = build_lr_scheduler(optimizer, train_steps=100, lr_decay_style="linear")
    newer = build_lr_scheduler(optimizer, train_steps=10, lr_decay_style="linear")
    state = {
        "model": model,
        "optimizer": optimizer,
        "extra_state": {"lr_scheduler": original.state_dict()},
        "_veomni_scheduler_object": original,
    }
    with pytest.raises(RuntimeError, match="scheduler object differs"):
        DistributedCheckpointer.save(str(tmp_path), state, parallel_state=SimpleNamespace(dp_mode="ddp"))
    assert not (tmp_path / "model").exists()
    assert newer is not None


def test_multi_optimizer_rejects_replaced_framework_scheduler_recipe(tmp_path):
    model, optimizer = _model_and_optimizer()
    scheduler = build_lr_scheduler(optimizer, train_steps=100, lr_decay_style="linear")

    def custom(step, *, factor=0.5):
        return factor

    scheduler["weight"].lr_lambdas[0] = custom
    state = {
        "model": model,
        "optimizer": optimizer,
        "extra_state": {"lr_scheduler": scheduler.state_dict()},
        "_veomni_scheduler_object": scheduler,
    }
    with pytest.raises(RuntimeError, match="scheduler recipe changed after construction"):
        DistributedCheckpointer.save(str(tmp_path), state, parallel_state=SimpleNamespace(dp_mode="ddp"))
    assert not (tmp_path / "model").exists()


def test_multi_optimizer_rejects_modified_scheduler_sidecar_before_loading(tmp_path):
    model, optimizer = _model_and_optimizer()
    scheduler = build_lr_scheduler(optimizer, train_steps=100)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {
            "model": model,
            "optimizer": optimizer,
            "extra_state": {"lr_scheduler": scheduler.state_dict()},
            "_veomni_scheduler_object": scheduler,
        },
        parallel_state=parallel,
    )
    sidecar = tmp_path / "model" / "extra_state" / "extra_state_rank_0.pt"
    stale = torch.load(sidecar, weights_only=False)
    stale["lr_scheduler"]["weight"]["last_epoch"] += 1
    torch.save(stale, sidecar)

    target, restored = _model_and_optimizer()
    restored_scheduler = build_lr_scheduler(restored, train_steps=100)
    original = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="(?i)optimizer.*layout"):
        DistributedCheckpointer.load(
            str(tmp_path),
            {
                "model": target,
                "optimizer": restored,
                "extra_state": {},
                "_veomni_scheduler_object": restored_scheduler,
            },
            parallel_state=parallel,
        )
    torch.testing.assert_close(target.weight, original, rtol=0, atol=0)
    assert restored_scheduler is not None


@pytest.mark.parametrize("swapped", ["optimizer", "extra_state"])
def test_multi_optimizer_rejects_mixed_checkpoint_generations(tmp_path, swapped):
    import shutil

    model, optimizer = _model_and_optimizer()
    parallel = SimpleNamespace(dp_mode="ddp")
    first, second = tmp_path / "first", tmp_path / "second"
    for index, path in enumerate((first, second), start=1):
        model.weight.grad = torch.full_like(model.weight, 0.1 * index)
        optimizer.step()
        optimizer.zero_grad()
        DistributedCheckpointer.save(
            str(path),
            {"model": model, "optimizer": optimizer, "extra_state": {"condition_rng": torch.tensor(index)}},
            parallel_state=parallel,
        )
    source = first / "model" / swapped
    destination = second / "model" / swapped
    shutil.rmtree(destination)
    shutil.copytree(source, destination)

    target, restored = _model_and_optimizer()
    original = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="(?i)generation|layout"):
        DistributedCheckpointer.load(
            str(second), {"model": target, "optimizer": restored, "extra_state": {}}, parallel_state=parallel
        )
    torch.testing.assert_close(target.weight, original, rtol=0, atol=0)


def test_model_generation_rechecked_by_actual_dcp_load(tmp_path, monkeypatch):
    import shutil

    import veomni.checkpoint.dcp_checkpointer as checkpoint

    model, optimizer = _model_and_optimizer()
    parallel = SimpleNamespace(dp_mode="ddp")
    first, second = tmp_path / "first", tmp_path / "second"
    for index, path in enumerate((first, second), start=1):
        model.weight.grad = torch.full_like(model.weight, 0.1 * index)
        optimizer.step()
        optimizer.zero_grad()
        DistributedCheckpointer.save(str(path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    original_preflight = checkpoint._preflight_multi_optimizer_layout

    def replace_after_preflight(*args, **kwargs):
        verified = original_preflight(*args, **kwargs)
        shutil.rmtree(second / "model" / "ckpt")
        shutil.copytree(first / "model" / "ckpt", second / "model" / "ckpt")
        return verified

    monkeypatch.setattr(checkpoint, "_preflight_multi_optimizer_layout", replace_after_preflight)
    target, restored = _model_and_optimizer()
    with pytest.raises(BaseException, match="(?i)checkpoint.*(changed|generation)"):
        DistributedCheckpointer.load(str(second), {"model": target, "optimizer": restored}, parallel_state=parallel)


def test_model_load_hook_cannot_change_optimizer_group_layout(tmp_path):
    def build():
        model = torch.nn.Linear(2, 2)
        leaf = torch.optim.AdamW([{"params": [model.weight], "lr": 0.01}, {"params": [model.bias], "lr": 0.02}])
        return model, MultiOptimizer(model, {"leaf": leaf}, ["leaf"])

    model, optimizer = build()
    model.weight.grad = torch.ones_like(model.weight)
    model.bias.grad = torch.ones_like(model.bias)
    optimizer.step()
    optimizer.zero_grad()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    target, restored = build()

    def change_groups(_module, _incompatible):
        leaf = restored.optimizers_dict["leaf"]
        leaf.param_groups[0]["params"].extend(leaf.param_groups.pop()["params"])

    target.register_load_state_dict_post_hook(change_groups)
    with pytest.raises(RuntimeError, match="Cannot prove model serialization hook"):
        DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)


@pytest.mark.parametrize("mutation", ["gradient", "state"])
def test_model_load_hook_cannot_populate_optimizer_state_or_gradient(tmp_path, mutation):
    model, optimizer = _model_and_optimizer()
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    optimizer.zero_grad()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    target, restored = _model_and_optimizer()

    def mutate(_module, _incompatible):
        if mutation == "gradient":
            target.weight.grad = torch.full_like(target.weight, 5)
        else:
            restored.optimizers_dict["weight"].state[target.weight]["step"] = torch.tensor(1.0)

    target.register_load_state_dict_post_hook(mutate)
    with pytest.raises(RuntimeError, match="Cannot prove model serialization hook"):
        DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)


def test_checkpoint_generation_marker_is_not_a_hf_parameter():
    from veomni.checkpoint.dcp_checkpointer import _MULTI_GENERATION_MARKER, _normalize_key

    assert _normalize_key(_MULTI_GENERATION_MARKER) is None


def test_muon_adam_resume_does_not_run_an_optimizer_step(tmp_path):
    def build():
        model = torch.nn.Linear(4, 4)
        muon = DistributedMuon([model.weight], ns_implementation="std")
        adam = torch.optim.AdamW([model.bias])
        return model, MultiOptimizer(model, {"muon": muon, "adam": adam}, ["muon", "adam"])

    model, optimizer = build()
    model.weight.grad = torch.full_like(model.weight, 0.1)
    optimizer.step()
    optimizer.zero_grad()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)

    target, restored = build()
    DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)
    torch.testing.assert_close(target.weight, model.weight, rtol=0, atol=0)
    torch.testing.assert_close(target.bias, model.bias, rtol=0, atol=0)
    assert target.bias not in restored.optimizers_dict["adam"].state
    model.weight.grad = torch.full_like(model.weight, 0.2)
    target.weight.grad = torch.full_like(target.weight, 0.2)
    model.bias.grad = torch.full_like(model.bias, 0.3)
    target.bias.grad = torch.full_like(target.bias, 0.3)
    optimizer.step()
    restored.step()
    torch.testing.assert_close(target.weight, model.weight, rtol=0, atol=0)
    torch.testing.assert_close(target.bias, model.bias, rtol=0, atol=0)


def test_muon_head_partition_change_requires_explicit_migration(tmp_path):
    def build(head_blocks):
        model = torch.nn.Linear(4, 8, bias=False)
        leaf = DistributedMuon([model.weight], ns_implementation="std", head_blocks=head_blocks)
        return model, MultiOptimizer(model, {"muon": leaf}, ["muon"])

    model, optimizer = build(2)
    model.weight.grad = torch.full_like(model.weight, 0.1)
    optimizer.step()
    optimizer.zero_grad()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    target, changed = build(4)
    with pytest.raises(RuntimeError, match="(?i)optimizer.*layout"):
        DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": changed}, parallel_state=parallel)


def test_multi_optimizer_layout_binds_extra_parallel_mesh_names():
    from veomni.checkpoint.optimizer_resume_layout import _local_layout

    model, optimizer = _model_and_optimizer()
    mesh = SimpleNamespace(mesh=torch.tensor([[0, 1], [2, 3]]), device_type="cpu", mesh_dim_names=("ep_fsdp", "ep"))
    spec = SimpleNamespace(para_name="ep", placement="Shard(0)", fsdp_shard_dim=1, para_fsdp_mesh=mesh)
    model._fqn2spec_info = {"weight": spec}
    before = _local_layout(model, optimizer, None, None, None, writing=False)
    mesh.mesh_dim_names = ("other_fsdp", "other")
    after = _local_layout(model, optimizer, None, None, None, writing=False)
    assert before != after


@pytest.mark.parametrize("target_frozen", [False, True])
def test_frozen_optimizer_history_survives_resume_then_unfreeze(tmp_path, target_frozen):
    def build():
        model = torch.nn.Linear(2, 2)
        weight = torch.optim.AdamW([model.weight], lr=0.01)
        bias = torch.optim.AdamW([model.bias], lr=0.02)
        return model, MultiOptimizer(model, {"weight": weight, "bias": bias}, ["weight", "bias"])

    model, optimizer = build()
    model.weight.grad = torch.full_like(model.weight, 0.1)
    optimizer.step()
    optimizer.zero_grad()
    model.weight.requires_grad_(False)
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)

    target, restored = build()
    target.weight.requires_grad_(not target_frozen)
    if not target_frozen:
        with pytest.raises(RuntimeError, match="(?i)optimizer.*layout"):
            DistributedCheckpointer.load(
                str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel
            )
        return
    DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)
    assert target.weight.requires_grad is not target_frozen
    model.weight.requires_grad_(True)
    target.weight.requires_grad_(True)
    model.weight.grad = torch.full_like(model.weight, 0.2)
    target.weight.grad = torch.full_like(target.weight, 0.2)
    optimizer.step()
    restored.step()
    torch.testing.assert_close(target.weight, model.weight, rtol=0, atol=0)
    source_state = optimizer.optimizers_dict["weight"].state[model.weight]
    restored_state = restored.optimizers_dict["weight"].state[target.weight]
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(restored_state[key], source_state[key], rtol=0, atol=0)


def test_multi_optimizer_rejects_checkpoint_with_pending_gradients(tmp_path):
    model, optimizer = _model_and_optimizer()
    model.weight.grad = torch.ones_like(model.weight)
    before = model.weight.detach().clone()
    with pytest.raises(RuntimeError, match="cannot omit pending gradients"):
        DistributedCheckpointer.save(
            str(tmp_path),
            {"model": model, "optimizer": optimizer},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
    assert not list(tmp_path.iterdir())


def test_unowned_pending_gradients_reject_save_and_load_before_model_copy(tmp_path):
    def build():
        model = torch.nn.Linear(2, 2)
        leaf = torch.optim.AdamW([model.weight])
        return model, MultiOptimizer(model, {"weight": leaf}, ["weight"])

    parallel = SimpleNamespace(dp_mode="ddp")
    source, optimizer = build()
    source.weight.grad = torch.ones_like(source.weight)
    optimizer.step()
    optimizer.zero_grad()
    source.bias.grad = torch.ones_like(source.bias)
    with pytest.raises(RuntimeError, match="cannot omit pending gradients"):
        DistributedCheckpointer.save(str(tmp_path), {"model": source, "optimizer": optimizer}, parallel_state=parallel)
    assert not list(tmp_path.iterdir())

    source.bias.grad = None
    DistributedCheckpointer.save(str(tmp_path), {"model": source, "optimizer": optimizer}, parallel_state=parallel)
    target, restored = build()
    target.bias.grad = torch.ones_like(target.bias)
    before = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="cannot omit pending gradients"):
        DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)
    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)


def test_incomplete_optimizer_source_state_cannot_be_saved(tmp_path):
    model, optimizer = _model_and_optimizer()
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    optimizer.zero_grad()
    del optimizer.optimizers_dict["weight"].state[model.weight]["exp_avg_sq"]
    with pytest.raises(RuntimeError, match="incomplete or unsupported parameter state"):
        DistributedCheckpointer.save(
            str(tmp_path),
            {"model": model, "optimizer": optimizer},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
    assert not list(tmp_path.iterdir())


def test_incomplete_optimizer_checkpoint_rejected_before_model_copy(tmp_path):
    import pickle

    from veomni.checkpoint.layout import optimizer_dir

    model, optimizer = _model_and_optimizer()
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    optimizer.zero_grad()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    path = Path(optimizer_dir(str(tmp_path))) / ".metadata"
    with path.open("rb") as handle:
        metadata = pickle.load(handle)
    del metadata.state_dict_metadata["optimizer.state.weight.exp_avg_sq"]
    with path.open("wb") as handle:
        pickle.dump(metadata, handle)
    target, restored = _model_and_optimizer()
    before = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="incomplete or unsupported parameter state|inventory"):
        DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)
    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)


@pytest.mark.parametrize("missing", ["group_lr", "entire_history"])
def test_missing_group_or_all_parameter_history_rejected_before_model_copy(tmp_path, missing):
    import pickle

    from veomni.checkpoint.layout import optimizer_dir

    source, optimizer = _model_and_optimizer()
    source.weight.grad = torch.ones_like(source.weight)
    optimizer.step()
    optimizer.zero_grad()
    optimizer.optimizers_dict["weight"].param_groups[0]["lr"] = 0.0025
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": source, "optimizer": optimizer}, parallel_state=parallel)
    path = Path(optimizer_dir(str(tmp_path))) / ".metadata"
    with path.open("rb") as handle:
        metadata = pickle.load(handle)
    if missing == "group_lr":
        keys = ["optimizer.param_groups.weight.lr"]
    else:
        keys = [key for key in metadata.state_dict_metadata if key.startswith("optimizer.state.weight.")]
        assert set(keys) == {
            "optimizer.state.weight.step",
            "optimizer.state.weight.exp_avg",
            "optimizer.state.weight.exp_avg_sq",
        }
    for key in keys:
        del metadata.state_dict_metadata[key]
    with path.open("wb") as handle:
        pickle.dump(metadata, handle)
    target, restored = _model_and_optimizer()
    before = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="inventory"):
        DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)
    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)


def test_model_save_hook_cannot_remove_checkpoint_optimizer(tmp_path):
    model, optimizer = _model_and_optimizer()
    state = {"model": model, "optimizer": optimizer}

    def remove_optimizer(*_):
        state["optimizer"] = None

    model.register_state_dict_post_hook(remove_optimizer)
    with pytest.raises(RuntimeError, match="Cannot prove model serialization hook"):
        DistributedCheckpointer.save(str(tmp_path), state, parallel_state=SimpleNamespace(dp_mode="ddp"))
    assert state["optimizer"] is optimizer


def test_multi_optimizer_rejects_stale_sidecar_before_overwrite(tmp_path):
    model, optimizer = _model_and_optimizer()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "extra_state": {"cursor": "old"}},
        parallel_state=parallel,
    )
    with pytest.raises(RuntimeError, match="sidecar cannot be reused"):
        DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)


def test_multi_optimizer_rejects_duplicate_ownership_before_checkpoint_write(tmp_path):
    model, optimizer = _model_and_optimizer()
    optimizer.optimizers_dict["bias"].param_groups[0]["params"].append(model.weight)
    with pytest.raises(RuntimeError, match="ownership is duplicated"):
        DistributedCheckpointer.save(
            str(tmp_path),
            {"model": model, "optimizer": optimizer},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
    assert not list(tmp_path.iterdir())


def test_multi_optimizer_rejects_empty_parameter_group_before_checkpoint_write(tmp_path):
    model = torch.nn.Linear(2, 2)
    leaf = torch.optim.AdamW([{"params": []}, {"params": [model.weight]}])
    optimizer = MultiOptimizer(model, {"leaf": leaf}, ["leaf"])
    with pytest.raises(RuntimeError, match="empty parameter group"):
        DistributedCheckpointer.save(
            str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=SimpleNamespace(dp_mode="ddp")
        )
    assert not list(tmp_path.iterdir())


def test_optimizer_root_names_must_match_checkpoint_model(tmp_path):
    weight = torch.nn.Parameter(torch.ones(2))
    model = torch.nn.Module()
    model.register_parameter("b", weight)
    optimizer_root = torch.nn.Module()
    optimizer_root.register_parameter("a", weight)
    leaf = torch.optim.AdamW([weight], lr=0.01)
    optimizer = MultiOptimizer(optimizer_root, {"leaf": leaf}, ["leaf"])
    with pytest.raises(RuntimeError, match="different canonical parameter identities"):
        DistributedCheckpointer.save(
            str(tmp_path),
            {"model": model, "optimizer": optimizer},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
    assert not list(tmp_path.iterdir())


def test_tensor_lr_dtype_change_rejected_before_model_load(tmp_path):
    def build(dtype):
        model = torch.nn.Linear(2, 2)
        leaf = torch.optim.AdamW([model.weight], lr=torch.tensor(0.03012345, dtype=dtype))
        leaf.param_groups[0]["lr"] = leaf.param_groups[0]["lr"].clone()
        return model, MultiOptimizer(model, {"leaf": leaf}, ["leaf"])

    model, optimizer = build(torch.float32)
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    target, restored = build(torch.bfloat16)
    before = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="layout|identity"):
        DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)
    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)


def test_shared_tensor_lr_across_groups_is_rejected_before_save(tmp_path):
    def build(shared):
        model = torch.nn.Linear(2, 2)
        lr = torch.tensor(0.01)
        other = lr if shared else lr.clone()
        leaf = torch.optim.AdamW([{"params": [model.weight], "lr": lr}, {"params": [model.bias], "lr": other}])
        return model, MultiOptimizer(model, {"leaf": leaf}, ["leaf"])

    model, optimizer = build(True)
    parallel = SimpleNamespace(dp_mode="ddp")
    with pytest.raises(RuntimeError, match="Shared mutable optimizer tensor option"):
        DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    assert not list(tmp_path.iterdir())


def test_model_checkpoint_extra_state_records_optimizer_global_step():
    from veomni.models.checkpoint.manager import ModelCheckpointManager

    _, optimizer = _model_and_optimizer()
    manager = object.__new__(ModelCheckpointManager)
    manager.runtime = SimpleNamespace(optimizer=optimizer, lr_scheduler=None, extra_state=lambda: {})
    saved = manager._extra_state(SimpleNamespace(global_step=3))
    assert saved["_veomni_optimizer_global_step"] == 3


def test_proven_optimizer_load_sets_identity_receipt(tmp_path):
    model, optimizer = _model_and_optimizer()
    parallel = SimpleNamespace(dp_mode="ddp")
    generation = DistributedCheckpointer.save(
        str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel
    )
    target, restored = _model_and_optimizer()
    state = {"model": target, "optimizer": restored}
    DistributedCheckpointer.load(str(tmp_path), state, parallel_state=parallel)
    assert state["_veomni_optimizer_identity_verified"] is True
    assert isinstance(generation, str) and state["_veomni_optimizer_checkpoint_generation"] == generation


def test_failed_second_load_clears_old_identity_receipt(tmp_path, monkeypatch):
    from veomni.checkpoint import dcp_checkpointer

    parallel = SimpleNamespace(dp_mode="ddp")
    first, first_optimizer = _model_and_optimizer()
    second, second_optimizer = _model_and_optimizer()
    with torch.no_grad():
        first.weight.fill_(1)
        second.weight.fill_(9)
    DistributedCheckpointer.save(
        str(tmp_path / "first"), {"model": first, "optimizer": first_optimizer}, parallel_state=parallel
    )
    DistributedCheckpointer.save(
        str(tmp_path / "second"), {"model": second, "optimizer": second_optimizer}, parallel_state=parallel
    )
    target, restored = _model_and_optimizer()
    state = {"model": target, "optimizer": restored}
    DistributedCheckpointer.load(str(tmp_path / "first"), state, parallel_state=parallel)
    assert state["_veomni_optimizer_identity_verified"] is True

    def fail_optimizer_apply(*_args, **_kwargs):
        raise RuntimeError("injected optimizer apply failure")

    monkeypatch.setattr(dcp_checkpointer._ProvenMultiOptimizerState, "load_state_dict", fail_optimizer_apply)
    with pytest.raises(BaseException, match="optimizer apply failure"):
        DistributedCheckpointer.load(str(tmp_path / "second"), state, parallel_state=parallel)
    assert torch.equal(target.weight, second.weight)
    assert "_veomni_optimizer_identity_verified" not in state
    assert "_veomni_optimizer_checkpoint_generation" not in state


def test_model_buffer_dtype_change_rejected_before_model_load(tmp_path):
    model, optimizer = _model_and_optimizer()
    model.register_buffer("sensitive", torch.tensor([0.1], dtype=torch.float32))
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    target, restored = _model_and_optimizer()
    target.register_buffer("sensitive", torch.tensor([0.2], dtype=torch.bfloat16))
    before = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="tensor dtype or shape differs"):
        DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)
    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)


def test_source_model_extra_buffer_rejected_before_model_load(tmp_path):
    model, optimizer = _model_and_optimizer()
    model.register_buffer("offset", torch.tensor([0.5]))
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    target, restored = _model_and_optimizer()
    before = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="model state inventory differs"):
        DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)
    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)


@pytest.mark.parametrize("buffer_alias", [False, True])
def test_distinct_model_tensors_with_overlapping_storage_are_rejected(tmp_path, buffer_alias):
    model = torch.nn.Module()
    model.register_parameter("p", torch.nn.Parameter(torch.ones(3)))
    if buffer_alias:
        model.register_buffer("q", model.p.detach())
        leaf = torch.optim.AdamW([model.p])
    else:
        model.register_parameter("q", torch.nn.Parameter(model.p.detach()))
        leaf = torch.optim.AdamW([model.p, model.q])
    optimizer = MultiOptimizer(model, {"leaf": leaf}, ["leaf"])
    with pytest.raises(RuntimeError, match="checkpoint model storage alias"):
        DistributedCheckpointer.save(
            str(tmp_path),
            {"model": model, "optimizer": optimizer},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["parameter_buffer", "buffers"])
def test_model_object_alias_graph_must_match_on_resume(tmp_path, kind):
    def build(shared):
        model = torch.nn.Module()
        model.register_parameter("p", torch.nn.Parameter(torch.ones(3)))
        if kind == "parameter_buffer":
            model.register_buffer("view", model.p if shared else model.p.detach().clone())
        else:
            model.register_buffer("view1", torch.ones(3))
            model.register_buffer("view2", model.view1 if shared else model.view1.clone())
        leaf = torch.optim.AdamW([model.p])
        return model, MultiOptimizer(model, {"leaf": leaf}, ["leaf"])

    model, optimizer = build(True)
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    target, restored = build(False)
    before = target.p.detach().clone()
    with pytest.raises(RuntimeError, match="layout|identity"):
        DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)
    torch.testing.assert_close(target.p, before, rtol=0, atol=0)


def test_multi_optimizer_rejects_unproven_step_hooks(tmp_path):
    model, optimizer = _model_and_optimizer()
    optimizer.optimizers_dict["weight"].register_step_pre_hook(lambda opt, *_: opt.param_groups[0].update(lr=0.5))
    with pytest.raises(RuntimeError, match="cannot prove custom optimizer"):
        DistributedCheckpointer.save(
            str(tmp_path),
            {"model": model, "optimizer": optimizer},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
    assert not list(tmp_path.iterdir())


def test_multi_optimizer_rejects_overridden_step(tmp_path):
    model, optimizer = _model_and_optimizer()
    optimizer.step = lambda: None
    with pytest.raises(RuntimeError, match="cannot prove overridden optimizer methods"):
        DistributedCheckpointer.save(
            str(tmp_path),
            {"model": model, "optimizer": optimizer},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("old_model_state", [True, False])
def test_multi_optimizer_save_preserves_completed_checkpoint(tmp_path, old_model_state):
    from veomni.checkpoint.layout import checkpoint_is_complete, write_manifest

    model, optimizer = _model_and_optimizer()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model}, parallel_state=parallel)
    write_manifest(str(tmp_path), global_step=0, world_size=1, model_state=old_model_state)
    assert checkpoint_is_complete(str(tmp_path))
    with pytest.raises(RuntimeError, match="Cannot overwrite a completed checkpoint"):
        DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    assert checkpoint_is_complete(str(tmp_path))


def test_failed_multi_optimizer_save_remains_incomplete(tmp_path, monkeypatch):
    from veomni.checkpoint.layout import checkpoint_is_complete

    model, optimizer = _model_and_optimizer()
    original = DistributedCheckpointer._create_storage_writer

    def failing_writer(path):
        if str(path).endswith("optimizer"):
            raise OSError("injected optimizer write failure")
        return original(path)

    monkeypatch.setattr(DistributedCheckpointer, "_create_storage_writer", failing_writer)
    with pytest.raises(OSError, match="injected optimizer write failure"):
        DistributedCheckpointer.save(
            str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=SimpleNamespace(dp_mode="ddp")
        )
    assert not checkpoint_is_complete(str(tmp_path))


@pytest.mark.parametrize("phase", ["save", "load"])
def test_model_value_changing_hooks_are_rejected(tmp_path, phase):
    model, optimizer = _model_and_optimizer()
    parallel = SimpleNamespace(dp_mode="ddp")
    if phase == "load":
        DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
        model, optimizer = _model_and_optimizer()

        def change_weight(module, _):
            with torch.no_grad():
                module.weight.add_(0.5)

        model.register_load_state_dict_post_hook(change_weight)
    else:

        def change_weight(module, saved, prefix, metadata):
            saved[prefix + "weight"] = saved[prefix + "weight"] + 0.5

        model.register_state_dict_post_hook(change_weight)
    before = model.weight.detach().clone()
    with pytest.raises(RuntimeError, match="Cannot prove model serialization hook"):
        if phase == "save":
            DistributedCheckpointer.save(
                str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel
            )
        else:
            DistributedCheckpointer.load(
                str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel
            )
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)


def test_model_hook_cannot_bypass_check_by_copying_fsdp_name(tmp_path):
    model, optimizer = _model_and_optimizer()

    def fake_sharded_hook(*_):
        with torch.no_grad():
            model.weight.add_(0.5)

    fake_sharded_hook.__module__ = "torch.distributed.fsdp._fully_shard._fsdp_param_group"
    fake_sharded_hook.__qualname__ = "FSDPParamGroup._register_state_dict_hooks.<locals>.to_sharded_hook"
    model.register_state_dict_pre_hook(fake_sharded_hook)
    before = model.weight.detach().clone()
    with pytest.raises(RuntimeError, match="Cannot prove model serialization hook"):
        DistributedCheckpointer.save(
            str(tmp_path),
            {"model": model, "optimizer": optimizer},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)


def _replicated_optimizer_metadata_worker(rank, checkpoint, divergent_field, endpoint):
    from datetime import timedelta

    import torch.distributed as dist
    from torch.distributed.fsdp import fully_shard

    from veomni.distributed.parallel_state import _init_parallel_state, get_parallel_state

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        _init_parallel_state(dp_size=2, dp_mode="fsdp2", device_type="cpu")
        parallel = get_parallel_state()
        model = torch.nn.Linear(4, 4)
        fully_shard(model, mesh=parallel.dp_shard_mesh)
        leaf = torch.optim.AdamW(model.parameters(), lr=0.01)
        optimizer = MultiOptimizer(model, {"leaf": leaf}, ["leaf"])
        model(torch.ones(2, 4)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        if rank == 1:
            if divergent_field == "lr":
                leaf.param_groups[0]["lr"] = 0.02
            else:
                next(iter(leaf.state.values()))["step"] += 1
        with pytest.raises(RuntimeError, match="Replicated optimizer metadata differs"):
            DistributedCheckpointer.save(checkpoint, {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("field", ["lr", "step"])
def test_replicated_optimizer_metadata_must_agree_before_save(tmp_path, field):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_replicated_optimizer_metadata_worker, args=(str(tmp_path), field), nprocs=2)
