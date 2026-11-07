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


def test_captured_layout_binds_sidecar_without_rechecking_runtime():
    from veomni.checkpoint.optimizer_resume_layout import (
        capture_multi_layout,
        coordinated_multi_layout,
        finish_multi_layout,
    )

    model, optimizer = _model_and_optimizer()
    captured = capture_multi_layout(model, optimizer, None, None)
    expected = coordinated_multi_layout(model, optimizer, None, None, "sidecar-digest", writing=True)
    assert finish_multi_layout(captured, "sidecar-digest") == expected
    assert finish_multi_layout(captured, "other-sidecar-digest") != expected


def _corrupt_inventory_worker(rank, endpoint):
    import hashlib
    import json
    from datetime import timedelta

    import torch.distributed as dist

    from veomni.checkpoint.dcp_checkpointer import _ProvenMultiOptimizerState
    from veomni.checkpoint.multi_optimizer_checkpoint import _MULTI_INVENTORY_MARKER
    from veomni.checkpoint.optimizer_resume_layout import MULTI_LAYOUT_MARKER

    class OptimizerProbe:
        def __init__(self):
            self.entered = False

        def _dcp_load_state_dict(self, state_dict, *, group=None):
            self.entered = True
            dist.all_reduce(torch.ones(1), group=group)

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        optimizer = OptimizerProbe()
        state = _ProvenMultiOptimizerState(
            model=torch.nn.Linear(1, 1),
            optimizer=optimizer,
            parallel_state=SimpleNamespace(dp_mode="ddp"),
            generation="generation",
            expected_layout="layout",
            group=None,
            load=True,
        )
        inventory = ["optimizer.weight"]
        digest = hashlib.sha256(json.dumps(inventory, separators=(",", ":")).encode()).hexdigest()
        marker = {"generation": "generation", "layout": "layout", "inventory_digests": [digest, digest]}
        if rank == 0:
            marker["inventory_digests"][0] = "corrupt"
        payload = {
            "weight": torch.ones(1),
            _MULTI_INVENTORY_MARKER: json.dumps(inventory),
            MULTI_LAYOUT_MARKER: json.dumps(marker),
        }
        with pytest.raises(RuntimeError, match="inventory identity"):
            state.load_state_dict(payload)
        assert not optimizer.entered
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_corrupt_optimizer_inventory_fails_on_all_ranks():
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_corrupt_inventory_worker, nprocs=2)


def _asymmetric_optimizer_policy_worker(rank, checkpoint, operation, failure, endpoint):
    from datetime import timedelta
    from unittest.mock import patch

    import torch.distributed as dist

    import veomni.checkpoint.dcp_checkpointer as checkpoint_module
    from veomni.optim.swap_adamw import SwapAdamW

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=6))
    try:
        model = torch.nn.Linear(2, 2)

        class FaultAdamW(torch.optim.AdamW):
            @property
            def _is_multi_optimizer(self):
                if failure == "type_property" and rank == 0:
                    raise OSError("injected optimizer type failure")
                return False

        leaf = FaultAdamW(model.parameters())
        optimizer = leaf
        if failure == "multi_mismatch" and rank == 0:
            optimizer = MultiOptimizer(model, {"all": leaf}, ["all"])
        if failure == "presence_mismatch" and rank == 0:
            optimizer = None
        if failure == "swap_mismatch" and rank == 0:
            optimizer = SwapAdamW(model.parameters(), pin_memory=False)
        state = {
            "model": model,
            "optimizer": optimizer,
            "_veomni_scheduler_object": None,
            "_veomni_optimizer_identity_verified": True,
            "_veomni_optimizer_checkpoint_generation": "stale",
        }
        original = checkpoint_module._requires_rank_local_optimizer_checkpoint

        def inspect_optimizer(value):
            if failure == "rank_local_check" and rank == 0:
                raise OSError("injected optimizer precision policy failure")
            return original(value)

        with patch.object(checkpoint_module, "_requires_rank_local_optimizer_checkpoint", inspect_optimizer):
            with pytest.raises((OSError, RuntimeError), match="injected|failed on another rank|optimizer.*differ"):
                getattr(DistributedCheckpointer, operation)(
                    checkpoint, state, parallel_state=SimpleNamespace(dp_mode="ddp")
                )
        if operation == "load":
            assert "_veomni_optimizer_identity_verified" not in state
            assert "_veomni_optimizer_checkpoint_generation" not in state
        assert not Path(checkpoint).exists()
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("operation", ["save", "load"])
@pytest.mark.parametrize(
    "failure", ["rank_local_check", "type_property", "multi_mismatch", "presence_mismatch", "swap_mismatch"]
)
def test_asymmetric_optimizer_policy_fails_before_checkpoint_io(tmp_path, operation, failure):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(
        _asymmetric_optimizer_policy_worker,
        args=(str(tmp_path / "checkpoint"), operation, failure),
        nprocs=2,
    )


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
        DistributedCheckpointer.save(
            checkpoint,
            {"model": source, "optimizer": optimizer, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
        target, restored = _model_and_optimizer()
        original_model = checkpoint_module._ProvenModelState
        original_optimizer = checkpoint_module._ProvenMultiOptimizerState
        original_reader = DistributedCheckpointer._create_storage_reader
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

        with (
            patch.object(checkpoint_module, "_ProvenModelState", side_effect=model_state),
            patch.object(checkpoint_module, "_ProvenMultiOptimizerState", side_effect=optimizer_state),
            patch.object(DistributedCheckpointer, "_create_storage_reader", side_effect=reader),
            pytest.raises((OSError, RuntimeError), match="injected|failed on another rank"),
        ):
            DistributedCheckpointer.load(
                checkpoint,
                {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
                parallel_state=parallel,
            )
        if rank == 1:
            assert injected, f"the {failure} fault did not reach its intended load stage"
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("failure", ["model_reader", "optimizer_reader", "model_state", "optimizer_state"])
def test_proven_load_preflight_failure_stops_all_ranks(tmp_path, failure):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_load_preflight_failure_worker, args=(str(tmp_path), failure), nprocs=2)


def test_multi_optimizer_async_checkpoint_rejected_before_writes(tmp_path):
    model, optimizer = _model_and_optimizer()
    with pytest.raises(RuntimeError, match="require synchronous DCP save"):
        DistributedCheckpointer.save(
            str(tmp_path),
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
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
    manager.config = SimpleNamespace(dcp_evict_source_cache=False)
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
            str(tmp_path),
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
    assert not list(tmp_path.iterdir())


def test_tensor_lr_aliased_to_optimizer_defaults_rejected_before_save(tmp_path):
    model = torch.nn.Linear(2, 2)
    leaf = torch.optim.AdamW([model.weight], lr=torch.tensor(0.01))
    optimizer = MultiOptimizer(model, {"leaf": leaf}, ["leaf"])
    with pytest.raises(RuntimeError, match="Shared mutable optimizer tensor option"):
        DistributedCheckpointer.save(
            str(tmp_path),
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
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
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": source, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target, restored = build()
    DistributedCheckpointer.load(
        str(tmp_path),
        {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
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
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
            stage_dir=str(stage),
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
    assert stale_sidecar.exists()
    assert not checkpoint.exists()


def test_staged_layout_finalization_failure_cleans_temporary_sidecar(tmp_path, monkeypatch):
    import veomni.checkpoint.dcp_checkpointer as checkpoint_module

    checkpoint = tmp_path / "final"
    stage = tmp_path / "stage"
    staged = stage / checkpoint_module._STAGE_ROOT / checkpoint_module._stage_key(str(checkpoint))
    model, optimizer = _model_and_optimizer()

    def fail(*args, **kwargs):
        raise RuntimeError("injected layout finalization failure")

    monkeypatch.setattr(checkpoint_module, "finish_multi_layout", fail)
    with pytest.raises(RuntimeError, match="injected layout finalization failure"):
        DistributedCheckpointer.save(
            str(checkpoint),
            {"model": model, "optimizer": optimizer, "extra_state": {}, "_veomni_scheduler_object": None},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
            stage_dir=str(stage),
        )
    assert not staged.exists()
    assert not (checkpoint / "model").exists()


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
            checkpoint,
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
            parallel_state=parallel,
            stage_dir=stage,
        )
        target, restored = _model_and_optimizer()
        DistributedCheckpointer.load(
            checkpoint,
            {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
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
            DistributedCheckpointer.save(
                checkpoint,
                {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
                parallel_state=parallel,
            )
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
        DistributedCheckpointer.save(
            checkpoint,
            {"model": source, "optimizer": optimizer, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
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
            DistributedCheckpointer.load(
                checkpoint,
                {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
                parallel_state=parallel,
            )
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
                checkpoint,
                {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
                parallel_state=SimpleNamespace(dp_mode="ddp"),
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


def _runtime_plan_mismatch_worker(rank, endpoint):
    import contextlib
    from datetime import timedelta

    import torch.distributed as dist
    from torch.distributed.tensor import Shard

    import veomni.distributed.torch_parallelize as parallelize_module
    import veomni.models.model_runtime as runtime_module
    from veomni.distributed.parallel_plan import ParallelPlan

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:

        class TestRuntime(runtime_module.VeOmniModelRuntime):
            def setup(self):
                pass

            def _build_model(self):
                self.model = torch.nn.Linear(2, 2)
                self.model.get_parallel_plan = lambda: ParallelPlan(extra_parallel_plan={"ep": {"weight": Shard(0)}})

            def _freeze_model_module(self):
                pass

            def _build_parallelized_model(self):
                pass

        runtime_module.use_parallel_state = lambda *_args, **_kwargs: contextlib.nullcontext()
        state = SimpleNamespace(
            any_extra_parallel_enabled=True,
            extra_parallel_sizes=("ep",),
            extra_parallel_enabled=lambda name: name == "ep",
        )
        runtime_module.get_parallel_state_by_name = lambda _name: state
        parallelize_module.get_parallel_state = lambda: state
        checkpoint = SimpleNamespace(extra_parallel_canonical_layout=True)
        args = SimpleNamespace(optimizer=SimpleNamespace(type="adamw"))
        with pytest.raises(RuntimeError, match="requires an ExtraParallel model plan"):
            TestRuntime(args, train=SimpleNamespace(checkpoint=checkpoint))
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_runtime_checkpoint_plan_mismatch_fails_all_ranks():
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_runtime_plan_mismatch_worker, args=(), nprocs=2)


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
        str(tmp_path),
        {"model": source, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
        trainable_only=True,
    )
    target, restored = make_model(1.0 if same_base else 2.0)
    before = target.bias.detach().clone()
    if same_base:
        DistributedCheckpointer.load(
            str(tmp_path),
            {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
            parallel_state=parallel,
            trainable_only=True,
        )
        torch.testing.assert_close(target.bias, source.bias, rtol=0, atol=0)
    else:
        with pytest.raises(RuntimeError, match="identity|layout|checkpoint"):
            DistributedCheckpointer.load(
                str(tmp_path),
                {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
                parallel_state=parallel,
                trainable_only=True,
            )
        torch.testing.assert_close(target.bias, before, rtol=0, atol=0)


def test_multi_optimizer_rejects_changed_group_identity(tmp_path):
    model, optimizer = _model_and_optimizer()
    model.weight.grad = torch.ones_like(model.weight)
    model.bias.grad = torch.ones_like(model.bias)
    optimizer.step()
    optimizer.zero_grad()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )

    target, changed = _model_and_optimizer(split=False)
    with pytest.raises(RuntimeError, match="(?i)optimizer.*(group|layout|identity)"):
        DistributedCheckpointer.load(
            str(tmp_path),
            {"model": target, "optimizer": changed, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )


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
            str(tmp_path),
            {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )


def test_nonfresh_target_is_rejected_before_any_checkpoint_load(tmp_path):
    model, optimizer = _model_and_optimizer()
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    optimizer.zero_grad()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )

    target, restored = _model_and_optimizer()
    restored.optimizers_dict["weight"].state[target.weight] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.zeros_like(target.weight, dtype=torch.bfloat16),
        "exp_avg_sq": torch.zeros_like(target.weight, dtype=torch.bfloat16),
    }
    original = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="fresh optimizer"):
        DistributedCheckpointer.load(
            str(tmp_path),
            {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
    torch.testing.assert_close(target.weight, original, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("param_dtype", "moment_dtype"),
    [(torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32)],
)
def test_anyprecision_mixed_state_dtype_save_resume_and_update(tmp_path, monkeypatch, param_dtype, moment_dtype):
    def build():
        model = torch.nn.Linear(2, 2, bias=False).to(param_dtype)
        leaf = AnyPrecisionAdamW(
            model.parameters(),
            momentum_dtype=moment_dtype,
            variance_dtype=moment_dtype,
            compensation_buffer_dtype=moment_dtype,
        )
        return model, MultiOptimizer(model, {"weight": leaf}, ["weight"])

    model, optimizer = build()
    model.weight.grad = torch.full_like(model.weight, 0.1)
    optimizer.step()
    optimizer.zero_grad()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target, restored = build()
    target_leaf = restored.optimizers_dict["weight"]
    original_load = AnyPrecisionAdamW.load_state_dict
    owned_pointers = []

    def tracked_load(self, state_dict):
        if self is target_leaf:
            before = {name: value.data_ptr() for name, value in self.state[target.weight].items()}
        result = original_load(self, state_dict)
        if self is target_leaf:
            after = {name: value.data_ptr() for name, value in self.state[target.weight].items()}
            owned_pointers.append((before, after))
        return result

    monkeypatch.setattr(AnyPrecisionAdamW, "load_state_dict", tracked_load)
    DistributedCheckpointer.load(
        str(tmp_path),
        {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    assert len(owned_pointers) == 1
    before, after = owned_pointers[0]
    assert before == after
    assert set(before) == {"step", "exp_avg", "exp_avg_sq", "compensation"}
    source_state = optimizer.optimizers_dict["weight"].state[model.weight]
    target_state = restored.optimizers_dict["weight"].state[target.weight]
    for field in ("exp_avg", "exp_avg_sq", "compensation"):
        assert source_state[field].dtype == target_state[field].dtype == moment_dtype
    torch.testing.assert_close(source_state, target_state, rtol=0, atol=0)
    torch.testing.assert_close(model.weight, target.weight, rtol=0, atol=0)
    model.weight.grad = torch.full_like(model.weight, 0.2)
    target.weight.grad = torch.full_like(target.weight, 0.2)
    optimizer.step()
    restored.step()
    torch.testing.assert_close(model.weight, target.weight, rtol=0, atol=0)
    torch.testing.assert_close(source_state, target_state, rtol=0, atol=0)


def test_anyprecision_direct_load_preserves_hooks_and_state_dtype():
    source = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
    first = AnyPrecisionAdamW([source], momentum_dtype=torch.bfloat16, variance_dtype=torch.bfloat16)
    source.grad = torch.full_like(source, 0.2)
    first.step()
    target = torch.nn.Parameter(source.detach().clone())
    second = AnyPrecisionAdamW([target], momentum_dtype=torch.bfloat16, variance_dtype=torch.bfloat16)
    calls = []
    second.register_load_state_dict_pre_hook(lambda opt, state: calls.append("pre"))
    second.register_load_state_dict_post_hook(lambda opt: calls.append("post"))
    second.load_state_dict(first.state_dict())
    assert calls == ["pre", "post"]
    assert second.state[target]["exp_avg"].dtype == torch.bfloat16
    assert second.state[target]["exp_avg_sq"].dtype == torch.bfloat16
    for field in ("step", "exp_avg", "exp_avg_sq", "compensation"):
        assert first.state[source][field].data_ptr() != second.state[target][field].data_ptr()
    source.grad = torch.full_like(source, 0.3)
    target.grad = torch.full_like(target, 0.3)
    first.step()
    second.step()
    torch.testing.assert_close(source, target, rtol=0, atol=0)


def test_anyprecision_direct_load_does_not_reuse_preexisting_alias():
    source = torch.nn.Parameter(torch.ones(2))
    target = torch.nn.Parameter(torch.ones(2))
    first = AnyPrecisionAdamW([source])
    second = AnyPrecisionAdamW([target])
    source.grad = torch.ones_like(source)
    first.step()
    second.state[target] = dict(first.state[source])
    second.load_state_dict(first.state_dict())
    for field in ("step", "exp_avg", "exp_avg_sq", "compensation"):
        assert first.state[source][field].data_ptr() != second.state[target][field].data_ptr()


def test_anyprecision_owned_buffer_reuse_requires_exact_view():
    from veomni.optim.optimizer import _same_owned_optimizer_tensor

    storage = torch.arange(8)
    assert _same_owned_optimizer_tensor(storage[:4], storage[:4])
    assert not _same_owned_optimizer_tensor(storage[:4], storage[1:5])
    assert not _same_owned_optimizer_tensor(storage[:4], storage[:8:2])
    assert not _same_owned_optimizer_tensor(storage[:0], storage[:0])


@pytest.mark.parametrize("bad_field", ["exp_avg", "exp_avg_sq", "compensation", "step"])
def test_anyprecision_direct_load_rejects_dtensor_state_for_plain_parameter(tmp_path, bad_field):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Replicate

    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rdzv'}", rank=0, world_size=1)
    try:
        param = torch.nn.Parameter(torch.ones(2))
        optimizer = AnyPrecisionAdamW([param])
        param.grad = torch.full_like(param, 0.2)
        optimizer.step()
        bad = optimizer.state_dict()
        mesh = init_device_mesh("cpu", (1,))
        value = bad["state"][0][bad_field]
        bad["state"][0][bad_field] = DTensor.from_local(value, mesh, [Replicate()])
        target = torch.nn.Parameter(param.detach().clone())
        restored = AnyPrecisionAdamW([target])
        with pytest.raises(RuntimeError, match="incompatible|scalar tensor"):
            restored.load_state_dict(bad)
        assert not restored.state
    finally:
        dist.destroy_process_group()


def test_anyprecision_direct_load_rejects_invalid_local_shard_shape(tmp_path):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard

    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rdzv'}", rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,))
        param = torch.nn.Parameter(DTensor.from_local(torch.ones(4, 4), mesh, [Shard(0)]))
        optimizer = AnyPrecisionAdamW([param])
        param.grad = torch.ones_like(param)
        optimizer.step()
        bad = optimizer.state_dict()
        bad["state"][0]["exp_avg"] = DTensor.from_local(
            torch.zeros(2, 4, dtype=torch.bfloat16), mesh, [Shard(0)], shape=torch.Size((4, 4)), stride=(4, 1)
        )
        restored = AnyPrecisionAdamW([torch.nn.Parameter(param.detach().clone())])
        with pytest.raises(RuntimeError, match="incompatible exp_avg layout"):
            restored.load_state_dict(bad)
        assert not restored.state
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("bad_field", ["replicated_moment", "wrong_local_shape", "plain_moment", "dtensor_step"])
def test_anyprecision_source_rejects_incompatible_state_layout(tmp_path, bad_field):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Replicate, Shard

    from veomni.checkpoint.multi_optimizer_checkpoint import _validate_source_optimizer_states

    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rdzv'}", rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,))
        param = torch.nn.Parameter(DTensor.from_local(torch.ones(2, 2), mesh, [Shard(0)]))
        model = torch.nn.Linear(2, 2, bias=False)
        model.weight = param
        leaf = AnyPrecisionAdamW(
            [param], momentum_dtype=torch.bfloat16, variance_dtype=torch.bfloat16, use_kahan_summation=False
        )
        optimizer = MultiOptimizer(model, {"weight": leaf}, ["weight"])
        state = {
            "step": torch.tensor(1.0),
            "exp_avg": torch.zeros_like(param, dtype=torch.bfloat16),
            "exp_avg_sq": torch.zeros_like(param, dtype=torch.bfloat16),
        }
        if bad_field == "replicated_moment":
            state["exp_avg"] = DTensor.from_local(torch.zeros(2, 2, dtype=torch.bfloat16), mesh, [Replicate()])
        elif bad_field == "wrong_local_shape":
            state["exp_avg"] = DTensor.from_local(
                torch.zeros(1, 2, dtype=torch.bfloat16), mesh, [Shard(0)], shape=torch.Size((2, 2)), stride=(2, 1)
            )
        elif bad_field == "plain_moment":
            state["exp_avg"] = torch.zeros(2, 2, dtype=torch.bfloat16)
        else:
            state["step"] = DTensor.from_local(torch.tensor(1.0), mesh, [Replicate()])
        leaf.state[param] = state
        with pytest.raises(RuntimeError, match="layout differs|ordinary scalar tensor"):
            _validate_source_optimizer_states(optimizer, model)
        assert leaf.state[param] is state
    finally:
        dist.destroy_process_group()


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
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target, restored = build()
    DistributedCheckpointer.load(
        str(tmp_path),
        {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
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
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
            trainable_only=True,
        )


def test_trainable_only_resume_rejects_frozen_parameter_with_saved_history(tmp_path):
    model, optimizer = _model_and_optimizer()
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    optimizer.zero_grad()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target, restored = _model_and_optimizer()
    target.weight.requires_grad_(False)
    original = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="trainable-only resume cannot prove"):
        DistributedCheckpointer.load(
            str(tmp_path),
            {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
            parallel_state=parallel,
            trainable_only=True,
        )
    torch.testing.assert_close(target.weight, original, rtol=0, atol=0)


def test_multi_optimizer_uninitialized_leaf_uses_its_first_real_step(tmp_path):
    model, optimizer = _model_and_optimizer()
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    optimizer.zero_grad()
    bias_before = model.bias.detach().clone()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    torch.testing.assert_close(model.bias, bias_before, rtol=0, atol=0)
    assert model.bias not in optimizer.optimizers_dict["bias"].state

    target, restored = _model_and_optimizer()
    DistributedCheckpointer.load(
        str(tmp_path),
        {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
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


def test_multi_model_generation_keeps_standard_torch_export_strictly_loadable(tmp_path):
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    model, optimizer = _model_and_optimizer()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
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
            {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {"condition_rng": torch.tensor(index)},
                "_veomni_scheduler_object": None,
            },
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
            str(second),
            {"model": target, "optimizer": restored, "extra_state": {}, "_veomni_scheduler_object": None},
            parallel_state=parallel,
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
        DistributedCheckpointer.save(
            str(path),
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
    original_preflight = checkpoint._preflight_multi_optimizer_layout

    def replace_after_preflight(*args, **kwargs):
        verified = original_preflight(*args, **kwargs)
        shutil.rmtree(second / "model" / "ckpt")
        shutil.copytree(first / "model" / "ckpt", second / "model" / "ckpt")
        return verified

    monkeypatch.setattr(checkpoint, "_preflight_multi_optimizer_layout", replace_after_preflight)
    target, restored = _model_and_optimizer()
    with pytest.raises(BaseException, match="(?i)checkpoint.*(changed|generation)"):
        DistributedCheckpointer.load(
            str(second),
            {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )


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
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )

    target, restored = build()
    DistributedCheckpointer.load(
        str(tmp_path),
        {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
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
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target, changed = build(4)
    with pytest.raises(RuntimeError, match="(?i)optimizer.*layout"):
        DistributedCheckpointer.load(
            str(tmp_path),
            {"model": target, "optimizer": changed, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )


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
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )

    target, restored = build()
    target.weight.requires_grad_(not target_frozen)
    if not target_frozen:
        with pytest.raises(RuntimeError, match="(?i)optimizer.*layout"):
            DistributedCheckpointer.load(
                str(tmp_path),
                {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
                parallel_state=parallel,
            )
        return
    DistributedCheckpointer.load(
        str(tmp_path),
        {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
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
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
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
        DistributedCheckpointer.save(
            str(tmp_path),
            {"model": source, "optimizer": optimizer, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
    assert not list(tmp_path.iterdir())

    source.bias.grad = None
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": source, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target, restored = build()
    target.bias.grad = torch.ones_like(target.bias)
    before = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="cannot omit pending gradients"):
        DistributedCheckpointer.load(
            str(tmp_path),
            {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
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
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
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
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    path = Path(optimizer_dir(str(tmp_path))) / ".metadata"
    with path.open("rb") as handle:
        metadata = pickle.load(handle)
    del metadata.state_dict_metadata["optimizer.state.weight.exp_avg_sq"]
    with path.open("wb") as handle:
        pickle.dump(metadata, handle)
    target, restored = _model_and_optimizer()
    before = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="incomplete or unsupported parameter state|inventory"):
        DistributedCheckpointer.load(
            str(tmp_path),
            {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
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
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": source, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
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
        DistributedCheckpointer.load(
            str(tmp_path),
            {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)


def test_multi_optimizer_rejects_stale_sidecar_before_overwrite(tmp_path):
    model, optimizer = _model_and_optimizer()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "extra_state": {"cursor": "old"}, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    with pytest.raises(RuntimeError, match="sidecar cannot be reused"):
        DistributedCheckpointer.save(
            str(tmp_path),
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )


def test_multi_optimizer_rejects_duplicate_ownership_before_checkpoint_write(tmp_path):
    model, optimizer = _model_and_optimizer()
    optimizer.optimizers_dict["bias"].param_groups[0]["params"].append(model.weight)
    with pytest.raises(RuntimeError, match="ownership is duplicated"):
        DistributedCheckpointer.save(
            str(tmp_path),
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
    assert not list(tmp_path.iterdir())


def test_multi_optimizer_rejects_empty_parameter_group_before_checkpoint_write(tmp_path):
    model = torch.nn.Linear(2, 2)
    leaf = torch.optim.AdamW([{"params": []}, {"params": [model.weight]}])
    optimizer = MultiOptimizer(model, {"leaf": leaf}, ["leaf"])
    with pytest.raises(RuntimeError, match="empty parameter group"):
        DistributedCheckpointer.save(
            str(tmp_path),
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
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
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
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
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target, restored = build(torch.bfloat16)
    before = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="layout|identity"):
        DistributedCheckpointer.load(
            str(tmp_path),
            {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
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
        DistributedCheckpointer.save(
            str(tmp_path),
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
    assert not list(tmp_path.iterdir())


def test_model_checkpoint_extra_state_records_optimizer_global_step():
    from veomni.models.checkpoint.manager import ModelCheckpointManager, _scheduler_recipe

    _, optimizer = _model_and_optimizer()
    manager = object.__new__(ModelCheckpointManager)
    manager.runtime = SimpleNamespace(optimizer=optimizer, lr_scheduler=None, extra_state=lambda: {})
    saved = manager._extra_state(SimpleNamespace(global_step=3))
    assert saved["_veomni_optimizer_global_step"] == 3
    assert saved["_veomni_scheduler_recipe"] == {"status": "supported", "value": _scheduler_recipe(None, optimizer)}


def test_proven_optimizer_load_sets_identity_receipt(tmp_path):
    model, optimizer = _model_and_optimizer()
    parallel = SimpleNamespace(dp_mode="ddp")
    generation = DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target, restored = _model_and_optimizer()
    state = {"model": target, "optimizer": restored, "_veomni_scheduler_object": None}
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
        str(tmp_path / "first"),
        {"model": first, "optimizer": first_optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    DistributedCheckpointer.save(
        str(tmp_path / "second"),
        {"model": second, "optimizer": second_optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target, restored = _model_and_optimizer()
    state = {"model": target, "optimizer": restored, "_veomni_scheduler_object": None}
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
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target, restored = _model_and_optimizer()
    target.register_buffer("sensitive", torch.tensor([0.2], dtype=torch.bfloat16))
    before = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="tensor dtype or shape differs"):
        DistributedCheckpointer.load(
            str(tmp_path),
            {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)


def test_source_model_extra_buffer_rejected_before_model_load(tmp_path):
    model, optimizer = _model_and_optimizer()
    model.register_buffer("offset", torch.tensor([0.5]))
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target, restored = _model_and_optimizer()
    before = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="model state inventory differs"):
        DistributedCheckpointer.load(
            str(tmp_path),
            {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)


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
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target, restored = build(False)
    before = target.p.detach().clone()
    with pytest.raises(RuntimeError, match="layout|identity"):
        DistributedCheckpointer.load(
            str(tmp_path),
            {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
    torch.testing.assert_close(target.p, before, rtol=0, atol=0)


def _save_destination_stat_error_worker(rank, checkpoint, location, endpoint):
    import errno
    import os
    from datetime import timedelta
    from unittest.mock import patch

    import torch.distributed as dist

    from veomni.checkpoint.layout import manifest_path, model_dir

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        root = Path(checkpoint)
        if rank == 0:
            root.mkdir(parents=True)
            (root / "sentinel").write_bytes(b"preserve-existing-data")
        dist.barrier()
        target = (
            manifest_path(checkpoint)
            if location == "manifest"
            else os.path.join(model_dir(checkpoint, ""), "extra_state", "extra_state_rank_1.pt")
        )
        original_stat = os.stat
        injected = False

        def stat(path, *args, **kwargs):
            nonlocal injected
            if rank == 1 and isinstance(path, (str, os.PathLike)) and os.fspath(path) == target:
                injected = True
                raise OSError(errno.EIO, "injected checkpoint stat failure")
            return original_stat(path, *args, **kwargs)

        model, optimizer = _model_and_optimizer()
        with (
            patch("veomni.checkpoint.dcp_checkpointer.os.stat", side_effect=stat),
            pytest.raises((OSError, RuntimeError), match="stat failure|failed on another rank") as captured,
        ):
            DistributedCheckpointer.save(
                checkpoint,
                {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
                parallel_state=SimpleNamespace(dp_mode="ddp"),
            )
        if rank == 1:
            assert injected and "injected checkpoint stat failure" in str(captured.value)
        else:
            assert "failed on another rank" in str(captured.value)
            assert (root / "sentinel").read_bytes() == b"preserve-existing-data"
            assert not Path(model_dir(checkpoint, "")).exists()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("location", ["manifest", "sidecar"])
def test_multi_optimizer_save_coordinates_destination_stat_errors(tmp_path, location):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_save_destination_stat_error_worker, args=(str(tmp_path / "ckpt"), location), nprocs=2)


@pytest.mark.parametrize("old_model_state", [True, False])
def test_multi_optimizer_save_preserves_completed_checkpoint(tmp_path, old_model_state):
    from veomni.checkpoint.layout import checkpoint_is_complete, write_manifest

    model, optimizer = _model_and_optimizer()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model}, parallel_state=parallel)
    write_manifest(str(tmp_path), global_step=0, world_size=1, model_state=old_model_state)
    assert checkpoint_is_complete(str(tmp_path))
    with pytest.raises(RuntimeError, match="Cannot overwrite a completed checkpoint"):
        DistributedCheckpointer.save(
            str(tmp_path),
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
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
            str(tmp_path),
            {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
    assert not checkpoint_is_complete(str(tmp_path))


def test_multi_checkpoint_requires_explicit_scheduler_declaration(tmp_path):
    model, optimizer = _model_and_optimizer()
    # Ordinary PyTorch schedulers can be attached without VeOmni's registry.
    # Omission must not silently be interpreted as a claim that none exists.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer.optimizers_dict["weight"], step_size=1)
    state = {"model": model, "optimizer": optimizer}
    with pytest.raises(RuntimeError, match="explicit scheduler declaration"):
        DistributedCheckpointer.save(str(tmp_path), state, parallel_state=SimpleNamespace(dp_mode="ddp"))
    assert not (tmp_path / "model").exists()
    state["_veomni_scheduler_object"] = scheduler
    with pytest.raises(RuntimeError, match="scheduler object has no matching scheduler binding"):
        DistributedCheckpointer.save(str(tmp_path), state, parallel_state=SimpleNamespace(dp_mode="ddp"))
    assert not (tmp_path / "model").exists()


def test_scheduler_recipe_rejects_schedules_outside_the_checkpoint_contract():
    from veomni.models.checkpoint.manager import _scheduler_recipe

    _, optimizer = _model_and_optimizer()
    leaf = optimizer.optimizers_dict["weight"]
    step = torch.optim.lr_scheduler.StepLR(leaf, step_size=1)
    with pytest.raises(RuntimeError, match="scheduler implementation or binding is unsupported"):
        _scheduler_recipe(step, leaf)

    scheduler = torch.optim.lr_scheduler.LambdaLR(leaf, lambda value, *, factor=0.5: value * factor)
    with pytest.raises(RuntimeError, match="scheduler captures unsupported mutable state"):
        _scheduler_recipe(scheduler, leaf)


@pytest.mark.parametrize("kind", ["function", "comprehension"])
def test_scheduler_recipe_rejects_global_in_nested_function(monkeypatch, kind):
    from veomni.checkpoint.optimizer_resume_layout import _scheduler_functions

    monkeypatch.setitem(globals(), "_TEST_NESTED_LR_FACTOR", 0.5)

    def schedule(step):
        def nested():
            return _TEST_NESTED_LR_FACTOR  # noqa: F821

        return step * nested()

    def comprehension(step):
        return sum(value * _TEST_NESTED_LR_FACTOR for value in range(step))  # noqa: F821

    # Changing this global changes the next learning-rate update even though
    # the outer code object never names it. Its identity is therefore unsafe.
    with pytest.raises(RuntimeError, match="unsupported global state"):
        _scheduler_functions(SimpleNamespace(lr_lambdas=[schedule if kind == "function" else comprehension]))


def test_scheduler_recipe_accepts_nested_pure_function():
    from veomni.checkpoint.optimizer_resume_layout import _scheduler_functions

    def schedule(step):
        def nested(value):
            return value * 0.5

        return nested(step)

    recipe = _scheduler_functions(SimpleNamespace(lr_lambdas=[schedule]))
    assert len(recipe) == 1
    assert schedule(2) == 1.0


def test_scheduler_recipe_accepts_local_temporary_cell():
    from veomni.checkpoint.optimizer_resume_layout import _scheduler_functions

    def schedule(step):
        value = step * 0.5
        return (lambda: value)()

    assert len(_scheduler_functions(SimpleNamespace(lr_lambdas=[schedule]))) == 1
    assert schedule(2) == 1.0


def test_scheduler_recipe_accepts_math_attribute_with_unrelated_global(monkeypatch):
    import math as math_module

    from veomni.checkpoint.optimizer_resume_layout import _scheduler_functions

    monkeypatch.setitem(globals(), "math", math_module)
    monkeypatch.setitem(globals(), "cos", "unrelated")

    def schedule(step):
        def nested(value):
            return math.cos(value)  # noqa: F821

        return nested(step)

    assert len(_scheduler_functions(SimpleNamespace(lr_lambdas=[schedule]))) == 1


@pytest.mark.parametrize("kind", ["set", "frozenset_constant"])
def test_scheduler_recipe_rejects_process_dependent_computation(kind):
    from veomni.checkpoint.optimizer_resume_layout import _scheduler_functions

    def unordered(step):
        return sum(float(value) for value in {"0.125", "1e16", "-1e16"})

    def unordered_constant(step):
        return step in {1, 2, 3}

    schedule = {"set": unordered, "frozenset_constant": unordered_constant}[kind]
    with pytest.raises(RuntimeError, match="unsupported.*state"):
        _scheduler_functions(SimpleNamespace(lr_lambdas=[schedule]))


def test_scheduler_recipe_accepts_deterministic_format_and_records_its_identity():
    from veomni.checkpoint.optimizer_resume_layout import _scheduler_functions

    def build_schedule(two_decimals):
        if two_decimals:

            def schedule(step):
                return float(f"{step / 300:.2f}")
        else:

            def schedule(step):
                return float(f"{step / 300:.3f}")

        return schedule

    two_decimals = build_schedule(True)
    three_decimals = build_schedule(False)
    assert two_decimals.__qualname__ == three_decimals.__qualname__
    two = _scheduler_functions(SimpleNamespace(lr_lambdas=[two_decimals]))
    three = _scheduler_functions(SimpleNamespace(lr_lambdas=[three_decimals]))
    assert two_decimals(13) != three_decimals(13)
    assert two != three


def test_scheduler_recipe_rejects_tuple_subclass_closure():
    from veomni.checkpoint.optimizer_resume_layout import _scheduler_functions

    class StatefulTuple(tuple):
        def __getitem__(self, index):
            return 0.5

    factor = StatefulTuple((1.0,))

    def schedule(step):
        return step * factor[0]

    with pytest.raises(RuntimeError, match="unsupported mutable state"):
        _scheduler_functions(SimpleNamespace(lr_lambdas=[schedule]))


def test_scheduler_recipe_rejects_nonlocal_mutation():
    from veomni.checkpoint.optimizer_resume_layout import _scheduler_functions

    state = 1.0

    def schedule(step):
        nonlocal state
        state += 1.0
        return step * state

    with pytest.raises(RuntimeError, match="unsupported dynamic state"):
        _scheduler_functions(SimpleNamespace(lr_lambdas=[schedule]))


@pytest.mark.parametrize("builtin_name", ["eval", "globals", "getattr", "__import__"])
def test_scheduler_recipe_rejects_dynamic_builtin_in_nested_function(builtin_name):
    from veomni.checkpoint.optimizer_resume_layout import _scheduler_functions

    def schedule_eval(step):
        return lambda: eval("step")  # noqa: S307

    def schedule_globals(step):
        return lambda: globals()

    def schedule_getattr(step):
        return lambda: getattr(step, str(step))

    def schedule_import(step):
        return lambda: __import__("math")

    schedules = {
        "eval": schedule_eval,
        "globals": schedule_globals,
        "getattr": schedule_getattr,
        "__import__": schedule_import,
    }
    with pytest.raises(RuntimeError, match="unsupported.*state|unsupported.*builtin"):
        _scheduler_functions(SimpleNamespace(lr_lambdas=[schedules[builtin_name]]))


def test_nested_global_scheduler_is_rejected_before_checkpoint_load(tmp_path, monkeypatch):
    model, optimizer = _model_and_optimizer()
    scheduler = build_lr_scheduler(optimizer, train_steps=10)
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
    target_scheduler = build_lr_scheduler(restored, train_steps=10)
    monkeypatch.setitem(globals(), "_TEST_NESTED_LR_FACTOR", 0.5)

    def schedule(step):
        def nested():
            return _TEST_NESTED_LR_FACTOR  # noqa: F821

        return step * nested()

    target_scheduler["weight"].lr_lambdas[0] = schedule
    before = {name: value.detach().clone() for name, value in target.named_parameters()}
    scheduler_before = target_scheduler.state_dict()
    with pytest.raises(RuntimeError, match="unsupported global state"):
        DistributedCheckpointer.load(
            str(tmp_path),
            {
                "model": target,
                "optimizer": restored,
                "extra_state": {},
                "_veomni_scheduler_object": target_scheduler,
            },
            parallel_state=parallel,
        )
    for name, value in target.named_parameters():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert target_scheduler.state_dict() == scheduler_before
    assert all(not leaf.state for leaf in restored.optimizers_dict.values())


def test_multi_checkpoint_accepts_explicit_no_scheduler(tmp_path):
    model, optimizer = _model_and_optimizer()
    model.weight.grad = torch.ones_like(model.weight)
    model.bias.grad = torch.ones_like(model.bias)
    optimizer.step()
    optimizer.zero_grad()
    parallel = SimpleNamespace(dp_mode="ddp")
    source = {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None}
    DistributedCheckpointer.save(str(tmp_path), source, parallel_state=parallel)
    target_model, target_optimizer = _model_and_optimizer()
    target = {"model": target_model, "optimizer": target_optimizer, "_veomni_scheduler_object": None}
    DistributedCheckpointer.load(str(tmp_path), target, parallel_state=parallel)
    for actual, expected in zip(target_model.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(target_optimizer.state_dict(), optimizer.state_dict(), rtol=0, atol=0)
    for source_param, target_param in zip(model.parameters(), target_model.parameters(), strict=True):
        source_param.grad = torch.full_like(source_param, 0.25)
        target_param.grad = torch.full_like(target_param, 0.25)
    optimizer.step()
    target_optimizer.step()
    for actual, expected in zip(target_model.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(target_optimizer.state_dict(), optimizer.state_dict(), rtol=0, atol=0)


def test_deterministic_formatted_scheduler_recovers_next_update(tmp_path):
    import weakref

    from veomni.optim.lr_scheduler import _CHECKPOINT_SCHEDULER_BINDINGS, MultiLRScheduler

    def build_formatted_scheduler(optimizer, *, precision=4):
        schedules = {}
        for name, leaf in optimizer.optimizers_dict.items():
            if precision == 4:

                def lr_lambda(step):
                    return float(f"{1 / (step + 1):.4f}")
            else:

                def lr_lambda(step):
                    return float(f"{1 / (step + 1):.3f}")

            schedule = torch.optim.lr_scheduler.LambdaLR(leaf, lr_lambda)
            _CHECKPOINT_SCHEDULER_BINDINGS[leaf] = weakref.ref(schedule)
            schedules[name] = schedule
        return MultiLRScheduler(schedules)

    torch.manual_seed(12)
    model, optimizer = _model_and_optimizer()
    scheduler = build_formatted_scheduler(optimizer)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
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
    changed, changed_optimizer = _model_and_optimizer()
    changed_scheduler = build_formatted_scheduler(changed_optimizer, precision=3)
    before = [parameter.detach().clone() for parameter in changed.parameters()]
    with pytest.raises(RuntimeError, match="(?i)(scheduler|identity|layout)"):
        DistributedCheckpointer.load(
            str(tmp_path),
            {
                "model": changed,
                "optimizer": changed_optimizer,
                "extra_state": {},
                "_veomni_scheduler_object": changed_scheduler,
            },
            parallel_state=parallel,
        )
    for parameter, original in zip(changed.parameters(), before, strict=True):
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)
    assert all(not leaf.state for leaf in changed_optimizer.optimizers_dict.values())

    target, target_optimizer = _model_and_optimizer()
    target_scheduler = build_formatted_scheduler(target_optimizer)
    restored = {
        "model": target,
        "optimizer": target_optimizer,
        "extra_state": {},
        "_veomni_scheduler_object": target_scheduler,
    }
    DistributedCheckpointer.load(str(tmp_path), restored, parallel_state=parallel)
    target_scheduler.load_state_dict(restored["extra_state"]["lr_scheduler"])
    for source_parameter, target_parameter in zip(model.parameters(), target.parameters(), strict=True):
        source_parameter.grad = torch.full_like(source_parameter, 0.25)
        target_parameter.grad = torch.full_like(target_parameter, 0.25)
    optimizer.step()
    target_optimizer.step()
    scheduler.step()
    target_scheduler.step()
    assert scheduler.state_dict() == target_scheduler.state_dict()
    for source_parameter, target_parameter in zip(model.parameters(), target.parameters(), strict=True):
        torch.testing.assert_close(source_parameter, target_parameter, rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state_dict(), target_optimizer.state_dict(), rtol=0, atol=0)


def test_direct_multi_checkpoint_applies_returned_scheduler_state(tmp_path):
    model, optimizer = _model_and_optimizer()
    scheduler = build_lr_scheduler(optimizer, train_steps=10, lr_decay_style="linear")
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    parallel = SimpleNamespace(dp_mode="ddp")
    source = {
        "model": model,
        "optimizer": optimizer,
        "extra_state": {"lr_scheduler": scheduler.state_dict()},
        "_veomni_scheduler_object": scheduler,
    }
    DistributedCheckpointer.save(str(tmp_path), source, parallel_state=parallel)
    target, target_optimizer = _model_and_optimizer()
    target_scheduler = build_lr_scheduler(target_optimizer, train_steps=10, lr_decay_style="linear")
    restored = {
        "model": target,
        "optimizer": target_optimizer,
        "extra_state": {},
        "_veomni_scheduler_object": target_scheduler,
    }
    DistributedCheckpointer.load(str(tmp_path), restored, parallel_state=parallel)
    target_scheduler.load_state_dict(restored["extra_state"]["lr_scheduler"])
    assert scheduler.state_dict() == target_scheduler.state_dict()
    for source_param, target_param in zip(model.parameters(), target.parameters(), strict=True):
        source_param.grad = torch.full_like(source_param, 0.25)
        target_param.grad = torch.full_like(target_param, 0.25)
    optimizer.step()
    target_optimizer.step()
    scheduler.step()
    target_scheduler.step()
    for actual, expected in zip(target.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert scheduler.state_dict() == target_scheduler.state_dict()
