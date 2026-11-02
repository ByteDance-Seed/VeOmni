"""Distributed coverage for trainable PLE shards with no local requests."""

import os
from importlib import import_module
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
from torch.utils.checkpoint import DefaultDeviceType, checkpoint

from scripts.ci.accelerator.launch_plan import LaunchSpec


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
_CI_CPU_2 = LaunchSpec(nproc=2, device="cpu")
_CI_CPU_4 = LaunchSpec(nproc=4, device="cpu")


def _model_module(backend):
    return import_module(f"veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_{backend}")


def _isolate_cpu_spawn(monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    for name in ("CUDA_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES"):
        monkeypatch.setenv(name, "")


def _configure_cpu_worker():
    assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
    for name in ("CUDA_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES"):
        assert os.environ.get(name) == ""
    DefaultDeviceType.set_device_type("cpu")
    assert DefaultDeviceType.get_device_type() == "cpu"


def _worker(rank, rendezvous, dtype, backend, world_size):
    _configure_cpu_worker()
    model_module = _model_module(backend)

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cpu", (world_size // 2, 2), mesh_dim_names=("ple_fsdp", "ple"))
        state = SimpleNamespace(
            extra_parallel_sizes={"ple": 2}, extra_parallel_flat_group=lambda name: dist.group.WORLD
        )
        model_module.get_parallel_state = lambda: state
        lookup = model_module.Qwen4ExpTextNGramEmbedding._lookup_local_rows

        for mode in ("plain", "non_reentrant", "reentrant"):
            tables = torch.nn.ModuleDict({f"shard_{i}": torch.nn.Embedding(4, 4) for i in range(3)})
            for index, table in enumerate(tables.values()):
                values = torch.arange(16, dtype=torch.float32).reshape(4, 4).add_(100 * index).to(dtype)
                table.weight = torch.nn.Parameter(distribute_tensor(values, mesh, [Shard(1), Shard(0)]))
            fixture = SimpleNamespace(ngram_embedding=tables)
            shard = rank % 2
            shard_ids = torch.tensor([shard, shard])
            row_ids = torch.tensor([0, 1])
            anchor = torch.ones((), requires_grad=True)

            def run(a, fixture=fixture, shard_ids=shard_ids, row_ids=row_ids):
                return lookup(fixture, shard_ids, row_ids) + a * 0

            if mode == "plain":
                output = run(anchor)
            else:
                output = checkpoint(run, anchor, use_reentrant=mode == "reentrant")
            expected = tables[f"shard_{shard}"].weight.to_local().detach()[row_ids]
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            output.sum().backward()
            for index, table in enumerate(tables.values()):
                grad = table.weight.grad
                if index == 2:
                    assert grad is None
                    continue
                assert grad is not None
                expected_grad = (
                    torch.ones_like(table.weight.to_local())
                    if index == shard
                    else torch.zeros_like(table.weight.to_local())
                )
                torch.testing.assert_close(grad.to_local(), expected_grad, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def _trainability_worker(rank, rendezvous, backend, mode):
    from datetime import timedelta

    _configure_cpu_worker()
    model_module = _model_module(backend)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:
        mesh = init_device_mesh("cpu", (1, 2), mesh_dim_names=("ple_fsdp", "ple"))
        state = SimpleNamespace(
            extra_parallel_sizes={"ple": 2}, extra_parallel_flat_group=lambda name: dist.group.WORLD
        )
        model_module.get_parallel_state = lambda: state
        tables = torch.nn.ModuleDict({f"shard_{i}": torch.nn.Embedding(4, 4) for i in range(2)})
        for index, table in enumerate(tables.values()):
            weight = torch.arange(16, dtype=torch.float32).reshape(4, 4).add_(100 * index)
            table.weight = torch.nn.Parameter(distribute_tensor(weight, mesh, [Shard(1), Shard(0)]))
            trainable = mode == "partial_agreed" and index == 0
            table.weight.requires_grad_(trainable)
        fixture = SimpleNamespace(ngram_embedding=tables)
        lookup = model_module.Qwen4ExpTextNGramEmbedding._lookup_local_rows
        selected_shard = 1 if mode == "partial_agreed" and rank == 1 else 0
        shard_ids, row_ids = torch.tensor([selected_shard, selected_shard]), torch.tensor([0, 1])
        calls = []
        all_reduce = dist.all_reduce

        def record_all_reduce(tensor, *args, **kwargs):
            calls.append(tensor.numel())
            return all_reduce(tensor, *args, **kwargs)

        dist.all_reduce = record_all_reduce
        try:
            actual = lookup(fixture, shard_ids, row_ids)
        finally:
            dist.all_reduce = all_reduce
        assert calls == ([] if mode == "all_frozen" else [len(tables)])
        expected = tables[f"shard_{selected_shard}"].weight.to_local().detach()[row_ids]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.requires_grad == (mode == "partial_agreed")
        if mode == "partial_agreed":
            actual.sum().backward()
            torch.testing.assert_close(
                tables["shard_0"].weight.grad.to_local(),
                torch.full_like(tables["shard_0"].weight.to_local(), 1.0 if rank == 0 else 0.0),
                rtol=0,
                atol=0,
            )
            assert tables["shard_1"].weight.grad is None
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("backend", ["gpu", "npu"])
@pytest.mark.parametrize("mode", ["all_frozen", "partial_agreed"])
@pytest.mark.launch_plan(provider=_CI_CPU_2)
def test_ple_shard_trainability_agreement(tmp_path, monkeypatch, backend, mode):
    _isolate_cpu_spawn(monkeypatch)
    mp.spawn(_trainability_worker, args=(str(tmp_path / "gloo"), backend, mode), nprocs=2, join=True)


def _distributed_partly_frozen_worker(rank, rendezvous, backend):
    from datetime import timedelta

    _configure_cpu_worker()
    model_module = _model_module(backend)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:
        mesh = init_device_mesh("cpu", (1, 2), mesh_dim_names=("ple_fsdp", "ple"))
        state = SimpleNamespace(
            extra_parallel_sizes={"ple": 2},
            extra_parallel_flat_group=lambda name: dist.group.WORLD,
            extra_parallel_fsdp_device_mesh={"ple": mesh},
            extra_parallel_2d_rank_table=lambda name: ((0, 1),),
            extra_parallel_gradient_divide_factor=lambda name: 2,
        )
        model_module.get_parallel_state = lambda: state
        for mode in ("plain", "non_reentrant", "reentrant"):
            tables = torch.nn.ModuleDict({f"shard_{i}": torch.nn.Embedding(4, 4) for i in range(2)})
            for index, table in enumerate(tables.values()):
                values = torch.arange(16, dtype=torch.float32).reshape(4, 4).add_(100 * index)
                table.weight = torch.nn.Parameter(distribute_tensor(values, mesh, [Shard(1), Shard(0)]))
                table.weight.requires_grad_(index == 0)
            fixture = SimpleNamespace(ngram_embedding=tables, padded_rows_per_shard=4)
            fixture._lookup_local_rows = MethodType(
                model_module.Qwen4ExpTextNGramEmbedding._lookup_local_rows, fixture
            )
            fixture._distributed_lookup = MethodType(
                model_module.Qwen4ExpTextNGramEmbedding._distributed_lookup, fixture
            )
            shard_ids = torch.tensor([rank, rank])
            row_ids = torch.tensor([2 * rank, 2 * rank + 1])
            anchor = torch.ones((), requires_grad=True)

            def run(a, fixture=fixture, shard_ids=shard_ids, row_ids=row_ids):
                return fixture._distributed_lookup(shard_ids, row_ids) + a * 0

            output = run(anchor) if mode == "plain" else checkpoint(run, anchor, use_reentrant=mode == "reentrant")
            expected = torch.arange(16, dtype=torch.float32).reshape(4, 4)[row_ids] + 100 * rank
            torch.testing.assert_close(output.detach(), expected, rtol=0, atol=0)
            output.sum().backward()
            grad = tables["shard_0"].weight.grad
            assert grad is not None
            torch.testing.assert_close(grad.to_local(), torch.full_like(grad.to_local(), 0.5 if rank == 0 else 0.0))
            assert tables["shard_1"].weight.grad is None
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("backend", ["gpu", "npu"])
@pytest.mark.launch_plan(provider=_CI_CPU_2)
def test_distributed_ple_partly_frozen_backward(tmp_path, monkeypatch, backend):
    _isolate_cpu_spawn(monkeypatch)
    mp.spawn(_distributed_partly_frozen_worker, args=(str(tmp_path / "gloo"), backend), nprocs=2, join=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("backend", ["gpu", "npu"])
@pytest.mark.launch_plan(provider=_CI_CPU_2)
def test_trainable_ple_empty_shards_preserve_outputs_and_gradients_under_checkpoint(
    tmp_path, monkeypatch, dtype, backend
):
    _isolate_cpu_spawn(monkeypatch)
    mp.spawn(
        _worker, args=(str(tmp_path / "gloo"), dtype, backend, _CI_CPU_2.nproc), nprocs=_CI_CPU_2.nproc, join=True
    )


@pytest.mark.parametrize("backend", ["gpu", "npu"])
def test_ple_empty_identity_does_not_save_weight_as_activation(backend):
    identity = _model_module(backend)._Qwen4ExpAttachZeroWeightGrad
    weight = torch.nn.Parameter(torch.randn(8, 4))
    output = torch.randn(2, 4, requires_grad=True)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: saved.append(tensor) or tensor, lambda tensor: tensor
    ):
        actual = identity.apply(output, weight)
        actual.sum().backward()
    assert not saved
    torch.testing.assert_close(weight.grad, torch.zeros_like(weight), rtol=0, atol=0)


def _distributed_lookup_worker(rank, rendezvous, dtype, backend, world_size):
    _configure_cpu_worker()
    model_module = _model_module(backend)

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cpu", (world_size // 2, 2), mesh_dim_names=("ple_fsdp", "ple"))
        state = SimpleNamespace(
            extra_parallel_sizes={"ple": 2},
            extra_parallel_flat_group=lambda name: dist.group.WORLD,
            extra_parallel_fsdp_device_mesh={"ple": mesh},
            extra_parallel_2d_rank_table=lambda name: tuple(
                tuple(2 * column + row for row in range(2)) for column in range(world_size // 2)
            ),
            extra_parallel_gradient_divide_factor=lambda name: world_size,
        )
        model_module.get_parallel_state = lambda: state
        for mode in ("plain", "non_reentrant", "reentrant"):
            tables = torch.nn.ModuleDict({f"shard_{i}": torch.nn.Embedding(4, 4) for i in range(3)})
            for index, table in enumerate(tables.values()):
                values = torch.arange(16, dtype=torch.float32).reshape(4, 4).add_(100 * index).to(dtype)
                table.weight = torch.nn.Parameter(distribute_tensor(values, mesh, [Shard(1), Shard(0)]))
            fixture = SimpleNamespace(ngram_embedding=tables, padded_rows_per_shard=4)
            fixture._lookup_local_rows = MethodType(
                model_module.Qwen4ExpTextNGramEmbedding._lookup_local_rows, fixture
            )
            fixture._distributed_lookup = MethodType(
                model_module.Qwen4ExpTextNGramEmbedding._distributed_lookup, fixture
            )
            shard = rank % 2
            shard_ids, row_ids = torch.tensor([shard, shard]), torch.tensor([0, 1])
            anchor = torch.ones((), requires_grad=True)

            def run(a, fixture=fixture, shard_ids=shard_ids, row_ids=row_ids):
                return fixture._distributed_lookup(shard_ids, row_ids) + a * 0

            output = run(anchor) if mode == "plain" else checkpoint(run, anchor, use_reentrant=mode == "reentrant")
            expected = torch.arange(16, dtype=torch.float32).reshape(4, 4)[:2].add_(100 * shard).to(dtype)
            torch.testing.assert_close(output.detach(), expected, rtol=0, atol=0)
            output.sum().backward()
            for index, table in enumerate(tables.values()):
                if index == 2:
                    assert table.weight.grad is None
                else:
                    assert table.weight.grad is not None
                    expected_grad = torch.full_like(table.weight.to_local(), 0.5 if rank % 2 == 0 else 0.0)
                    torch.testing.assert_close(table.weight.grad.to_local(), expected_grad, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("backend", ["gpu", "npu"])
@pytest.mark.launch_plan(provider=_CI_CPU_2)
def test_distributed_ple_empty_owner_under_checkpoint(tmp_path, monkeypatch, dtype, backend):
    _isolate_cpu_spawn(monkeypatch)
    mp.spawn(
        _distributed_lookup_worker,
        args=(str(tmp_path / "gloo"), dtype, backend, _CI_CPU_2.nproc),
        nprocs=_CI_CPU_2.nproc,
        join=True,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("backend", ["gpu", "npu"])
@pytest.mark.launch_plan(provider=_CI_CPU_4)
def test_trainable_ple_column_shards_with_empty_requests(tmp_path, monkeypatch, dtype, backend):
    _isolate_cpu_spawn(monkeypatch)
    mp.spawn(
        _worker, args=(str(tmp_path / "gloo"), dtype, backend, _CI_CPU_4.nproc), nprocs=_CI_CPU_4.nproc, join=True
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("backend", ["gpu", "npu"])
@pytest.mark.launch_plan(provider=_CI_CPU_4)
def test_distributed_ple_column_shards_with_empty_owner(tmp_path, monkeypatch, dtype, backend):
    _isolate_cpu_spawn(monkeypatch)
    mp.spawn(
        _distributed_lookup_worker,
        args=(str(tmp_path / "gloo"), dtype, backend, _CI_CPU_4.nproc),
        nprocs=_CI_CPU_4.nproc,
        join=True,
    )
