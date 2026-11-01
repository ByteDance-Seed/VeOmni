"""Coordinated CUDA precision preflight without requiring a CUDA device."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import LaunchSpec


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_cuda_domain_uses_training_mesh_even_with_no_local_optimizer():
    from types import SimpleNamespace

    from veomni.optim.optimizer import _qwen38_cuda_domain

    def parallel(device):
        return SimpleNamespace(device_mesh=SimpleNamespace(device_type=device))

    assert _qwen38_cuda_domain("qwen38_muon", parallel("cuda"), [])
    assert not _qwen38_cuda_domain("qwen38_muon", parallel("cpu"), [])
    assert not _qwen38_cuda_domain("qwen38_muon", parallel("npu"), [])
    assert not _qwen38_cuda_domain("adamw", parallel("cuda"), [])


def test_framework_cuda_guard_rejects_incomplete_training_domains(monkeypatch):
    from types import SimpleNamespace

    from veomni.optim.optimizer import _qwen38_cuda_training_group

    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 4 if group is None else 2)
    with pytest.raises(ValueError, match="TP=PP=1"):
        _qwen38_cuda_training_group(SimpleNamespace(tp_size=2, pp_size=1, fsdp_group=object()))
    with pytest.raises(ValueError, match="complete training process group"):
        _qwen38_cuda_training_group(SimpleNamespace(tp_size=1, pp_size=1, fsdp_group=object()))


def _group_worker(rank, rendezvous, output_dir):
    from veomni.optim import optimizer as factory
    from veomni.optim import qwen38_muon as implementation

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4, timeout=timedelta(seconds=45)
    )
    try:
        first = dist.new_group([0, 1])
        second = dist.new_group([2, 3])
        group = first if rank < 2 else second
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.tensor([1.0]))
        model.weight.grad = torch.ones_like(model.weight)
        adam = torch.optim.AdamW([model.weight], lr=0.01)
        combined = factory.MultiOptimizer(
            model, {"adam": adam}, ["adam"], qwen38_cuda_guard=True, qwen38_cuda_group=group
        )
        factory._optimizer_uses_cuda = lambda _: True
        implementation._check_cuda_fp32_policy = lambda: None
        combined.step()
        before = model.weight.detach().clone()
        before_state = {name: value.clone() for name, value in adam.state[model.weight].items()}

        def second_policy():
            if rank == 1:
                raise RuntimeError("rank-local unsafe precision")

        implementation._check_cuda_fp32_policy = second_policy
        failed = False
        try:
            combined.step()
        except RuntimeError:
            failed = True
        unchanged = torch.equal(model.weight, before) and all(
            torch.equal(adam.state[model.weight][name], value) for name, value in before_state.items()
        )
        torch.save({"failed": failed, "unchanged": unchanged}, Path(output_dir) / f"rank{rank}.pt")
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=4, device="cpu"))
def test_cuda_policy_preflight_is_group_local_and_preserves_updates_on_failure(tmp_path):
    mp.spawn(_group_worker, args=(str(tmp_path / "rdzv"), str(tmp_path)), nprocs=4, join=True)
    results = [torch.load(tmp_path / f"rank{rank}.pt", weights_only=True) for rank in range(4)]
    assert results[:2] == [{"failed": True, "unchanged": True}] * 2
    assert results[2:] == [{"failed": False, "unchanged": False}] * 2


def _empty_rank_worker(rank, rendezvous, output_dir):
    from veomni.optim import optimizer as factory
    from veomni.optim import qwen38_muon as implementation

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.tensor([1.0]))
        model.weight.grad = torch.ones_like(model.weight)
        adam = torch.optim.AdamW([model.weight], lr=0.01)
        children = {"adam": adam} if rank == 0 else {}
        combined = factory.MultiOptimizer(
            model, children, list(children), qwen38_cuda_guard=True, qwen38_cuda_group=dist.group.WORLD
        )

        def policy():
            if rank == 1:
                raise RuntimeError("empty rank has unsafe precision")

        implementation._check_cuda_fp32_policy = policy
        failed = False
        try:
            combined.step()
        except RuntimeError:
            failed = True
        torch.save(
            {
                "failed": failed,
                "unchanged": torch.equal(model.weight, torch.tensor([1.0])),
                "state_empty": not adam.state,
            },
            Path(output_dir) / f"empty-rank{rank}.pt",
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_explicit_cuda_guard_includes_rank_with_no_local_optimizer(tmp_path):
    mp.spawn(_empty_rank_worker, args=(str(tmp_path / "rdzv-empty"), str(tmp_path)), nprocs=2, join=True)
    results = [torch.load(tmp_path / f"empty-rank{rank}.pt", weights_only=True) for rank in range(2)]
    assert results == [{"failed": True, "unchanged": True, "state_empty": True}] * 2


def _closure_worker(rank, rendezvous, output_dir):
    from veomni.optim import qwen38_muon as implementation

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        param = torch.nn.Parameter(torch.ones(2, 2))
        param.grad = torch.ones_like(param)
        opt = implementation.Qwen38MuonAdamW(
            [{"params": [param], "qwen38_recipe": implementation.RECIPE, "qwen38_plan": '[["muon", 0, 2]]'}],
            cuda_precision_group=dist.group.WORLD,
        )
        implementation._optimizer_uses_cuda = lambda _: True
        implementation._check_cuda_fp32_policy = lambda: None
        opt.step()
        before = param.detach().clone()
        before_state = {key: value.clone() for key, value in opt.state[param].items()}
        called = []
        failed = False
        message = ""
        try:
            opt.step(closure=(lambda: called.append(True)) if rank == 1 else None)
        except RuntimeError as exc:
            failed = True
            message = str(exc)
        torch.save(
            {
                "failed": failed,
                "unchanged": torch.equal(param, before)
                and all(torch.equal(opt.state[param][key], value) for key, value in before_state.items()),
                "closure_called": bool(called),
                "message": message,
            },
            Path(output_dir) / f"closure-rank{rank}.pt",
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_cuda_closure_on_one_rank_rejects_all_ranks_before_update(tmp_path):
    mp.spawn(_closure_worker, args=(str(tmp_path / "rdzv-closure"), str(tmp_path)), nprocs=2, join=True)
    results = [torch.load(tmp_path / f"closure-rank{rank}.pt", weights_only=True) for rank in range(2)]
    assert all(result["failed"] and result["unchanged"] and not result["closure_called"] for result in results)
    assert "another rank" in results[0]["message"]
    assert "closures" in results[1]["message"]


def _checkpoint_worker(rank, rendezvous, output_dir):
    from types import SimpleNamespace

    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.optim import optimizer as factory
    from veomni.optim import qwen38_muon as implementation

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    original_reader = DistributedCheckpointer._create_storage_reader
    try:

        def make():
            model = torch.nn.Module()
            model.a = torch.nn.Parameter(torch.ones(2, 2))
            model.b = torch.nn.Parameter(torch.ones(2, 2))
            opts = {
                name: implementation.Qwen38MuonAdamW(
                    [{"params": [param], "qwen38_recipe": implementation.RECIPE, "qwen38_plan": '[["muon", 0, 2]]'}]
                )
                for name, param in (("a", model.a), ("b", model.b))
            }
            return model, opts, factory.MultiOptimizer(model, opts, ["a", "b"])

        results = []
        for mode in ("layout", "prepare", "incomplete", "late_error"):
            source, _, source_opt = make()
            source.a.grad = torch.ones_like(source.a)
            source.b.grad = torch.ones_like(source.b)
            source_opt.step()
            source_opt.zero_grad()
            parallel = SimpleNamespace(dp_mode="ddp")
            path = str(Path(output_dir) / f"safe-{mode}")
            DistributedCheckpointer.save(path, {"model": source, "optimizer": source_opt}, parallel_state=parallel)
            target, opts, target_opt = make()
            with torch.no_grad():
                target.a.add_(3)
                target.b.add_(5)
            before = {name: param.detach().clone() for name, param in target.named_parameters()}
            if mode == "layout" and rank == 1:
                target.a.requires_grad_(False)
            attempts = 0

            def reader(cls, path, mode=mode):
                nonlocal attempts
                actual = original_reader(path)
                if Path(path).name != "optimizer":
                    return actual
                attempts += 1
                if rank == 1 and mode == "prepare" and attempts == 2:
                    raise OSError("rank-local inventory reader failure")
                if rank == 1 and mode in ("incomplete", "late_error") and attempts == 1:
                    original_metadata = actual.read_metadata
                    suffix = "state.b.exp_avg" if mode == "incomplete" else "param_groups.b.qwen38_plan"

                    def incomplete_metadata(original_metadata=original_metadata, suffix=suffix):
                        metadata = original_metadata()
                        keys = [key for key in metadata.state_dict_metadata if key.endswith(suffix)]
                        assert len(keys) == 1
                        del metadata.state_dict_metadata[keys[0]]
                        return metadata

                    actual.read_metadata = incomplete_metadata
                return actual

            DistributedCheckpointer._create_storage_reader = classmethod(reader)
            message = ""
            try:
                DistributedCheckpointer.load(path, {"model": target, "optimizer": target_opt}, parallel_state=parallel)
            except (RuntimeError, ValueError, OSError, KeyError) as exc:
                message = str(exc)
            finally:
                DistributedCheckpointer._create_storage_reader = classmethod(lambda cls, path: original_reader(path))
            unchanged = all(torch.equal(param, before[name]) for name, param in target.named_parameters())
            unchanged &= all(not opt.state for opt in opts.values())
            results.append({"failed": bool(message), "unchanged": unchanged, "message": message})
            dist.barrier()
        torch.save(results, Path(output_dir) / f"checkpoint-rank{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_cuda_checkpoint_preflight_rejects_all_ranks_before_model_copy(tmp_path):
    mp.spawn(_checkpoint_worker, args=(str(tmp_path / "rdzv-checkpoint"), str(tmp_path)), nprocs=2, join=True)
    results = [torch.load(tmp_path / f"checkpoint-rank{rank}.pt", weights_only=True) for rank in range(2)]
    assert all(result["failed"] and result["unchanged"] for rank in results for result in rank)
    assert all("trainability differs" in results[rank][0]["message"] for rank in range(2))
    assert "inventory reader" in results[1][1]["message"]
    assert all("inventory" in results[1][index]["message"] for index in (2, 3))


def _qwen_frozen_worker(rank, rendezvous, output_dir):
    from types import SimpleNamespace

    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.optim import optimizer as factory
    from veomni.optim import qwen38_muon as implementation

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.ones(2, 2))
        opt = implementation.Qwen38MuonAdamW(
            [{"params": [model.weight], "qwen38_recipe": implementation.RECIPE, "qwen38_plan": '[["muon", 0, 2]]'}]
        )
        combined = factory.MultiOptimizer(model, {"qwen": opt}, ["qwen"])
        model.weight.grad = torch.ones_like(model.weight)
        combined.step()
        combined.zero_grad()
        combined._qwen38_cuda_guard = True
        combined._qwen38_cuda_group = dist.group.WORLD
        before = model.weight.detach().clone()
        if rank == 1:
            model.weight.requires_grad_(False)
        message = ""
        try:
            DistributedCheckpointer.save(
                str(Path(output_dir) / "frozen-checkpoint"),
                {"model": model, "optimizer": combined},
                parallel_state=SimpleNamespace(dp_mode="ddp"),
            )
        except RuntimeError as exc:
            message = str(exc)
        torch.save(
            {
                "message": message,
                "unchanged": torch.equal(model.weight, before),
                "step": opt.state[model.weight]["step"],
            },
            Path(output_dir) / f"frozen-qwen-rank{rank}.pt",
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_qwen_multioptimizer_freeze_rejected_on_all_checkpoint_ranks(tmp_path):
    mp.spawn(_qwen_frozen_worker, args=(str(tmp_path / "rdzv-frozen-qwen"), str(tmp_path)), nprocs=2, join=True)
    results = [torch.load(tmp_path / f"frozen-qwen-rank{rank}.pt", weights_only=True) for rank in range(2)]
    assert all("trainability differs" in result["message"] for result in results)
    assert all(result["unchanged"] and result["step"].item() == 1 for result in results)
