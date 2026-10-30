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
    import copy
    from types import SimpleNamespace

    from veomni.checkpoint import dcp_checkpointer as checkpoint
    from veomni.optim import optimizer as factory
    from veomni.optim import qwen38_muon as implementation

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:

        def make(guard=False):
            model = torch.nn.Module()
            model.a = torch.nn.Parameter(torch.ones(2, 2))
            model.b = torch.nn.Parameter(torch.ones(2, 2))
            opts = {
                name: implementation.Qwen38MuonAdamW(
                    [{"params": [param], "qwen38_recipe": implementation.RECIPE, "qwen38_plan": '[["muon", 0, 2]]'}]
                )
                for name, param in (("a", model.a), ("b", model.b))
            }
            combined = factory.MultiOptimizer(
                model, opts, ["a", "b"], qwen38_cuda_guard=guard, qwen38_cuda_group=dist.group.WORLD if guard else None
            )
            return model, opts, combined

        results = []
        for mode in ("layout", "prepare", "incomplete", "late_error"):
            source, _, source_opt = make()
            source.a.grad = torch.ones_like(source.a)
            source.b.grad = torch.ones_like(source.b)
            source_opt.step()
            source_opt.step()
            saved = copy.deepcopy(source_opt.state_dict(for_load=False))
            target, opts, target_opt = make()
            target.a.grad = torch.ones_like(target.a)
            target.b.grad = torch.ones_like(target.b)
            target_opt.step()
            target_opt._qwen38_cuda_guard = True
            target_opt._qwen38_cuda_group = dist.group.WORLD
            old = {name: copy.deepcopy(opt.state_dict()) for name, opt in opts.items()}
            if mode == "layout":
                target_opt._veomni_resume_zero_optimizer_state = True
                target_opt._veomni_absent_optimizer_state_bases = frozenset({"b"})
            if rank == 1 and mode == "layout":
                from torch.distributed.tensor import Shard

                target._fqn2spec_info = {"a": SimpleNamespace(placement=Shard(0), fsdp_shard_dim=0)}
            elif rank == 1 and mode == "incomplete":
                del saved["state.b.exp_avg"]
            elif rank == 1 and mode == "late_error":
                del saved["param_groups.b.qwen38_plan"]
            original = checkpoint._install_absent_optimizer_state_hooks
            if rank == 1 and mode == "prepare":

                def fail_prepare(*args):
                    raise RuntimeError("rank-local checkpoint preparation failure")

                checkpoint._install_absent_optimizer_state_hooks = fail_prepare
            wrapper = checkpoint.OptimizerState(
                target,
                target_opt,
                parallel_state=SimpleNamespace(dp_mode="fsdp2"),
                load=True,
                coordinated_group_validation=True,
                coordinated_group=dist.group.WORLD,
            )
            failed = False
            message = ""
            try:
                wrapper.load_state_dict(saved)
            except (RuntimeError, KeyError) as exc:
                failed = True
                message = str(exc)
            finally:
                checkpoint._install_absent_optimizer_state_hooks = original
            unchanged = True
            for name, opt in opts.items():
                current = opt.state_dict()
                unchanged &= current["param_groups"] == old[name]["param_groups"]
                unchanged &= all(
                    torch.equal(current["state"][0][key], value) for key, value in old[name]["state"][0].items()
                )
            if mode == "layout":
                unchanged &= target_opt._veomni_resume_zero_optimizer_state is True
                unchanged &= target_opt._veomni_absent_optimizer_state_bases == frozenset({"b"})
            results.append({"failed": failed, "unchanged": unchanged, "message": message})
            dist.barrier()
        torch.save(results, Path(output_dir) / f"checkpoint-rank{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_cuda_checkpoint_wrapper_rolls_back_all_ranks_on_rank_local_failure(tmp_path):
    mp.spawn(_checkpoint_worker, args=(str(tmp_path / "rdzv-checkpoint"), str(tmp_path)), nprocs=2, join=True)
    results = [torch.load(tmp_path / f"checkpoint-rank{rank}.pt", weights_only=True) for rank in range(2)]
    assert all(result["failed"] and result["unchanged"] for rank in results for result in rank)
    assert all("another rank" in result["message"] for result in results[0])
    for result, expected in zip(
        results[1],
        ("Unversioned same-dimension", "checkpoint preparation failure", "Incomplete Muon", "qwen38_plan"),
        strict=True,
    ):
        assert expected in result["message"]


def _qwen_frozen_worker(rank, rendezvous, output_dir):
    from types import SimpleNamespace

    from veomni.checkpoint.dcp_checkpointer import OptimizerState
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
        combined._qwen38_cuda_guard = True
        combined._qwen38_cuda_group = dist.group.WORLD
        before = model.weight.detach().clone()
        if rank == 1:
            model.weight.requires_grad_(False)
        wrapper = OptimizerState(
            model,
            combined,
            parallel_state=SimpleNamespace(dp_mode="ddp"),
            coordinated_group_validation=True,
            coordinated_group=dist.group.WORLD,
        )
        message = ""
        try:
            wrapper.state_dict()
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
    assert "another rank" in results[0]["message"]
    assert "frozen parameter still owned" in results[1]["message"]
    assert all(result["unchanged"] and result["step"].item() == 1 for result in results)
