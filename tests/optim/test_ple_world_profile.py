"""Full-world negotiation for the opt-in PLE file-state optimizer."""

import copy
import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.optim import ple_world_profile as profile


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class BrokenProfile:
    @property
    def type(self):
        raise ValueError("injected profile inspection failure")


def _worker(rank, world_size, rendezvous, root, mode):
    from veomni.utils import dist_utils

    dist_utils.get_device_type = lambda: "cpu"
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size, timeout=timedelta(seconds=15)
    )
    try:
        file_mode = not mode.startswith("nonfile") and not (mode == "mixed-mode" and rank == 1)
        opt = SimpleNamespace(
            type="qwen38_muon" if file_mode else "adamw",
            qwen38_ple_moments_world_profile=file_mode,
            qwen38_ple_moments_scratch_dir=root if file_mode else None,
        )
        if mode == "mixed-scratch" and rank == 1:
            opt.qwen38_ple_moments_scratch_dir += "-other"
        if mode == "world-without-scratch":
            opt.qwen38_ple_moments_scratch_dir = None
        accelerator = SimpleNamespace(ep_size=2, ulysses_size=1, cp_size=1)
        checkpoint = SimpleNamespace(save_async=mode == "async")
        if mode == "nonfile-nonserializable":
            opt.extension = lambda: None
            accelerator.extension = object()
        assert vars(copy.deepcopy(opt)) == vars(opt)
        if mode == "inspect-profile-error" and rank == 0:
            opt = BrokenProfile()
        failed = False
        try:
            if mode == "missing-preflight":
                profile.validate_constructor(root, True, None, None)
            else:
                owner = object() if mode == "owner-invalid-before" else None
                receipt = profile.establish(opt, accelerator, checkpoint, owner=owner)
                if mode == "owner-invalid-before":
                    if rank == 0:
                        profile._receipt = SimpleNamespace(owner_id=id(owner))
                    receipt = profile.for_job(owner)
                if mode == "mutated" and rank == 1:
                    opt.type = "adamw"
                if mode in ("missing-before", "invalid-before") and rank == 0:
                    receipt = None if mode == "missing-before" else object()
                if mode == "inspect-before" and rank == 0:
                    opt = BrokenProfile()
                profile.validate_before_build(opt, accelerator, checkpoint, receipt)
                if file_mode:
                    if mode in ("missing-factory", "invalid-factory") and rank == 0:
                        receipt = None if mode == "missing-factory" else object()
                    if mode == "inspect-factory" and rank == 0:
                        opt = BrokenProfile()
                    profile.validate_build_config("qwen38_muon", opt, receipt)
                    profile.validate_constructor(root, True, receipt, profile._FactoryPermit(receipt))
        except (RuntimeError, ValueError, TypeError):
            failed = True
        count = torch.tensor(int(failed))
        dist.all_reduce(count)
        expected = 0 if mode == "file" or mode.startswith("nonfile") else world_size
        assert int(count) == expected, (mode, int(count))
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize(
    "mode",
    [
        "file",
        "nonfile",
        "nonfile-nonserializable",
        "mixed-mode",
        "mixed-scratch",
        "world-without-scratch",
        "async",
        "missing-preflight",
        "missing-before",
        "invalid-before",
        "owner-invalid-before",
        "missing-factory",
        "invalid-factory",
        "inspect-profile-error",
        "inspect-before",
        "inspect-factory",
        "mutated",
    ],
)
def test_ple_profile_is_agreed_before_optimizer_branch(tmp_path, mode):
    mp.start_processes(
        _worker, args=(2, str(tmp_path / "rendezvous"), str(tmp_path), mode), nprocs=2, join=True, start_method="fork"
    )


def _ordinary_single_rank_worker(rank, rendezvous):
    from veomni.distributed.parallel_state import _init_parallel_state
    from veomni.optim.optimizer import build_optimizer

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=10)
    )
    try:
        _init_parallel_state(dp_size=2, dp_shard_size=2, device_type="cpu")
        if rank == 0:
            config = SimpleNamespace(type="adamw", state_offload="none")
            profile.validate_before_build(config, SimpleNamespace(), SimpleNamespace(), None)
            model = torch.nn.Linear(4, 2)
            optimizer = build_optimizer(model, lr=0.001, optimizer_type="adamw", optimizer_config=config)
            assert isinstance(optimizer, torch.optim.AdamW)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_ordinary_optimizer_can_be_built_by_a_single_world_rank(tmp_path):
    mp.start_processes(
        _ordinary_single_rank_worker,
        args=(str(tmp_path / "ordinary-rendezvous"),),
        nprocs=2,
        join=True,
        start_method="fork",
    )


def _ordinary_inspection_error_worker(rank, rendezvous, stage):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=10)
    )
    try:
        if rank == 0:
            with pytest.raises(ValueError, match="injected profile inspection failure"):
                if stage == "before":
                    profile.validate_before_build(BrokenProfile(), SimpleNamespace(), SimpleNamespace(), None)
                else:
                    profile.validate_build_config("adamw", BrokenProfile(), None)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("stage", ["before", "factory"])
def test_ordinary_inspection_error_is_rank_local(tmp_path, stage):
    mp.start_processes(
        _ordinary_inspection_error_worker,
        args=(str(tmp_path / "ordinary-error-rendezvous"), stage),
        nprocs=2,
        join=True,
        start_method="fork",
    )


def _factory_worker(rank, world_size, rendezvous, root, mode):
    from veomni.distributed.parallel_state import _init_parallel_state
    from veomni.optim import file_backed_state, qwen38_muon
    from veomni.optim import optimizer as factory
    from veomni.utils import dist_utils

    dist_utils.get_device_type = lambda: "cpu"
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size, timeout=timedelta(seconds=20)
    )
    try:
        _init_parallel_state(dp_size=world_size, dp_shard_size=world_size, device_type="cpu")
        file_backed_state._filesystem_type = lambda path: "ext4"
        qwen38_muon.FileBackedFP32State = lambda path: file_backed_state.FileBackedFP32State(
            path, minimum_free_bytes=0
        )
        opt = SimpleNamespace(
            type="qwen38_muon",
            state_offload="none",
            qwen38_ple_moments_world_profile=True,
            qwen38_ple_moments_scratch_dir=root,
            cpu_offload_param_patterns=[".+"],
            cpu_offload_cpu_moment_dtype="float32",
            cpu_offload_resident_moment_dtype="float32",
        )
        receipt = profile.establish(opt, SimpleNamespace(), SimpleNamespace(save_async=False))
        model = torch.nn.Module()
        model.ngram_embedding = torch.nn.Embedding(4, 3)
        model.lm_head = torch.nn.Linear(3, 4, bias=False)
        if mode == "extra-constructor-error":
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.tensor import Shard, distribute_tensor

            mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("ep_fsdp",))
            model.lm_head.weight = torch.nn.Parameter(distribute_tensor(torch.ones(4, 3), mesh, [Shard(0)]))
        params = list(model.parameters())
        if mode == "order" and rank == 1:
            params.reverse()
        if mode in ("constructor-error", "extra-constructor-error") and rank == 1:

            def fail(*args, **kwargs):
                raise OSError("injected constructor failure")

            factory.Qwen38MuonAdamW = fail
        factory._should_build_extra_parallel_aware = lambda model: mode == "extra-constructor-error"
        if mode == "extra-constructor-error":
            factory.get_parallel_state = lambda: SimpleNamespace(extra_parallel_names=["ep"])
        failed = False
        try:
            factory.build_optimizer(
                model,
                lr=0.01 if mode == "lr" and rank == 1 else 0.001,
                optimizer_type="qwen38_muon",
                optimizer_config=opt,
                ple_profile_receipt=receipt,
                param_groups=[
                    {
                        "params": params,
                        "weight_decay": 0.0,
                        "muon_momentum": 0.99 if mode == "momentum" and rank == 1 else 0.95,
                    }
                ],
            )
        except (OSError, RuntimeError, ValueError):
            failed = True
        result = torch.tensor(int(failed))
        dist.all_reduce(result)
        assert int(result) == (0 if mode == "success" else world_size), mode
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize(
    "mode", ["success", "order", "lr", "momentum", "constructor-error", "extra-constructor-error"]
)
def test_file_optimizer_factory_coordinates_real_group_failures(tmp_path, mode):
    mp.start_processes(
        _factory_worker,
        args=(2, str(tmp_path / "factory-rendezvous"), str(tmp_path), mode),
        nprocs=2,
        join=True,
        start_method="fork",
    )


def _file_step_worker(rank, rendezvous, root, mode):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    from veomni.distributed.parallel_state import _init_parallel_state
    from veomni.optim import file_backed_state, qwen38_muon
    from veomni.optim import optimizer as factory
    from veomni.utils import dist_utils

    dist_utils.get_device_type = lambda: "cpu"
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:
        _init_parallel_state(dp_size=2, dp_shard_size=2, device_type="cpu")
        file_backed_state._filesystem_type = lambda path: "ext4"
        qwen38_muon.FileBackedFP32State = lambda path: file_backed_state.FileBackedFP32State(
            path, minimum_free_bytes=0
        )
        opt_cfg = SimpleNamespace(
            type="qwen38_muon",
            state_offload="none",
            qwen38_ple_moments_world_profile=True,
            qwen38_ple_moments_scratch_dir=root,
            cpu_offload_param_patterns=[".+"],
            cpu_offload_cpu_moment_dtype="float32",
            cpu_offload_resident_moment_dtype="float32",
        )
        receipt = profile.establish(opt_cfg, SimpleNamespace(), SimpleNamespace(save_async=False))
        model = torch.nn.Module()
        model.ngram_embedding = torch.nn.Embedding(4, 4)
        model.o_proj = torch.nn.Linear(4, 4, bias=False)
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("muon",))
        model.o_proj.weight = torch.nn.Parameter(distribute_tensor(torch.ones(4, 4), mesh, [Shard(0)]))
        optimizer = factory.build_optimizer(
            model,
            lr=0.001,
            optimizer_type="qwen38_muon",
            optimizer_config=opt_cfg,
            ple_profile_receipt=receipt,
            param_groups=[{"params": list(model.parameters()), "weight_decay": 0.0}],
        )
        leaf = next(iter(optimizer.optimizers_dict.values()))
        p = model.o_proj.weight
        p.grad = distribute_tensor(torch.full((4, 4), 0.2), mesh, [Shard(0)])
        if mode == "grad_mismatch" and rank == 1:
            p.grad = None
        model.ngram_embedding.weight.grad = torch.full_like(model.ngram_embedding.weight, 0.1)
        original = model.ngram_embedding.weight.detach().clone()
        if mode == "disk_failure" and rank == 1:

            def fail_allocate(*args, **kwargs):
                raise OSError("injected file-backed allocation failure")

            leaf._ple_moment_storage.zeros = fail_allocate
        error = None
        try:
            optimizer.step()
        except (RuntimeError, ValueError, OSError) as exc:
            error = exc
        result = torch.tensor(int(error is not None))
        dist.all_reduce(result)
        assert int(result) == (0 if mode == "success" else 2), (rank, mode, error)
        if mode == "grad_mismatch":
            torch.testing.assert_close(original, model.ngram_embedding.weight, rtol=0, atol=0)
        if mode == "disk_failure":
            assert leaf._ple_state_poisoned
            with pytest.raises(RuntimeError, match="checkpoint|state failed"):
                optimizer.step()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("mode", ["success", "grad_mismatch", "disk_failure"])
def test_file_optimizer_step_coordinates_before_muon_collectives(tmp_path, mode):
    mp.start_processes(
        _file_step_worker,
        args=(str(tmp_path / "step-rendezvous"), str(tmp_path), mode),
        nprocs=2,
        join=True,
        start_method="fork",
    )


def _file_dcp_fsdp_worker(rank, rendezvous, root, failure=None, rows=8):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.distributed.parallel_state import _init_parallel_state
    from veomni.optim import file_backed_state, qwen38_muon
    from veomni.optim import optimizer as factory
    from veomni.utils import dist_utils

    dist_utils.get_device_type = lambda: "cpu"
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        parallel = _init_parallel_state(dp_size=2, dp_shard_size=2, dp_mode="fsdp2", device_type="cpu")
        file_backed_state._filesystem_type = lambda path: "ext4"
        qwen38_muon.FileBackedFP32State = lambda path: file_backed_state.FileBackedFP32State(
            path, minimum_free_bytes=0
        )
        opt_cfg = SimpleNamespace(
            type="qwen38_muon",
            state_offload="none",
            qwen38_ple_moments_world_profile=True,
            qwen38_ple_moments_scratch_dir=root,
            cpu_offload_param_patterns=[".+"],
            cpu_offload_cpu_moment_dtype="float32",
            cpu_offload_resident_moment_dtype="float32",
        )
        receipt = profile.establish(opt_cfg, SimpleNamespace(), SimpleNamespace(save_async=False))
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("fsdp",))

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ngram_embedding = torch.nn.Embedding(rows, 4)
                torch.nn.init.constant_(self.ngram_embedding.weight, 0.125)

            def forward(self):
                return self.ngram_embedding(torch.tensor([rank, rank + 2]) % rows).sum()

        def build():
            model = fully_shard(Tiny(), mesh=mesh)
            optim = factory.build_optimizer(
                model,
                lr=0.001,
                optimizer_type="qwen38_muon",
                optimizer_config=opt_cfg,
                ple_profile_receipt=receipt,
                param_groups=[{"params": list(model.parameters()), "weight_decay": 0.0}],
            )
            return model, optim

        def step(model, optim):
            model().backward()
            optim.step()
            optim.zero_grad()

        source, source_opt = build()
        step(source, source_opt)
        if rows == 1:
            assert source.ngram_embedding.weight.to_local().numel() == (4 if rank == 0 else 0)
            assert any(getattr(leaf, "_ple_file_mode_enabled", False) for leaf in source_opt.optimizers_dict.values())
        path = os.path.join(root, "checkpoint-fsdp")
        if failure is not None:
            from unittest.mock import patch

            import veomni.checkpoint.dcp_checkpointer as checkpoint_module

            if failure == "mode_mismatch" and rank == 1:
                leaf = next(leaf for leaf in source_opt.optimizers_dict.values() if leaf._ple_file_mode_enabled)
                leaf._ple_file_mode_enabled = False
            original = checkpoint_module._iter_optimizer_leaves

            def inspect_leaves(optimizer):
                if failure == "inspect_error" and rank == 1:
                    raise OSError("injected file-backed optimizer inspection failure")
                return original(optimizer)

            with patch.object(checkpoint_module, "_iter_optimizer_leaves", inspect_leaves):
                with pytest.raises((RuntimeError, OSError), match="file-backed optimizer|injected"):
                    DistributedCheckpointer.save(
                        path,
                        {"model": source, "optimizer": source_opt, "_veomni_scheduler_object": None},
                        parallel_state=parallel,
                    )
            assert not os.path.exists(path)
            completed = torch.tensor(1)
            dist.all_reduce(completed)
            assert completed.item() == 2
            return
        DistributedCheckpointer.save(
            path, {"model": source, "optimizer": source_opt, "_veomni_scheduler_object": None}, parallel_state=parallel
        )
        target, target_opt = build()
        DistributedCheckpointer.load(
            path, {"model": target, "optimizer": target_opt, "_veomni_scheduler_object": None}, parallel_state=parallel
        )
        step(source, source_opt)
        step(target, target_opt)
        original = source.ngram_embedding.weight
        loaded = target.ngram_embedding.weight
        torch.testing.assert_close(original.to_local(), loaded.to_local(), rtol=0, atol=0)
        a = next(iter(source_opt.optimizers_dict.values())).state[original]
        b = next(iter(target_opt.optimizers_dict.values())).state[loaded]
        for key in ("step", "master_weight", "exp_avg", "exp_avg_sq"):
            x = a[key].to_local() if hasattr(a[key], "to_local") else a[key]
            y = b[key].to_local() if hasattr(b[key], "to_local") else b[key]
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("rows", [8, 1])
def test_file_optimizer_fsdp2_save_restore_and_next_update(tmp_path, rows):
    mp.start_processes(
        _file_dcp_fsdp_worker,
        args=(str(tmp_path / "fsdp-rendezvous"), str(tmp_path), None, rows),
        nprocs=2,
        join=True,
        start_method="fork",
    )


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("failure", ["mode_mismatch", "inspect_error"])
def test_file_optimizer_policy_failure_is_coordinated_before_save(tmp_path, failure):
    mp.start_processes(
        _file_dcp_fsdp_worker,
        args=(str(tmp_path / "fsdp-rendezvous"), str(tmp_path), failure),
        nprocs=2,
        join=True,
        start_method="fork",
    )


def test_file_optimizer_group_signature_preserves_tensor_lr_bits():
    from veomni.optim.optimizer import _file_profile_group_signature

    model = torch.nn.Linear(1, 1, bias=False)
    first = torch.tensor(0.100001)
    second = torch.tensor(0.100002)
    assert repr(first) == repr(second)
    a = _file_profile_group_signature(model, [{"params": list(model.parameters()), "lr": first}])
    b = _file_profile_group_signature(model, [{"params": list(model.parameters()), "lr": second}])
    assert a != b


def test_direct_single_rank_profile_requires_scratch():
    from veomni.optim.qwen38_muon import Qwen38MuonAdamW, add_qwen38_plans

    model = torch.nn.Module()
    model.ngram_embedding = torch.nn.Embedding(2, 2)
    groups = add_qwen38_plans(model, [{"params": list(model.parameters()), "weight_decay": 0.0}])
    with pytest.raises(ValueError, match="scratch"):
        Qwen38MuonAdamW(groups, ple_moments_world_profile=True)


def _file_owner_worker(rank, rendezvous, root):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    from veomni.distributed.parallel_state import _init_parallel_state
    from veomni.optim import file_backed_state, qwen38_muon
    from veomni.optim import optimizer as factory
    from veomni.utils import dist_utils

    dist_utils.get_device_type = lambda: "cpu"
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4, timeout=timedelta(seconds=25)
    )
    try:
        _init_parallel_state(dp_size=4, dp_shard_size=4, device_type="cpu")
        file_backed_state._filesystem_type = lambda path: "ext4"
        qwen38_muon.FileBackedFP32State = lambda path: file_backed_state.FileBackedFP32State(
            path, minimum_free_bytes=0
        )
        cfg = SimpleNamespace(
            type="qwen38_muon",
            state_offload="none",
            qwen38_ple_moments_world_profile=True,
            qwen38_ple_moments_scratch_dir=root,
            cpu_offload_param_patterns=[".+"],
            cpu_offload_cpu_moment_dtype="float32",
            cpu_offload_resident_moment_dtype="float32",
        )
        receipt = profile.establish(cfg, SimpleNamespace(), SimpleNamespace(save_async=False))
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("owner", "owner_fsdp"))
        owner_mesh = mesh["owner_fsdp"]
        model = torch.nn.Module()
        model.ngram_embedding = torch.nn.Embedding(4, 3)
        model.ngram_embedding.weight = torch.nn.Parameter(
            distribute_tensor(torch.full((4, 3), 1.0 + rank // 2), owner_mesh, [Shard(0)])
        )
        optimizer = factory.build_optimizer(
            model,
            lr=0.001,
            optimizer_type="qwen38_muon",
            optimizer_config=cfg,
            ple_profile_receipt=receipt,
            param_groups=[{"params": list(model.parameters()), "weight_decay": 0.0}],
        )
        param = model.ngram_embedding.weight
        param.grad = distribute_tensor(torch.ones(4, 3), owner_mesh, [Shard(0)])
        before = param.to_local().detach().clone()
        optimizer.step()
        assert not torch.equal(before, param.to_local())
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=4, device="cpu"))
def test_file_optimizer_accepts_distinct_ple_owner_mesh_members(tmp_path):
    mp.start_processes(
        _file_owner_worker,
        args=(str(tmp_path / "owner-rendezvous"), str(tmp_path)),
        nprocs=4,
        join=True,
        start_method="fork",
    )
