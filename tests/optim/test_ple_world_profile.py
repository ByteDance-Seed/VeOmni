"""Full-world negotiation for the opt-in PLE file-state optimizer."""

import copy
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
            config = SimpleNamespace(type="adamw")
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
    from veomni.optim import file_backed_state, qwen38_muon
    from veomni.optim import optimizer as factory
    from veomni.utils import dist_utils

    dist_utils.get_device_type = lambda: "cpu"
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size, timeout=timedelta(seconds=20)
    )
    try:
        file_backed_state._filesystem_type = lambda path: "ext4"
        qwen38_muon.FileBackedFP32State = lambda path: file_backed_state.FileBackedFP32State(
            path, minimum_free_bytes=0
        )
        opt = SimpleNamespace(
            type="qwen38_muon",
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
                param_groups=[{"params": params, "weight_decay": 0.0}],
            )
        except (OSError, RuntimeError, ValueError):
            failed = True
        result = torch.tensor(int(failed))
        dist.all_reduce(result)
        assert int(result) == (0 if mode == "success" else world_size), mode
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("mode", ["success", "order", "lr", "constructor-error", "extra-constructor-error"])
def test_file_optimizer_factory_coordinates_real_group_failures(tmp_path, mode):
    mp.start_processes(
        _factory_worker,
        args=(2, str(tmp_path / "factory-rendezvous"), str(tmp_path), mode),
        nprocs=2,
        join=True,
        start_method="fork",
    )
