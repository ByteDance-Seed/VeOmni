from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.dcp_checkpointer import OptimizerState, _ModelStrictLoadPlanner
from veomni.optim.cpu_offload_adamw import CPUOffloadAdamW
from veomni.optim.master_fp32_adamw import FullFP32AdamW, MasterFP32AdamW
from veomni.optim.optimizer import AnyPrecisionAdamW, MultiOptimizer


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.mark.parametrize("kind", ["adamw", "master", "cpu", "anyprecision"])
@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_dcp_lazy_state_is_absent_for_load_hooks_and_hook_changes_persist(tmp_path, kind, multi, partial):
    def make():
        model = torch.nn.Linear(2, 2)
        cls = {
            "adamw": torch.optim.AdamW,
            "master": MasterFP32AdamW,
            "cpu": CPUOffloadAdamW,
            "anyprecision": AnyPrecisionAdamW,
        }[kind]
        optimizer = cls(model.parameters(), lr=0.01)
        wrapped = MultiOptimizer(model, {"all": optimizer}, ["all"]) if multi else optimizer
        return model, optimizer, wrapped

    source_model, source_child, source = make()
    if partial:
        source_model.weight.grad = torch.ones_like(source_model.weight)
        source.step()
    ps = SimpleNamespace(dp_mode="ddp")
    dcp.save({"optimizer": OptimizerState(source_model, source, parallel_state=ps)}, checkpoint_id=tmp_path)

    target_model, target_child, target = make()
    events = []

    def before(_optimizer, incoming):
        state = incoming["state"]
        ids = incoming["param_groups"][0]["params"]
        assert (ids[0] in state) is partial
        assert ids[1] not in state
        events.append("before")

    def after(optimizer):
        assert (target_model.weight in optimizer.state) is partial
        assert target_model.bias not in optimizer.state
        optimizer.state[target_model.bias]["hook_marker"] = 42
        events.append("after")

    target_child.register_load_state_dict_pre_hook(before)
    target_child.register_load_state_dict_post_hook(after)
    dcp.load(
        {"optimizer": OptimizerState(target_model, target, parallel_state=ps, load=True)},
        checkpoint_id=tmp_path,
        planner=_ModelStrictLoadPlanner(strict_model=False, strict_optimizer_dtype=kind != "adamw", optimizer=target),
    )
    assert events == ["before", "after"]
    assert target_child.state[target_model.bias]["hook_marker"] == 42


def _distributed_save_hook_worker(rank, rendezvous, path, asynchronous):
    import time
    from datetime import timedelta

    import torch.distributed as dist

    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=12)
    )
    try:
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        first = torch.optim.AdamW(model[0].parameters(), lr=0.01)
        second = torch.optim.AdamW(model[1].parameters(), lr=0.01)
        optimizer = MultiOptimizer(model, {"first": first, "second": second}, ["first", "second"])
        seen_first, seen_second = [], []

        def fail_on_first(_optimizer):
            seen_first.append(True)
            if rank == 0:
                raise RuntimeError("injected local save hook failure")

        first.register_state_dict_pre_hook(fail_on_first)
        second.register_state_dict_pre_hook(lambda _optimizer: seen_second.append(True))
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="state-dict hooks|failed on another rank"):
            DistributedCheckpointer.save(
                path=path,
                state={"model": model, "optimizer": optimizer},
                global_steps=1,
                save_async=asynchronous,
                save_timeout_seconds=12,
                parallel_state=SimpleNamespace(dp_mode="ddp"),
            )
        assert time.monotonic() - started < 9
        assert not seen_first and not seen_second
        if asynchronous:
            DistributedCheckpointer.wait_for_pending_save()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_dcp_save_hook_error_is_coordinated_between_optimizer_children(tmp_path, asynchronous):
    import torch.multiprocessing as mp

    mp.spawn(
        _distributed_save_hook_worker,
        args=(str(tmp_path / "rdzv"), str(tmp_path / "checkpoint"), asynchronous),
        nprocs=2,
        join=True,
    )


def _distributed_load_hook_worker(rank, rendezvous, path):
    import time
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=12)
    )
    try:

        def make():
            model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
            first = torch.optim.AdamW(model[0].parameters(), lr=0.01)
            second = torch.optim.AdamW(model[1].parameters(), lr=0.01)
            return model, first, second, MultiOptimizer(model, {"first": first, "second": second}, ["first", "second"])

        ps = SimpleNamespace(dp_mode="ddp")
        model, _, _, optimizer = make()
        model(torch.ones(2, 2)).sum().backward()
        optimizer.step()
        dcp.save(
            {"optimizer": OptimizerState(model, optimizer, parallel_state=ps, coordinated_group_validation=True)},
            checkpoint_id=path,
        )
        target_model, first, second, target = make()
        seen_first, seen_second = [], []

        def fail_on_first(_optimizer, _state):
            seen_first.append(True)
            if rank == 0:
                raise RuntimeError("injected local load hook failure")

        first.register_load_state_dict_pre_hook(fail_on_first)
        second.register_load_state_dict_pre_hook(lambda _optimizer, _state: seen_second.append(True))
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="state-dict hooks|failed on another rank"):
            dcp.load(
                {
                    "optimizer": OptimizerState(
                        target_model, target, parallel_state=ps, load=True, coordinated_group_validation=True
                    )
                },
                checkpoint_id=path,
                planner=_ModelStrictLoadPlanner(strict_model=False, optimizer=target),
            )
        assert time.monotonic() - started < 9
        assert not seen_first and not seen_second
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_dcp_load_hook_error_is_coordinated_between_optimizer_children(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(
        _distributed_load_hook_worker, args=(str(tmp_path / "rdzv"), str(tmp_path / "checkpoint")), nprocs=2, join=True
    )


@pytest.mark.parametrize(
    "source_dtype,target_dtype", [(torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32)]
)
def test_anyprecision_rejects_checkpoint_target_stale_moment_dtype(source_dtype, target_dtype):
    param = torch.nn.Parameter(torch.ones(2))
    target = AnyPrecisionAdamW([param], momentum_dtype=target_dtype, variance_dtype=target_dtype)
    param.grad = torch.ones_like(param)
    target.step()
    target.state[param]["exp_avg"] = target.state[param]["exp_avg"].to(dtype=source_dtype)
    with pytest.raises(RuntimeError, match="differs from its policy"):
        with target.checkpoint_state(load=True):
            pass


@pytest.mark.parametrize("stepped", [False, True])
def test_anyprecision_rejects_checkpoint_group_precision_change(stepped):
    p = torch.nn.Parameter(torch.ones(2))
    source = AnyPrecisionAdamW([p], momentum_dtype=torch.bfloat16, variance_dtype=torch.bfloat16)
    if stepped:
        p.grad = torch.ones_like(p)
        source.step()
    target = AnyPrecisionAdamW(
        [torch.nn.Parameter(torch.ones(2))], momentum_dtype=torch.float32, variance_dtype=torch.float32
    )
    with pytest.raises(RuntimeError, match="precision policy differs"):
        target.load_state_dict(source.state_dict())


def _subgroup_model_cold_mesh_worker(rank, rendezvous):
    from datetime import timedelta

    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import Shard

    from veomni.checkpoint.dcp_checkpointer import ModelState

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4, timeout=timedelta(seconds=12)
    )
    try:
        mesh = DeviceMesh("cpu", [[0, 1], [2, 3]], mesh_dim_names=("expert_fsdp", "expert"))
        subgroup = dist.new_group(ranks=[0, 1], backend="gloo", timeout=timedelta(seconds=12))
        model = torch.nn.Linear(2, 2, bias=False)
        model._fqn2spec_info = {
            "weight": SimpleNamespace(
                placement=Shard(0), persistent_fsdp_shard_dim=None, para_fsdp_mesh=mesh, fsdp_shard_dim=0
            )
        }
        if rank in (0, 1):
            wrapper = ModelState(
                model, parallel_state=SimpleNamespace(dp_mode="fsdp2"), coordinated=True, coordinated_group=subgroup
            )
            with pytest.raises(ValueError, match="requires the global ExtraParallel meshes to be precreated"):
                wrapper.state_dict()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=4))
def test_cold_subgroup_model_checkpoint_rejects_uncreated_global_mesh(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_subgroup_model_cold_mesh_worker, args=(str(tmp_path / "rdzv"),), nprocs=4, join=True)


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("operation", ["save", "load"])
def test_dcp_rejects_duplicate_parameter_groups(tmp_path, multi, operation):
    def make(duplicate):
        model = torch.nn.Linear(2, 2, bias=False)
        parameters = [model.weight, model.weight] if duplicate else [model.weight]
        if duplicate:
            with pytest.warns(UserWarning, match="duplicate parameters"):
                child = torch.optim.AdamW(parameters, lr=0.01)
        else:
            child = torch.optim.AdamW(parameters, lr=0.01)
        wrapped = MultiOptimizer(model, {"all": child}, ["all"]) if multi else child
        return model, wrapped

    ps = SimpleNamespace(dp_mode="ddp")
    if operation == "save":
        model, optimizer = make(True)
        with pytest.raises(RuntimeError, match="duplicate parameters"):
            OptimizerState(model, optimizer, parallel_state=ps).state_dict()
    else:
        model, optimizer = make(False)
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        dcp.save({"optimizer": OptimizerState(model, optimizer, parallel_state=ps)}, checkpoint_id=tmp_path)
        target_model, target = make(True)
        with pytest.raises(RuntimeError, match="duplicate parameters"):
            dcp.load(
                {"optimizer": OptimizerState(target_model, target, parallel_state=ps, load=True)},
                checkpoint_id=tmp_path,
                planner=_ModelStrictLoadPlanner(strict_model=False, optimizer=target),
            )


class _NestedRecipeSGD(torch.optim.Optimizer):
    def __init__(self, param, scale=None):
        group = {"params": [param]}
        if scale is not None:
            group["recipe"] = {"inner": {"scale": scale}}
        super().__init__([group], {"lr": 0.01})

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    scale = group.get("recipe", {}).get("inner", {}).get("scale", 1.0)
                    param.add_(param.grad, alpha=-group["lr"] * scale)


@pytest.mark.parametrize("multi", [False, True])
def test_dcp_rejects_unsupported_nested_optimizer_group_fields(tmp_path, multi):
    from torch.distributed.checkpoint.api import CheckpointException

    def make(scale):
        model = torch.nn.Linear(1, 1, bias=False)
        child = _NestedRecipeSGD(model.weight, scale)
        return model, MultiOptimizer(model, {"all": child}, ["all"]) if multi else child

    ps = SimpleNamespace(dp_mode="ddp")
    model, optimizer = make(0.5)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    dcp.save({"optimizer": OptimizerState(model, optimizer, parallel_state=ps)}, checkpoint_id=tmp_path)
    target_model, target = make(None)
    with pytest.raises(CheckpointException, match="unsupported saved field"):
        dcp.load(
            {"optimizer": OptimizerState(target_model, target, parallel_state=ps, load=True)},
            checkpoint_id=tmp_path,
            planner=_ModelStrictLoadPlanner(strict_model=False, optimizer=target, model=target_model),
        )


def _distributed_model_save_hook_worker(rank, rendezvous, path, asynchronous):
    import time
    from datetime import timedelta

    import torch.distributed as dist

    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=12)
    )
    try:
        model = torch.nn.Linear(2, 2)

        def fail_on_rank_zero(_module, _prefix, _keep_vars):
            if rank == 0:
                raise RuntimeError("injected local model save hook failure")

        model.register_state_dict_pre_hook(fail_on_rank_zero)
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="failure|failed on another rank"):
            DistributedCheckpointer.save(
                path=path,
                state={"model": model},
                global_steps=1,
                save_async=asynchronous,
                save_timeout_seconds=12,
                parallel_state=SimpleNamespace(dp_mode="ddp"),
            )
        assert time.monotonic() - started < 9
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_dcp_model_save_hook_failure_is_coordinated(tmp_path, asynchronous):
    import torch.multiprocessing as mp

    mp.spawn(
        _distributed_model_save_hook_worker,
        args=(str(tmp_path / "rdzv"), str(tmp_path / "checkpoint"), asynchronous),
        nprocs=2,
        join=True,
    )


def _async_slot_timeout_worker(rank, rendezvous, path):
    import time
    from datetime import timedelta

    import torch.distributed as dist

    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=2)
    )
    try:
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        import veomni.checkpoint.dcp_checkpointer as checkpoint_module

        original = checkpoint_module.optimizer_state_dict_for_checkpoint

        def slow_on_rank_zero(*args, **kwargs):
            if rank == 0:
                time.sleep(4)
            return original(*args, **kwargs)

        checkpoint_module.optimizer_state_dict_for_checkpoint = slow_on_rank_zero
        DistributedCheckpointer.save(
            path=path,
            state={"model": model, "optimizer": optimizer},
            global_steps=1,
            save_async=True,
            save_timeout_seconds=12,
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
        DistributedCheckpointer.wait_for_pending_save()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_async_checkpoint_state_preparation_uses_slot_timeout(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(
        _async_slot_timeout_worker, args=(str(tmp_path / "rdzv"), str(tmp_path / "checkpoint")), nprocs=2, join=True
    )


@pytest.mark.parametrize("multi", [False, True])
def test_dcp_rejects_adam_coupled_to_adamw_migration(tmp_path, multi):
    def make(kind):
        model = torch.nn.Linear(1, 1, bias=False)
        child = kind(model.parameters(), lr=0.01, weight_decay=0.5)
        return model, MultiOptimizer(model, {"all": child}, ["all"]) if multi else child

    ps = SimpleNamespace(dp_mode="ddp")
    model, source = make(torch.optim.Adam)
    model.weight.grad = torch.ones_like(model.weight)
    source.step()
    dcp.save({"optimizer": OptimizerState(model, source, parallel_state=ps)}, checkpoint_id=tmp_path)
    target_model, target = make(torch.optim.AdamW)
    with pytest.raises(RuntimeError, match="decoupled weight decay"):
        dcp.load(
            {"optimizer": OptimizerState(target_model, target, parallel_state=ps, load=True)},
            checkpoint_id=tmp_path,
            planner=_ModelStrictLoadPlanner(strict_model=False, optimizer=target, model=target_model),
        )


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize(
    "source_dtype,target_dtype", [(torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32)]
)
def test_distributed_muon_rejects_checkpoint_momentum_precision_change(tmp_path, multi, source_dtype, target_dtype):
    from torch.distributed.checkpoint.api import CheckpointException

    from veomni.optim.muon import DistributedMuon

    def make(dtype):
        model = torch.nn.Linear(2, 2, bias=False, dtype=dtype)
        child = DistributedMuon(model.parameters(), lr=0.01, ns_implementation="std")
        wrapped = MultiOptimizer(model, {"all": child}, ["all"]) if multi else child
        return model, child, wrapped

    ps = SimpleNamespace(dp_mode="ddp")
    model, source_child, source = make(source_dtype)
    model.weight.grad = torch.ones_like(model.weight)
    source.step()
    target_model, target_child, target = make(target_dtype)
    with pytest.raises(RuntimeError, match="momentum dtype/shape differs"):
        target_child.load_state_dict(source_child.state_dict())
    dcp.save({"optimizer": OptimizerState(model, source, parallel_state=ps)}, checkpoint_id=tmp_path)
    with pytest.raises(CheckpointException, match="dtype mismatch|precision|dtype"):
        dcp.load(
            {"optimizer": OptimizerState(target_model, target, parallel_state=ps, load=True)},
            checkpoint_id=tmp_path,
            planner=_ModelStrictLoadPlanner(
                strict_model=False, strict_optimizer_dtype=True, optimizer=target, model=target_model
            ),
        )


def _mixed_empty_state_save_worker(rank, rendezvous, path):
    from datetime import timedelta

    import torch.distributed as dist

    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=12)
    )
    try:
        model = torch.nn.Linear(2, 2, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        if rank == 1:
            model.weight.grad = torch.ones_like(model.weight)
            optimizer.step()
        with pytest.raises(RuntimeError, match="cannot preserve differing non-PLE owner state/step"):
            DistributedCheckpointer.save(
                path=path,
                state={"model": model, "optimizer": optimizer},
                global_steps=1,
                parallel_state=SimpleNamespace(dp_mode="ddp"),
            )
        dist.barrier()
        if rank == 0:
            from pathlib import Path

            assert not (Path(path) / "global_step_1" / "model" / "optimizer" / ".metadata").exists()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_dcp_save_rejects_unrepresentable_rank_local_empty_state(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(
        _mixed_empty_state_save_worker,
        args=(str(tmp_path / "rdzv"), str(tmp_path / "checkpoint")),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("multi", [False, True])
def test_cpu_offload_rejects_lazy_moment_precision_policy_change(tmp_path, partial, multi):
    def make(cpu_dtype):
        model = torch.nn.Linear(2, 2)
        child = CPUOffloadAdamW(
            [{"params": [model.weight], "cpu_offload": False}, {"params": [model.bias], "cpu_offload": True}],
            lr=0.01,
            resident_moment_dtype=torch.float32,
            cpu_moment_dtype=cpu_dtype,
        )
        wrapped = MultiOptimizer(model, {"all": child}, ["all"]) if multi else child
        return model, child, wrapped

    ps = SimpleNamespace(dp_mode="ddp")
    source_model, source_child, source = make(torch.float32)
    if partial:
        source_model.weight.grad = torch.ones_like(source_model.weight)
        source.step()
    target_model, target_child, target = make(torch.bfloat16)
    with pytest.raises(RuntimeError, match="precision policy differs"):
        target_child.load_state_dict(source_child.state_dict())
    dcp.save({"optimizer": OptimizerState(source_model, source, parallel_state=ps)}, checkpoint_id=tmp_path)
    with pytest.raises(RuntimeError, match="precision policy differs"):
        dcp.load(
            {"optimizer": OptimizerState(target_model, target, parallel_state=ps, load=True)},
            checkpoint_id=tmp_path,
            planner=_ModelStrictLoadPlanner(
                strict_model=False, strict_optimizer_dtype=True, optimizer=target, model=target_model
            ),
        )


def _mixed_per_parameter_state_save_worker(rank, rendezvous, path):
    from datetime import timedelta

    import torch.distributed as dist

    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=12)
    )
    try:
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        model.weight.grad = torch.ones_like(model.weight)
        if rank == 1:
            model.bias.grad = torch.ones_like(model.bias)
        optimizer.step()
        assert bool(optimizer.state)
        with pytest.raises(RuntimeError, match="cannot preserve differing non-PLE owner state/step"):
            DistributedCheckpointer.save(
                path=path,
                state={"model": model, "optimizer": optimizer},
                global_steps=1,
                parallel_state=SimpleNamespace(dp_mode="ddp"),
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_dcp_save_rejects_per_parameter_rank_local_lazy_state(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(
        _mixed_per_parameter_state_save_worker,
        args=(str(tmp_path / "rdzv"), str(tmp_path / "checkpoint")),
        nprocs=2,
        join=True,
    )


def _mixed_shard_steps_save_worker(rank, rendezvous, path):
    from datetime import timedelta

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=12)
    )
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp",))
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(distribute_tensor(torch.ones(4, 2), mesh, [Shard(0)]))
        optimizer = CPUOffloadAdamW([{"params": [model.weight], "cpu_offload": True}], lr=0.01)
        grad = distribute_tensor(torch.ones(4, 2), mesh, [Shard(0)])
        model.weight.grad = grad
        optimizer.step()
        if rank == 1:
            model.weight.grad = grad
            optimizer.step()
        with pytest.raises(RuntimeError, match="cannot preserve differing non-PLE owner state/step"):
            DistributedCheckpointer.save(
                path=path,
                state={"model": model, "optimizer": optimizer},
                global_steps=1,
                parallel_state=SimpleNamespace(dp_mode="ddp"),
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_dcp_save_rejects_distinct_shard_optimizer_steps(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(
        _mixed_shard_steps_save_worker,
        args=(str(tmp_path / "rdzv"), str(tmp_path / "checkpoint")),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize(
    "source_type,target_type", [(MasterFP32AdamW, FullFP32AdamW), (FullFP32AdamW, MasterFP32AdamW)]
)
def test_master_adamw_rejects_zero_step_moment_precision_change(source_type, target_type):
    source = source_type([torch.nn.Parameter(torch.ones(2))])
    target = target_type([torch.nn.Parameter(torch.ones(2))])
    with pytest.raises(RuntimeError, match="precision policy differs"):
        target.load_state_dict(source.state_dict())


@pytest.mark.parametrize("optimizer_type", [MasterFP32AdamW, FullFP32AdamW])
def test_master_adamw_dynamic_group_restores_precision_policy(optimizer_type):
    source = optimizer_type([torch.nn.Parameter(torch.ones(2))])
    source.add_param_group({"params": [torch.nn.Parameter(torch.ones(2))]})
    target = optimizer_type([torch.nn.Parameter(torch.ones(2))])
    target.add_param_group({"params": [torch.nn.Parameter(torch.ones(2))]})
    for group in source.param_groups:
        for param in group["params"]:
            param.grad = torch.full_like(param, 0.125)
    source.step()
    target.load_state_dict(source.state_dict())
    assert all(
        group["veomni_adam_moment_dtype"] == str(optimizer_type.moment_dtype)[6:] for group in target.param_groups
    )


def test_cpu_offload_dynamic_group_preserves_placement_and_precision():
    from veomni.optim.cpu_offload_adamw import CPUOffloadAdamW

    source = CPUOffloadAdamW([torch.nn.Parameter(torch.ones(2))], resident_moment_dtype=torch.bfloat16)
    added = torch.nn.Parameter(torch.ones(2))
    source.add_param_group({"params": [added], "cpu_offload": True})
    target = CPUOffloadAdamW([torch.nn.Parameter(torch.ones(2))], resident_moment_dtype=torch.bfloat16)
    target_added = torch.nn.Parameter(torch.ones(2))
    target.add_param_group({"params": [target_added], "cpu_offload": True})
    added.grad = torch.ones_like(added)
    source.step()
    target.load_state_dict(source.state_dict())
    assert [group["veomni_adam_moment_dtype"] for group in target.param_groups] == ["bfloat16", "float32"]
    assert id(target_added) in target._offloaded
    assert target.state[target_added]["exp_avg"].dtype == torch.float32


@pytest.mark.parametrize("multi", [False, True])
def test_dcp_rejects_unrepresentable_empty_optimizer_group(tmp_path, multi):
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

    model = torch.nn.Linear(1, 1, bias=False)
    child = torch.optim.SGD([{"params": [], "lr": 0.0123}, {"params": [model.weight], "lr": 0.5}])
    optimizer = MultiOptimizer(model, {"all": child}, ["all"]) if multi else child
    with pytest.raises(Exception, match="empty parameter group"):
        DistributedCheckpointer.save(
            path=tmp_path,
            state={"model": model, "optimizer": optimizer},
            global_steps=1,
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )


def _rank_local_empty_group_worker(rank, rendezvous, path):
    from datetime import timedelta

    import torch.distributed as dist

    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        model = torch.nn.Linear(1, 1, bias=False)
        groups = [{"params": [model.weight]}]
        if rank == 0:
            groups.insert(0, {"params": []})
        optimizer = torch.optim.SGD(groups, lr=0.1)
        with pytest.raises(RuntimeError, match="empty parameter group|validating optimizer checkpoint parameters"):
            DistributedCheckpointer.save(
                path=path,
                state={"model": model, "optimizer": optimizer},
                global_steps=1,
                parallel_state=SimpleNamespace(dp_mode="ddp"),
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_dcp_coordinately_rejects_rank_local_empty_group(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(
        _rank_local_empty_group_worker,
        args=(str(tmp_path / "rdzv"), str(tmp_path / "checkpoint")),
        nprocs=2,
        join=True,
    )


def _rank_local_empty_group_load_worker(rank, rendezvous, path):
    from datetime import timedelta

    import torch.distributed as dist

    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        DistributedCheckpointer.save(
            path=path,
            state={"model": model, "optimizer": optimizer},
            global_steps=1,
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
        target_model = torch.nn.Linear(1, 1, bias=False)
        groups = [{"params": [target_model.weight]}]
        if rank == 0:
            groups.insert(0, {"params": []})
        target = torch.optim.SGD(groups, lr=0.1)
        with pytest.raises(RuntimeError, match="empty parameter group|validating optimizer checkpoint parameters"):
            DistributedCheckpointer.load(
                path=str(Path(path) / "global_step_1"),
                state={"model": target_model, "optimizer": target},
                parallel_state=SimpleNamespace(dp_mode="ddp"),
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_dcp_coordinately_rejects_rank_local_empty_group_at_load(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(
        _rank_local_empty_group_load_worker,
        args=(str(tmp_path / "rdzv"), str(tmp_path / "checkpoint")),
        nprocs=2,
        join=True,
    )
