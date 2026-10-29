from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint import dcp_checkpointer as checkpoint


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _model_and_optimizer():
    model = torch.nn.Linear(2, 2, bias=False)
    model._fqn2spec_info = {"weight": SimpleNamespace(persistent_fsdp_shard_dim=1)}
    optimizer = torch.optim.AdamW(model.parameters())
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    return model, optimizer


def _mock_process_group(monkeypatch, other):
    monkeypatch.setattr(checkpoint.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(checkpoint.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(checkpoint, "raise_if_any_rank_failed", lambda error, *args, **kwargs: _raise(error))

    def gather(result, local, group=None):
        result[:] = [local, other(local)]

    monkeypatch.setattr(checkpoint.dist, "all_gather_object", gather)


def _raise(error):
    if error is not None:
        raise error


def test_persistent_steps_reject_uncoordinated_save(monkeypatch):
    model, optimizer = _model_and_optimizer()
    _mock_process_group(monkeypatch, lambda rows: rows)
    with pytest.raises(RuntimeError, match="requires coordinated DCP validation"):
        checkpoint._validate_persistent_optimizer_steps(model, optimizer)


def test_persistent_steps_allow_equal_and_reject_different_steps(monkeypatch):
    model, optimizer = _model_and_optimizer()
    _mock_process_group(monkeypatch, lambda rows: rows)
    checkpoint._validate_persistent_optimizer_steps(model, optimizer, coordinated=True)

    def different(rows):
        result = dict(rows)
        mode, active, signature = result["weight"]
        fields, dtype, value = signature
        result["weight"] = (mode, active, (fields, dtype, value + 1))
        return result

    _mock_process_group(monkeypatch, different)
    with pytest.raises(RuntimeError, match="different persistent owner optimizer states/steps"):
        checkpoint._validate_persistent_optimizer_steps(model, optimizer, coordinated=True)


def test_plain_parameter_does_not_get_owner_grid_exemption(monkeypatch):
    model, optimizer = _model_and_optimizer()
    optimizer.requires_exact_recipe = True
    optimizer.param_groups[0]["qwen38_ple_lookup"] = True
    optimizer.param_groups[0]["qwen38_ple_step_layout"] = '["owner-grid-v1", null]'
    _mock_process_group(monkeypatch, lambda rows: rows)
    with pytest.raises(RuntimeError, match="requires coordinated DCP validation"):
        checkpoint._validate_persistent_optimizer_steps(model, optimizer)


def test_persistent_save_rejects_hooks_and_wholly_lazy_state(monkeypatch):
    model, optimizer = _model_and_optimizer()
    _mock_process_group(monkeypatch, lambda rows: rows)
    hook = optimizer.register_state_dict_pre_hook(lambda _: None)
    with pytest.raises(RuntimeError, match="do not support state-dict hooks"):
        checkpoint._validate_persistent_optimizer_steps(model, optimizer, coordinated=True)
    hook.remove()
    optimizer.state.clear()
    with pytest.raises(RuntimeError, match="uninitialized persistent optimizer state"):
        checkpoint._validate_persistent_optimizer_steps(model, optimizer, coordinated=True)


def test_persistent_serialized_step_must_match_live_state():
    model, optimizer = _model_and_optimizer()
    rows = checkpoint._persistent_optimizer_rows(model, optimizer)
    saved = {"state.weight.step": torch.tensor(1.0)}
    checkpoint._verify_serialized_persistent_steps(model, rows, saved)
    saved["state.weight.step"] = torch.tensor(2.0)
    with pytest.raises(RuntimeError, match="changed during checkpoint serialization"):
        checkpoint._verify_serialized_persistent_steps(model, rows, saved)


def test_checkpoint_group_can_use_one_complete_hsdp_replica():
    from torch.distributed.tensor import Replicate, Shard

    mesh = SimpleNamespace(mesh=torch.arange(8).reshape(2, 2, 2))
    placements = [Replicate(), Shard(1), Shard(0)]
    assert checkpoint._checkpoint_group_has_complete_replica(mesh, placements, {0, 1, 2, 3})
    assert checkpoint._checkpoint_group_has_complete_replica(mesh, placements, {0, 1, 6, 7})
    assert not checkpoint._checkpoint_group_has_complete_replica(mesh, placements, {0, 1, 4, 5})


def test_persistent_uninitialized_parameter_is_rejected_when_other_state_exists():
    model = torch.nn.Linear(2, 2)
    model._fqn2spec_info = {"weight": SimpleNamespace(persistent_fsdp_shard_dim=1)}
    optimizer = torch.optim.AdamW(model.parameters())
    model.bias.grad = torch.ones_like(model.bias)
    optimizer.step()
    with pytest.raises(RuntimeError, match="uninitialized persistent optimizer state"):
        checkpoint._persistent_optimizer_rows(model, optimizer)


def test_frozen_persistent_parameter_with_state_is_rejected():
    model, optimizer = _model_and_optimizer()
    model.weight.requires_grad_(False)
    with pytest.raises(RuntimeError, match="frozen persistent optimizer parameter"):
        checkpoint._persistent_optimizer_rows(model, optimizer)


def test_frozen_persistent_parameter_without_state_is_rejected():
    model = torch.nn.Linear(2, 2)
    model._fqn2spec_info = {"weight": SimpleNamespace(persistent_fsdp_shard_dim=1)}
    optimizer = torch.optim.AdamW(model.parameters())
    model.bias.grad = torch.ones_like(model.bias)
    optimizer.step()
    model.weight.requires_grad_(False)
    with pytest.raises(RuntimeError, match="frozen persistent optimizer parameter"):
        checkpoint._persistent_optimizer_rows(model, optimizer)


def test_single_process_frozen_persistent_parameter_save_is_rejected():
    model, optimizer = _model_and_optimizer()
    model.weight.requires_grad_(False)
    wrapper = checkpoint.OptimizerState(model, optimizer, parallel_state=SimpleNamespace(dp_mode="ddp"))
    with pytest.raises(RuntimeError, match="frozen persistent optimizer parameter"):
        wrapper.state_dict()


def test_frozen_persistent_restore_target_is_rejected(tmp_path):
    model, optimizer = _model_and_optimizer()
    ps = SimpleNamespace(dp_mode="ddp")
    dcp.save({"optimizer": checkpoint.OptimizerState(model, optimizer, parallel_state=ps)}, checkpoint_id=tmp_path)
    target, target_optimizer = _model_and_optimizer()
    target.weight.requires_grad_(False)
    wrapper = checkpoint.OptimizerState(target, target_optimizer, parallel_state=ps, load=True)
    with pytest.raises(RuntimeError, match="frozen persistent optimizer parameter"):
        dcp.load({"optimizer": wrapper}, checkpoint_id=tmp_path)


def test_empty_scalar_owner_is_rejected_before_optimizer_serialization():
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.empty(0, 2))
    model._fqn2spec_info = {"weight": SimpleNamespace(persistent_fsdp_shard_dim=1)}
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(RuntimeError, match="uninitialized persistent optimizer state"):
        checkpoint._persistent_optimizer_rows(model, optimizer)


def _persistent_step_worker(rank, rendezvous, differing):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2, 1), mesh_dim_names=("ple_fsdp", "ple"))
        model = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        model.weight = torch.nn.Parameter(distribute_tensor(model.weight.detach(), mesh, [Shard(1), Shard(0)]))
        model._fqn2spec_info = {"weight": SimpleNamespace(persistent_fsdp_shard_dim=1)}
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        if differing and rank == 0:
            optimizer.step()
        wrapper = checkpoint.OptimizerState(
            model, optimizer, parallel_state=SimpleNamespace(dp_mode="ddp"), coordinated_group_validation=True
        )
        if differing:
            with pytest.raises(RuntimeError, match="(different persistent|differing non-PLE) owner"):
                wrapper.state_dict()
        else:
            state = wrapper.state_dict()
            assert state["state"]["weight"]["step"].item() == 1
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("differing", [False, True])
def test_coordinated_persistent_optimizer_steps(tmp_path, differing):
    mp.start_processes(
        _persistent_step_worker,
        args=(str(tmp_path / "rendezvous"), differing),
        nprocs=2,
        join=True,
        start_method="fork",
    )


def _incomplete_group_worker(rank, rendezvous):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2, 1), mesh_dim_names=("ple_fsdp", "ple"))
        group = dist.new_group(ranks=[0])
        model = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        model.weight = torch.nn.Parameter(distribute_tensor(model.weight.detach(), mesh, [Shard(1), Shard(0)]))
        model._fqn2spec_info = {"weight": SimpleNamespace(persistent_fsdp_shard_dim=1)}
        optimizer = torch.optim.AdamW(model.parameters())
        if rank == 0:
            wrapper = checkpoint.OptimizerState(
                model,
                optimizer,
                parallel_state=SimpleNamespace(dp_mode="ddp"),
                coordinated_group_validation=True,
                coordinated_group=group,
            )
            with pytest.raises(RuntimeError, match="DCP group does not cover extra-parallel parameter owners"):
                wrapper.state_dict()
            ep_model = torch.nn.Linear(4, 4, bias=False)
            ep_model._fqn2spec_info = {
                "weight": SimpleNamespace(persistent_fsdp_shard_dim=None, para_fsdp_mesh=mesh, placement=Shard(0))
            }
            ep_optimizer = torch.optim.AdamW(ep_model.parameters())
            ep_wrapper = checkpoint.OptimizerState(
                ep_model,
                ep_optimizer,
                parallel_state=SimpleNamespace(dp_mode="ddp"),
                coordinated_group_validation=True,
                coordinated_group=group,
            )
            with pytest.raises(RuntimeError, match="DCP group does not cover extra-parallel parameter owners"):
                ep_wrapper.state_dict()
            checkpoint._validate_persistent_optimizer_ownership(
                model, optimizer, group, coordinated=True, check_coverage=False
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_checkpoint_group_must_cover_persistent_mesh(tmp_path):
    mp.start_processes(
        _incomplete_group_worker,
        args=(str(tmp_path / "rdzv-subgroup"),),
        nprocs=2,
        join=True,
        start_method="fork",
    )


def _rank_local_frozen_target_worker(rank, rendezvous):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        model, optimizer = _model_and_optimizer()
        if rank == 0:
            model.weight.requires_grad_(False)
        wrapper = checkpoint.OptimizerState(
            model,
            optimizer,
            parallel_state=SimpleNamespace(dp_mode="ddp"),
            load=True,
            coordinated_group_validation=True,
        )
        with pytest.raises(RuntimeError, match="(frozen persistent|failed on another rank)"):
            wrapper.state_dict()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_rank_local_frozen_target_fails_on_every_rank(tmp_path):
    mp.start_processes(
        _rank_local_frozen_target_worker,
        args=(str(tmp_path / "rdzv-frozen"),),
        nprocs=2,
        join=True,
        start_method="fork",
    )
