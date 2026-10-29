from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.dcp_checkpointer import OptimizerState, _ModelStrictLoadPlanner
from veomni.optim.optimizer import MultiOptimizer
from veomni.optim.qwen38_muon import Qwen38MuonAdamW, add_qwen38_plans


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _make_model_and_optimizer():
    model = torch.nn.Module()
    for name in ("first", "second"):
        block = torch.nn.Module()
        block.down_proj = torch.nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
        setattr(model, name, block)
    children = {}
    for name in ("first", "second"):
        param = getattr(model, name).down_proj.weight
        groups = add_qwen38_plans(model, [{"params": [param], "cpu_offload": False}])
        children[name] = Qwen38MuonAdamW(groups, lr=0.001)
    return model, MultiOptimizer(model, children, list(children))


def test_dcp_preserves_dormant_muon_recipe_without_creating_state(tmp_path):
    torch.manual_seed(123)
    model, optimizer = _make_model_and_optimizer()
    model.first.down_proj.weight.grad = torch.ones_like(model.first.down_proj.weight)
    optimizer.step()
    optimizer.zero_grad()
    saved_model = {name: value.clone() for name, value in model.state_dict().items()}
    dormant = optimizer.optimizers_dict["second"]
    calls = []
    hook = dormant.register_step_pre_hook(lambda *args: calls.append(True))
    parallel_state = SimpleNamespace(dp_mode="ddp")
    saved = OptimizerState(model, optimizer, parallel_state=parallel_state)
    dcp.save({"optimizer": saved}, checkpoint_id=tmp_path)
    hook.remove()
    metadata = dcp.FileSystemReader(tmp_path).read_metadata().state_dict_metadata
    assert "optimizer.param_groups.second.down_proj.weight.qwen38_plan" in metadata
    assert not calls and not dormant.state

    resumed_model, resumed_optimizer = _make_model_and_optimizer()
    resumed_model.load_state_dict(saved_model)
    dcp.load(
        {"optimizer": OptimizerState(resumed_model, resumed_optimizer, parallel_state=parallel_state, load=True)},
        checkpoint_id=tmp_path,
        planner=_ModelStrictLoadPlanner(False, True, True, optimizer=resumed_optimizer),
    )
    assert not resumed_optimizer.optimizers_dict["second"].state
    for current in (model, resumed_model):
        for param in current.parameters():
            param.grad = torch.ones_like(param)
    optimizer.step()
    resumed_optimizer.step()
    for expected, actual in zip(model.parameters(), resumed_model.parameters(), strict=True):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    for name in ("first", "second"):
        left = optimizer.optimizers_dict[name].state[getattr(model, name).down_proj.weight]
        right = resumed_optimizer.optimizers_dict[name].state[getattr(resumed_model, name).down_proj.weight]
        assert left.keys() == right.keys()
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_frozen_muon_parameter_save_fails_before_state_is_lost():
    model, optimizer = _make_model_and_optimizer()
    model.first.down_proj.weight.grad = torch.ones_like(model.first.down_proj.weight)
    optimizer.step()
    model.first.down_proj.weight.requires_grad_(False)
    wrapper = OptimizerState(model, optimizer, parallel_state=SimpleNamespace(dp_mode="ddp"))
    with pytest.raises(RuntimeError, match="frozen Muon optimizer parameters"):
        wrapper.state_dict()


def test_frozen_muon_restore_target_is_rejected(tmp_path):
    model, optimizer = _make_model_and_optimizer()
    model.first.down_proj.weight.grad = torch.ones_like(model.first.down_proj.weight)
    optimizer.step()
    ps = SimpleNamespace(dp_mode="ddp")
    dcp.save({"optimizer": OptimizerState(model, optimizer, parallel_state=ps)}, checkpoint_id=tmp_path)
    target, target_optimizer = _make_model_and_optimizer()
    target.first.down_proj.weight.requires_grad_(False)
    wrapper = OptimizerState(target, target_optimizer, parallel_state=ps, load=True)
    with pytest.raises(RuntimeError, match="frozen Muon optimizer parameters"):
        dcp.load({"optimizer": wrapper}, checkpoint_id=tmp_path)


def test_frozen_ordinary_optimizer_restore_target_is_rejected(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    ps = SimpleNamespace(dp_mode="ddp")
    dcp.save({"optimizer": OptimizerState(model, optimizer, parallel_state=ps)}, checkpoint_id=tmp_path)
    target = torch.nn.Linear(2, 2)
    target_optimizer = torch.optim.AdamW(target.parameters())
    target.weight.requires_grad_(False)
    wrapper = OptimizerState(target, target_optimizer, parallel_state=ps, load=True)
    with pytest.raises(RuntimeError, match="frozen optimizer parameters"):
        dcp.load({"optimizer": wrapper}, checkpoint_id=tmp_path)


def test_non_file_coordinated_load_keeps_validation_enabled():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    wrapper = OptimizerState(
        model, optimizer, parallel_state=SimpleNamespace(dp_mode="ddp"), load=True, coordinated=True
    )
    wrapper.state_dict()
    assert wrapper._coordinated_group_validation


def _global_marker_worker(rank, rendezvous, checkpoint_dir, all_dormant):
    import torch.distributed as dist

    from veomni.checkpoint.dcp_checkpointer import _ZERO_OPTIMIZER_STATE_MARKER

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        wrapper = OptimizerState(
            model, optimizer, parallel_state=SimpleNamespace(dp_mode="ddp"), coordinated_group_validation=True
        )
        # Represent a rank that has no local optimizer chunks and another that
        # contributes a chunk; the zero-state marker describes the whole DCP.
        wrapper._state_dict_impl = lambda: (
            {"param_groups.weight.lr": 0.001, "state.weight.step": torch.tensor(1.0)}
            if rank == 0 and not all_dormant
            else {"param_groups.weight.lr": 0.001, _ZERO_OPTIMIZER_STATE_MARKER: True}
        )
        saved = wrapper.state_dict()
        assert (_ZERO_OPTIMIZER_STATE_MARKER in saved) is all_dormant
        dcp.save({"optimizer": saved}, checkpoint_id=checkpoint_dir)
        metadata = dcp.FileSystemReader(checkpoint_dir).read_metadata().state_dict_metadata
        assert (f"optimizer.{_ZERO_OPTIMIZER_STATE_MARKER}" in metadata) is all_dormant
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_zero_optimizer_marker_uses_all_ranks(tmp_path):
    for all_dormant in (False, True):
        mp.start_processes(
            _global_marker_worker,
            args=(str(tmp_path / f"rdzv-{all_dormant}"), str(tmp_path / f"ckpt-{all_dormant}"), all_dormant),
            nprocs=2,
            join=True,
            start_method="fork",
        )


def _rank_dependent_muon_state_worker(rank, rendezvous):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        model, optimizer = _make_model_and_optimizer()
        if rank == 0:
            model.first.down_proj.weight.grad = torch.ones_like(model.first.down_proj.weight)
            optimizer.step()
        wrapper = OptimizerState(
            model, optimizer, parallel_state=SimpleNamespace(dp_mode="ddp"), coordinated_group_validation=True
        )
        with pytest.raises(RuntimeError, match="differing (Muon|non-PLE) owner state/step"):
            wrapper.state_dict()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_rank_dependent_muon_state_fails_closed(tmp_path):
    mp.start_processes(
        _rank_dependent_muon_state_worker,
        args=(str(tmp_path / "rdzv-lazy"),),
        nprocs=2,
        join=True,
        start_method="fork",
    )


def _ordinary_adam_hook_worker(rank, rendezvous):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        calls = []
        hook = optimizer.register_state_dict_pre_hook(lambda _: calls.append(True))
        wrapper = OptimizerState(
            model, optimizer, parallel_state=SimpleNamespace(dp_mode="ddp"), coordinated_group_validation=True
        )
        # The bounded-DCP path validates optimizer state before serialization,
        # and therefore rejects hooks that could change it after validation.
        with pytest.raises(RuntimeError, match="does not support external optimizer state-dict hooks"):
            wrapper.state_dict()
        assert not calls
        hook.remove()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_ordinary_adam_state_dict_hooks_are_preserved(tmp_path):
    mp.start_processes(
        _ordinary_adam_hook_worker,
        args=(str(tmp_path / "rdzv-hook"),),
        nprocs=2,
        join=True,
        start_method="fork",
    )


def _empty_ple_owner_worker(rank, rendezvous, checkpoint_dir):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2, 1), mesh_dim_names=("ple_fsdp", "ple"))

        def make():
            model = torch.nn.Module()
            model.ngram_embedding = torch.nn.Embedding(8, 8, dtype=torch.bfloat16)
            model.ngram_embedding.weight = torch.nn.Parameter(
                distribute_tensor(torch.ones(8, 8, dtype=torch.bfloat16), mesh, [Shard(1), Shard(0)])
            )
            model._fqn2spec_info = {
                "ngram_embedding.weight": SimpleNamespace(
                    placement=Shard(0), fsdp_shard_dim=1, persistent_fsdp_shard_dim=1
                )
            }
            groups = add_qwen38_plans(model, [{"params": [model.ngram_embedding.weight], "weight_decay": 0.0}])
            leaf = Qwen38MuonAdamW(groups, lr=0.001)
            return model, leaf, MultiOptimizer(model, {"ple": leaf}, ["ple"])

        model, leaf, optimizer = make()
        ps = SimpleNamespace(
            dp_mode="fsdp2", extra_parallel_names=["ple"], extra_parallel_fsdp_device_mesh={"ple": mesh}
        )
        wrapper = OptimizerState(model, optimizer, parallel_state=ps, coordinated_group_validation=True)
        calls = []
        hook = leaf.register_step_pre_hook(lambda *args: calls.append(True))
        saved = wrapper.state_dict()
        step = saved["state.ngram_embedding.weight.step"]
        assert isinstance(step, DTensor) and step.to_local().item() == 0
        assert not calls and not leaf.state
        dcp.save({"optimizer": wrapper}, checkpoint_id=checkpoint_dir)
        hook.remove()
        assert not calls and not leaf.state

        resumed_model, resumed_leaf, resumed_optimizer = make()
        dcp.load(
            {
                "optimizer": OptimizerState(
                    resumed_model, resumed_optimizer, parallel_state=ps, load=True, coordinated_group_validation=True
                )
            },
            checkpoint_id=checkpoint_dir,
            planner=_ModelStrictLoadPlanner(False, True, True, optimizer=resumed_optimizer),
        )
        assert not resumed_leaf.state
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_empty_ple_owner_checkpoint_keeps_optimizer_lazy(tmp_path):
    mp.start_processes(
        _empty_ple_owner_worker,
        args=(str(tmp_path / "rdzv-ple"), str(tmp_path / "ckpt-ple")),
        nprocs=2,
        join=True,
        start_method="fork",
    )
