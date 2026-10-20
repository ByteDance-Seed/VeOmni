"""CPU reference and cold-state policy checks for optimizer-only offload."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException

from veomni.checkpoint.dcp_checkpointer import (
    _ZERO_OPTIMIZER_STATE_MARKER,
    OptimizerState,
    _ModelStrictLoadPlanner,
    _requires_exact_optimizer_recipe,
    _requires_exact_optimizer_state_dtypes,
)
from veomni.optim.cpu_offload_adamw import CPUOffloadAdamW
from veomni.optim.master_fp32_adamw import FullFP32AdamW, MasterFP32AdamW


@pytest.mark.parametrize("offload", [False, True])
def test_cpu_update_and_cold_resume_match_fp32_reference(offload):
    torch.manual_seed(3)
    p = torch.nn.Parameter(torch.randn(16, dtype=torch.bfloat16))
    reference = torch.nn.Parameter(p.float().detach())
    opt = CPUOffloadAdamW([{"params": [p], "cpu_offload": offload}], lr=0.001)
    ref = torch.optim.AdamW([reference], lr=0.001, betas=(0.9, 0.95), weight_decay=0)
    for _ in range(3):
        p.grad = torch.randn_like(p)
        reference.grad = p.grad.float()
        opt.step()
        ref.step()
        torch.testing.assert_close(opt.state[p]["master_weight"], reference, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(p, reference.bfloat16(), rtol=0, atol=0)
    clone = torch.nn.Parameter(p.detach().clone())
    resumed = CPUOffloadAdamW([{"params": [clone], "cpu_offload": offload}], lr=0.001)
    resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
    clone.grad = p.grad.clone()
    opt.step()
    resumed.step()
    torch.testing.assert_close(clone, p, rtol=0, atol=0)
    for name in ("master_weight", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(resumed.state[clone][name], opt.state[p][name], rtol=0, atol=0)


@pytest.mark.parametrize("missing", ["master_weight", "exp_avg", "exp_avg_sq", "step", "all"])
def test_dcp_rejects_missing_master_optimizer_state(tmp_path, missing):
    fields = {
        "master_weight": torch.tensor([2.0]),
        "exp_avg": torch.zeros(1),
        "exp_avg_sq": torch.zeros(1),
        "step": torch.tensor(1.0),
    }
    saved = {k: v for k, v in fields.items() if k != missing} if missing != "all" else {}
    dcp.save(
        {"optimizer": {"state": {"weight": saved}}, "model": {"weight": torch.tensor([2.0])}}, checkpoint_id=tmp_path
    )
    target = {"optimizer": {"state": {"weight": copy.deepcopy(fields)}}, "model": {"weight": torch.tensor([9.0])}}
    with pytest.raises(CheckpointException, match="precision-sensitive optimizer state"):
        dcp.load(
            target,
            checkpoint_id=tmp_path,
            planner=_ModelStrictLoadPlanner(strict_model=True, strict_optimizer_dtype=True),
        )


def test_master_optimizer_rejects_implicit_precision_migration():
    p = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    source = FullFP32AdamW([p])
    p.grad = torch.ones_like(p)
    source.step()
    with pytest.raises(RuntimeError, match="migration"):
        MasterFP32AdamW([p]).load_state_dict(source.state_dict())


def test_optimizer_state_wrapper_preserves_strict_optimizer_policy():
    p = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    optimizer = CPUOffloadAdamW([p], lr=0.001)
    wrapper = SimpleNamespace(optimizer=optimizer)
    assert _requires_exact_optimizer_state_dtypes(wrapper)
    assert not _requires_exact_optimizer_recipe(wrapper)
    assert _requires_exact_optimizer_recipe(SimpleNamespace(optimizer=SimpleNamespace(requires_exact_recipe=True)))


@pytest.mark.parametrize("field", ["master_weight", "exp_avg", "exp_avg_sq"])
@pytest.mark.parametrize("saved_shape", [(2,), (4,), (1, 3)])
def test_dcp_rejects_optimizer_shape_mismatch_before_read(tmp_path, field, saved_shape):
    fields = {name: torch.zeros(3) for name in ("master_weight", "exp_avg", "exp_avg_sq")}
    fields["step"] = torch.tensor(1.0)
    saved = copy.deepcopy(fields)
    saved[field] = torch.ones(saved_shape)
    dcp.save({"optimizer": {"state": {"weight": saved}}}, checkpoint_id=tmp_path)
    target = {"optimizer": {"state": {"weight": copy.deepcopy(fields)}}}
    before = copy.deepcopy(target)
    with pytest.raises(CheckpointException, match="optimizer shape mismatch"):
        dcp.load(
            target,
            checkpoint_id=tmp_path,
            planner=_ModelStrictLoadPlanner(strict_model=False, strict_optimizer_dtype=True),
        )
    for name, tensor in target["optimizer"]["state"]["weight"].items():
        torch.testing.assert_close(tensor, before["optimizer"]["state"]["weight"][name], rtol=0, atol=0)


@pytest.mark.parametrize("partial_target", [False, True])
@pytest.mark.parametrize("container", [False, True])
def test_dcp_validates_optimizer_inventory_with_partial_target(tmp_path, partial_target, container):
    model = torch.nn.Linear(3, 2).bfloat16()
    opt = CPUOffloadAdamW(model.parameters(), lr=0.001)
    # Persist complete state, then resume into either a fresh or partially initialized optimizer.
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    opt.step()
    opt.zero_grad(set_to_none=True)
    state = OptimizerState(model, opt, parallel_state=SimpleNamespace(dp_mode="ddp"))
    dcp.save({"optimizer": state}, checkpoint_id=tmp_path)
    target_model = torch.nn.Linear(3, 2).bfloat16()
    target_opt = CPUOffloadAdamW(target_model.parameters(), lr=0.001)
    if partial_target:
        target_model.weight.grad = torch.ones_like(target_model.weight)
        target_opt.step()
        target_opt.zero_grad(set_to_none=True)
    inventory = SimpleNamespace(optimizers_dict={"precise": target_opt}) if container else target_opt
    target_state = OptimizerState(target_model, target_opt, parallel_state=SimpleNamespace(dp_mode="ddp"), load=True)
    planner = _ModelStrictLoadPlanner(strict_model=False, strict_optimizer_dtype=True, optimizer=inventory)
    if partial_target:
        with pytest.raises(CheckpointException, match="Incomplete precision-sensitive optimizer target state"):
            dcp.load({"optimizer": target_state}, checkpoint_id=tmp_path, planner=planner)
    else:
        dcp.load({"optimizer": target_state}, checkpoint_id=tmp_path, planner=planner)
        for param, target in zip(model.parameters(), target_model.parameters(), strict=True):
            for key in ("master_weight", "exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(opt.state[param][key], target_opt.state[target][key], rtol=0, atol=0)


def test_dcp_keeps_partial_ordinary_adamw_state_supported(tmp_path):
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    model.weight.grad = torch.ones_like(model.weight)
    opt.step()
    opt.zero_grad(set_to_none=True)
    dcp.save(
        {"optimizer": OptimizerState(model, opt, parallel_state=SimpleNamespace(dp_mode="ddp"))},
        checkpoint_id=tmp_path,
    )
    target = torch.nn.Linear(3, 2)
    resumed = torch.optim.AdamW(target.parameters(), lr=0.001)
    target.weight.grad = torch.ones_like(target.weight)
    resumed.step()
    resumed.zero_grad(set_to_none=True)
    dcp.load(
        {"optimizer": OptimizerState(target, resumed, parallel_state=SimpleNamespace(dp_mode="ddp"), load=True)},
        checkpoint_id=tmp_path,
        planner=_ModelStrictLoadPlanner(strict_model=False, optimizer=resumed),
    )
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(opt.state[model.weight][key], resumed.state[target.weight][key], rtol=0, atol=0)
    assert target.bias not in resumed.state


def test_dcp_keeps_default_state_for_unstepped_parameter(tmp_path):
    fields = {
        "master_weight": torch.tensor([2.0]),
        "exp_avg": torch.tensor([0.5]),
        "exp_avg_sq": torch.tensor([0.25]),
        "step": torch.tensor(3.0),
    }
    dcp.save({"optimizer": {"state": {"weight": copy.deepcopy(fields)}}}, checkpoint_id=tmp_path)
    target_weight = copy.deepcopy(fields)
    target_bias = {name: torch.zeros_like(tensor) for name, tensor in fields.items()}
    target = {"optimizer": {"state": {"weight": target_weight, "bias": target_bias}}}

    weight = torch.nn.Parameter(torch.zeros(1))
    bias = torch.nn.Parameter(torch.zeros(1))
    optimizer = SimpleNamespace(
        requires_exact_state_dtypes=True,
        param_groups=[{"params": [weight, bias]}],
        state={weight: copy.deepcopy(fields), bias: copy.deepcopy(target_bias)},
    )
    dcp.load(
        target,
        checkpoint_id=tmp_path,
        planner=_ModelStrictLoadPlanner(strict_model=False, strict_optimizer_dtype=True, optimizer=optimizer),
    )
    for name in fields:
        torch.testing.assert_close(target["optimizer"]["state"]["weight"][name], fields[name], rtol=0, atol=0)
        torch.testing.assert_close(
            target["optimizer"]["state"]["bias"][name], torch.zeros_like(fields[name]), rtol=0, atol=0
        )


def test_dcp_drops_wholly_absent_optimizer_state_after_resume(tmp_path):
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    source_model = torch.nn.Linear(1, 1).bfloat16()
    source_optimizer = CPUOffloadAdamW(source_model.parameters(), lr=0.001)
    source_model.weight.grad = torch.ones_like(source_model.weight)
    source_optimizer.step()
    source_optimizer.zero_grad(set_to_none=True)
    assert source_model.bias not in source_optimizer.state
    parallel_state = SimpleNamespace(dp_mode="ddp")
    dcp.save(
        {"optimizer": OptimizerState(source_model, source_optimizer, parallel_state=parallel_state)},
        checkpoint_id=tmp_path,
    )

    target_model = torch.nn.Linear(1, 1).bfloat16()
    target_optimizer = CPUOffloadAdamW(target_model.parameters(), lr=0.001)
    # Simulate a fused resume that materializes DCP placeholders before the
    # restored model value is available. A stale master/step would survive.
    get_optimizer_state_dict(target_model, target_optimizer)
    with torch.no_grad():
        target_model.bias.fill_(7.0)
    dcp.load(
        {"optimizer": OptimizerState(target_model, target_optimizer, parallel_state=parallel_state, load=True)},
        checkpoint_id=tmp_path,
        planner=_ModelStrictLoadPlanner(strict_model=False, strict_optimizer_dtype=True, optimizer=target_optimizer),
    )
    assert target_model.bias not in target_optimizer.state
    assert target_optimizer.state[target_model.weight]["step"].item() == 1

    target_model.weight.grad = torch.zeros_like(target_model.weight)
    target_model.bias.grad = torch.zeros_like(target_model.bias)
    target_optimizer.step()
    bias_state = target_optimizer.state[target_model.bias]
    assert bias_state["step"].item() == 1
    torch.testing.assert_close(bias_state["master_weight"], torch.full_like(bias_state["master_weight"], 7.0))
    assert target_optimizer.state[target_model.weight]["step"].item() == 2


def test_dcp_drops_wholly_absent_state_from_ddp_wrapped_model(tmp_path):
    import torch.distributed as dist
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
    from torch.nn.parallel import DistributedDataParallel

    if dist.is_initialized():
        pytest.skip("requires ownership of the default process group")

    dist.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'ddp-init'}",
        rank=0,
        world_size=1,
    )
    try:
        parallel_state = SimpleNamespace(dp_mode="ddp")
        source_model = DistributedDataParallel(torch.nn.Linear(1, 1).bfloat16())
        source_optimizer = CPUOffloadAdamW(source_model.parameters(), lr=0.001)
        source_model.module.weight.grad = torch.ones_like(source_model.module.weight)
        source_optimizer.step()
        source_optimizer.zero_grad(set_to_none=True)
        assert source_model.module.bias not in source_optimizer.state
        dcp.save(
            {"optimizer": OptimizerState(source_model, source_optimizer, parallel_state=parallel_state)},
            checkpoint_id=tmp_path / "checkpoint",
        )

        target_model = DistributedDataParallel(torch.nn.Linear(1, 1).bfloat16())
        target_optimizer = CPUOffloadAdamW(target_model.parameters(), lr=0.001)
        get_optimizer_state_dict(target_model, target_optimizer)
        with torch.no_grad():
            target_model.module.bias.fill_(7.0)
        dcp.load(
            {
                "optimizer": OptimizerState(
                    target_model,
                    target_optimizer,
                    parallel_state=parallel_state,
                    load=True,
                )
            },
            checkpoint_id=tmp_path / "checkpoint",
            planner=_ModelStrictLoadPlanner(
                strict_model=False,
                strict_optimizer_dtype=True,
                optimizer=target_optimizer,
            ),
        )
        assert target_model.module.bias not in target_optimizer.state
        assert target_optimizer.state[target_model.module.weight]["step"].item() == 1

        target_model.module.weight.grad = torch.zeros_like(target_model.module.weight)
        target_model.module.bias.grad = torch.zeros_like(target_model.module.bias)
        target_optimizer.step()
        bias_state = target_optimizer.state[target_model.module.bias]
        assert bias_state["step"].item() == 1
        torch.testing.assert_close(bias_state["master_weight"], torch.full_like(bias_state["master_weight"], 7.0))
        assert target_optimizer.state[target_model.module.weight]["step"].item() == 2
    finally:
        dist.destroy_process_group()


def test_dcp_rejects_adamw_multi_optimizer_as_muon(tmp_path):
    from veomni.optim.optimizer import MultiOptimizer
    from veomni.optim.qwen38_muon import Qwen38MuonAdamW, add_qwen38_plans

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = torch.nn.Linear(4, 2, bias=False)

    parallel_state = SimpleNamespace(dp_mode="ddp")
    source_model = Model().bfloat16()
    source_child = CPUOffloadAdamW([{"params": [source_model.down_proj.weight]}], lr=0.001)
    source = MultiOptimizer(source_model, {"all": source_child}, ["all"])
    source_model.down_proj.weight.grad = torch.ones_like(source_model.down_proj.weight)
    source.step()
    source.zero_grad()
    dcp.save(
        {"optimizer": OptimizerState(source_model, source, parallel_state=parallel_state)},
        checkpoint_id=tmp_path,
    )

    target_model = Model().bfloat16()
    groups = add_qwen38_plans(target_model, [{"params": [target_model.down_proj.weight], "cpu_offload": False}])
    target_child = Qwen38MuonAdamW(groups, lr=0.001)
    target = MultiOptimizer(target_model, {"all": target_child}, ["all"])
    with pytest.raises(CheckpointException, match="recipe/state is incomplete"):
        dcp.load(
            {"optimizer": OptimizerState(target_model, target, parallel_state=parallel_state, load=True)},
            checkpoint_id=tmp_path,
            planner=_ModelStrictLoadPlanner(
                strict_model=False,
                strict_optimizer_dtype=True,
                strict_optimizer_recipe=True,
                optimizer=target,
            ),
        )


def test_dcp_zero_step_sparse_multi_optimizer_restores_lazy_state(tmp_path):
    from torch.distributed.checkpoint import FileSystemReader

    from veomni.optim.optimizer import MultiOptimizer

    model = torch.nn.Linear(3, 2).bfloat16()
    child = CPUOffloadAdamW(model.parameters(), lr=0.001)
    optimizer = MultiOptimizer(model, {"all": child}, ["all"])
    parallel_state = SimpleNamespace(dp_mode="ddp")
    dcp.save(
        {"optimizer": OptimizerState(model, optimizer, parallel_state=parallel_state)},
        checkpoint_id=tmp_path,
    )
    metadata = FileSystemReader(tmp_path).read_metadata()
    assert "optimizer.__veomni_zero_optimizer_state__" in metadata.state_dict_metadata

    target_model = torch.nn.Linear(3, 2).bfloat16()
    target_child = CPUOffloadAdamW(target_model.parameters(), lr=0.001)
    target = MultiOptimizer(target_model, {"all": target_child}, ["all"])
    dcp.load(
        {"optimizer": OptimizerState(target_model, target, parallel_state=parallel_state, load=True)},
        checkpoint_id=tmp_path,
        planner=_ModelStrictLoadPlanner(strict_model=False, strict_optimizer_dtype=True, optimizer=target),
    )
    assert not target_child.state
    assert not target._veomni_resume_zero_optimizer_state


def test_dcp_rejects_zero_state_marker_with_tensor_state_and_clears_stale_flag(tmp_path):
    from veomni.optim.optimizer import MultiOptimizer

    model = torch.nn.Linear(3, 2).bfloat16()
    child = CPUOffloadAdamW(model.parameters(), lr=0.001)
    optimizer = MultiOptimizer(model, {"all": child}, ["all"])
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    parallel_state = SimpleNamespace(dp_mode="ddp")
    state_dict = OptimizerState(model, optimizer, parallel_state=parallel_state).state_dict()
    assert any(key.startswith("state.") for key in state_dict)
    state_dict[_ZERO_OPTIMIZER_STATE_MARKER] = True
    dcp.save({"optimizer": state_dict}, checkpoint_id=tmp_path)

    target_model = torch.nn.Linear(3, 2).bfloat16()
    target_child = CPUOffloadAdamW(target_model.parameters(), lr=0.001)
    target = MultiOptimizer(target_model, {"all": target_child}, ["all"])
    target._veomni_resume_zero_optimizer_state = True
    with pytest.raises(CheckpointException, match="zero-state optimizer marker conflicts"):
        dcp.load(
            {"optimizer": OptimizerState(target_model, target, parallel_state=parallel_state, load=True)},
            checkpoint_id=tmp_path,
            planner=_ModelStrictLoadPlanner(strict_model=False, strict_optimizer_dtype=True, optimizer=target),
        )
    assert not target._veomni_resume_zero_optimizer_state


def test_dcp_zero_step_exact_recipe_multi_optimizer_restores_lazy_state(tmp_path):
    from veomni.optim.optimizer import MultiOptimizer
    from veomni.optim.qwen38_muon import Qwen38MuonAdamW, add_qwen38_plans

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = torch.nn.Linear(4, 2, bias=False)

    def make_optimizer(model):
        groups = add_qwen38_plans(model, [{"params": [model.down_proj.weight], "cpu_offload": True}])
        child = Qwen38MuonAdamW(groups, lr=0.001)
        return child, MultiOptimizer(model, {"all": child}, ["all"])

    parallel_state = SimpleNamespace(dp_mode="ddp")
    model = Model().bfloat16()
    _, optimizer = make_optimizer(model)
    dcp.save(
        {"optimizer": OptimizerState(model, optimizer, parallel_state=parallel_state)},
        checkpoint_id=tmp_path,
    )

    target_model = Model().bfloat16()
    target_child, target = make_optimizer(target_model)
    dcp.load(
        {"optimizer": OptimizerState(target_model, target, parallel_state=parallel_state, load=True)},
        checkpoint_id=tmp_path,
        planner=_ModelStrictLoadPlanner(
            strict_model=False,
            strict_optimizer_dtype=True,
            strict_optimizer_recipe=True,
            optimizer=target,
        ),
    )
    assert not target_child.state
    assert not target._veomni_resume_zero_optimizer_state


def _extra_parallel_resume_worker(rank, world, rendezvous, hsdp=False):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Replicate, Shard

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world)
    try:
        if hsdp:
            mesh = init_device_mesh("cpu", (2, 2, 2), mesh_dim_names=("replicate", "fsdp", "ep"))
            replicate, fsdp, ep = mesh.get_coordinate()
            state_mesh = mesh["replicate", "fsdp"]
            state_placements = [Replicate(), Shard(0)]
        else:
            mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("fsdp", "ep"))
            fsdp, ep = mesh.get_coordinate()
            state_mesh = mesh["fsdp"]
            state_placements = [Shard(0)]
        # Deliberately uneven FSDP/EP shards exercise DTensor metadata recovery.
        global_shape = (5, 7)
        full = torch.arange(global_shape[0] * global_shape[1], dtype=torch.float32).reshape(global_shape)
        local = torch.tensor_split(torch.tensor_split(full, 2, dim=0)[fsdp], 2, dim=1)[ep].clone()
        param = torch.nn.Parameter(
            DTensor.from_local(
                local,
                mesh,
                [Replicate(), Shard(0), Shard(1)] if hsdp else [Shard(0), Shard(1)],
                run_check=False,
                shape=global_shape,
                stride=(global_shape[1], 1),
            )
        )
        state_tensors = {
            name: DTensor.from_local(local.clone() + index, state_mesh, state_placements, run_check=False)
            for index, name in enumerate(("master_weight", "exp_avg", "exp_avg_sq"), start=1)
        }
        optimizer = CPUOffloadAdamW([{"params": [param], "cpu_offload": True}], lr=0.001)
        optimizer.state[param] = {"step": torch.tensor(2.0), **state_tensors}
        saved = copy.deepcopy(optimizer.state_dict())

        resumed = CPUOffloadAdamW([{"params": [param], "cpu_offload": True}], lr=0.001)
        resumed.load_state_dict(saved)
        for name, expected in state_tensors.items():
            actual = resumed.state[param][name]
            assert actual.device_mesh == param.device_mesh
            assert actual.placements == param.placements
            torch.testing.assert_close(actual.to_local(), expected.to_local(), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("hsdp", [False, True])
def test_extra_parallel_fsdp_state_resume_restores_parameter_mesh(tmp_path, hsdp):
    import torch.multiprocessing as mp

    mp.start_processes(
        _extra_parallel_resume_worker,
        args=(8 if hsdp else 4, str(tmp_path / "rdzv"), hsdp),
        nprocs=8 if hsdp else 4,
        join=True,
        start_method="fork",
    )


def _restore_fsdp_only_extra_parallel_worker(rank, world, rendezvous):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard

    from veomni.checkpoint.dcp_checkpointer import drop_extra_parallel_dim, restore_extra_parallel_dim

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ep_fsdp", "ep"))
        fsdp_rank, ep_rank = mesh.get_coordinate()
        fsdp_mesh = mesh["ep_fsdp"]
        # The EP extent is even globally, but each FSDP-only DTensor exposes
        # only its EP-local extent, so the original shape cannot be inferred.
        full = torch.arange(6 * 8, dtype=torch.float32).reshape(6, 8)
        ep_local = torch.tensor_split(full, 2, dim=0)[ep_rank]
        local = torch.tensor_split(ep_local, 2, dim=1)[fsdp_rank].clone()
        fsdp_only = DTensor.from_local(
            local,
            fsdp_mesh,
            [Shard(1)],
            run_check=False,
            shape=ep_local.shape,
            stride=(ep_local.shape[1], 1),
        )

        with pytest.raises(RuntimeError, match="pass the original global_shape"):
            restore_extra_parallel_dim(fsdp_only, mesh, fsdp_mesh)

        restored = restore_extra_parallel_dim(fsdp_only, mesh, fsdp_mesh, global_shape=full.shape)
        assert restored.device_mesh == mesh
        assert tuple(restored.placements) == (Shard(1), Shard(0))
        assert restored.shape == full.shape
        torch.testing.assert_close(restored.to_local(), local, rtol=0, atol=0)

        full_layout = DTensor.from_local(
            local,
            mesh,
            [Shard(1), Shard(0)],
            run_check=False,
            shape=full.shape,
            stride=(full.shape[1], 1),
        )
        restored_full = restore_extra_parallel_dim(full_layout, mesh, fsdp_mesh, global_shape=full.shape)
        assert restored_full.shape == full.shape
        torch.testing.assert_close(restored_full.to_local(), local, rtol=0, atol=0)

        # When EP and FSDP shard the same tensor dim, global checkpoint metadata
        # must follow EP-outer / FSDP-inner physical storage. A same-config
        # round-trip alone can miss a block-transposed global layout, so rebuild
        # the complete tensor and compare its values.
        same_dim_full = torch.arange(6, dtype=torch.float32)
        same_dim_ep_local = torch.tensor_split(same_dim_full, 2, dim=0)[ep_rank]
        same_dim_local = torch.tensor_split(same_dim_ep_local, 2, dim=0)[fsdp_rank].clone()
        same_dim_fsdp_only = DTensor.from_local(
            same_dim_local,
            fsdp_mesh,
            [Shard(0)],
            run_check=False,
            shape=same_dim_ep_local.shape,
            stride=(1,),
        )
        same_dim_restored = restore_extra_parallel_dim(
            same_dim_fsdp_only,
            mesh,
            fsdp_mesh,
            ep_shard_dim=0,
            fsdp_shard_dim=0,
            global_shape=same_dim_full.shape,
        )
        assert same_dim_restored.device_mesh.mesh_dim_names == ("ep", "ep_fsdp")
        torch.testing.assert_close(same_dim_restored.to_local(), same_dim_local, rtol=0, atol=0)
        torch.testing.assert_close(same_dim_restored.full_tensor(), same_dim_full, rtol=0, atol=0)

        # A 6-element extent on a 2x2 mesh is the smallest case where flat and
        # EP-outer / FSDP-inner chunking differ: [2, 1, 2, 1] versus
        # [2, 2, 1, 1]. The drop must reconstruct the parameter mesh and retain
        # this rank's physical local shard without rejecting uneven ranks.
        same_dim_dropped = drop_extra_parallel_dim(
            same_dim_restored,
            fsdp_mesh,
            fsdp_shard_dim=0,
            extra_parallel_name="ep",
        )
        assert same_dim_dropped.device_mesh == fsdp_mesh
        assert tuple(same_dim_dropped.placements) == (Shard(0),)
        assert same_dim_dropped.shape == (same_dim_full.shape[0] // 2,)
        torch.testing.assert_close(same_dim_dropped.to_local(), same_dim_local, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_restore_fsdp_only_extra_parallel_uses_original_global_shape(tmp_path):
    import torch.multiprocessing as mp

    mp.start_processes(
        _restore_fsdp_only_extra_parallel_worker,
        args=(4, str(tmp_path / "rdzv")),
        nprocs=4,
        join=True,
        start_method="fork",
    )


def _extra_parallel_dcp_resume_worker(rank, world, rendezvous, checkpoint, hsdp, optimizer_kind, same_dim=False):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    from veomni.distributed.parallel_plan import SpecInfo
    from veomni.optim.master_fp32_adamw import FullFP32AdamW
    from veomni.optim.optimizer import MultiOptimizer

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world)
    try:
        if hsdp:
            mesh = init_device_mesh("cpu", (2, 2, 2), mesh_dim_names=("replicate", "fsdp", "ep"))
            placements = [Replicate(), Shard(0), Shard(0)] if same_dim else [Replicate(), Shard(1), Shard(0)]
        else:
            mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("fsdp", "ep"))
            placements = [Shard(0), Shard(0)] if same_dim else [Shard(1), Shard(0)]
        torch.manual_seed(17)
        full = torch.randn(5, 7, dtype=torch.bfloat16)
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(distribute_tensor(full, mesh, placements))
        model._fqn2spec_info = {
            "weight": SpecInfo(
                para_name="ep",
                placement=Shard(0),
                fqn="weight",
                para_fsdp_mesh=mesh,
                fsdp_shard_dim=0 if same_dim else 1,
            )
        }
        parallel_state = SimpleNamespace(
            dp_mode="fsdp2",
            any_extra_parallel_enabled=True,
            extra_parallel_names=("ep",),
            extra_parallel_fsdp_device_mesh={"ep": mesh},
        )

        def make_optimizer(model, gradient=None):
            if optimizer_kind == "cpu_offload_adamw":
                child = CPUOffloadAdamW([{"params": [model.weight], "cpu_offload": True}], lr=0.001)
            else:
                child = FullFP32AdamW([model.weight], lr=0.001)
            optimizer = MultiOptimizer(model, {"ep": child}, ["ep"])
            if gradient is not None:
                model.weight.grad = distribute_tensor(gradient, mesh, placements)
                optimizer.step()
                model.weight.grad = None
            return child, optimizer

        source_child, source = make_optimizer(model, torch.randn_like(full))
        dcp.save(
            {"optimizer": OptimizerState(model, source, parallel_state=parallel_state)},
            checkpoint_id=checkpoint,
        )
        dist.barrier()

        target_model = torch.nn.Module()
        target_model.weight = torch.nn.Parameter(distribute_tensor(torch.zeros_like(full), mesh, placements))
        target_model._fqn2spec_info = model._fqn2spec_info
        target_child, target = make_optimizer(target_model)
        dcp.load(
            {"optimizer": OptimizerState(target_model, target, parallel_state=parallel_state, load=True)},
            checkpoint_id=checkpoint,
            planner=_ModelStrictLoadPlanner(strict_model=False, strict_optimizer_dtype=True, optimizer=target),
        )
        for name in ("master_weight", "exp_avg", "exp_avg_sq"):
            expected = source_child.state[model.weight][name]
            actual = target_child.state[target_model.weight][name]
            assert actual.device_mesh == target_model.weight.device_mesh
            assert actual.placements == target_model.weight.placements
            assert actual.shape == target_model.weight.shape
            torch.testing.assert_close(actual.to_local(), expected.to_local(), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("optimizer_kind", ["cpu_offload_adamw", "full_fp32_adamw"])
@pytest.mark.parametrize("hsdp", [False, True])
@pytest.mark.parametrize("same_dim", [False, True])
def test_extra_parallel_dcp_resume_restores_parameter_layout(tmp_path, hsdp, optimizer_kind, same_dim):
    import torch.multiprocessing as mp

    world = 8 if hsdp else 4
    mp.start_processes(
        _extra_parallel_dcp_resume_worker,
        args=(world, str(tmp_path / "rdzv"), str(tmp_path / "checkpoint"), hsdp, optimizer_kind, same_dim),
        nprocs=world,
        join=True,
        start_method="fork",
    )
