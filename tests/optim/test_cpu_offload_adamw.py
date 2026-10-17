"""CPU reference and cold-state policy checks for optimizer-only offload."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException

from veomni.checkpoint.dcp_checkpointer import OptimizerState, _ModelStrictLoadPlanner
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
    with pytest.raises(CheckpointException, match="missing precision-sensitive optimizer state"):
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
