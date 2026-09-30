"""CPU reference and cold-state policy checks for optimizer-only offload."""

import copy

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException

from veomni.checkpoint.dcp_checkpointer import _ModelStrictLoadPlanner
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
