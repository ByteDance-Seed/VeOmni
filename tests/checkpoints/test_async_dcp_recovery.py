# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Exercise real asynchronous DCP without MultiOptimizer."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer


def _build():
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 2)
    return model, torch.optim.AdamW(model.parameters(), lr=0.01)


def _step(model, optimizer, inputs):
    loss = model(inputs).square().sum()
    loss.backward()
    grads = {name: param.grad.detach().clone() for name, param in model.named_parameters()}
    optimizer.step()
    optimizer.zero_grad()
    return loss.detach(), grads


def _snapshot(model, optimizer):
    return {
        name: {
            "param": param.detach().clone(),
            **{key: value.detach().clone() for key, value in optimizer.state[param].items() if torch.is_tensor(value)},
        }
        for name, param in model.named_parameters()
    }


def _assert_equal(left, right):
    assert left.keys() == right.keys()
    for name, fields in left.items():
        assert fields.keys() == right[name].keys()
        for field, value in fields.items():
            assert torch.equal(value, right[name][field]), f"{name}.{field} differs"


def _async_resume_worker(rank, checkpoint, endpoint):
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        parallel = SimpleNamespace(dp_mode="ddp")
        model, optimizer = _build()
        _step(model, optimizer, torch.tensor([[1.0, 2.0]]))
        before = _snapshot(model, optimizer)
        DistributedCheckpointer.save(
            checkpoint, {"model": model, "optimizer": optimizer}, save_async=True, parallel_state=parallel
        )
        DistributedCheckpointer.wait_for_pending_save()
        resumed, restored_optimizer = _build()
        with torch.no_grad():
            for param in resumed.parameters():
                param.fill_(-1)
        DistributedCheckpointer.load(
            checkpoint, {"model": resumed, "optimizer": restored_optimizer}, parallel_state=parallel
        )
        _assert_equal(before, _snapshot(resumed, restored_optimizer))
        loss, grads = _step(model, optimizer, torch.tensor([[3.0, 4.0]]))
        restored_loss, restored_grads = _step(resumed, restored_optimizer, torch.tensor([[3.0, 4.0]]))
        assert torch.equal(loss, restored_loss)
        assert grads.keys() == restored_grads.keys()
        assert all(torch.equal(value, restored_grads[name]) for name, value in grads.items())
        _assert_equal(_snapshot(model, optimizer), _snapshot(resumed, restored_optimizer))
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_async_adamw_recovery_preserves_next_update(tmp_path):
    spawn_with_rendezvous(_async_resume_worker, args=(str(tmp_path / "checkpoint"),), nprocs=2)
