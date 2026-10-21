# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``optimizer_type="adafactor"``: parity with HF Adafactor, on one device and under FSDP2 DTensors."""

from __future__ import annotations

import os
import subprocess
import sys
from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from transformers.optimization import Adafactor

from tests.tools.launch_utils import find_free_port
from veomni.checkpoint.dcp_checkpointer import drop_extra_parallel_dim, restore_extra_parallel_dim
from veomni.optim import build_optimizer
from veomni.optim.optimizer import DistributedAdafactor


def _optimizer(parameters, cls):
    return cls(
        parameters,
        lr=1.125e-4,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=1e-4,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )


@pytest.mark.parametrize("shape", [(4, 6), (7,), (2, 3, 5)])
def test_matches_hf_adafactor_on_plain_tensors(shape):
    torch.manual_seed(0)
    dense = torch.randn(shape)
    reference = torch.nn.Parameter(dense.clone())
    parameter = torch.nn.Parameter(dense.clone())
    reference_optimizer = _optimizer([reference], Adafactor)
    optimizer = _optimizer([parameter], DistributedAdafactor)
    for _ in range(3):
        grad = torch.randn(shape)
        reference.grad = grad.clone()
        parameter.grad = grad.clone()
        reference_optimizer.step()
        optimizer.step()
        torch.testing.assert_close(parameter, reference, rtol=0, atol=0)


def test_build_optimizer_builds_distributed_adafactor():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.1, optimizer_type="adafactor")
    assert isinstance(optimizer, DistributedAdafactor)
    assert {group["weight_decay"] for group in optimizer.param_groups} <= {0.0, 0.1}
    assert all(group["lr"] == 1e-3 for group in optimizer.param_groups)


def _distributed_gate() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = dist.get_world_size()
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))

    for shard_dim in (0, 1):
        dense = torch.arange(24, dtype=torch.float32, device=device).reshape(4, 6) / 100
        dense_grad = torch.arange(24, 48, dtype=torch.float32, device=device).reshape(4, 6) / 100
        reference = torch.nn.Parameter(dense.clone())
        reference_optimizer = _optimizer([reference], Adafactor)
        reference.grad = dense_grad.clone()
        reference_optimizer.step()

        parameter = torch.nn.Parameter(distribute_tensor(dense, mesh, [Shard(shard_dim)]))
        parameter.grad = distribute_tensor(dense_grad, mesh, [Shard(shard_dim)])
        optimizer = _optimizer([parameter], DistributedAdafactor)
        optimizer.step()
        torch.testing.assert_close(parameter.full_tensor(), reference, rtol=2e-6, atol=2e-6)

        second_grad = dense_grad.flip(0).contiguous()
        reference.grad = second_grad.clone()
        reference_optimizer.step()
        parameter.grad = distribute_tensor(second_grad, mesh, [Shard(shard_dim)])
        optimizer.step()
        torch.testing.assert_close(parameter.full_tensor(), reference, rtol=2e-6, atol=2e-6)
        state = optimizer.state[parameter]
        assert isinstance(state["exp_avg_sq_row"], DTensor)
        assert isinstance(state["exp_avg_sq_col"], DTensor)

        # Resume must preserve both factored statistics, not merely permit further updates.
        saved_optimizer_state = deepcopy(optimizer.state_dict())
        resumed_dense = parameter.full_tensor().detach().clone()
        resumed_parameter = torch.nn.Parameter(distribute_tensor(resumed_dense, mesh, [Shard(shard_dim)]))
        resumed_optimizer = _optimizer([resumed_parameter], DistributedAdafactor)
        resumed_optimizer.load_state_dict(saved_optimizer_state)
        third_grad = dense_grad.flip(1).contiguous()
        parameter.grad = distribute_tensor(third_grad, mesh, [Shard(shard_dim)])
        optimizer.step()
        resumed_parameter.grad = distribute_tensor(third_grad, mesh, [Shard(shard_dim)])
        resumed_optimizer.step()
        torch.testing.assert_close(resumed_parameter.full_tensor(), parameter.full_tensor(), rtol=2e-6, atol=2e-6)

    # The EP checkpoint wrapper appends an ``ep`` Shard(0) behind the FSDP placement
    # on the ``(ep_fsdp, ep)`` mesh. Adafactor keeps sharded row statistics and
    # replicated column statistics; both must round-trip.
    ep_mesh = init_device_mesh("cuda", (world_size, 1), mesh_dim_names=("ep_fsdp", "ep"))
    fsdp_mesh = ep_mesh["ep_fsdp"]
    for placement in (Shard(1), Replicate()):
        dense_state = torch.arange(24, dtype=torch.float32, device=device).reshape(4, 6)
        state = distribute_tensor(dense_state, fsdp_mesh, [placement])
        restored = restore_extra_parallel_dim(state, ep_mesh, fsdp_mesh)
        assert restored.placements == (placement, Shard(0)), restored.placements
        torch.testing.assert_close(restored.full_tensor(), dense_state)
        dropped = drop_extra_parallel_dim(restored, fsdp_mesh)
        assert dropped.placements == (placement,), dropped.placements
        torch.testing.assert_close(dropped.full_tensor(), dense_state)

    if rank == 0:
        print("ADAFACTOR_FSDP2_OK", flush=True)
    dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_fsdp2_adafactor_2gpu():
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc_per_node=2",
        f"--master_port={find_free_port()}",
        os.path.abspath(__file__),
    ]
    env = os.environ.copy()
    env.setdefault("NCCL_DEBUG", "WARN")
    result = subprocess.run(cmd, env=env, check=True)
    assert result.returncode == 0


if __name__ == "__main__":
    _distributed_gate()
