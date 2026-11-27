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

"""Gloo coverage for Ulysses all-to-all on axes beyond the first two."""

from __future__ import annotations

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.distributed.sequence_parallel.ulysses import all_to_all_tensor, gather_seq_scatter_heads


_WORLD_SIZE = 2
_CI_CPU_2 = LaunchSpec(nproc=_WORLD_SIZE, device="cpu")


def _local_tensor(rank: int) -> torch.Tensor:
    return torch.arange(2 * 4 * 2 * 3, dtype=torch.float32).reshape(2, 4, 2, 3) + rank * 100


def _expected_forward(rank: int) -> torch.Tensor:
    """Cat every rank's head slice ``rank`` along the sequence axis."""
    return torch.cat([_local_tensor(source)[:, :, rank : rank + 1, :] for source in range(_WORLD_SIZE)], dim=1)


def _grad_out(rank: int) -> torch.Tensor:
    return torch.arange(2 * 8 * 1 * 3, dtype=torch.float32).reshape(2, 8, 1, 3) + rank * 1000


def _expected_grad(rank: int) -> torch.Tensor:
    """Inverse exchange: sequence chunks gathered back onto the head axis."""
    return torch.cat([_grad_out(source).split(4, dim=1)[rank] for source in range(_WORLD_SIZE)], dim=2)


def _worker(rank: int, rendezvous: str) -> None:
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(_WORLD_SIZE))
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=_WORLD_SIZE, timeout=timedelta(seconds=60)
    )
    original_all_to_all = dist.all_to_all

    def reject_list_all_to_all(*args, **kwargs):
        raise RuntimeError("Backend gloo does not support alltoall")

    dist.all_to_all = reject_list_all_to_all
    try:
        group = dist.group.WORLD
        local = _local_tensor(rank)
        forward = all_to_all_tensor(local, scatter_dim=2, gather_dim=1, group=group)
        torch.testing.assert_close(forward, _expected_forward(rank))

        waited = all_to_all_tensor(local, scatter_dim=2, gather_dim=1, group=group, async_op=True)
        torch.testing.assert_close(waited(), _expected_forward(rank))

        leaf = local.detach().requires_grad_(True)
        gathered = gather_seq_scatter_heads(leaf, seq_dim=1, head_dim=2, group=group)
        gathered.backward(_grad_out(rank))
        assert leaf.grad is not None
        torch.testing.assert_close(leaf.grad, _expected_grad(rank))
    finally:
        dist.all_to_all = original_all_to_all
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=_CI_CPU_2)
@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires CPU Gloo")
def test_high_dim_all_to_all_uses_single(tmp_path) -> None:
    torch.multiprocessing.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=_WORLD_SIZE, join=True)
