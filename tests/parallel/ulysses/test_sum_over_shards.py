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

"""``SumOverShards``: the forward reports the group sum, the backward scales by ``sp_size``.

The scale cancels FSDP's average over ``dp_shard_sp``: each SP rank holds the
derivative of its own shard only, so averaging over ``sp_size`` of them would
deliver ``1 / sp_size`` of the total gradient.
"""

from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import CasePlan, LaunchSpec
from veomni.distributed.sequence_parallel.loss import SumOverShards, sum_over_ulysses_shards
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_torch_device


_BACKENDS = [
    pytest.param("gloo", marks=pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo required")),
    pytest.param(
        dist.Backend.NCCL,
        marks=pytest.mark.skipif(
            not IS_CUDA_AVAILABLE or not dist.is_nccl_available() or get_torch_device().device_count() < 2,
            reason="Two CUDA devices and NCCL required",
        ),
    ),
]


def _check_sum_over_shards(rank, init_method, backend):
    device = "cpu"
    if backend == dist.Backend.NCCL:
        get_torch_device().set_device(rank)
        device = get_device_type()
    dist.init_process_group(backend, init_method=init_method, rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        weight = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
        shard = torch.tensor([1.0, -1.0, 0.5], device=device) * (rank + 1)
        local_sum = (weight * shard).sum()

        total = SumOverShards.apply(local_sum, dist.group.WORLD, 2)
        total.backward()

        expected_total = (weight.detach() * torch.tensor([1.0, -1.0, 0.5], device=device) * 3).sum()
        torch.testing.assert_close(total.detach(), expected_total, rtol=0, atol=0)
        torch.testing.assert_close(weight.grad, shard * 2, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def _sum_over_shards_launch(context):
    """Two processes for both backends: Gloo runs on CPU, NCCL needs two accelerators."""
    if context.params["backend"] == "gloo":
        return CasePlan.sequential(LaunchSpec(nproc=2, device="cpu"))
    return CasePlan.sequential(LaunchSpec(nproc=2, device="accelerator"))


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.launch_plan(provider=_sum_over_shards_launch)
def test_sum_over_shards_sums_forward_and_scales_backward(tmp_path, backend):
    mp.spawn(_check_sum_over_shards, args=((tmp_path / "rendezvous").as_uri(), backend), nprocs=2)


def test_sum_over_ulysses_shards_is_identity_without_ulysses():
    tensor = torch.tensor(3.0, requires_grad=True)
    state = SimpleNamespace(ulysses_enabled=False)
    with mock.patch("veomni.distributed.parallel_state.get_parallel_state", return_value=state):
        assert sum_over_ulysses_shards(tensor) is tensor
