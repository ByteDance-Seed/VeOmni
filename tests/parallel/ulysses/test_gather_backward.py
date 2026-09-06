# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Gather backward must not modify gradients shared with another graph branch."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from veomni.distributed.sequence_parallel.ulysses import _Gather
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_torch_device


def _check_gather_backward(rank, init_method, backend, layout, sum_grad, scale_grad):
    device = "cpu"
    if backend == "nccl":
        get_torch_device().set_device(rank)
        device = get_device_type()
    dist.init_process_group(backend, init_method=init_method, rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        local = torch.full((2, 3), float(rank), device=device, requires_grad=True)
        gathered = _Gather.apply(dist.group.WORLD, local, 0, scale_grad, sum_grad)
        values = torch.arange(1, 13, dtype=torch.float32, device=device).reshape(4, 3) + rank * 10
        if layout == "contiguous":
            upstream = values.clone()
        elif layout == "transposed":
            upstream = values.T.contiguous().T
        elif layout == "narrowed":
            backing = torch.zeros((4, 5), device=device)
            backing[:, 1:4] = values
            upstream = backing[:, 1:4]
        else:
            upstream = torch.tensor(float(rank + 1), device=device).expand(4, 3)
        original = upstream.clone()
        expected = original.clone()
        if sum_grad:
            dist.all_reduce(expected, group=dist.group.WORLD)
        if scale_grad:
            expected = expected * 2

        # AddBackward shares its incoming gradient with both branches. A gather
        # that reduces it in place also changes the unrelated bias gradient.
        bias = torch.zeros_like(gathered, requires_grad=True)
        local_grad, bias_grad = torch.autograd.grad(gathered + bias, (local, bias), grad_outputs=upstream)

        torch.testing.assert_close(local_grad, expected[rank * 2 : (rank + 1) * 2], rtol=0, atol=0)
        torch.testing.assert_close(bias_grad, original, rtol=0, atol=0)
        torch.testing.assert_close(upstream, original, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("gloo", marks=pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo required")),
        pytest.param(
            "nccl",
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE or not dist.is_nccl_available() or get_torch_device().device_count() < 2,
                reason="Two CUDA devices and NCCL required",
            ),
        ),
    ],
)
@pytest.mark.parametrize("layout", ["contiguous", "transposed", "narrowed", "expanded"])
@pytest.mark.parametrize("sum_grad", [False, True])
@pytest.mark.parametrize("scale_grad", [False, True])
def test_gather_backward_preserves_shared_gradients(tmp_path, backend, layout, sum_grad, scale_grad):
    mp.spawn(
        _check_gather_backward,
        args=((tmp_path / "rendezvous").as_uri(), backend, layout, sum_grad, scale_grad),
        nprocs=2,
    )
