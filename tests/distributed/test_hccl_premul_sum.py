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

"""CPU-runnable tests for the HCCL PREMUL_SUM compatibility patch."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

from veomni.distributed.hccl_premul_sum import (
    _WRAPPED_MARKER,
    apply_hccl_premul_sum_patch,
    hccl_premul_sum_wrapper,
)


class _MockPremulSum:
    def __init__(self, factor: float):
        self.factor = factor

    def __eq__(self, other):
        return other is ReduceOp.PREMUL_SUM

    def __getstate__(self):
        return ("PREMUL_SUM", self.factor)


class _FakeWork:
    def __init__(self, completion=None):
        self.completion = completion
        self.wait_count = 0

    def wait(self):
        self.wait_count += 1
        if self.completion is not None:
            self.completion()
        return True


def test_keyword_premul_sum_decomposes_to_sum_then_scale():
    calls = []

    def collective(tensor, op=ReduceOp.SUM, group=None, async_op=False):
        calls.append((tensor, op, group, async_op))

    original = torch.tensor([2.0, 4.0, 6.0])
    output = original.clone()
    wrapper = hccl_premul_sum_wrapper(collective, "tensor")

    result = wrapper(tensor=output, op=_MockPremulSum(0.5))

    assert result is None
    assert len(calls) == 1
    assert calls[0][1] == ReduceOp.SUM
    torch.testing.assert_close(output, original * 0.5)


def test_positional_premul_sum_decomposes_to_sum_then_scale():
    calls = []

    def collective(tensor, op=ReduceOp.SUM, group=None, async_op=False):
        calls.append((tensor, op, group, async_op))

    original = torch.tensor([3.0, 6.0])
    output = original.clone()
    wrapper = hccl_premul_sum_wrapper(collective, "tensor")

    wrapper(output, _MockPremulSum(1 / 3))

    assert calls[0][1] == ReduceOp.SUM
    torch.testing.assert_close(output, original / 3)


def test_extracts_factor_from_pytorch_premul_sum_object():
    calls = []

    def collective(tensor, op=ReduceOp.SUM, group=None, async_op=False):
        calls.append((tensor, op, group, async_op))

    output = torch.tensor([8.0, 12.0])
    wrapper = hccl_premul_sum_wrapper(collective, "tensor")

    wrapper(output, op=dist._make_nccl_premul_sum(0.25))

    assert calls[0][1] == ReduceOp.SUM
    torch.testing.assert_close(output, torch.tensor([2.0, 3.0]))


def test_non_premul_async_collective_passes_through_without_waiting():
    work = _FakeWork()
    calls = []

    def collective(tensor, op=ReduceOp.SUM, group=None, async_op=False):
        calls.append((tensor, op, group, async_op))
        return work

    output = torch.tensor([1.0, 2.0, 3.0])
    original = output.clone()
    wrapper = hccl_premul_sum_wrapper(collective, "tensor")

    result = wrapper(output, op=ReduceOp.SUM, async_op=True)

    assert result is work
    assert work.wait_count == 0
    assert calls[0][1] == ReduceOp.SUM
    assert calls[0][3] is True
    torch.testing.assert_close(output, original)


def test_async_premul_sum_waits_before_scaling():
    output = torch.zeros(2)
    work = _FakeWork(lambda: output.copy_(torch.tensor([4.0, 8.0])))

    def collective(tensor, op=ReduceOp.SUM, group=None, async_op=False):
        assert tensor is output
        assert op == ReduceOp.SUM
        assert async_op is True
        return work

    wrapper = hccl_premul_sum_wrapper(collective, "tensor")

    result = wrapper(output, op=_MockPremulSum(0.5), async_op=True)

    assert result is work
    assert work.wait_count == 1
    torch.testing.assert_close(output, torch.tensor([2.0, 4.0]))


def test_apply_patch_wraps_all_collectives_once(monkeypatch):
    def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
        return None

    def reduce_scatter(output, input_list, op=ReduceOp.SUM, group=None, async_op=False):
        return None

    def reduce_scatter_tensor(output, input, op=ReduceOp.SUM, group=None, async_op=False):
        return None

    originals = {
        "all_reduce": all_reduce,
        "reduce_scatter": reduce_scatter,
        "reduce_scatter_tensor": reduce_scatter_tensor,
    }
    for name, function in originals.items():
        monkeypatch.setattr(dist, name, function)

    apply_hccl_premul_sum_patch()
    first_install = {name: getattr(dist, name) for name in originals}
    apply_hccl_premul_sum_patch()

    for name, original in originals.items():
        installed = getattr(dist, name)
        assert installed is first_install[name]
        assert installed is not original
        assert getattr(installed, _WRAPPED_MARKER)
        assert installed.__wrapped__ is original
