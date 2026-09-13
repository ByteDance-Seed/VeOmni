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

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import torch
import torch_npu


@dataclass(frozen=True)
class _GmmInterleaveBinding:
    schedule: Any
    capture_backward: bool = False
    schedule_replay_forward: bool = False


_gmm_interleave_binding: ContextVar[Optional[_GmmInterleaveBinding]] = ContextVar(
    "gmm_interleave_binding",
    default=None,
)


@contextmanager
def gmm_backward_interleave(schedule: Any) -> Iterator[None]:
    token = _gmm_interleave_binding.set(_GmmInterleaveBinding(schedule, capture_backward=True))
    try:
        yield
    finally:
        _gmm_interleave_binding.reset(token)


@contextmanager
def gmm_replay_forward_interleave(schedule: Any) -> Iterator[None]:
    token = _gmm_interleave_binding.set(_GmmInterleaveBinding(schedule, schedule_replay_forward=True))
    try:
        yield
    finally:
        _gmm_interleave_binding.reset(token)


class GmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, group_list):
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list
        binding = _gmm_interleave_binding.get()
        ctx.backward_interleave_schedule = (
            binding.schedule if binding is not None and binding.capture_backward else None
        )

        replay_ticket = None
        if binding is not None and binding.schedule_replay_forward:
            replay_ticket = binding.schedule.before_replay_gmm()

        marker = (
            torch.autograd.profiler.record_function(f"ilp::gmm::Fprime::{replay_ticket}")
            if replay_ticket is not None
            else nullcontext()
        )
        with marker:
            fwd_output = torch_npu.npu_grouped_matmul(
                [x], [weight], bias=None, group_list=group_list, split_item=2, group_type=0, group_list_type=1
            )[0]
        if replay_ticket is not None:
            binding.schedule.after_replay_gmm(replay_ticket)
        return fwd_output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight = ctx.saved_tensors
        group_list = ctx.group_list
        schedule = ctx.backward_interleave_schedule
        backward_ticket = schedule.before_backward_gmm() if schedule is not None else None

        marker = (
            torch.autograd.profiler.record_function(f"ilp::gmm::B::{backward_ticket}")
            if backward_ticket is not None
            else nullcontext()
        )
        with marker:
            weight = torch.transpose(weight, 1, 2)
            grad_input = torch_npu.npu_grouped_matmul(
                [grad_output],
                [weight],
                bias=None,
                group_list=group_list,
                split_item=2,
                group_type=0,
                group_list_type=1,
            )[0]

            grad_weight = torch_npu.npu_grouped_matmul(
                [input_tensor.T],
                [grad_output],
                bias=None,
                group_list=group_list,
                split_item=3,
                group_type=2,
                group_list_type=1,
            )[0]
        if backward_ticket is not None:
            schedule.after_backward_gmm(backward_ticket)

        return grad_input, grad_weight, None


def npu_group_gemm(x, weight, group_list):
    output = GmmFunction.apply(x, weight, group_list)
    return output
