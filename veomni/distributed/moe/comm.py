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


from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator, Optional

import torch
import torch.distributed as dist


MoeBackwardPhaseCallback = Callable[[str], None]
MoeBackwardCollectiveCallback = Callable[[str, str], None]
_moe_backward_phase_callback: ContextVar[Optional[MoeBackwardPhaseCallback]] = ContextVar(
    "moe_backward_phase_callback", default=None
)
_moe_backward_collective_callback: ContextVar[Optional[MoeBackwardCollectiveCallback]] = ContextVar(
    "moe_backward_collective_callback", default=None
)


@contextmanager
def moe_backward_phase_callback(callback: Optional[MoeBackwardPhaseCallback]) -> Iterator[None]:
    token = _moe_backward_phase_callback.set(callback)
    try:
        yield
    finally:
        _moe_backward_phase_callback.reset(token)


@contextmanager
def moe_backward_collective_callback(callback: Optional[MoeBackwardCollectiveCallback]) -> Iterator[None]:
    token = _moe_backward_collective_callback.set(callback)
    try:
        yield
    finally:
        _moe_backward_collective_callback.reset(token)


def register_expert_backward_boundary(
    tensor: torch.Tensor,
    callback: Optional[MoeBackwardPhaseCallback] = None,
) -> torch.Tensor:
    callback = callback if callback is not None else _moe_backward_phase_callback.get()
    if callback is None or not tensor.requires_grad:
        return tensor

    def backward_hook(grad: torch.Tensor) -> torch.Tensor:
        callback("experts")
        return grad

    tensor.register_hook(backward_hook)
    return tensor


class AsyncCollectiveHandle:
    """Own a Work object without returning it through autograd.Function."""

    def __init__(self) -> None:
        self._work = None
        self._waited = False

    def set_work(self, work) -> None:
        self._work = work

    def wait(self) -> None:
        if self._waited:
            return
        work = self._work
        try:
            if work is not None:
                work.wait()
        finally:
            # HCCL Work may retain collective input/output storage. A handle is
            # reused by autograd after forward, so release Work at first wait.
            work = None
            self._work = None
            self._waited = True


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes, phase):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.phase = phase
        ctx.callback = _moe_backward_phase_callback.get()
        ctx.collective_callback = _moe_backward_collective_callback.get()

        world_size = dist.get_world_size(group=group)

        if world_size == 1:
            return input

        input = input.contiguous()

        if output_split_sizes is None:
            output = torch.empty_like(input)
        else:
            output = torch.empty(size=(sum(output_split_sizes), input.size(1)), dtype=input.dtype, device=input.device)
        dist.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        collective_callback = ctx.collective_callback
        if collective_callback is not None and ctx.phase is not None:
            collective_callback(ctx.phase, "before")
        grad_input = _AllToAll.apply(
            ctx.group,
            *grad_output,
            ctx.input_split_sizes,
            ctx.output_split_sizes,
            None,
        )
        if collective_callback is not None and ctx.phase is not None:
            collective_callback(ctx.phase, "after")
        callback = ctx.callback
        if callback is not None and ctx.phase is not None:
            callback(ctx.phase)
        return (
            None,
            grad_input,
            None,
            None,
            None,
        )


class _AllToAll_Async(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes, handle, phase, callback):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.handle = handle
        ctx.phase = phase
        ctx.callback = callback if callback is not None else _moe_backward_phase_callback.get()
        ctx.collective_callback = _moe_backward_collective_callback.get()

        world_size = dist.get_world_size(group=group)

        if world_size == 1:
            handle.set_work(None)
            return input

        input = input.contiguous()

        if output_split_sizes is None:
            output = torch.empty_like(input)
        else:
            output = torch.empty(size=(sum(output_split_sizes), input.size(1)), dtype=input.dtype, device=input.device)
        work = dist.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True,
        )
        handle.set_work(work)
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        ctx.handle.wait()
        collective_callback = ctx.collective_callback
        if collective_callback is not None and ctx.phase is not None:
            collective_callback(ctx.phase, "before")
        grad_input = _AllToAll.apply(
            ctx.group,
            *grad_output,
            ctx.input_split_sizes,
            ctx.output_split_sizes,
            None,
        )
        if collective_callback is not None and ctx.phase is not None:
            collective_callback(ctx.phase, "after")
        callback = ctx.callback
        if callback is not None and ctx.phase is not None:
            callback(ctx.phase)
        return (
            None,
            grad_input,
            None,
            None,
            None,
            None,
            None,
        )


def all_to_all(group, input, output_split_size=None, input_split_size=None, phase=None):
    return _AllToAll.apply(group, input, output_split_size, input_split_size, phase)


def all_to_all_async(group, input, output_split_size, input_split_size, phase=None, callback=None):
    handle = AsyncCollectiveHandle()
    output = _AllToAll_Async.apply(group, input, output_split_size, input_split_size, handle, phase, callback)
    return output, handle
