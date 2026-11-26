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


from functools import lru_cache

import torch
import torch.distributed as dist


@lru_cache(maxsize=None)
def _npu_p2p_coalesces(device):
    from torch_npu import npu
    from torch_npu.distributed import distributed_c10d
    from torch_npu.utils import get_cann_version

    # Match TorchNPU's CANN gate. The base wheel compares version strings.
    cann_supported = getattr(distributed_c10d, "_CANN_VERSION", None)
    if cann_supported is None:
        cann_supported = (get_cann_version("CANN") or "") >= "9.0.0"
    if not cann_supported:
        return False

    name = npu.get_device_name(device)
    return "Ascend910B1" <= name <= "Ascend910B4_1" or "Ascend910_9362" <= name <= "Ascend910_9392"


def batch_ep_p2p(operations):
    """Submit EP P2P operations in the backend's supported batch layout."""
    device = operations[0].tensor.device
    if len(operations) == 1 or device.type != "npu" or _npu_p2p_coalesces(device):
        return dist.batch_isend_irecv(operations)

    # Native HCCL accepts one message per directed rank pair in a batch.
    # Both endpoints assign the nth message on a pair to the same round.
    batches = []
    next_round = {}
    for operation in operations:
        key = (operation.op, operation.peer)
        round_index = next_round.get(key, 0)
        if round_index == len(batches):
            batches.append([])
        batches[round_index].append(operation)
        next_round[key] = round_index + 1
    if len(batches) == 1:
        return dist.batch_isend_irecv(operations)
    return [work for batch in batches for work in dist.batch_isend_irecv(batch)]


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

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
        return (
            None,
            _AllToAll.apply(ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes),
            None,
            None,
        )


class _AllToAll_Async(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = dist.get_world_size(group=group)

        if world_size == 1:
            return input

        input = input.contiguous()

        if output_split_sizes is None:
            output = torch.empty_like(input)
        else:
            output = torch.empty(size=(sum(output_split_sizes), input.size(1)), dtype=input.dtype, device=input.device)
        async_handle = dist.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True,
        )
        return output, async_handle

    @staticmethod
    def backward(ctx, *grad_output, grad_async_handle):
        return (
            None,
            _AllToAll_Async.apply(ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes),
            None,
            None,
        )


def all_to_all(group, input, output_split_size=None, input_split_size=None):
    return _AllToAll.apply(group, input, output_split_size, input_split_size)


def all_to_all_async(group, input, output_split_size, input_split_size):
    return _AllToAll_Async.apply(group, input, output_split_size, input_split_size)
