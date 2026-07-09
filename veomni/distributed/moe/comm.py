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


from typing import List, Optional

import torch
import torch.distributed as dist


_EP_RANK_SEQ_LENS: Optional[List[int]] = None


def get_ep_rank_seq_lens() -> Optional[List[int]]:
    global _EP_RANK_SEQ_LENS
    return _EP_RANK_SEQ_LENS


def set_ep_rank_seq_lens(seq_len: int, ep_group: dist.ProcessGroup, device) -> None:
    global _EP_RANK_SEQ_LENS
    local_seq_len = torch.tensor([seq_len], dtype=torch.long, device=device)
    gathered_seq_lens = torch.empty(dist.get_world_size(ep_group), dtype=torch.long, device=device)
    dist.all_gather_into_tensor(gathered_seq_lens, local_seq_len, group=ep_group, async_op=False)
    _EP_RANK_SEQ_LENS = gathered_seq_lens.tolist()


def reset_ep_rank_seq_lens() -> None:
    global _EP_RANK_SEQ_LENS
    _EP_RANK_SEQ_LENS = None


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


def _gather_along_first_dim(
    local_tokens: torch.Tensor, group: dist.ProcessGroup, token_counts: Optional[List[int]] = None
) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return local_tokens

    local_tokens = local_tokens.contiguous()
    if token_counts is None:
        token_counts = get_ep_rank_seq_lens()
    if token_counts is None:
        token_counts = [local_tokens.shape[0] for _ in range(world_size)]

    if len(set(token_counts)) == 1:
        total_tokens = token_counts[0] * world_size
        token_shape = [total_tokens] + list(local_tokens.shape[1:])
        gathered_tokens = torch.empty(*token_shape, dtype=local_tokens.dtype, device=local_tokens.device)
        dist.all_gather_into_tensor(gathered_tokens, local_tokens, group=group)
        return gathered_tokens
    else:
        gathered_list = [
            torch.empty(count, *local_tokens.shape[1:], dtype=local_tokens.dtype, device=local_tokens.device)
            for count in token_counts
        ]
        dist.all_gather(gathered_list, local_tokens, group=group)
        return torch.cat(gathered_list, dim=0)


def _reduce_scatter_along_first_dim(
    full_tokens: torch.Tensor, group: dist.ProcessGroup, token_counts: Optional[List[int]] = None
) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return full_tokens

    full_tokens = full_tokens.contiguous()
    total_tokens = full_tokens.shape[0]
    hidden_dim = full_tokens.shape[1]

    if token_counts is None:
        token_counts = get_ep_rank_seq_lens()
    if token_counts is None:
        local_tokens_size = total_tokens // world_size
        token_counts = [local_tokens_size] * world_size
    else:
        local_tokens_size = token_counts[dist.get_rank(group)]

    if len(set(token_counts)) == 1:
        if total_tokens % world_size != 0:
            raise ValueError(f"total_tokens ({total_tokens}) must be divisible by EP group size ({world_size}).")
        local_output = torch.empty(
            local_tokens_size, *full_tokens.shape[1:], dtype=full_tokens.dtype, device=full_tokens.device
        )
        dist.reduce_scatter_tensor(output=local_output, input=full_tokens, group=group)
        return local_output
    else:
        input_list = list(torch.split(full_tokens, token_counts, dim=0))
        local_output = torch.empty(local_tokens_size, hidden_dim, dtype=full_tokens.dtype, device=full_tokens.device)
        dist.reduce_scatter(local_output, input_list, group=group)
        return local_output


class _AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input):
        ctx.group = group
        ctx.token_counts = get_ep_rank_seq_lens()
        return _gather_along_first_dim(input, group, ctx.token_counts)

    @staticmethod
    def backward(ctx, grad_output):
        return None, _reduce_scatter_along_first_dim(grad_output, ctx.group, ctx.token_counts)


class _ReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input):
        ctx.group = group
        ctx.token_counts = get_ep_rank_seq_lens()
        return _reduce_scatter_along_first_dim(input, group, ctx.token_counts)

    @staticmethod
    def backward(ctx, grad_output):
        return None, _gather_along_first_dim(grad_output, ctx.group, ctx.token_counts)


def allgather_tokens_in_ep(input, ep_group=None):
    return _AllGather.apply(ep_group, input)


def reduce_scatter_tokens_in_ep(input, ep_group=None):
    return _ReduceScatter.apply(ep_group, input)
