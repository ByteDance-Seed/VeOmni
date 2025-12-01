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


from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from ...utils.device import get_device_id
from .comm import (
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
)
from .utils import (
    pad_tensor,
    unpad_tensor,
)


def _all_gather(
    x: Tensor,
    group: dist.ProcessGroup,
):
    device = x.device
    dtype = x.dtype
    group = get_ulysses_sequence_parallel_group() if group is None else group
    sp_world_size = dist.get_world_size(group)
    x_size = torch.tensor(x.size()).to(device)
    size_list = [torch.zeros(x_size.size(), dtype=torch.int64, device=device) for i in range(sp_world_size)]
    dist.all_gather(size_list, x_size, group=group)
    tensor_list = [torch.zeros(torch.Size(size_list[i]), dtype=dtype, device=device) for i in range(sp_world_size)]
    dist.all_gather(tensor_list, x, group=group)
    return tensor_list, size_list


def _all_gather_into_tensor(
    x: Tensor,
    group: dist.ProcessGroup,
):
    dim_size = list(x.size())

    group = get_ulysses_sequence_parallel_group() if group is None else group
    sp_world_size = dist.get_world_size(group)
    dim_size[0] = dim_size[0] * sp_world_size
    output = torch.empty(dim_size, dtype=x.dtype, device=get_device_id())
    dist.all_gather_into_tensor(output, x, group=group)
    return output


def _all_to_all(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    group = get_ulysses_sequence_parallel_group() if group is None else group
    seq_world_size = dist.get_world_size(group)
    input_list = [t.contiguous() for t in torch.tensor_split(local_input, seq_world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
    comm = dist.all_to_all(output_list, input_list, group=group, async_op=async_op)
    if async_op:

        def wait():
            comm.wait()
            return torch.cat(output_list, dim=gather_dim).contiguous()

        return wait
    return torch.cat(output_list, dim=gather_dim).contiguous()


def _all_to_all_single(
    x: Tensor, scatter_dim: int, gather_dim: int, group: Optional[dist.ProcessGroup] = None, async_op: bool = False
):
    """
    A function to do all-to-all on the first two dim
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    sp_world_size = dist.get_world_size(group)
    assert scatter_dim <= 1, "scatter_dim must be 0 or 1 when using all_to_all_single!"
    assert gather_dim <= 1, "gather_dim must be 0 or 1 when using all_to_all_single!"
    if scatter_dim != 0:
        gather_dim_bef = x.shape[gather_dim]
        scatter_dim_bef = x.shape[scatter_dim]
        x = (
            x.reshape([gather_dim_bef, sp_world_size, scatter_dim_bef // sp_world_size] + list(x.shape[2:]))
            .transpose(0, 1)
            .reshape([gather_dim_bef * sp_world_size, scatter_dim_bef // sp_world_size] + list(x.shape[2:]))
            .contiguous()
        )

    output = torch.empty_like(x)
    comm = dist.all_to_all_single(output, x.contiguous(), group=group, async_op=async_op)

    if async_op:

        def wait():
            comm.wait()
            if scatter_dim == 0:
                return torch.cat(output.split(x.size(0) // sp_world_size), dim=gather_dim)
            else:
                return output

        return wait

    if scatter_dim == 0:
        output = torch.cat(output.split(x.size(0) // sp_world_size), dim=gather_dim)
    return output


def all_to_all_tensor(
    x: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup,
    async_op: bool = False,
):
    if scatter_dim <= 1 and gather_dim <= 1:
        return _all_to_all_single(x, scatter_dim, gather_dim, group, async_op)
    else:
        return _all_to_all(x, scatter_dim, gather_dim, group, async_op)


class _SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_input: Tensor,
        scatter_dim: int,
        gather_dim: int,
        async_op: bool,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.async_op = async_op
        return all_to_all_tensor(local_input, scatter_dim, gather_dim, group, async_op)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        if ctx.async_op:
            input_t = torch.cat(grad_output[1:], dim=ctx.gather_dim).contiguous()
        else:
            input_t = grad_output[0]
        return (
            None,
            all_to_all_tensor(input_t, ctx.gather_dim, ctx.scatter_dim, ctx.group, False),
            None,
            None,
            None,
            None,
        )


class _Slice(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, local_input: Tensor, dim: int, scale_grad: bool) -> Tensor:
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        seq_world_size = dist.get_world_size(group)
        ctx.seq_world_size = seq_world_size
        ctx.dim = dim
        ctx.scale_grad = scale_grad
        dim_size = local_input.shape[dim]
        return local_input.split(dim_size // seq_world_size, dim=dim)[ctx.rank].contiguous()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[None, Tensor, None]:
        dim_size = list(grad_output.size())
        split_size = dim_size[0]
        output = _all_gather_into_tensor(grad_output, group=ctx.group)
        if ctx.scale_grad:
            output = output / ctx.seq_world_size
        return (None, torch.cat(output.split(split_size), dim=ctx.dim), None, None)


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_input: Tensor,
        dim: int,
        grad_scale: Optional[bool] = False,
    ) -> Tensor:
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        seq_world_size = dist.get_world_size(group)
        ctx.seq_world_size = seq_world_size
        output, size_list = _all_gather(local_input.contiguous(), group=ctx.group)
        dim_size_list = [size_list[i][dim].item() for i in range(seq_world_size)]
        ctx.dim_size_list = dim_size_list
        return torch.cat(output, dim=dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[None, Tensor]:
        if ctx.grad_scale:
            grad_output = grad_output * ctx.seq_world_size
        return (
            None,
            grad_output.split(ctx.dim_size_list, dim=ctx.dim)[ctx.rank].contiguous(),
            None,
            None,
        )


def gather_heads_scatter_seq(x: Tensor, head_dim: int, seq_dim: int, group: ProcessGroup = None) -> Tensor:
    """
    A func to sync attention result with alltoall in sequence parallel
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if not group:
        return x
    dim_size = x.size(seq_dim)
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    if dim_size % sp_world != 0:
        padding_size = sp_world - (dim_size % sp_world)
        x = pad_tensor(x, seq_dim, padding_size)
    return _SeqAllToAll.apply(group, x, seq_dim, head_dim, False)


def gather_seq_scatter_heads(
    x: Tensor,
    seq_dim: int,
    head_dim: int,
    unpadded_dim_size: int = 0,
    async_op: bool = False,
    group: ProcessGroup = None,
) -> Tensor:
    """
    A func to sync embedding input with alltoall in sequence parallel
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if not group:
        return x
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    if async_op:
        return _SeqAllToAll.apply(group, x, head_dim, seq_dim, async_op)
    else:
        x = _SeqAllToAll.apply(group, x, head_dim, seq_dim, async_op)
        if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
            padding_size = x.size(seq_dim) - unpadded_dim_size
            x = unpad_tensor(x, seq_dim, padding_size)
        return x


def gather_seq_scatter_heads_qkv(
    qkv_tensor: Tensor,
    seq_dim: int,
    unpadded_dim_size: Optional[int] = None,
    restore_shape: bool = True,
    async_op: bool = False,
    group: ProcessGroup = None,
) -> Tensor:
    """
    A func to sync splited qkv tensor
    qkv_tensor: the tensor we want to do alltoall with. The last dim must
        be the projection_idx, which we will split into 3 part. After
        spliting, the gather idx will be projecttion_idx + 1
    seq_dim: gather_dim for all2all comm
    restore_shape: if True, output will has the same shape length as input
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if not group:
        return qkv_tensor
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    orig_shape = qkv_tensor.shape
    scatter_dim = qkv_tensor.dim()
    bef_all2all_shape = list(orig_shape)
    qkv_proj_dim = bef_all2all_shape[-1]
    bef_all2all_shape = bef_all2all_shape[:-1] + [3, qkv_proj_dim // 3]
    qkv_tensor = qkv_tensor.view(bef_all2all_shape)
    if async_op:
        return _SeqAllToAll.apply(group, qkv_tensor, scatter_dim, seq_dim, async_op)
    else:
        qkv_tensor = _SeqAllToAll.apply(group, qkv_tensor, scatter_dim, seq_dim, async_op)

        if restore_shape:
            out_shape = list(orig_shape)
            out_shape[seq_dim] *= sp_world
            out_shape[-1] = qkv_proj_dim // sp_world
            qkv_tensor = qkv_tensor.view(out_shape)

        # remove padding
        if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
            padding_size = qkv_tensor.size(seq_dim) - unpadded_dim_size
            qkv_tensor = unpad_tensor(qkv_tensor, seq_dim, padding_size)

        return qkv_tensor


class _AlltoAllRegion(torch.autograd.Function):
    """balance the intermediate tensors in the sequence parallel region"""

    @staticmethod
    def forward(ctx, group, x, input_splits, output_splits):
        ctx.group = group
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        input_tensor_list = list(x.split(input_splits, dim=0))
        input_tensor_list = [t.contiguous() for t in input_tensor_list]
        output_tensor_list = [torch.empty([o, *x.shape[1:]], dtype=x.dtype, device=x.device) for o in output_splits]
        dist.all_to_all(output_tensor_list, input_tensor_list, group=group)
        return torch.cat(output_tensor_list, dim=0)

    def backward(ctx, dy):
        dx_list = [torch.empty([i, *dy.shape[1:]], dtype=dy.dtype, device=dy.device) for i in ctx.input_splits]
        dy_list = list(dy.split(ctx.output_splits, dim=0))
        dist.all_to_all(dx_list, dy_list, group=ctx.group)
        return None, torch.cat(dx_list, dim=0), None, None


def all_to_all_images(image_embeds, in_splits, out_splits):
    if not in_splits:
        return image_embeds
    image_embeds = image_embeds[: sum(in_splits)]
    group = get_ulysses_sequence_parallel_group()
    return _AlltoAllRegion.apply(group, image_embeds, in_splits, out_splits)
