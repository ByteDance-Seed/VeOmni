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


from collections.abc import Sequence
from typing import Mapping

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule


_SUPPORTED_TRANSPORT_DTYPES = frozenset((torch.bfloat16, torch.float16))


def _accumulate_received(
    received: torch.Tensor,
    output: torch.Tensor,
    world_size: int,
    scale: float,
) -> None:
    reduced = torch.sum(received.view(world_size, -1), dim=0, dtype=torch.float32)
    if scale != 1.0:
        reduced.mul_(scale)
    output.copy_(reduced.view(output.shape))


class BF16FP16ReduceScatterWithFP32Accumulation:
    """FSDP2 ReduceScatter using BF16/FP16 transport and FP32 accumulation.

    The collective is decomposed into a BF16 or FP16 all-to-all followed by an
    FP32 sum and scale on the destination rank. FSDP must be configured to pass
    an unscaled SUM and leave all gradient scaling to this implementation.
    """

    def __init__(self, reduction_scale: float) -> None:
        self._reduction_scale = reduction_scale

    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.empty(size, dtype=dtype, device=device)

    @torch.no_grad()
    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: dist.ReduceOp,
        async_op: bool = False,
    ) -> dist.Work | None:
        if input_tensor.dtype not in _SUPPORTED_TRANSPORT_DTYPES:
            raise TypeError(
                "BF16/FP16 FP32-accumulation ReduceScatter requires a BF16 or FP16 input tensor, "
                f"got {input_tensor.dtype}."
            )

        if async_op:
            raise NotImplementedError("BF16/FP16 FP32-accumulation ReduceScatter does not support async_op=True.")

        if output_tensor.dtype != input_tensor.dtype:
            raise TypeError(
                "BF16/FP16 reduce-scatter requires matching input and output dtypes, "
                f"got {input_tensor.dtype} and {output_tensor.dtype}."
            )
        if input_tensor.device != output_tensor.device:
            raise ValueError(
                "Reduce-scatter input and output must be on the same device, "
                f"got {input_tensor.device} and {output_tensor.device}."
            )
        if not input_tensor.is_contiguous() or not output_tensor.is_contiguous():
            raise ValueError("Reduce-scatter input and output must be contiguous.")

        world_size = dist.get_world_size(group)
        if input_tensor.numel() != output_tensor.numel() * world_size:
            raise ValueError(
                "Reduce-scatter input must contain one equally sized output shard per rank, "
                f"got input numel {input_tensor.numel()}, output numel {output_tensor.numel()}, "
                f"and world size {world_size}."
            )

        if op != dist.ReduceOp.SUM:
            raise ValueError(f"BF16/FP16 FP32-accumulation ReduceScatter requires SUM, got {op}.")

        received = torch.empty_like(input_tensor)
        dist.all_to_all_single(
            received,
            input_tensor,
            group=group,
            async_op=False,
        )
        _accumulate_received(received, output_tensor, world_size, self._reduction_scale)
        return None


def register_bf16_fp16_reduce_scatter_with_fp32_accumulation(
    model: torch.nn.Module,
    *,
    reduction_scales: Mapping[torch.nn.Module, float],
) -> int:
    """Register custom communication and move gradient scaling into the hook."""
    count = 0
    for module in model.modules():
        if isinstance(module, FSDPModule) and module in reduction_scales:
            module.set_gradient_divide_factor(1.0)
            module.set_force_sum_reduction_for_comms(True)
            comm = BF16FP16ReduceScatterWithFP32Accumulation(reduction_scale=reduction_scales[module])
            module.set_custom_reduce_scatter(comm)
            count += 1
    return count
