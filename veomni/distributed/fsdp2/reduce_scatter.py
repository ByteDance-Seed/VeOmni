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
from typing import Mapping, Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule


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


class BF16ReduceScatterWithFP32Accumulation:
    """FSDP2 ReduceScatter using BF16 transport and local FP32 accumulation.

    The collective is decomposed into a BF16 all-to-all followed by an FP32
    sum on the destination rank. Non-BF16 tensors use the native collective so
    modules excluded from FSDP mixed precision keep their existing behavior.
    PyTorch 2.11 does not expose the scalar stored in a PREMUL_SUM operation,
    so ``premul_sum_factor`` must be the factor configured on this FSDP state.
    """

    def __init__(self, premul_sum_factor: Optional[float] = None) -> None:
        self._premul_sum_factor = premul_sum_factor

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
        if input_tensor.dtype != torch.bfloat16:
            return dist.reduce_scatter_tensor(
                output_tensor,
                input_tensor,
                group=group,
                op=op,
                async_op=async_op,
            )

        if async_op:
            raise NotImplementedError("BF16 FP32-accumulation ReduceScatter does not support async_op=True.")

        if output_tensor.dtype != torch.bfloat16:
            raise TypeError(f"BF16 reduce-scatter requires a BF16 output tensor, got {output_tensor.dtype}.")
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

        if op == dist.ReduceOp.SUM:
            scale = 1.0
        elif op == dist.ReduceOp.AVG:
            scale = 1.0 / world_size
        elif op == dist.ReduceOp.PREMUL_SUM:
            if self._premul_sum_factor is None:
                raise ValueError("PREMUL_SUM requires premul_sum_factor to be configured.")
            scale = self._premul_sum_factor
        else:
            raise ValueError(f"Unsupported reduce operation: {op}.")

        received = torch.empty_like(input_tensor)
        dist.all_to_all_single(
            received,
            input_tensor,
            group=group,
            async_op=False,
        )
        _accumulate_received(received, output_tensor, world_size, scale)
        return None


def register_bf16_reduce_scatter_with_fp32_accumulation(
    model: torch.nn.Module,
    *,
    premul_sum_factors: Optional[Mapping[torch.nn.Module, float]] = None,
) -> int:
    """Register custom communication, binding known PREMUL factors per FSDP state."""
    premul_sum_factors = premul_sum_factors or {}
    count = 0
    for module in model.modules():
        if isinstance(module, FSDPModule):
            comm = BF16ReduceScatterWithFP32Accumulation(premul_sum_factor=premul_sum_factors.get(module))
            module.set_custom_reduce_scatter(comm)
            count += 1
    return count
