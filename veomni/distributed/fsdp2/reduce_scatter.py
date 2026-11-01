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
from pathlib import Path
from typing import Mapping
from uuid import UUID

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule

from ...utils import logging


_SUPPORTED_TRANSPORT_DTYPES = frozenset((torch.bfloat16, torch.float16))
logger = logging.get_logger(__name__)


def _get_node_id() -> str | None:
    """Use the Linux kernel boot identity, not a container hostname or rank layout."""
    try:
        node_id = UUID(Path("/proc/sys/kernel/random/boot_id").read_text().strip())
    except (OSError, ValueError):
        return None
    return str(node_id) if node_id.int not in (0, (1 << 128) - 1) else None


class ReduceScatterTransportPolicy:
    """Cache placement decisions during one model's FSDP initialization."""

    def __init__(self) -> None:
        self._node_id = _get_node_id()
        self._shard_status: dict[dist.ProcessGroup, str] = {}
        self._decisions: dict[tuple[dist.ProcessGroup, dist.ProcessGroup | None], bool] = {}

    def can_use(self, mesh) -> bool:
        """Select transport for a named 1D shard mesh or 2D replica/shard mesh."""
        if len(mesh.mesh_dim_names) not in (1, 2):
            raise ValueError("Low-precision ReduceScatter requires a 1D or 2D FSDP mesh.")
        shard_group = mesh.get_group(mesh.mesh_dim_names[-1])
        replica_group = mesh.get_group(mesh.mesh_dim_names[0]) if len(mesh.mesh_dim_names) == 2 else None
        key = (shard_group, replica_group)
        if key in self._decisions:
            return self._decisions[key]

        shard_size = dist.get_world_size(shard_group)
        if shard_group not in self._shard_status:
            if shard_size == 1:
                status = "single_rank"
            else:
                node_ids = [None] * shard_size
                # Unavailable identity is a collective fallback; communication failures must propagate.
                dist.all_gather_object(node_ids, self._node_id, group=shard_group)
                status = "unknown" if None in node_ids else "node_local" if len(set(node_ids)) == 1 else "cross_node"
            self._shard_status[shard_group] = status
        status = self._shard_status[shard_group]

        if replica_group is not None and shard_size > 1:
            # Never skip this on a rejected shard row: HSDP peers must agree on
            # the custom scaling/force-SUM contract before their native AllReduce.
            statuses = [None] * dist.get_world_size(replica_group)
            dist.all_gather_object(statuses, status, group=replica_group)
            if "unknown" in statuses:
                status = "unknown"
            elif "cross_node" in statuses:
                status = "cross_node"

        enabled = status == "node_local"
        self._decisions[key] = enabled
        if dist.get_rank(shard_group) == 0:
            ranks = dist.get_process_group_ranks(shard_group)
            if enabled:
                logger.info(f"Low-precision ReduceScatter transport enabled for node-local shard group {ranks}.")
            elif status == "single_rank":
                logger.info(f"Using native ReduceScatter for single-rank shard group {ranks}.")
            else:
                reason = "node identity is unavailable" if status == "unknown" else "a shard group spans nodes"
                logger.warning(
                    f"Using native ReduceScatter for shard group {ranks}: {reason} in this FSDP mesh. "
                    "Replica-linked shard groups retain consistent native scaling."
                )
        return enabled


class FP32ReduceScatterWithLowPrecisionTransport:
    """FSDP2 FP32 reduction using BF16/FP16 transport.

    FSDP keeps its FP32 reduction input and output contract. This implementation
    converts only the wire buffers to BF16 or FP16, then performs the destination-
    local sum and scale directly into the FP32 output. FSDP must pass an unscaled
    SUM and leave all gradient scaling to this implementation. BF16 preserves the
    FP32 exponent range; FP16 transport follows normal cast semantics and can
    overflow finite FP32 values outside the FP16 range. Unscaled FP32 sums may
    overflow even for finite BF16 inputs whose average is representable; this
    does not preserve native AVG's overflow behavior.
    """

    def __init__(self, transport_dtype: torch.dtype, reduction_scale: float) -> None:
        if transport_dtype not in _SUPPORTED_TRANSPORT_DTYPES:
            raise ValueError(
                "Low-precision ReduceScatter transport must be torch.bfloat16 or torch.float16, "
                f"got {transport_dtype}."
            )
        self._transport_dtype = transport_dtype
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
        if input_tensor.dtype != torch.float32 or output_tensor.dtype != torch.float32:
            raise TypeError(
                "Low-precision transport requires FP32 ReduceScatter input and output tensors, "
                f"got {input_tensor.dtype} and {output_tensor.dtype}."
            )

        if async_op:
            # FSDP2 calls synchronously on its reduction stream; stream events provide overlap.
            raise NotImplementedError("Low-precision transport ReduceScatter does not support async_op=True.")

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
            raise ValueError(f"Low-precision transport ReduceScatter requires SUM, got {op}.")

        transport_input = input_tensor.to(self._transport_dtype)
        received = torch.empty_like(transport_input)
        dist.all_to_all_single(
            received,
            transport_input,
            group=group,
            async_op=False,
        )
        torch.sum(
            received.view(world_size, -1),
            dim=0,
            dtype=torch.float32,
            out=output_tensor.view(-1),
        )
        if self._reduction_scale != 1.0:
            output_tensor.mul_(self._reduction_scale)
        return None


def register_fp32_reduce_scatter_with_low_precision_transport(
    model: torch.nn.Module,
    *,
    transport_dtype: torch.dtype,
    reduction_scales: Mapping[torch.nn.Module, float],
) -> int:
    """Register custom communication and move gradient scaling into the hook."""
    count = 0
    for module in model.modules():
        if isinstance(module, FSDPModule) and module in reduction_scales:
            # Together these disable FSDP pre/post scaling and force SUM; the hook owns all scaling.
            module.set_gradient_divide_factor(1.0)
            module.set_force_sum_reduction_for_comms(True)
            comm = FP32ReduceScatterWithLowPrecisionTransport(
                transport_dtype=transport_dtype,
                reduction_scale=reduction_scales[module],
            )
            module.set_custom_reduce_scatter(comm)
            count += 1
    return count
