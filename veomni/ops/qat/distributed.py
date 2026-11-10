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

"""Distributed-operand handling shared by the fake-quantization primitives.

A fake-quantizing row is a *tensor-level* op: it rounds what it is handed and
passes it to ``F.linear``. It issues no collectives, so it can only run on an
operand that is already whole on the calling rank. These helpers express that
requirement once so every format applies it identically.
"""

from __future__ import annotations

from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate


__all__ = [
    "local_shard",
]


def local_shard(tensor: Tensor) -> Tensor:
    """Unwrap a ``Replicate`` DTensor, and refuse one this op cannot combine.

    ``Replicate`` qualifies -- the local shard is the entire tensor, and FSDP2
    presents its parameters that way after the all-gather. A ``Shard`` placement
    does not: the local pieces are partial operands, and completing the GEMM
    needs a collective the row has no way to issue. Sharding the output
    dimension would need a concat afterwards, and sharding the input dimension
    needs an all-reduce; quantizing each piece locally is fine arithmetic, but
    the result would be a partial product presented as a complete one.

    So a sharded placement raises rather than returning wrong numbers. Use a
    fully sharded FSDP2 mesh, which all-gathers before the forward, or keep this
    row off the sharded axis.

    The bias goes through this too. Leaving it a DTensor while the weight became
    a plain tensor makes ``addmm`` reject the call with "got mixed torch.Tensor
    and DTensor", and the bias is not quantized, so its local form is the same
    value.

    Args:
        tensor: A plain tensor, or a DTensor produced by a parallelism plan.

    Returns:
        A plain tensor that is whole on this rank.

    Raises:
        TypeError: if ``tensor`` is sharded.
    """
    if not isinstance(tensor, DTensor):
        return tensor
    if not all(isinstance(placement, Replicate) for placement in tensor.placements):
        raise TypeError(
            "fake quantization needs an operand that is whole on this rank, got placements "
            f"{tuple(str(p) for p in tensor.placements)}. These rows do not issue collectives: a sharded "
            "weight would produce a partial product, so it is refused instead of silently mis-multiplied. "
            "FSDP2 all-gathers before the forward and is therefore unaffected."
        )
    return tensor.to_local()
