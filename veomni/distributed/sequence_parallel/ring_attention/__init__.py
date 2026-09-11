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

from typing import Optional

import torch
from torch import Tensor
from torch.distributed import ProcessGroup

from ....utils.device import get_device_type
from .comm import RingComm
from .gpu import (
    FA_BACKEND,
    ring_flash_attn_func,
    update_out_and_lse,
    zigzag_ring_flash_attn_func,
    zigzag_ring_flash_attn_varlen_func,
)
from .layout import (
    local_cu_seqlens,
    zigzag_block_order,
    zigzag_reorder,
    zigzag_reorder_packed,
    zigzag_reorder_varlen,
    zigzag_undo,
)
from .npu import zigzag_ring_npu_flash_attn_func, zigzag_ring_npu_flash_attn_varlen_func


__all__ = [
    "ring_attention",
    "RingComm",
    "FA_BACKEND",
    "local_cu_seqlens",
    "zigzag_block_order",
    "zigzag_reorder",
    "zigzag_reorder_packed",
    "zigzag_reorder_varlen",
    "zigzag_undo",
    "ring_flash_attn_func",
    "update_out_and_lse",
    "zigzag_ring_flash_attn_func",
    "zigzag_ring_flash_attn_varlen_func",
    "zigzag_ring_npu_flash_attn_func",
    "zigzag_ring_npu_flash_attn_varlen_func",
]


def _load_backend(device_type: str):
    if device_type == "npu":
        from . import npu

        return npu

    from . import gpu

    return gpu


def ring_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    group: ProcessGroup,
    cp_size: int,
    cu_seqlens: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
    softmax_scale: Optional[float] = None,
    dropout_p: float = 0.0,
    causal: bool = True,
    device_type: Optional[str] = None,
) -> Tensor:
    """Run the device-specific balanced causal Ring Attention implementation."""
    if not causal:
        raise NotImplementedError("context-parallel (cp_size>1) ring attention requires causal attention")
    if attention_mask is not None:
        raise NotImplementedError(
            "context-parallel (cp_size>1) ring attention does not support explicit attention masks"
        )

    backend = _load_backend(get_device_type() if device_type is None else device_type)
    is_packed = cu_seqlens is not None and cu_seqlens.numel() > 2
    if not is_packed:
        return backend.forward(
            query,
            key,
            value,
            softmax_scale=softmax_scale,
            causal=True,
            group=group,
            dropout_p=dropout_p,
        )

    local_cu = local_cu_seqlens(cu_seqlens.to(torch.int32), cp_size)
    local_lengths = local_cu[1:] - local_cu[:-1]
    local_max_seqlen = int(local_lengths.max().item()) if local_lengths.numel() else 0
    output = backend.packed_forward(
        query.squeeze(0),
        key.squeeze(0),
        value.squeeze(0),
        local_cu,
        local_max_seqlen,
        softmax_scale=softmax_scale,
        causal=True,
        group=group,
        dropout_p=dropout_p,
    )
    return output.unsqueeze(0)
