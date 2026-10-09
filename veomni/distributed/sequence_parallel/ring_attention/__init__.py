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
    usp_zigzag_slice,
    vit_cp_cu_seqlens,
    vit_usp_slice,
    zigzag_block_order,
    zigzag_reorder,
    zigzag_reorder_packed,
    zigzag_reorder_varlen,
    zigzag_undo,
)
from .npu import (
    ring_npu_flash_attn_varlen_func,
    zigzag_ring_npu_flash_attn_func,
    zigzag_ring_npu_flash_attn_varlen_func,
)


__all__ = [
    "ring_attention",
    "zigzag_ring_attention",
    "bidirectional_ring_attention",
    "RingComm",
    "FA_BACKEND",
    "local_cu_seqlens",
    "usp_zigzag_slice",
    "vit_cp_cu_seqlens",
    "vit_usp_slice",
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
    "ring_npu_flash_attn_varlen_func",
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
    ulysses_size: int = 1,
    cu_seqlens: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
    softmax_scale: Optional[float] = None,
    dropout_p: float = 0.0,
    causal: bool = True,
    device_type: Optional[str] = None,
) -> Tensor:
    """Dispatch context-parallel Ring Attention by mask type.

    ``causal=True`` runs :func:`zigzag_ring_attention` (text, zig-zag layout) and ``causal=False``
    runs :func:`bidirectional_ring_attention` (ViT, vit seqs split into per-rank shards).
    """
    if attention_mask is not None:
        raise NotImplementedError(
            "context-parallel (cp_size>1) ring attention does not support explicit attention masks"
        )
    kwargs = dict(
        group=group,
        cp_size=cp_size,
        cu_seqlens=cu_seqlens,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        device_type=device_type,
    )
    if causal:
        return zigzag_ring_attention(query, key, value, **kwargs)
    return bidirectional_ring_attention(query, key, value, ulysses_size=ulysses_size, **kwargs)


def zigzag_ring_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    group: ProcessGroup,
    cp_size: int,
    cu_seqlens: Optional[Tensor] = None,
    softmax_scale: Optional[float] = None,
    dropout_p: float = 0.0,
    device_type: Optional[str] = None,
) -> Tensor:
    """Run the device-specific balanced causal Ring Attention implementation."""
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


def bidirectional_ring_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    group: ProcessGroup,
    cp_size: int,
    ulysses_size: int,
    cu_seqlens: Optional[Tensor],
    softmax_scale: Optional[float] = None,
    dropout_p: float = 0.0,
    device_type: Optional[str] = None,
) -> Tensor:
    """Run bidirectional packed Ring Attention for ViT inputs laid out by :func:`vit_usp_slice`.

    ``cu_seqlens`` are the GLOBAL offsets of the real vit seqs. Every cp rank's local offsets are
    derived here, so the backend receives one ``cu_seqlens`` per cp rank and pairs the local query
    with the key/value of the source rank at each ring step.
    """
    if cu_seqlens is None:
        raise ValueError("bidirectional ring attention requires the global cu_seqlens of the vit sequences")
    backend = _load_backend(get_device_type() if device_type is None else device_type)

    global_cu_seqlens = cu_seqlens.tolist()
    cu_seqlens_per_rank = [
        vit_cp_cu_seqlens(global_cu_seqlens, cp_size, cp_rank, ulysses_size) for cp_rank in range(cp_size)
    ]
    local_len = cu_seqlens_per_rank[0][-1]
    if query.shape[1] != local_len:
        raise ValueError(
            f"local sequence length {query.shape[1]} != expected cp local length {local_len}; "
            "the vit inputs must be sliced with vit_usp_slice using the same cp/ulysses sizes"
        )

    output = backend.bidirectional_packed_forward(
        query.squeeze(0),
        key.squeeze(0),
        value.squeeze(0),
        cu_seqlens_per_rank,
        softmax_scale=softmax_scale,
        group=group,
        dropout_p=dropout_p,
    )
    return output.unsqueeze(0)
