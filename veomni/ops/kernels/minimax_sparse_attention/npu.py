# Copyright 2026 The MiniMax AI Team, HuggingFace Team, and the VeOmni Team. All rights reserved.
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

"""Ascend fused attention on MiniMax block-selected tokens."""

import torch

from ._common import _run_padded_attention


def _npu_fusion_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    keep: torch.Tensor,
    *,
    scale: float,
    dropout_p: float,
) -> torch.Tensor:
    import torch_npu

    output = torch_npu.npu_fusion_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        head_num=query.shape[1],
        input_layout="BNSD",
        atten_mask=(~keep).contiguous(),
        sparse_mode=1,
        inner_precise=2,
        scale=scale,
        keep_prob=1.0 - dropout_p,
    )[0]
    return output.transpose(1, 2).contiguous()


def minimax_sparse_attention_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    block_size: int,
    scale: float,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Apply NPU fused attention to packed Q/K/V using per-group block indices."""
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("MiniMax NPU attention requires matching FP16 or BF16 query, key and value.")
    return _run_padded_attention(
        q,
        k,
        v,
        block_indices,
        cu_seqlens,
        max_seqlen,
        block_size=block_size,
        scale=scale,
        dropout_p=dropout_p,
        attend=_npu_fusion_attention,
        via_bfloat16=True,
    )
