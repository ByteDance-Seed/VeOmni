# Copyright 2026 ByteDance Ltd. and/or its affiliates
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

"""Qwen4-Exp QSA TileLang sparse-attention autograd wrapper."""

import torch

from . import tilelang_qsa_bwd as qsa_bwd
from . import tilelang_qsa_fwd as qsa_fwd


def _validate_qsa_inputs(query, key, value, selected_indices):
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            f"query/key/value must be rank-4 tensors; got query={query.shape}, key={key.shape}, value={value.shape}"
        )
    if key.shape != value.shape:
        raise ValueError(f"key/value shapes must match; got key={key.shape}, value={value.shape}")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError(f"query/key batch and head dimensions must match; got query={query.shape}, key={key.shape}")
    if selected_indices.ndim != 3 or selected_indices.shape[:2] != (query.shape[0], query.shape[2]):
        raise ValueError(
            "query batch and sequence dimensions must match selected_indices; "
            f"got query={query.shape}, selected_indices={selected_indices.shape}"
        )
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError(f"query heads ({query.shape[1]}) must be divisible by kv heads ({key.shape[1]})")
    if selected_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"selected_indices must use int32 or int64, got {selected_indices.dtype}")
    if key.shape[2] == 0:
        raise ValueError("key/value must contain at least one token")
    if torch.any((selected_indices < -1) | (selected_indices >= key.shape[2])):
        raise ValueError(
            f"selected_indices must contain -1 padding or valid global KV token indices in [0, {key.shape[2]})"
        )


class Qwen4ExpSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, selected_indices, sm_scale=None):
        o, lse = qsa_fwd.qsa_fwd_interface(q, k, v, selected_indices, sm_scale=sm_scale)

        ctx.save_for_backward(q, k, v, selected_indices, o.clone(), lse)
        ctx.sm_scale = sm_scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, selected_indices, o, lse = ctx.saved_tensors

        dq, dk, dv = qsa_bwd.qsa_bwd_interface(
            q, k, v, o, do.contiguous(), selected_indices, lse, sm_scale=ctx.sm_scale
        )

        return dq, dk, dv, None, None


def qsa_attn_tilelang(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_indices: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Sparse GQA attention over compact selected KV indices.

    Every query row attends to exactly the K/V rows listed in its
    ``selected_indices`` entry; ``-1`` slots are skipped. The selection already
    encodes causality and packed-sample boundaries, so the kernel applies no
    further masking. Valid indices within each query row must be unique, as
    guaranteed by the QSA indexer.

    Args:
        query:            [B, H, S, D] bf16
        key:              [B, H_kv, S_kv, D] bf16
        value:            [B, H_kv, S_kv, D] bf16
        selected_indices: [B, S, topk] int32/int64, unique valid indices with -1 padding
        sm_scale:         softmax scale, defaults to ``1/sqrt(D)``

    Returns:
        [B, S, H, D] bf16, matching the layout the eager QSA reference returns.
    """
    # The kernels are compiled for bf16 operands. Callers run under autocast,
    # whose fp32 op policy (sum, rsqrt, ...) can silently promote an upstream
    # tensor, so reject the mismatch here instead of feeding the kernel garbage.
    if query.dtype is not torch.bfloat16 or key.dtype is not torch.bfloat16 or value.dtype is not torch.bfloat16:
        raise ValueError(
            "Qwen4-Exp TileLang QSA requires bfloat16 query/key/value, got "
            f"query={query.dtype}, key={key.dtype}, value={value.dtype}"
        )
    _validate_qsa_inputs(query, key, value, selected_indices)
    head_dim = query.shape[-1]
    if query.shape[-1] != 1 << (query.shape[-1] - 1).bit_length():
        raise ValueError(f"QSA TileLang requires a power-of-two head dim; got {query.shape[-1]}.")
    # Resolve the default from the logical head width before the TileLang
    # interfaces pad small dimensions to their 64-wide MMA floor. The same
    # explicit value is saved for backward, keeping both passes consistent.
    if sm_scale is None:
        sm_scale = head_dim**-0.5

    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()
    indices = selected_indices
    if indices.dtype != torch.int32:
        indices = indices.int()
    indices = indices.contiguous()
    return Qwen4ExpSparseAttention.apply(q, k, v, indices, sm_scale)
