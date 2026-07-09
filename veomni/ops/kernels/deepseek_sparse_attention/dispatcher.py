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

"""Backend dispatcher for DeepSeek sparse MLA attention."""

from __future__ import annotations

import torch


def sparse_mla_attention_with_cudnn_backward(
    backend: str,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    q_nope_absorbed: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run the selected sparse MLA forward with cuDNN FE DSA backward."""
    if backend == "flashmla_cudnn":
        from veomni.ops.kernels.deepseek_sparse_attention.flashmla_cudnn import (
            flash_mla_sparse_attention_with_cudnn_backward,
        )

        return flash_mla_sparse_attention_with_cudnn_backward(
            q_pe,
            k_pe,
            kv_cache,
            q_nope_absorbed,
            topk_indices,
            attn_sink=attn_sink,
            topk_length=topk_length,
            softmax_scale=softmax_scale,
        )
    if backend == "fa4_cudnn":
        from veomni.ops.kernels.deepseek_sparse_attention.fa4_cudnn import (
            fa4_sparse_attention_with_cudnn_backward,
        )

        return fa4_sparse_attention_with_cudnn_backward(
            q_pe,
            k_pe,
            kv_cache,
            q_nope_absorbed,
            topk_indices,
            attn_sink=attn_sink,
            topk_length=topk_length,
            softmax_scale=softmax_scale,
        )
    raise ValueError(f"Unknown dsa_attention_backend={backend!r}")


__all__ = ["sparse_mla_attention_with_cudnn_backward"]
