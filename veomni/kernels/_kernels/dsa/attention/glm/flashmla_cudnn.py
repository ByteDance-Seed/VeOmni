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
# See the License for the specific language governing limitations
# under the License.

"""GLM-DSA FlashMLA forward + cuDNN FE backward adapter."""

from __future__ import annotations

from torch import Tensor


def wrapper(
    q_pe: Tensor,
    k_pe: Tensor,
    kv_cache: Tensor,
    q_nope_absorbed: Tensor,
    topk_indices: Tensor,
    *,
    softmax_scale: float | None = None,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """FlashMLA sparse prefill with cuDNN backward. Same face as eager.

    ``q_pe`` / ``q_nope_absorbed`` are ``[B, S, H, D]``. ``k_pe`` and
    ``kv_cache`` are MQA ``[B, S_kv, 1, D]``.
    ``attention_mask`` is accepted for call-face parity with eager. The
    fused row applies causality through top-k.
    """
    del attention_mask
    from ....vendor.flashmla_cudnn import flash_mla_sparse_attention_with_cudnn_backward

    return flash_mla_sparse_attention_with_cudnn_backward(
        q_pe,
        k_pe,
        kv_cache,
        q_nope_absorbed,
        topk_indices,
        softmax_scale=softmax_scale,
    )
