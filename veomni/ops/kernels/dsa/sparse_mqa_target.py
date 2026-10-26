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

"""Head-summed compressed-slice teacher for the DSA indexer loss.

DeepSeek-V3.2 eq. 4. Not a registered kernel row. TileLang is optional
and GPU-only; importing this module must not load the vendor kernel.
"""

from __future__ import annotations

import torch

from veomni.utils.device import IS_CUDA_AVAILABLE, get_gpu_compute_capability


def _require_tilelang_sm90() -> None:
    """Require the NVIDIA architecture supported by the TileLang target kernel."""
    if torch.version.hip is not None or not IS_CUDA_AVAILABLE or get_gpu_compute_capability() < 90:
        raise RuntimeError("DeepSeek V4 TileLang kernels require an SM90 or later NVIDIA CUDA GPU")


def sparse_mqa_target_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Head-summed attention mass over the compressed index slice.

    ``lse`` is the full base-2 CSA log-sum-exp from sparse MQA forward,
    including the sliding window and the folded attention sink. The returned
    tensor is unnormalised; L1-normalise in the caller.
    """
    _require_tilelang_sm90()
    from .vendor.tilelang_sparse_mla_target import sparse_mqa_target_fwd_interface

    return sparse_mqa_target_fwd_interface(q, kv, topk_idxs, lse, sm_scale)


def sparse_attn_tilelang(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: float | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Sparse MQA over selected gathered KV entries via TileLang (SM90+ only)."""
    _require_tilelang_sm90()
    from .vendor.tilelang_sparse_mla import sparse_attn_tilelang as _impl

    return _impl(q, kv, attn_sink, topk_idxs, sm_scale=sm_scale, return_lse=return_lse)


__all__ = [
    "sparse_attn_tilelang",
    "sparse_mqa_target_fwd",
]
