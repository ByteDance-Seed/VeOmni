# Copyright 2026 the Miles contributors and ByteDance Ltd. and/or its affiliates
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

"""DeepSeek V4 model-specific SM90+ GPU kernels adapted from radixark/miles.

Imports stay inside the public wrappers because TileLang is an optional,
GPU-only dependency. Importing VeOmni on CPU or NPU must not load it.
"""

from typing import TYPE_CHECKING

import torch

from ....utils.device import IS_CUDA_AVAILABLE, get_gpu_compute_capability


if TYPE_CHECKING:
    from collections.abc import Callable


def _require_tilelang_sm90() -> None:
    if torch.version.hip is not None or not IS_CUDA_AVAILABLE or get_gpu_compute_capability() < 90:
        raise RuntimeError("DeepSeek V4 TileLang kernels require an SM90 or later NVIDIA CUDA GPU")


def sparse_attn_tilelang(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    _require_tilelang_sm90()
    from .tilelang_sparse_mla import sparse_attn_tilelang as impl

    return impl(q, kv, attn_sink, topk_idxs, sm_scale)


def v4_lighting_indexer(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    compress_ratio: int,
    topk: int,
    topk_indices: torch.Tensor | None = None,
    cu_seqlen_ks: torch.Tensor | None = None,
    cu_seqlen_ke: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_tilelang_sm90()
    from .tilelang_indexer import v4_lighting_indexer as impl

    return impl(
        index_q,
        index_k,
        weights,
        compress_ratio,
        topk,
        topk_indices,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    _require_tilelang_sm90()
    from .act_quant import act_quant as impl

    return impl(x, block_size, scale_fmt, scale_dtype, inplace)


def linear_bf16_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from .precision_aligned_ops import linear_bf16_fp32 as impl

    return impl(x, weight)


__all__ = ["act_quant", "linear_bf16_fp32", "sparse_attn_tilelang", "v4_lighting_indexer"]
