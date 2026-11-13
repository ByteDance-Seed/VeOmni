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

"""First-party mHC math and TileKernels input validation helpers.

Eager inlines unweighted RMSNorm. TileLang adapters validate CUDA placement,
BF16 dtype, rank-4 layout, and ``hc_mult=4`` here. Registry requirements
separately restrict those rows to NVIDIA SM90 or later and require the
``tile_kernels`` package.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


POST_MULTIPLIER = 2.0

#: Query-axis tile the TileKernels mHC kernels index without a length guard.
#: ``mhc_pre_norm_fn`` and its backward walk the token axis in steps of this size,
#: so a packed sequence whose per-sample query span is not a multiple of it reads
#: and writes past the tensor -- which is what turned a dynamic-batch step's
#: ``grad_norm`` into NaN on a 256k run. Callers pad to it and slice the result.
TOKEN_ALIGN = 128
#: The inference-only fused collapse (``mhc_pre_big_fuse``) tiles more coarsely.
FUSED_TOKEN_ALIGN = 32


def unweighted_rms(x: Tensor, eps: float) -> Tensor:
    """RMSNorm without an affine weight. Matches ``DeepseekV4UnweightedRMSNorm``."""
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)


def require_tilelang_input(x: Tensor, hc_mult: int) -> None:
    """Validate tensor placement, dtype, shape, and hyper-connection width."""
    if not x.is_cuda:
        raise ValueError("TileKernels mHC requires a CUDA tensor")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"TileKernels mHC requires bfloat16 activations, got {x.dtype}")
    if x.ndim != 4:
        raise ValueError(f"TileKernels mHC expects [batch, sequence, hc, hidden], got {tuple(x.shape)}")
    if hc_mult != 4 or x.shape[-2] != 4:
        raise ValueError(f"TileKernels mHC backward currently requires hc_mult=4, got {hc_mult}")


def pad_token_axis(x: Tensor, align: int) -> tuple[Tensor, int]:
    """Pad the query axis of a ``[batch, sequence, hc, hidden]`` tensor to ``align``.

    Returns ``(padded_or_original, seq_len)``. The caller slices every output of
    the kernel back to ``seq_len``; a zero row carries no information the kernel
    could leak into a kept row, because each query position's mix depends only on
    its own row plus the shared ``fn`` / ``scale`` / ``base``.
    """
    seq_len = x.shape[1]
    pad = -seq_len % align
    if not pad:
        return x.contiguous(), seq_len
    # ``contiguous()`` after the pad, not before: ``F.pad`` preserves a
    # ``channels_last`` memory format, and the kernels ``view(...)`` the result to
    # flatten batch and sequence, which a preserved stride layout breaks.
    return F.pad(x, (0, 0, 0, 0, 0, pad)).contiguous(), seq_len
