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

#: Query-axis tile the TileKernels mHC backward indexes without a length guard.
#: ``_mhc_pre_norm_fn_bwd_mul`` walks the token axis in steps of this size, so a
#: packed sequence whose per-sample query span is not a multiple of it reads and
#: writes past the tensor -- which is what turned a dynamic-batch step's
#: ``grad_norm`` into NaN on a 256k run. Callers pad to it and slice the result.
#: The forward halves of ``mhc_pre_norm_fn`` tile at ``TOKEN_BLOCK`` instead.
TOKEN_ALIGN = 128
#: The tile the forward half of these kernels uses. It is *finer* than
#: ``TOKEN_ALIGN``, not coarser: both ``mhc_pre_norm_fn``'s forward
#: (``_mhc_pre_norm_fn_fwd_mul``) and the fused inference collapse
#: (``mhc_pre_big_fuse``, which launches per token and calls that same forward)
#: walk the token axis in blocks of this size without a length guard. Padding a
#: forward-only call to 32 rather than 128 skips up to 96 wasted rows per sample.
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
    the kernel back to ``seq_len`` with :func:`trim_token_axis`; a zero row carries
    no information the kernel could leak into a kept row, because each query
    position's mix depends only on its own row plus the shared ``fn`` /
    ``scale`` / ``base``.

    The returned tensor is always contiguous, and may be the input itself even when
    nothing was padded -- every call runs ``.contiguous()``. That is why a caller
    must decide whether to trim from the *shape* (``padded.shape[1] != seq_len``)
    and not from an identity check against ``x``: an already-aligned but
    non-contiguous input comes back as a different tensor with the same length.
    """
    seq_len = x.shape[1]
    pad = -seq_len % align
    if not pad:
        return x.contiguous(), seq_len
    # ``contiguous()`` after the pad, not before: ``F.pad`` preserves a
    # ``channels_last`` memory format, and the kernels ``view(...)`` the result to
    # flatten batch and sequence, which a preserved stride layout breaks.
    return F.pad(x, (0, 0, 0, 0, 0, pad)).contiguous(), seq_len


def trim_token_axis(t: Tensor, seq_len: int) -> Tensor:
    """Drop the padding :func:`pad_token_axis` added, returning a contiguous tensor.

    Slicing alone is not enough for ``batch > 1``: the padded buffer is
    ``[batch, aligned, ...]``, so ``t[:, :seq_len]`` of it strides over the gap
    between samples and is not contiguous. A non-contiguous view also keeps the
    whole padded allocation alive, which is a per-layer cost in a training loop.
    When nothing was padded the slice is already the whole tensor and contiguous,
    so the clone is skipped.
    """
    if t.shape[1] == seq_len:
        return t if t.is_contiguous() else t.contiguous()
    return t[:, :seq_len].contiguous()
