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

"""mHC pre Triton adapter (AMD ROCm)."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from ..helper import POST_MULTIPLIER, require_triton_input, require_triton_tensors


def wrapper(
    hidden_streams: Tensor,
    fn: Tensor,
    scale: Tensor,
    base: Tensor,
    norm_eps: float,
    hc_mult: int,
    sinkhorn_iters: int,
    hc_eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Triton norm / mix tail / Sinkhorn / collapse around the FP32 mix GEMM.

    Same contract as the eager row: ``post`` and ``comb`` are FP32 and
    ``collapsed`` keeps the activation dtype.
    """
    require_triton_input(hidden_streams, hc_mult)
    channels = (2 + hc_mult) * hc_mult
    expected_fn = (channels, hc_mult * hidden_streams.shape[-1])
    if fn.shape != expected_fn:
        raise ValueError(f"Triton mHC pre expects fn shape {expected_fn}, got {tuple(fn.shape)}")
    if scale.shape != (3,):
        raise ValueError(f"Triton mHC pre expects scale shape (3,), got {tuple(scale.shape)}")
    if base.shape != (channels,):
        raise ValueError(f"Triton mHC pre expects base shape ({channels},), got {tuple(base.shape)}")
    require_triton_tensors(hidden_streams, fn=fn, scale=scale, base=base)

    from ..vendor.triton_mhc import hc_collapse, mhc_mix_tail, sinkhorn_normalize, unweighted_rms_norm

    normed = unweighted_rms_norm(hidden_streams.flatten(start_dim=2), norm_eps)
    mixes = F.linear(normed, fn.float())
    pre, post, comb = mhc_mix_tail(mixes, scale, base, hc_mult, POST_MULTIPLIER, hc_eps)
    comb = sinkhorn_normalize(comb, sinkhorn_iters, hc_eps)
    return post, comb, hc_collapse(hidden_streams, pre)
