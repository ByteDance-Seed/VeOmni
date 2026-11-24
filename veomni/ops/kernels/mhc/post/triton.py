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

"""mHC post Triton adapter (AMD ROCm)."""

from __future__ import annotations

import torch
from torch import Tensor

from ..helper import require_triton_input, require_triton_tensors


def wrapper(
    output: Tensor,
    residual: Tensor,
    post: Tensor,
    comb: Tensor,
) -> Tensor:
    """Mix *output* back into the residual streams in one Triton kernel.

    *post* is ``[B, S, H]`` and *comb* ``[B, S, H, H]``. Like the TileKernels
    row, the mix coefficients stay in FP32 rather than being rounded to the
    residual dtype first. The result keeps the residual dtype, so *output* may
    be narrower than *residual* (BF16 sublayer output into FP32 streams) but
    not wider.
    """
    hc_mult = residual.shape[-2] if residual.ndim >= 2 else 0
    require_triton_input(residual, hc_mult)
    expected_output = residual.shape[:-2] + residual.shape[-1:]
    if output.shape != expected_output:
        raise ValueError(f"Triton mHC post expects output shape {expected_output}, got {tuple(output.shape)}")
    if post.shape != residual.shape[:-1]:
        raise ValueError(f"Triton mHC post expects post shape {residual.shape[:-1]}, got {tuple(post.shape)}")
    expected_comb = residual.shape[:-1] + (hc_mult,)
    if comb.shape != expected_comb:
        raise ValueError(f"Triton mHC post expects comb shape {expected_comb}, got {tuple(comb.shape)}")
    require_triton_tensors(residual, output=output, post=post, comb=comb)
    if torch.promote_types(output.dtype, residual.dtype) != residual.dtype:
        raise TypeError(
            f"Triton mHC post returns the residual dtype {residual.dtype}, which cannot hold output dtype {output.dtype}"
        )

    from ..vendor.triton_mhc import hc_post_mix

    return hc_post_mix(residual, output, post.float(), comb.float())
