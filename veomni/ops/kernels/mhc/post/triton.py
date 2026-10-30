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

from torch import Tensor

from ..helper import require_triton_input


def wrapper(
    output: Tensor,
    residual: Tensor,
    post: Tensor,
    comb: Tensor,
) -> Tensor:
    """Mix *output* back into the residual streams in one Triton kernel.

    *post* is ``[B, S, H]`` and *comb* ``[B, S, H, H]``. Like the TileKernels
    row, the mix coefficients stay in FP32 rather than being rounded to the
    residual dtype first.
    """
    from ..vendor.triton_mhc import hc_post_mix

    require_triton_input(residual, residual.shape[-2])
    return hc_post_mix(residual, output, post.float(), comb.float())
