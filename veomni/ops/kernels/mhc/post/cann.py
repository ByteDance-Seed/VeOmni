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

"""mHC post on Ascend CANN (fused residual post-mix)."""

from __future__ import annotations

from torch import Tensor

from ..vendor.npu_mhc import npu_mhc_post


def wrapper(
    output: Tensor,
    residual: Tensor,
    post: Tensor,
    comb: Tensor,
) -> Tensor:
    """Mix *output* back into the residual streams, as the eager row does.

    *post* is ``[B, S, H]`` and *comb* is ``[B, S, H, H]``. The result has
    *residual*'s shape, and ``cann_ops_transformer`` owns the backward.
    """
    return npu_mhc_post(output, residual, post, comb)
