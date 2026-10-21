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

"""mHC pre on Ascend CANN (fused projection, Sinkhorn normalization and collapse)."""

from __future__ import annotations

from torch import Tensor

from ..vendor.npu_mhc import npu_mhc_pre_sinkhorn


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
    """Return ``(post, comb, collapsed)``, the eager row's outputs and shapes.

    *hidden_streams* is ``[B, S, H, D]`` BF16. The kernel reads the module
    parameters in FP32, and ``cann_ops_transformer`` owns the backward.
    """
    return npu_mhc_pre_sinkhorn(
        hidden_streams,
        fn.float().contiguous(),
        scale.float().contiguous(),
        base.float().contiguous(),
        hc_mult,
        sinkhorn_iters,
        hc_eps,
        norm_eps,
    )
