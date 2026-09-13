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

"""First-party mHC helpers.

Eager inlines unweighted RMSNorm. TileLang adapters share the SM90 / bf16 /
``hc_mult=4`` guard used by TileKernels.
"""

from __future__ import annotations

import torch
from torch import Tensor


POST_MULTIPLIER = 2.0


def unweighted_rms(x: Tensor, eps: float) -> Tensor:
    """RMSNorm without an affine weight. Matches ``DeepseekV4UnweightedRMSNorm``."""
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)


def require_tilelang_input(x: Tensor, hc_mult: int) -> None:
    """Reject layouts TileKernels mHC cannot train."""
    if not x.is_cuda:
        raise ValueError("TileKernels mHC requires a CUDA tensor")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"TileKernels mHC requires bfloat16 activations, got {x.dtype}")
    if x.ndim != 4:
        raise ValueError(f"TileKernels mHC expects [batch, sequence, hc, hidden], got {tuple(x.shape)}")
    if hc_mult != 4 or x.shape[-2] != 4:
        raise ValueError(f"TileKernels mHC backward currently requires hc_mult=4, got {hc_mult}")
