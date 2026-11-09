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

"""First-party mHC math and fused-kernel input validation helpers.

Eager inlines unweighted RMSNorm. TileLang adapters validate CUDA placement,
BF16 dtype, rank-4 layout, and ``hc_mult=4`` here; Triton adapters validate
GPU placement, rank-4 layout, and the stream count. Registry requirements
separately restrict the TileLang rows to NVIDIA SM90 or later with the
``tile_kernels`` package, and the Triton rows to AMD ROCm.
"""

from __future__ import annotations

import torch
from torch import Tensor


POST_MULTIPLIER = 2.0
TRITON_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


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


def require_triton_input(x: Tensor, hc_mult: int) -> None:
    """Validate tensor placement, layout, and stream count for the Triton rows."""
    if not x.is_cuda:
        raise ValueError("Triton mHC requires a GPU tensor")
    if x.dtype not in TRITON_FLOAT_DTYPES:
        raise TypeError(f"Triton mHC requires float16, bfloat16, or float32 activations, got {x.dtype}")
    if x.ndim != 4:
        raise ValueError(f"Triton mHC expects [batch, sequence, hc, hidden], got {tuple(x.shape)}")
    if x.shape[-2] != hc_mult:
        raise ValueError(f"Triton mHC got {x.shape[-2]} residual streams for hc_mult={hc_mult}")
    if x.shape[-1] == 0:
        raise ValueError("Triton mHC requires a nonzero hidden dimension")


def require_triton_tensors(reference: Tensor, **tensors: Tensor) -> None:
    """Require auxiliary tensors to share the activation device and use supported floating dtypes."""
    for name, tensor in tensors.items():
        if tensor.device != reference.device:
            raise ValueError(f"Triton mHC requires {name} on {reference.device}, got {tensor.device}")
        if tensor.dtype not in TRITON_FLOAT_DTYPES:
            raise TypeError(f"Triton mHC requires {name} to use float16, bfloat16, or float32, got {tensor.dtype}")
