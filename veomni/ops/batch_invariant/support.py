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
# See the License for the specific language governing limitations
# under the License.

"""CPU-safe helpers for batch-invariant ATen fallbacks."""

from __future__ import annotations

import torch
from torch import Tensor


def addmm_can_fuse_bias(bias: Tensor | None, n: int, *, beta: object = 1, alpha: object = 1) -> bool:
    """Return whether persistent addmm can encode this bias / scale pair.

    The fused kernel loads a contiguous 1D bias of length ``N`` and hardcodes
    ``alpha=1``, ``beta=1``. Anything else must fall back to ``mm`` plus add.
    """
    if not isinstance(beta, (int, float)) or not isinstance(alpha, (int, float)):
        return False
    if beta != 1 or alpha != 1:
        return False
    if bias is None:
        return True
    return bias.dim() == 1 and bias.numel() == n and bias.stride(0) == 1


def mean_keep_fp32_until_divide(
    input: Tensor,
    dim: tuple[int, ...] | list[int],
    keepdim: bool = False,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Reduce several dims in FP32, then cast. Avoids FP16 overflow on the sum."""
    if len(dim) == 0:
        dim = list(range(input.ndim))
    n_elems = 1
    for axis in dim:
        n_elems *= input.shape[axis]
    reduced = torch.sum(input, dim=dim, keepdim=keepdim, dtype=torch.float32)
    return (reduced / n_elems).to(dtype or input.dtype)
