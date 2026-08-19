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

"""Shared normalization semantics for gated delta-rule producers."""

from __future__ import annotations

import torch


def producer_dtype_l2norm(x: torch.Tensor, *, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Normalize in the producer's active arithmetic context and storage dtype.

    This is the historical NPU GDR expression.  In particular, it must not be
    replaced by an unconditional fp32 reduction followed by a cast: KCP and the
    local GDR core must consume the exact same normalized key tensor.
    """

    original_dtype = x.dtype
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return (x * inv_norm).to(original_dtype)


__all__ = ["producer_dtype_l2norm"]
