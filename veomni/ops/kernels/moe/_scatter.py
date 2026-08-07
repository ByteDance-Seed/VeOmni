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

"""MoE dispatch bookkeeping helpers shared by the Triton / Quack backends.

The scatter-index maps each ``(token, top-k slot)`` pair to its position in
the expert-sorted flattened buffer. The reference implementation in prior
versions of these kernels was:

    perm = flat.argsort(stable=True)   # 1
    scatter_index = perm.argsort()     # 2

That's two O(N log N) sorts back-to-back, and step 2 is inverting a
permutation of ``[0..N)`` — an inherently O(N) operation. The helper below
inlines that observation and materializes the inverse permutation with a
single ``arange`` + scatter, so the total cost drops to one sort + one
linear-time index write.

Kept as a plain-torch helper on purpose: it needs to work on GPU, NPU, and
CPU (the last is needed for lightweight unit tests). The Triton path in
``group_gemm.py`` and the Quack path in ``quack_gemm.py`` both consume this.
"""

from __future__ import annotations

import torch


def compute_expert_scatter_index(
    expert_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(sorted_order, scatter_index)`` for MoE dispatch.

    Args:
        expert_index: ``[T, topk]`` (or any 2-D shape) of expert assignments
            per token / top-k slot. Dtype is treated as an integer key; the
            helper is agnostic to the actual integer width.

    Returns:
        sorted_order: 1-D int64 tensor of length ``T * topk``. Element ``i``
            holds the flat index into ``expert_index.flatten()`` whose expert
            assignment sorts to position ``i``. ``stable=True`` so ties keep
            the natural (token, top-k slot) order — this stability is
            required by the downstream group-GEMM which assumes tokens for
            the same expert are contiguous in original order.
        scatter_index: int32 tensor with the same shape as ``expert_index``.
            ``scatter_index[t, k]`` is the row in the expert-sorted buffer
            that ``(t, k)`` maps to. The int32 dtype matches the prior
            behavior (the Triton MoE kernels take int32 indices).

    Design note (why not ``argsort().argsort()``):
        ``sorted_order.argsort()`` inverts a permutation and is unnecessarily
        O(N log N). We instead materialize the inverse via
        ``inv[sorted_order] = arange(N)``, which is O(N) and single-launch.
    """
    flat = expert_index.flatten()
    sorted_order = flat.argsort(stable=True)

    inv = torch.empty_like(sorted_order)
    # ``sorted_order`` is a permutation of [0..N); assign each of its
    # positions its rank, giving the inverse permutation in one pass.
    inv[sorted_order] = torch.arange(
        sorted_order.numel(),
        dtype=sorted_order.dtype,
        device=sorted_order.device,
    )
    scatter_index = inv.to(torch.int32).view(expert_index.shape)
    return sorted_order, scatter_index
