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


def compute_max_expert_tokens(
    expert_index: torch.Tensor,
    top_k: int,
    assume_distinct_experts: bool = False,
) -> int:
    """Safe, tight per-expert row bound (``max_M``) for the grouped GEMM grid.

    The grouped-GEMM kernels size their launch grid as
    ``cdiv(max_M, BLOCK_M)`` row-tiles *per expert*; every tile whose row range
    lies past the expert's real token count early-exits. Correctness only needs

        max_M >= max_e counts[e]

    where ``counts[e]`` is the number of scattered rows routed to expert ``e``.
    Any value above that just launches extra tiles that immediately return.

    Applicability range (why ``T`` works and is tighter than ``T * top_k``):
        With standard top-k gating the router does ``torch.topk(logits, top_k)``,
        which returns ``top_k`` *distinct* experts per token. A distinct-expert
        constraint means each of the ``T = expert_index.shape[0]`` tokens
        contributes at most one row to any single expert, hence
        ``max_e counts[e] <= T``. So ``T`` is a valid bound and is ``top_k``x
        smaller than the full scattered row count ``T * top_k``
        (``scatter_output.shape[0]``), shrinking the launched grid accordingly.
        A caller that has verified its router is distinct opts in with
        ``assume_distinct_experts=True`` (e.g. every ``torch.topk``-gated non-EP
        fused-MoE path in VeOmni: Qwen3-MoE, Qwen3.5-MoE, gpt-oss, DeepSeek-V4
        ``DeepseekV4TopKRouter``).

    Why the default is conservative (``assume_distinct_experts=False``):
        Some routers select experts *without* a distinct guarantee — e.g. the
        DeepSeek-V4 hash router looks up a frozen ``tid2eid`` table whose per-token
        rows are not verified to be distinct. If one token maps to the same expert
        more than once, that expert can receive up to ``T * top_k`` rows; with the
        tight ``T`` bound the overflow tiles are never launched, so those output
        rows keep the **uninitialized memory** of ``torch.empty`` (not zeros, not a
        detectable dropped-row pattern) — silent, unreproducible garbage. Failing
        toward this bound would be unsafe, so ``T`` must be an explicit opt-in and
        the default returns the conservative ``T * top_k`` bound
        (== ``scatter_output.shape[0]``), which reproduces the pre-change behaviour
        and is safe for arbitrary routing.

    Args:
        expert_index: ``[T, top_k]`` integer expert assignments per token. Must
            be 2-D: ``T = expert_index.shape[0]`` is the token count and both
            return values are derived from it, so a flattened ``[T * top_k]``
            index would make ``T`` wrong (``T * top_k``) and silently corrupt
            both bounds. The 2-D contract is enforced with a ``ValueError``.
        top_k: number of expert slots per token (``expert_index.shape[1]``),
            taken explicitly so callers that already hold it avoid a redundant
            ``.shape`` read; it is only used for the conservative branch.
        assume_distinct_experts: when ``True`` return the tight ``T`` bound,
            valid only under distinct top-k routing; when ``False`` (default)
            return the conservative ``T * top_k`` bound safe for arbitrary
            routing. The default is conservative on purpose: under-bounding
            ``max_M`` leaves uninitialized rows in the grouped-GEMM output
            (see ``group_gemm_same_nk``), so the tight bound must be opted into
            explicitly by a caller that has verified its router is distinct.

    Returns:
        A Python ``int`` usable directly as ``max_M`` (no host/device sync,
        unlike ``int(counts.max())``).
    """
    if expert_index.ndim != 2:
        raise ValueError(
            "compute_max_expert_tokens expects a 2-D [T, top_k] expert_index; "
            f"got shape {tuple(expert_index.shape)}. A flattened index would make "
            "the token count T ambiguous and corrupt the max_M bound."
        )
    T = expert_index.shape[0]
    if assume_distinct_experts:
        return T
    return T * top_k
