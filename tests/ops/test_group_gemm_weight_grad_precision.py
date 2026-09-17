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

"""What the grouped expert weight-gradient GEMM rounds, and where.

``group_gemm_same_mn`` accumulates in FP32 and casts to ``c.dtype`` when it
stores. It asserts a dtype for ``a`` and ``b`` but not for ``c``, so the output
buffer's dtype is the one thing that decides whether a weight gradient is
rounded: a BF16 ``c`` rounds every rank's *local* partial sum before the
cross-rank gradient reduction, while an FP32 ``c`` keeps them exact.

That distinction is what separates EP=1 from EP=2 in a BF16 run. Both group the
same token contributions into per-expert sums, but they group them differently,
so under a BF16 write-back they disagree by the rounding of the partials rather
than by anything mathematical. FSDP2 provides the cross-rank reduction itself:
the expert module is ``fully_shard``-ed on the ``<para>_fsdp`` submesh
(``veomni/distributed/torch_parallelize.py``) and each rank's local weight
gradient is reduce-scattered from there, so what the kernel stores is what gets
combined.

Measured on L20/SM89 with toy DeepSeek-V4 expert shapes (measurements, not the
thresholds this file enforces): a BF16 write-back is ~1.7e-3 relative L2 against
an FP64 reference of the same BF16 operands, an FP32 write-back is ~6e-8, and a
two-partial BF16 combination is ~2.4e-3 where the FP32 one is ~1e-7.
"""

import pytest
import torch

from veomni.ops.kernels.moe._kernels.kernel.group_gemm import group_gemm_same_mn
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type
from veomni.utils.import_utils import is_fused_moe_available


def _skip_if_unsupported():
    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA is required for the Triton grouped GEMM.")
    if not is_fused_moe_available():
        pytest.skip("Triton fused MoE is not available in this environment.")


def _relative_l2(actual, reference):
    reference = reference.double()
    delta = actual.double() - reference
    return (delta.norm() / reference.norm().clamp_min(1e-300)).item()


def _per_expert_fp64(a, b, cumsum_K):
    """``c[g] = a[rows_g].T @ b[rows_g]`` in FP64 over the same BF16 operands."""
    out = None
    start = 0
    for expert in range(cumsum_K.numel()):
        end = int(cumsum_K[expert].item())
        if end > start:
            piece = a[start:end].double().t() @ b[start:end].double()
        else:
            piece = torch.zeros(a.shape[1], b.shape[1], dtype=torch.float64, device=a.device)
        if out is None:
            out = torch.empty((cumsum_K.numel(),) + piece.shape, dtype=torch.float64, device=a.device)
        out[expert] = piece
        start = end
    return out


def _grouped_wgrad(a, b, counts, c_dtype):
    cumsum = torch.cumsum(counts, dim=0)
    c = torch.empty(counts.numel(), a.shape[1], b.shape[1], device=a.device, dtype=c_dtype)
    group_gemm_same_mn(a=a, b=b, c=c, cumsum_K=cumsum, max_K=a.shape[0], transpose_a=True, transpose_b=False)
    return c


def _inputs(num_tokens=256, hidden_dim=64, ffn_dim=96, counts=(0, 41, 41, 174), seed=0):
    device = torch.device(get_device_type())
    torch.manual_seed(seed)
    a = 0.1 * torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)
    b = 0.1 * torch.randn(num_tokens, ffn_dim, device=device, dtype=torch.bfloat16)
    # Skewed, non-uniform payloads that include an empty expert and token counts
    # crossing the kernel's block boundaries.
    return a, b, torch.tensor(counts, device=device, dtype=torch.int32)


def test_grouped_wgrad_writeback_dtype_controls_partial_sum_rounding():
    """Only the output buffer's dtype decides whether a local partial sum rounds."""
    _skip_if_unsupported()
    a, b, counts = _inputs()
    reference = _per_expert_fp64(a, b, torch.cumsum(counts, dim=0))

    c_bf16 = _grouped_wgrad(a, b, counts, torch.bfloat16)
    c_fp32 = _grouped_wgrad(a, b, counts, torch.float32)

    bf16_error = _relative_l2(c_bf16, reference)
    fp32_error = _relative_l2(c_fp32, reference)
    # FP32 accumulation over BF16 operands is exact to a few FP32 epsilons; the
    # BF16 store must land inside the one-BF16-epsilon band the store itself
    # defines. Both are ranges, so a better accumulation order or a retuned
    # BLOCK_K does not move the result out of them.
    eps = torch.finfo(torch.bfloat16).eps
    assert fp32_error < 0.02 * eps, fp32_error
    assert 0.05 * eps < bf16_error < 2 * eps, bf16_error

    # The BF16 result is the FP32 result rounded once at the epilogue: the
    # accumulation is dtype-independent, only the store cast differs. Bitwise
    # equality is the invariant this file exists to pin; if a kernel change ever
    # makes the two stores round differently, that is the signal.
    assert torch.equal(c_bf16.float(), c_fp32.to(torch.bfloat16).float())

    # An expert with no tokens still has its region zeroed.
    assert torch.count_nonzero(c_fp32[0]) == 0
    assert torch.count_nonzero(c_bf16[0]) == 0


def test_grouped_wgrad_bf16_partial_sums_reproduce_the_ep_gap():
    """A BF16 write-back of two EP-style partials costs what the EP gap measures.

    Emulates two ranks that each accumulate a disjoint half of every expert's
    token rows, using the real kernel for each rank's partial. Combining the two
    partials with an FP32 buffer lands on the FP64 reference; combining them with
    a BF16 buffer is ~2^-9-scale away. That rounding of the partials, not a
    different sum, is what separates EP=1 from EP=2 in a BF16 run.
    """
    _skip_if_unsupported()
    a, b, counts = _inputs()
    reference = _per_expert_fp64(a, b, torch.cumsum(counts, dim=0))

    # Each expert's rows are split in half between the two emulated ranks; the
    # per-rank inputs stay grouped by expert so the kernel's cumsum_K applies.
    rows_a = [[], []]
    rows_b = [[], []]
    counts_rank = [[], []]
    start = 0
    for count in counts.tolist():
        end = start + count
        half = (count + 1) // 2
        for rank, (lo, hi) in enumerate(((start, start + half), (start + half, end))):
            rows_a[rank].append(a[lo:hi])
            rows_b[rank].append(b[lo:hi])
            counts_rank[rank].append(hi - lo)
        start = end

    # The split must cover every expert block exactly once, in order.
    for expert_index, count in enumerate(counts.tolist()):
        assert counts_rank[0][expert_index] + counts_rank[1][expert_index] == count
    rebuilt = torch.cat(
        [
            piece
            for expert_index in range(len(counts_rank[0]))
            for piece in (rows_a[0][expert_index], rows_a[1][expert_index])
        ]
    )
    assert torch.equal(rebuilt, a)
    assert torch.equal(
        torch.cat([piece for index in range(len(counts_rank[0])) for piece in (rows_b[0][index], rows_b[1][index])]), b
    )

    partials = []
    for rank in (0, 1):
        rank_counts = torch.tensor(counts_rank[rank], device=a.device, dtype=torch.int32)
        a_rank = torch.cat(rows_a[rank]).contiguous()
        b_rank = torch.cat(rows_b[rank]).contiguous()
        partials.append(_grouped_wgrad(a_rank, b_rank, rank_counts, torch.float32))

    exact = partials[0] + partials[1]
    rounded = partials[0].to(torch.bfloat16) + partials[1].to(torch.bfloat16)

    exact_error = _relative_l2(exact, reference)
    rounded_error = _relative_l2(rounded, reference)
    eps = torch.finfo(torch.bfloat16).eps
    assert exact_error < 0.02 * eps, exact_error
    assert 0.05 * eps < rounded_error < 2 * eps, rounded_error

    # The rounding of the partials, not a different sum, is the whole difference.
    assert rounded_error > 10 * exact_error
