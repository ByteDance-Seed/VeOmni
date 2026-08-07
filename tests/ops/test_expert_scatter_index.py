# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Numerical parity tests for ``compute_expert_scatter_index``.

The helper replaces the ``argsort(stable=True).argsort()`` pair that the
Triton MoE (``group_gemm.py``) and Quack MoE (``quack_gemm.py``) kernels
used to build the scatter-index tensor. The second ``argsort`` was
inverting a permutation of ``[0..N)`` — an O(N) operation that used to be
implemented as an O(N log N) sort. This test verifies bit-exact parity
with the old expression and the inverse-permutation invariant.

All checks run on CPU: the helper is device-agnostic (composes
``argsort`` + ``arange`` + scatter) and the semantic parity is what
matters for downstream kernels.
"""

import pytest
import torch

from veomni.ops.kernels.moe._scatter import compute_expert_scatter_index


def _reference_scatter_index(expert_index: torch.Tensor) -> torch.Tensor:
    """Old expression, kept as the numeric ground truth."""
    return expert_index.flatten().argsort(stable=True).argsort().to(torch.int32).view(expert_index.shape)


@pytest.mark.parametrize(
    "num_tokens,num_experts,topk",
    [
        (1, 4, 1),
        (16, 8, 2),
        (32, 4, 2),
        (128, 16, 4),
        (7, 3, 1),
        (7, 3, 3),
    ],
)
def test_scatter_index_matches_argsort_argsort(num_tokens, num_experts, topk):
    torch.manual_seed(0xC0FFEE)
    expert_index = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int64)

    _, scatter_index = compute_expert_scatter_index(expert_index)
    reference = _reference_scatter_index(expert_index)

    assert scatter_index.shape == expert_index.shape
    assert scatter_index.dtype == torch.int32
    assert torch.equal(scatter_index, reference), (
        f"scatter_index mismatch for shape ({num_tokens}, {topk}), "
        f"num_experts={num_experts}. got={scatter_index}, ref={reference}"
    )


def test_scatter_index_is_a_permutation_of_range():
    torch.manual_seed(1)
    expert_index = torch.randint(0, 16, (64, 4), dtype=torch.int64)

    _, scatter_index = compute_expert_scatter_index(expert_index)
    flat = scatter_index.flatten().to(torch.int64)
    assert torch.equal(flat.sort().values, torch.arange(flat.numel(), dtype=torch.int64))


def test_sorted_order_is_stable_and_experts_are_contiguous():
    """The MoE kernels rely on ``stable=True`` so tokens for the same expert
    stay in original (token, top-k slot) order in the expert-sorted buffer."""
    expert_index = torch.tensor(
        [[0, 1], [1, 0], [0, 2], [2, 1]],
        dtype=torch.int64,
    )
    sorted_order, _ = compute_expert_scatter_index(expert_index)

    flat = expert_index.flatten()
    experts_in_sorted_order = flat[sorted_order]
    # experts_in_sorted_order should be non-decreasing (contiguous per-expert runs)
    assert torch.all(experts_in_sorted_order[1:] >= experts_in_sorted_order[:-1])

    # Within the same expert, original flat positions must be increasing (stability).
    for e in torch.unique(flat):
        positions = sorted_order[experts_in_sorted_order == e]
        assert torch.all(positions[1:] > positions[:-1]), (
            f"stability violated for expert {e.item()}: {positions.tolist()}"
        )


def test_scatter_index_dtype_and_device_preserved():
    expert_index = torch.tensor([[0, 3, 2], [1, 0, 2]], dtype=torch.int64)
    sorted_order, scatter_index = compute_expert_scatter_index(expert_index)

    assert sorted_order.dtype == torch.int64  # argsort returns int64
    assert scatter_index.dtype == torch.int32  # documented int32 for Triton kernels
    assert scatter_index.device == expert_index.device


def test_scatter_index_inverts_sorted_order():
    """scatter_index and sorted_order must be inverse permutations."""
    torch.manual_seed(42)
    expert_index = torch.randint(0, 8, (33, 3), dtype=torch.int64)

    sorted_order, scatter_index = compute_expert_scatter_index(expert_index)
    N = sorted_order.numel()
    flat_scatter = scatter_index.flatten().to(torch.int64)

    # sorted_order[scatter_index[i]] == i  for all i in [0, N).
    inv_check = sorted_order[flat_scatter]
    assert torch.equal(inv_check, torch.arange(N, dtype=torch.int64))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
