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

"""Numerical parity tests for ``compute_expert_scatter_index``.

The helper replaces the ``argsort(stable=True).argsort()`` pair that the
Triton and Quack MoE kernels used to build the scatter-index tensor. The
second ``argsort`` was inverting a permutation of ``[0..N)`` -- an O(N)
operation that used to be implemented as an O(N log N) sort. These tests
verify bit-exact parity with the old expression and the inverse-permutation
invariant.

All checks run on CPU: the helper is device-agnostic (composes ``argsort`` +
``arange`` + scatter) and the semantic parity is what matters downstream.
"""

import pytest
import torch

from veomni.ops.kernels.moe_experts.shared.scatter import compute_expert_scatter_index


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
    """Equal-expert entries must retain their original token/top-k order."""
    expert_index = torch.tensor(
        [[0, 1], [1, 0], [0, 2], [2, 1]],
        dtype=torch.int64,
    )
    sorted_order, _ = compute_expert_scatter_index(expert_index)

    flat = expert_index.flatten()
    experts_in_sorted_order = flat[sorted_order]
    assert torch.all(experts_in_sorted_order[1:] >= experts_in_sorted_order[:-1])

    for expert in torch.unique(flat):
        positions = sorted_order[experts_in_sorted_order == expert]
        assert torch.all(positions[1:] > positions[:-1]), (
            f"stability violated for expert {expert.item()}: {positions.tolist()}"
        )


def test_scatter_index_dtype_and_device_preserved():
    expert_index = torch.tensor([[0, 3, 2], [1, 0, 2]], dtype=torch.int64)
    sorted_order, scatter_index = compute_expert_scatter_index(expert_index)

    assert sorted_order.dtype == torch.int64
    assert scatter_index.dtype == torch.int32
    assert scatter_index.device == expert_index.device


def test_scatter_index_inverts_sorted_order():
    torch.manual_seed(42)
    expert_index = torch.randint(0, 8, (33, 3), dtype=torch.int64)

    sorted_order, scatter_index = compute_expert_scatter_index(expert_index)
    count = sorted_order.numel()
    flat_scatter = scatter_index.flatten().to(torch.int64)

    assert torch.equal(sorted_order[flat_scatter], torch.arange(count, dtype=torch.int64))
