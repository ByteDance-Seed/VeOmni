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

"""Expert scatter-index parity, inverse-permutation, and stability tests."""

import pytest
import torch

from veomni.ops.kernels.moe_experts.shared.scatter import compute_expert_scatter_index, compute_max_expert_tokens


def _reference_scatter_index(expert_index: torch.Tensor) -> torch.Tensor:
    """Build an independent inverse-permutation oracle with stable sorts."""
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

    sorted_order, scatter_index = compute_expert_scatter_index(expert_index)
    reference = _reference_scatter_index(expert_index)

    assert sorted_order.shape == (num_tokens * topk,)
    assert sorted_order.dtype == torch.int64
    assert sorted_order.device == expert_index.device
    assert scatter_index.shape == expert_index.shape
    assert scatter_index.dtype == torch.int32
    assert scatter_index.device == expert_index.device
    assert torch.equal(scatter_index, reference), (
        f"scatter_index mismatch for shape ({num_tokens}, {topk}), "
        f"num_experts={num_experts}. got={scatter_index}, ref={reference}"
    )
    flat_scatter = scatter_index.flatten().to(torch.int64)
    expected_order = torch.arange(sorted_order.numel(), device=expert_index.device)
    assert torch.equal(sorted_order[flat_scatter], expected_order)


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


def _max_expert_count(expert_index: torch.Tensor, num_experts: int) -> int:
    """Real ``max_e counts[e]`` after scatter — the value ``max_M`` must cover."""
    counts = torch.bincount(expert_index.flatten(), minlength=num_experts)
    return int(counts.max().item())


@pytest.mark.parametrize(
    "num_tokens,num_experts,topk",
    [
        (1, 4, 1),
        (16, 8, 2),
        (32, 4, 2),
        (128, 16, 4),
        (7, 3, 3),
    ],
)
def test_max_expert_tokens_conservative_is_scatter_row_count(num_tokens, num_experts, topk):
    expert_index = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int64)
    assert compute_max_expert_tokens(expert_index, topk) == num_tokens * topk
    assert compute_max_expert_tokens(expert_index, topk, assume_distinct_experts=False) == num_tokens * topk


@pytest.mark.parametrize(
    "num_tokens,num_experts,topk",
    [
        (1, 4, 1),
        (16, 8, 2),
        (128, 16, 4),
        (7, 3, 3),
    ],
)
def test_max_expert_tokens_distinct_is_token_count(num_tokens, num_experts, topk):
    expert_index = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int64)
    assert compute_max_expert_tokens(expert_index, topk, assume_distinct_experts=True) == num_tokens


def test_max_expert_tokens_tight_bound_covers_distinct_topk_routing():
    torch.manual_seed(0xBEEF)
    num_tokens, num_experts, topk = 96, 8, 4
    logits = torch.randn(num_tokens, num_experts)
    expert_index = torch.topk(logits, topk, dim=-1).indices

    tight = compute_max_expert_tokens(expert_index, topk, assume_distinct_experts=True)
    assert tight == num_tokens
    assert tight >= _max_expert_count(expert_index, num_experts)


def test_max_expert_tokens_conservative_bound_covers_non_distinct_routing():
    topk = 3
    expert_index = torch.zeros((4, topk), dtype=torch.int64)
    num_experts = 4
    real_max = _max_expert_count(expert_index, num_experts)
    assert real_max == 4 * topk

    tight = compute_max_expert_tokens(expert_index, topk, assume_distinct_experts=True)
    conservative = compute_max_expert_tokens(expert_index, topk, assume_distinct_experts=False)
    assert tight < real_max
    assert conservative >= real_max


@pytest.mark.parametrize("bad_shape", [(16,), (2, 3, 4), ()])
def test_max_expert_tokens_rejects_non_2d_index(bad_shape):
    expert_index = torch.zeros(bad_shape, dtype=torch.int64)
    with pytest.raises(ValueError, match="2-D"):
        compute_max_expert_tokens(expert_index, top_k=2)
