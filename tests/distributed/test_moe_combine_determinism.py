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
"""The MoE top-k combine has to be reproducible when inference asks for it.

``unpermute`` adds ``topk`` expert outputs back onto one token row with a
``scatter_add_`` over repeated indices. On CUDA that is an atomicAdd, so the
summation order is decided by a thread race and fp32 addition is not
associative: identical inputs give different last bits run to run, and eight
ranks running the same prompt land on eight different sets of logits. That is
enough to flip an argmax at a near-tie, and a generation FSM whose termination
is judged per rank then deadlocks.
"""

import pytest
import torch

from veomni.arguments.omni_arguments_types import OmniInferArguments
from veomni.distributed.moe.moe_utils import unpermute


def _combine_operands(num_tokens: int, num_experts: int, topk: int, hidden: int, device: str):
    """Operands for ``unpermute``, ordered the way ``token_pre_all2all`` leaves them.

    ``unpermute`` recovers each row's weight with
    ``routing_weights.T.masked_select(routing_map)``, so the rows must be
    grouped by expert -- expert-major, exactly as the all-to-all delivers them.
    """
    routing_map = torch.zeros(num_experts, num_tokens, dtype=torch.bool, device=device)
    routing_weights = torch.zeros(num_tokens, num_experts, device=device)
    for token in range(num_tokens):
        for slot in range(topk):
            expert = (token + slot) % num_experts
            routing_map[expert, token] = True
            routing_weights[token, expert] = 1.0 / topk

    mapping = torch.tensor(
        [token for expert in range(num_experts) for token in range(num_tokens) if routing_map[expert, token]],
        dtype=torch.long,
        device=device,
    )
    tokens = torch.randn(mapping.numel(), hidden, dtype=torch.bfloat16, device=device)
    return tokens, routing_weights, torch.Size([num_tokens, hidden]), mapping, routing_map


def test_inference_asks_for_full_determinism_by_default():
    """Unlike training: the same prompt and seed should give the same answer twice."""
    assert OmniInferArguments().enable_full_determinism is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="scatter_add_ only races on CUDA")
def test_combine_is_bitwise_reproducible_under_full_determinism():
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        # topk > 1 is what puts repeated indices in the mapping; a hidden size in
        # the thousands is what gives the atomics enough contention to diverge.
        operands = _combine_operands(num_tokens=256, num_experts=8, topk=6, hidden=2048, device="cuda")
        first = unpermute(*operands)
        for _ in range(4):
            assert torch.equal(unpermute(*operands), first)
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)
