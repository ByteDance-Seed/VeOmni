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

"""``InferenceMixin._sample_token`` returns a token id, on both branches.

Every SeedOmni family's ``generate()`` appends the result straight into the
text token cache, and the cache is handed to ``tokenizer.decode`` a whole
decode step later. A tensor that escapes here therefore fails far from the line
that produced it -- as a nested-list TypeError inside the tokenizer.
"""

import pytest
import torch

from veomni.arguments.omni_arguments_types import OmniInferArguments
from veomni.distributed.moe.moe_utils import unpermute
from veomni.models.seed_omni.modules.text_encoder.modeling import InferenceMixin
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


VOCAB = 32


class _Sampler(InferenceMixin):
    """The mixin needs only a projection and a device to sample."""

    def __init__(self):
        self.device = torch.device("cpu")

    def _project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.zeros(hidden_states.shape[0], VOCAB)


def test_greedy_returns_an_id():
    token = _Sampler()._sample_token(torch.zeros(1, 4, 8), do_sample=False)
    assert isinstance(token, int)
    assert 0 <= token < VOCAB


def test_sampling_returns_an_id():
    """The default branch: ``do_sample`` defaults to True."""
    token = _Sampler()._sample_token(torch.zeros(1, 4, 8))
    assert isinstance(token, int)
    assert 0 <= token < VOCAB


def test_sampling_survives_a_tokenizer_round_trip():
    """What the cache holds has to be decodable, which is where this used to fail."""
    ids = [_Sampler()._sample_token(torch.zeros(1, 4, 8)) for _ in range(3)]
    assert all(isinstance(token, int) for token in ids)
    # `decode` rejects a nested list with "'list' object cannot be interpreted
    # as an integer"; indexing a plain list of ids is the same access it makes.
    assert [ids[index] for index in range(len(ids))] == ids


def test_top_p_and_temperature_still_return_an_id():
    token = _Sampler()._sample_token(torch.zeros(1, 4, 8), temperature=0.7, top_p=0.9)
    assert isinstance(token, int)


class TestRanksAgree:
    """The sampled id is the FSM's stop condition, so it has to be one id.

    Each rank runs its own copy of the generation loop while the forward
    feeding it is full of collectives. A rank that samples EOS on its own
    leaves the loop and strands every other rank in the next all-to-all.
    """

    def test_single_process_is_left_alone(self):
        """No process group: nothing to agree with, and no collective to attempt."""
        assert _Sampler()._agree_on_token(7) == 7

    def test_broadcasts_from_rank_zero(self, monkeypatch):
        """With a group, every rank returns rank 0's id rather than its own."""
        import torch.distributed as dist

        from veomni.models.seed_omni.modules.text_encoder import modeling

        def fake_broadcast(tensor, src):
            assert src == 0
            tensor.fill_(11)

        monkeypatch.setattr(modeling.dist, "is_available", lambda: True)
        monkeypatch.setattr(modeling.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(modeling.dist, "broadcast", fake_broadcast)
        del dist

        assert _Sampler()._agree_on_token(4) == 11

    def test_both_branches_agree(self, monkeypatch):
        """Greedy is deterministic per rank but not across them: shards differ."""
        from veomni.models.seed_omni.modules.text_encoder import modeling

        monkeypatch.setattr(modeling.dist, "is_available", lambda: True)
        monkeypatch.setattr(modeling.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(modeling.dist, "broadcast", lambda tensor, src: tensor.fill_(3))

        sampler = _Sampler()
        assert sampler._sample_token(torch.zeros(1, 4, 8), do_sample=False) == 3
        assert sampler._sample_token(torch.zeros(1, 4, 8)) == 3


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


def test_inference_full_determinism_is_opt_in():
    """Serving default matches training / vLLM: bitwise kernels are opt-in."""
    assert OmniInferArguments().enable_full_determinism is False


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="scatter_add_ only races on CUDA")
def test_combine_is_bitwise_reproducible_under_full_determinism():
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        # topk > 1 is what puts repeated indices in the mapping; a hidden size in
        # the thousands is what gives the atomics enough contention to diverge.
        operands = _combine_operands(num_tokens=256, num_experts=8, topk=6, hidden=2048, device=get_device_type())
        first = unpermute(*operands)
        for _ in range(4):
            assert torch.equal(unpermute(*operands), first)
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)
