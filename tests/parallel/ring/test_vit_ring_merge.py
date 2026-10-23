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

"""CPU tests for the Qwen3-VL USP (cp>1) visual-token merge helpers.

Each rank is simulated in-process: the ViT merger output of every rank is
modelled as the token ids of its local patch rows (``vit_usp_slice`` layout),
and the ulysses-then-cp all-gather is replaced by concatenating those per-rank
outputs. The test checks that every local placeholder receives exactly its own
vit token (and deepstack token) in local order.
"""

from dataclasses import dataclass

import pytest
import torch

from veomni.distributed.sequence_parallel.ring_attention.layout import usp_zigzag_slice, vit_usp_slice
from veomni.models.transformers.qwen3_vl.generated import patched_modeling_qwen3_vl_gpu as qwen3_vl


@dataclass
class _FakeState:
    cp_size: int
    cp_rank: int
    ulysses_size: int
    ulysses_rank: int

    @property
    def ulysses_enabled(self):
        return self.ulysses_size > 1


def _vit_tokens_by_rank(vit_seqlens, cp_size, ulysses_size):
    """Token id of every merger output row on each rank (pad tokens are -1), in gather order."""
    rows = torch.arange(sum(vit_seqlens))
    outputs = []
    for cp_rank in range(cp_size):
        for ulysses_rank in range(ulysses_size):
            local = vit_usp_slice(rows, vit_seqlens, 0, cp_size, cp_rank, ulysses_size, ulysses_rank, pad_value=-1)
            first = local[::4]
            outputs.append(torch.where(first >= 0, first // 4, torch.full_like(first, -1)))
    return outputs


@pytest.mark.parametrize(
    ("vit_seqlens", "cp_size", "ulysses_size"),
    [([1380, 112], 2, 2), ([12, 8], 2, 2), ([20, 4, 28, 8], 4, 1), ([880] * 4, 4, 2), ([16], 2, 1)],
)
def test_vit_gathered_token_index_is_a_bijection(vit_seqlens, cp_size, ulysses_size):
    index = qwen3_vl._vit_gathered_token_index(vit_seqlens, cp_size, ulysses_size)
    gathered = torch.cat(_vit_tokens_by_rank(vit_seqlens, cp_size, ulysses_size))
    num_tokens = sum(vit_seqlens) // 4
    assert index.numel() == num_tokens
    assert torch.equal(gathered[index], torch.arange(num_tokens))


@pytest.mark.parametrize(("cp_size", "ulysses_size"), [(2, 1), (2, 2), (4, 1)])
def test_usp_merge_visual_embeds_fills_local_placeholders(monkeypatch, cp_size, ulysses_size):
    # doc 1: 16 tokens, image A (grid [1,4,6] -> 6 vit tokens) at positions 2..7;
    # doc 2: 8 tokens, image B (grid [1,2,2] -> 1 vit token) at position 3.
    grid_thw_list = [[1, 4, 6], [1, 2, 2]]
    vit_seqlens = [24, 4]
    doc_cu_seqlens = [0, 16, 24]
    full_mask = torch.zeros(1, 24, dtype=torch.bool)
    full_mask[0, 2:8] = True
    full_mask[0, 19] = True
    hidden = 3

    # Global expectation: the k-th placeholder holds vit token k.
    expected_full = torch.zeros(1, 24, hidden)
    expected_full[full_mask] = torch.arange(7, dtype=torch.float32).unsqueeze(-1).expand(7, hidden)
    gathered_tokens = torch.cat(_vit_tokens_by_rank(vit_seqlens, cp_size, ulysses_size)).float()
    gathered = gathered_tokens.unsqueeze(-1).expand(-1, hidden)
    # The fake gather returns the gathered vit tokens shifted by the (constant) local input value,
    # so the merger output (0) and each deepstack layer (k) can be told apart.
    monkeypatch.setattr(qwen3_vl, "_gather_vit_tokens", lambda local: gathered + local.flatten()[0])

    for cp_rank in range(cp_size):
        for ulysses_rank in range(ulysses_size):
            state = _FakeState(cp_size, cp_rank, ulysses_size, ulysses_rank)
            monkeypatch.setattr(qwen3_vl, "get_parallel_state", lambda state=state: state)
            slice_args = (doc_cu_seqlens, -2, cp_size, cp_rank, ulysses_size, ulysses_rank)
            local_len = 24 // (cp_size * ulysses_size)

            inputs_embeds = torch.zeros(1, local_len, hidden)
            vit_local = torch.zeros(5, hidden)  # local merger output; content comes from the fake gather
            deepstack_local = [torch.full((5, hidden), float(k)) for k in (1, 2, 3)]
            merged, deepstack, local_mask = qwen3_vl._usp_merge_visual_embeds(
                inputs_embeds, vit_local, deepstack_local, full_mask, grid_thw_list, doc_cu_seqlens
            )

            expected_local = usp_zigzag_slice(expected_full, *slice_args)
            assert torch.equal(local_mask, usp_zigzag_slice(full_mask, doc_cu_seqlens, -1, *slice_args[2:]))
            torch.testing.assert_close(merged, expected_local)
            local_tokens = expected_local[local_mask]
            for k, embed in zip((1, 2, 3), deepstack):
                torch.testing.assert_close(embed, local_tokens + k)
