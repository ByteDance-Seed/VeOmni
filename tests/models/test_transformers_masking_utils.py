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

from types import SimpleNamespace

import pytest
import torch
from torch.nn.attention.flex_attention import BlockMask
from transformers.configuration_utils import PreTrainedConfig

from veomni.models.transformers import masking_utils
from veomni.ops.kernels.attention import flex as flex_attention


_FLEX_IMPLEMENTATION = "veomni_flex_attention_with_sp"


def test_packed_sequence_ids_from_cu_seq_lens():
    sequence_ids = masking_utils._packed_sequence_ids_from_cu_seq_lens(
        torch.tensor([0, 3, 5, 8], dtype=torch.int32),
        batch_size=1,
        sequence_length=8,
        device=torch.device("cpu"),
    )

    assert torch.equal(sequence_ids, torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2]]))


@pytest.mark.parametrize(
    ("cu_seq_lens_q", "error_type", "match"),
    [
        (torch.tensor([[0, 5]], dtype=torch.int32), ValueError, "1D tensor"),
        (torch.tensor([0.0, 5.0]), TypeError, "integer dtype"),
        (torch.tensor([0], dtype=torch.int32), ValueError, "initial and final offsets"),
    ],
)
def test_packed_sequence_ids_reject_invalid_structure(cu_seq_lens_q, error_type, match):
    with pytest.raises(error_type, match=match):
        masking_utils._packed_sequence_ids_from_cu_seq_lens(
            cu_seq_lens_q,
            batch_size=1,
            sequence_length=5,
            device=torch.device("cpu"),
        )


def test_packed_sequence_ids_allow_empty_segments():
    sequence_ids = masking_utils._packed_sequence_ids_from_cu_seq_lens(
        torch.tensor([0, 3, 3, 5], dtype=torch.int32),
        batch_size=1,
        sequence_length=5,
        device=torch.device("cpu"),
    )

    assert torch.equal(sequence_ids, torch.tensor([[0, 0, 0, 2, 2]]))


@pytest.mark.parametrize(
    ("wrapper_name", "delegate_name"),
    [
        ("create_causal_mask", "_hf_create_causal_mask"),
        ("create_sliding_window_causal_mask", "_hf_create_sliding_window_causal_mask"),
    ],
)
def test_flex_mask_wrapper_adds_packed_sequence_predicate(monkeypatch, wrapper_name, delegate_name):
    captured = {}
    sentinel = object()

    def fake_delegate(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(masking_utils, delegate_name, fake_delegate)
    inputs_embeds = torch.zeros(1, 5, 4)
    attention_mask = torch.ones(1, 5)

    result = getattr(masking_utils, wrapper_name)(
        config=SimpleNamespace(_attn_implementation=_FLEX_IMPLEMENTATION),
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        and_mask_function=lambda _batch_idx, _head_idx, _q_idx, kv_idx: kv_idx != 1,
        cu_seq_lens_q=torch.tensor([0, 3, 5], dtype=torch.int32),
    )

    assert result is sentinel
    mask_function = captured["and_mask_function"]
    index = torch.tensor
    assert mask_function(index(0), index(0), index(2), index(0))
    assert not mask_function(index(0), index(0), index(2), index(1))  # Existing model-specific predicate is preserved.
    assert not mask_function(
        index(0), index(0), index(3), index(2)
    )  # Packed samples cannot attend across their boundary.
    assert mask_function(index(0), index(0), index(4), index(3))


@pytest.mark.parametrize("cu_seq_lens_q", [None, torch.tensor([0, 5], dtype=torch.int32)])
def test_flex_mask_wrapper_without_packing_preserves_original_predicate(monkeypatch, cu_seq_lens_q):
    captured = {}

    def original_mask_function(_batch_idx, _head_idx, _q_idx, _kv_idx):
        return True

    def fake_delegate(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(masking_utils, "_hf_create_causal_mask", fake_delegate)
    masking_utils.create_causal_mask(
        config=SimpleNamespace(_attn_implementation=_FLEX_IMPLEMENTATION),
        inputs_embeds=torch.zeros(1, 5, 4),
        attention_mask=torch.ones(1, 5),
        past_key_values=None,
        and_mask_function=original_mask_function,
        cu_seq_lens_q=cu_seq_lens_q,
    )

    assert captured["and_mask_function"] is original_mask_function


def test_non_flex_mask_wrapper_does_not_consume_packing_metadata(monkeypatch):
    captured = {}

    def original_mask_function(_batch_idx, _head_idx, _q_idx, _kv_idx):
        return True

    def fake_delegate(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(masking_utils, "_hf_create_causal_mask", fake_delegate)
    masking_utils.create_causal_mask(
        config=SimpleNamespace(_attn_implementation="flash_attention_2"),
        inputs_embeds=torch.zeros(1, 5, 4),
        attention_mask=torch.ones(1, 5),
        past_key_values=None,
        and_mask_function=original_mask_function,
        cu_seq_lens_q=torch.tensor([0, 3, 5], dtype=torch.int32),
    )

    assert captured["and_mask_function"] is original_mask_function


def test_packed_flex_mask_rejects_cache():
    with pytest.raises(ValueError, match="past_key_values"):
        masking_utils.create_causal_mask(
            config=SimpleNamespace(_attn_implementation=_FLEX_IMPLEMENTATION),
            inputs_embeds=torch.zeros(1, 5, 4),
            attention_mask=torch.ones(1, 5),
            past_key_values=object(),
            cu_seq_lens_q=torch.tensor([0, 3, 5], dtype=torch.int32),
        )


def test_packed_flex_mask_rejects_prepared_attention_mask():
    with pytest.raises(ValueError, match="2D attention mask"):
        masking_utils.create_causal_mask(
            config=SimpleNamespace(_attn_implementation=_FLEX_IMPLEMENTATION),
            inputs_embeds=torch.zeros(1, 5, 4),
            attention_mask=torch.ones(1, 1, 5, 5),
            past_key_values=None,
            cu_seq_lens_q=torch.tensor([0, 3, 5], dtype=torch.int32),
        )


def test_causal_mask_builds_pack_aware_block_mask(monkeypatch):
    monkeypatch.setattr(
        flex_attention,
        "get_parallel_state",
        lambda: SimpleNamespace(ulysses_enabled=False),
    )
    config = PreTrainedConfig()
    config._attn_implementation = _FLEX_IMPLEMENTATION

    block_mask = masking_utils.create_causal_mask(
        config=config,
        inputs_embeds=torch.zeros(1, 5, 4),
        attention_mask=torch.ones(1, 5),
        past_key_values=None,
        cu_seq_lens_q=torch.tensor([0, 3, 5], dtype=torch.int32),
    )

    assert isinstance(block_mask, BlockMask)
    index = torch.tensor
    assert block_mask.mask_mod(index(0), index(0), index(2), index(1))
    assert not block_mask.mask_mod(index(0), index(0), index(3), index(2))
    assert not block_mask.mask_mod(index(0), index(0), index(0), index(1))


def test_causal_mask_uses_global_packed_boundaries_with_ulysses(monkeypatch):
    monkeypatch.setattr(
        flex_attention,
        "get_parallel_state",
        lambda: SimpleNamespace(ulysses_enabled=True, ulysses_size=2),
    )
    config = PreTrainedConfig()
    config._attn_implementation = _FLEX_IMPLEMENTATION

    block_mask = masking_utils.create_causal_mask(
        config=config,
        inputs_embeds=torch.zeros(1, 4, 4),
        attention_mask=torch.ones(1, 8),
        past_key_values=None,
        cu_seq_lens_q=torch.tensor([0, 3, 5, 8], dtype=torch.int32),
    )

    assert isinstance(block_mask, BlockMask)
    assert block_mask.shape == (1, 1, 8, 8)
    index = torch.tensor
    assert block_mask.mask_mod(index(0), index(0), index(7), index(5))
    assert not block_mask.mask_mod(index(0), index(0), index(5), index(4))
