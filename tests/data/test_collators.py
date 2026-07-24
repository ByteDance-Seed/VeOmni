import types

import pytest
import torch

from veomni.utils.constants import IGNORE_INDEX
from veomni.utils.device import IS_NPU_AVAILABLE


def _fake_ps(sp_enabled: bool, sp_size: int = 1, sp_rank: int = 0):
    return types.SimpleNamespace(sp_enabled=sp_enabled, sp_size=sp_size, sp_rank=sp_rank)


@pytest.fixture
def features_two_samples():
    # Two samples with different lengths
    f1 = {
        "input_ids": torch.tensor([11, 12, 13], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
        "labels": torch.tensor([2], dtype=torch.long),  # sample-level label
    }
    f2 = {
        "input_ids": torch.tensor([21, 22], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1], dtype=torch.long),
        "labels": torch.tensor([1], dtype=torch.long),
    }
    return [f1, f2]


def test_seqcls_collator_sp_disabled(monkeypatch, features_two_samples):
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=False))

    collator = m.MainCollator(seq_classification=True)
    out = collator(features_two_samples)
    exp_input_ids = torch.tensor([[11, 12, 13, 21, 22]], dtype=torch.long)
    exp_attn = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long)
    exp_pos = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long)
    exp_labels = torch.tensor([[2, 1]], dtype=torch.long)
    exp_cu_seq_lens = torch.tensor([0, 3, 5], dtype=torch.int32)
    exp_max_length = 3

    assert torch.equal(out["input_ids"], exp_input_ids)
    assert torch.equal(out["attention_mask"], exp_attn)
    assert torch.equal(out["position_ids"], exp_pos)
    assert torch.equal(out["labels"], exp_labels)
    assert torch.equal(out["cu_seq_lens_q"], exp_cu_seq_lens)
    assert torch.equal(out["cu_seq_lens_k"], exp_cu_seq_lens)
    assert out["max_length_q"] == exp_max_length
    assert out["max_length_k"] == exp_max_length


def test_seqcls_collator_sp_enabled(monkeypatch, features_two_samples):
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=True))

    collator = m.MainCollator(seq_classification=True)
    out = collator(features_two_samples)
    exp_input_ids = torch.tensor([[11, 12, 13, 21, 22]], dtype=torch.long)
    exp_attn = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long)
    exp_pos = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long)
    exp_labels = torch.tensor([[2, 1]], dtype=torch.long)
    exp_cu_seq_lens = torch.tensor([0, 3, 5], dtype=torch.int32)
    exp_max_length = 3

    assert torch.equal(out["input_ids"], exp_input_ids)
    assert torch.equal(out["attention_mask"], exp_attn)
    assert torch.equal(out["position_ids"], exp_pos)
    assert torch.equal(out["labels"], exp_labels)
    assert torch.equal(out["cu_seq_lens_q"], exp_cu_seq_lens)
    assert torch.equal(out["cu_seq_lens_k"], exp_cu_seq_lens)
    assert out["max_length_q"] == exp_max_length
    assert out["max_length_k"] == exp_max_length


def test_main_collator_emits_packed_source_metadata_in_position_reset_order(monkeypatch):
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=False))
    features = [
        {
            "input_ids": torch.tensor([11, 12, 13]),
            "attention_mask": torch.ones(3, dtype=torch.long),
            "labels": torch.tensor([11, 12, 13]),
            "_veomni_source_metadata": {
                "schema_version": 1,
                "source_id": 7,
                "source_name": "a",
                "row_id": 42,
                "sample_id": "sample-a",
                "part_index": 1,
            },
        },
        {
            "input_ids": torch.tensor([21, 22]),
            "attention_mask": torch.ones(2, dtype=torch.long),
            "labels": torch.tensor([21, 22]),
            "_veomni_source_metadata": {
                "schema_version": 1,
                "source_id": "7",
                "source_name": "b",
            },
        },
    ]

    batch = m.MainCollator()(features)

    assert "_veomni_source_metadata" not in batch
    assert batch["_veomni_packed_source_metadata"] == {
        "schema_version": 1,
        "coordinate_space": "packed_pre_sp",
        "valid_token_count": 5,
        "segments": [
            {
                "source_id": 7,
                "source_name": "a",
                "row_id": 42,
                "sample_id": "sample-a",
                "part_index": 1,
                "segment_index": 0,
                "sample_index": 0,
                "subsegment_index": 0,
                "token_start": 0,
                "token_length": 3,
            },
            {
                "source_id": "7",
                "source_name": "b",
                "segment_index": 1,
                "sample_index": 1,
                "subsegment_index": 0,
                "token_start": 3,
                "token_length": 2,
            },
        ],
    }


def test_packed_source_metadata_excludes_padding_and_is_identical_across_sp_ranks(monkeypatch):
    import veomni.data.data_collator as m

    def make_features():
        return [
            {
                "input_ids": torch.tensor([11, 12, 13]),
                "attention_mask": torch.ones(3, dtype=torch.long),
                "labels": torch.tensor([11, 12, 13]),
                "_veomni_source_metadata": {
                    "schema_version": 1,
                    "source_id": "a",
                    "source_name": "source-a",
                },
            },
            {
                "input_ids": torch.tensor([21, 22]),
                "attention_mask": torch.ones(2, dtype=torch.long),
                "labels": torch.tensor([21, 22]),
                "_veomni_source_metadata": {
                    "schema_version": 1,
                    "source_id": "b",
                    "source_name": "source-b",
                },
            },
        ]

    packed_by_rank = []
    for sp_rank in (0, 1):
        monkeypatch.setattr(
            m,
            "get_parallel_state",
            lambda sp_rank=sp_rank: _fake_ps(sp_enabled=True, sp_size=2, sp_rank=sp_rank),
        )
        batch = m.MainCollator(pad_to_length=8)(make_features())
        packed_by_rank.append(batch["_veomni_packed_source_metadata"])

    assert packed_by_rank[0] == packed_by_rank[1]
    assert packed_by_rank[0]["valid_token_count"] == 5
    assert packed_by_rank[0]["segments"] == [
        {
            "source_id": "a",
            "source_name": "source-a",
            "segment_index": 0,
            "sample_index": 0,
            "subsegment_index": 0,
            "token_start": 0,
            "token_length": 3,
        },
        {
            "source_id": "b",
            "source_name": "source-b",
            "segment_index": 1,
            "sample_index": 1,
            "subsegment_index": 0,
            "token_start": 3,
            "token_length": 2,
        },
    ]


def test_data_collator_pad_to_length_sp_disabled(monkeypatch, features_two_samples):
    if IS_NPU_AVAILABLE:
        pytest.skip("NPU does not support this padding test yet.")
    import veomni.data.data_collator as m

    pad_to_length = 8
    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=False))
    token_labels = [
        {
            **features_two_samples[0],
            "labels": torch.tensor([2, 3, 4], dtype=torch.long),
        },
        {
            **features_two_samples[1],
            "labels": torch.tensor([1, 2], dtype=torch.long),
        },
    ]
    collator = m.MainCollator(pad_to_length=pad_to_length)
    out = collator(token_labels)

    exp_input_ids = torch.tensor([[11, 12, 13, 21, 22, 0, 0, 0]], dtype=torch.long)
    exp_attn = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long)
    exp_pos = torch.tensor([[0, 1, 2, 0, 1, 0, 0, 0]], dtype=torch.long)
    exp_labels = torch.tensor([[2, 3, 4, IGNORE_INDEX, 2, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]], dtype=torch.long)
    # pad_to_length tail is coalesced for both FA and linear-attn cu-seqlens.
    exp_cu_seq_lens = torch.tensor([0, 3, 5, 8], dtype=torch.int32)
    exp_max_length = 3

    assert torch.equal(out["input_ids"], exp_input_ids)
    assert torch.equal(out["attention_mask"], exp_attn)
    assert torch.equal(out["position_ids"], exp_pos)
    assert torch.equal(out["labels"], exp_labels)
    assert torch.equal(out["cu_seq_lens_q"], exp_cu_seq_lens)
    assert torch.equal(out["cu_seq_lens_k"], exp_cu_seq_lens)
    assert torch.equal(out["linear_attn_cu_seq_lens_q"], exp_cu_seq_lens)
    assert int(out["tail_padding_length"]) == 3
    assert out["max_length_q"] == exp_max_length
    assert out["max_length_k"] == exp_max_length


def test_seqcls_collator_pad_to_length_sp_enabled(monkeypatch, features_two_samples):
    import veomni.data.data_collator as m

    pad_to_length = 8
    sp_size = 2
    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=True, sp_size=sp_size, sp_rank=0))
    token_labels = [
        {
            **features_two_samples[0],
            "labels": torch.tensor([2, 3, 4], dtype=torch.long),
        },
        {
            **features_two_samples[1],
            "labels": torch.tensor([1, 2], dtype=torch.long),
        },
    ]
    collator = m.MainCollator(pad_to_length=pad_to_length)
    out = collator(token_labels)
    # SP slicing; lengths stay at pad_to_length // sp_size.
    # attention mask is not sliced
    exp_input_ids = torch.tensor([[11, 12, 13, 21]], dtype=torch.long)
    exp_attn = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long)
    exp_pos = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)
    exp_labels = torch.tensor([[3, 4, IGNORE_INDEX, 2]], dtype=torch.long)
    exp_cu_seq_lens = torch.tensor([0, 3, 5, 8], dtype=torch.int32)
    exp_max_length = 3

    assert torch.equal(out["input_ids"], exp_input_ids)
    assert torch.equal(out["attention_mask"], exp_attn)
    assert torch.equal(out["position_ids"], exp_pos)
    assert torch.equal(out["labels"], exp_labels)
    assert torch.equal(out["cu_seq_lens_q"], exp_cu_seq_lens)
    assert torch.equal(out["cu_seq_lens_k"], exp_cu_seq_lens)
    assert torch.equal(out["linear_attn_cu_seq_lens_q"], exp_cu_seq_lens)
    assert int(out["tail_padding_length"]) == 3
    assert out["max_length_q"] == exp_max_length
    assert out["max_length_k"] == exp_max_length

    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=True, sp_size=sp_size, sp_rank=1))
    collator = m.MainCollator(pad_to_length=pad_to_length)
    out = collator(token_labels)
    # SP slicing; lengths stay at pad_to_length // sp_size.
    # attention mask is not sliced
    exp_input_ids = torch.tensor([[22, 0, 0, 0]], dtype=torch.long)
    exp_attn = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long)
    exp_pos = torch.tensor([[1, 0, 0, 0]], dtype=torch.long)
    exp_labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]], dtype=torch.long)
    exp_cu_seq_lens = torch.tensor([0, 3, 5, 8], dtype=torch.int32)
    exp_max_length = 3

    assert torch.equal(out["input_ids"], exp_input_ids)
    assert torch.equal(out["attention_mask"], exp_attn)
    assert torch.equal(out["position_ids"], exp_pos)
    assert torch.equal(out["labels"], exp_labels)
    assert torch.equal(out["cu_seq_lens_q"], exp_cu_seq_lens)
    assert torch.equal(out["cu_seq_lens_k"], exp_cu_seq_lens)
    assert torch.equal(out["linear_attn_cu_seq_lens_q"], exp_cu_seq_lens)
    assert int(out["tail_padding_length"]) == 3
    assert out["max_length_q"] == exp_max_length
    assert out["max_length_k"] == exp_max_length


def test_packing_collator_clamps_linear_attn_tail_padding_length(monkeypatch, features_two_samples):
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=True))
    monkeypatch.setattr(m.PackingCollator, "pad_batch_to_length", lambda _, batch: batch)

    token_labels = [
        {
            **features_two_samples[0],
            "labels": torch.tensor([2, 3, 4], dtype=torch.long),
        },
        {
            **features_two_samples[1],
            "labels": torch.tensor([1, 2], dtype=torch.long),
        },
    ]
    collator = m.PackingCollator(pad_to_length=4)

    out = collator(token_labels)

    assert m._LINEAR_ATTN_TAIL_PADDING_LENGTH not in out


# TODO: add omni data ci test
