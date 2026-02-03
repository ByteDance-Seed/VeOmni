import types

import pytest
import torch

from veomni.data.constants import IGNORE_INDEX
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
    exp_cu_seq_lens = torch.tensor([0, 3, 5, 6, 7, 8], dtype=torch.int32)
    exp_max_length = 3

    assert torch.equal(out["input_ids"], exp_input_ids)
    assert torch.equal(out["attention_mask"], exp_attn)
    assert torch.equal(out["position_ids"], exp_pos)
    assert torch.equal(out["labels"], exp_labels)
    assert torch.equal(out["cu_seq_lens_q"], exp_cu_seq_lens)
    assert torch.equal(out["cu_seq_lens_k"], exp_cu_seq_lens)
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
    exp_cu_seq_lens = torch.tensor([0, 3, 5, 6, 7, 8], dtype=torch.int32)
    exp_max_length = 3

    assert torch.equal(out["input_ids"], exp_input_ids)
    assert torch.equal(out["attention_mask"], exp_attn)
    assert torch.equal(out["position_ids"], exp_pos)
    assert torch.equal(out["labels"], exp_labels)
    assert torch.equal(out["cu_seq_lens_q"], exp_cu_seq_lens)
    assert torch.equal(out["cu_seq_lens_k"], exp_cu_seq_lens)
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
    exp_cu_seq_lens = torch.tensor([0, 3, 5, 6, 7, 8], dtype=torch.int32)
    exp_max_length = 3

    assert torch.equal(out["input_ids"], exp_input_ids)
    assert torch.equal(out["attention_mask"], exp_attn)
    assert torch.equal(out["position_ids"], exp_pos)
    assert torch.equal(out["labels"], exp_labels)
    assert torch.equal(out["cu_seq_lens_q"], exp_cu_seq_lens)
    assert torch.equal(out["cu_seq_lens_k"], exp_cu_seq_lens)
    assert out["max_length_q"] == exp_max_length
    assert out["max_length_k"] == exp_max_length


# TODO: add omni data ci test
