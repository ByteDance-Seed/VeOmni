import types

import pytest
import torch

from veomni.trainer.text_dpo_trainer import DPOCollator
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


# ===================== DPOCollator tests =====================


def _make_dpo_sample(chosen_ids, rejected_ids):
    """Build a single DPO sample with shape [2, L] (chosen=row0, rejected=row1)."""
    c_len = len(chosen_ids)
    r_len = len(rejected_ids)
    max_len = max(c_len, r_len)

    def _pad(ids, pad_val):
        return ids + [pad_val] * (max_len - len(ids))

    return {
        "input_ids": torch.tensor([_pad(chosen_ids, 0), _pad(rejected_ids, 0)]),
        "attention_mask": torch.tensor(
            [_pad([1] * c_len, 0), _pad([1] * r_len, 0)],
        ),
        "labels": torch.tensor(
            [_pad(chosen_ids, IGNORE_INDEX), _pad(rejected_ids, IGNORE_INDEX)],
        ),
    }


def test_dpo_collator_single_sample():
    """B=1: collator should preserve [2, L] layout with chosen first."""
    sample = _make_dpo_sample([10, 20, 30], [40, 50, 60])
    collator = DPOCollator()
    batch = collator([sample])

    assert batch["input_ids"].shape == (2, 3)
    assert torch.equal(batch["input_ids"][0], torch.tensor([10, 20, 30]))
    assert torch.equal(batch["input_ids"][1], torch.tensor([40, 50, 60]))


def test_dpo_collator_multiple_samples_ordering():
    """B=2: first B rows must be chosen, last B rows must be rejected."""
    s1 = _make_dpo_sample([1, 2, 3], [4, 5, 6])
    s2 = _make_dpo_sample([7, 8, 9], [10, 11, 12])
    collator = DPOCollator()
    batch = collator([s1, s2])

    assert batch["input_ids"].shape == (4, 3)
    assert torch.equal(batch["input_ids"][0], torch.tensor([1, 2, 3]))
    assert torch.equal(batch["input_ids"][1], torch.tensor([7, 8, 9]))
    assert torch.equal(batch["input_ids"][2], torch.tensor([4, 5, 6]))
    assert torch.equal(batch["input_ids"][3], torch.tensor([10, 11, 12]))


def test_dpo_collator_padding_with_different_lengths():
    """Samples with different sequence lengths should be padded to the max."""
    s1 = _make_dpo_sample([1, 2], [3, 4])  # L=2
    s2 = _make_dpo_sample([5, 6, 7, 8], [9, 10, 11, 12])  # L=4
    collator = DPOCollator()
    batch = collator([s1, s2])

    assert batch["input_ids"].shape == (4, 4)
    assert torch.equal(batch["input_ids"][0], torch.tensor([1, 2, 0, 0]))
    assert torch.equal(batch["input_ids"][1], torch.tensor([5, 6, 7, 8]))
    assert torch.equal(batch["input_ids"][2], torch.tensor([3, 4, 0, 0]))
    assert torch.equal(batch["input_ids"][3], torch.tensor([9, 10, 11, 12]))


def test_dpo_collator_pad_values():
    """Verify correct pad values: input_ids=0, attention_mask=0, labels=IGNORE_INDEX."""
    s1 = _make_dpo_sample([1, 2, 3], [4, 5, 6])
    s2 = _make_dpo_sample([7], [8])
    collator = DPOCollator()
    batch = collator([s1, s2])

    # s2 chosen row is at index 1, padded from length 1 to 3
    assert batch["input_ids"][1, 1].item() == 0
    assert batch["input_ids"][1, 2].item() == 0
    assert batch["attention_mask"][1, 1].item() == 0
    assert batch["attention_mask"][1, 2].item() == 0
    assert batch["labels"][1, 1].item() == IGNORE_INDEX
    assert batch["labels"][1, 2].item() == IGNORE_INDEX


def test_dpo_collator_three_samples_ordering():
    """B=3: verify ordering generalizes beyond B=2."""
    s1 = _make_dpo_sample([1], [10])
    s2 = _make_dpo_sample([2], [20])
    s3 = _make_dpo_sample([3], [30])
    collator = DPOCollator()
    batch = collator([s1, s2, s3])

    assert batch["input_ids"].shape == (6, 1)
    assert batch["input_ids"][0].item() == 1
    assert batch["input_ids"][1].item() == 2
    assert batch["input_ids"][2].item() == 3
    assert batch["input_ids"][3].item() == 10
    assert batch["input_ids"][4].item() == 20
    assert batch["input_ids"][5].item() == 30


# TODO: add omni data ci test
