import types

import pytest
import torch

from veomni.data.constants import IGNORE_INDEX


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


def test_classification_data_collator_packing_and_position_ids(monkeypatch, features_two_samples):
    import veomni.data.data_collator as m

    # mock parallel state
    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=False))

    called = {"add_fa": 0}

    def fake_add_fa(batch):
        called["add_fa"] += 1
        # mimic return signature: cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k
        # Here, packed position_ids resets per sample, so lengths = [3,2]
        cu = torch.tensor([0, 3, 5], dtype=torch.int32)
        return cu, cu, 3, 3

    monkeypatch.setattr(m, "add_flash_attention_kwargs_from_position_ids", fake_add_fa)

    collator = m.DataCollatorWithPositionIDs(mask_boundary_labels=False)
    batch = collator(features_two_samples)

    assert batch["input_ids"].shape == (1, 5)
    assert batch["attention_mask"].shape == (1, 5)
    assert batch["position_ids"].shape == (1, 5)
    assert called["add_fa"] == 1


def test_classification_data_collator_sp_enabled_uses_prepare_fa(monkeypatch, features_two_samples):
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=True))

    called = {"prep": 0}

    def fake_prepare(position_ids):
        called["prep"] += 1
        cu = torch.tensor([0, 3, 5], dtype=torch.int32)
        # return ((cu_q, max_q), (cu_k, max_k)) or whatever your real fn returns
        return (cu, 3), (cu, 3)

    monkeypatch.setattr(m, "prepare_fa_kwargs_from_position_ids", fake_prepare)

    collator = m.DataCollatorWithPositionIDs(mask_boundary_labels=False)
    batch = collator(features_two_samples)

    assert batch["input_ids"].shape == (1, 5)
    assert "position_ids" in batch
    assert called["prep"] == 1


def test_seqcls_text_sequence_shard_collator_no_shift_and_no_mask(monkeypatch):
    import veomni.data.data_collator as m

    # sp_size=2, rank=0
    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=True, sp_size=2, sp_rank=0))

    # build a packed-like batch: [1, T]
    T = 5
    input_ids = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long)
    attention_mask = torch.ones((1, T), dtype=torch.long)
    position_ids = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long)

    # token-level labels: only last token of each sample has class id
    # suppose two samples boundaries are at idx 2 and 4 -> last tokens maybe at 2 and 4
    labels = torch.full((1, T), IGNORE_INDEX, dtype=torch.long)
    labels[0, 2] = 3
    labels[0, 4] = 1

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "labels": labels,
    }

    # Avoid depending on real FA kwargs computation; mark it called
    def fake_add_fa(b):
        b["_fa_called"] = True
        return b

    monkeypatch.setattr(m, "add_flash_attention_kwargs_from_position_ids", fake_add_fa)

    collator = m.TextSequenceShardCollator(
        rmpad=False,
        rmpad_with_pos_ids=False,
        pad_token_id=0,
        shift_labels=False,
        mask_boundary_labels=False,
    )
    out = collator(batch)

    # chunk_size = ceil(5/2)=3, pad to 6, slice rank0 -> first 3 tokens
    assert out["input_ids"].shape[-1] == 3
    assert out["labels"].shape[-1] == 3

    # no shift: label at index 2 should remain in rank0 chunk
    assert (out["labels"] == 3).any()

    # and we didn't mask it
    assert 3 in out["labels"].view(-1).tolist()

    assert out.get("_fa_called", False) is True


def test_seqcls_text_sequence_shard_collator_shapes(monkeypatch):
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=True, sp_size=4, sp_rank=2))

    T = 7
    input_ids = torch.arange(T).view(1, T)
    labels = torch.full((1, T), IGNORE_INDEX, dtype=torch.long)
    labels[0, T - 1] = 2
    attention_mask = torch.ones((1, T), dtype=torch.long)
    position_ids = torch.arange(T).view(1, T)

    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    monkeypatch.setattr(m, "add_flash_attention_kwargs_from_position_ids", lambda b: b)

    collator = m.TextSequenceShardCollator(
        rmpad=False,
        rmpad_with_pos_ids=False,
        pad_token_id=0,
        shift_labels=False,
        mask_boundary_labels=False,
    )
    out = collator(batch)

    # seq_len=7, sp_size=4 => chunk=2, pad to 8, each rank gets 2
    assert out["input_ids"].shape == (1, 2)
    assert out["labels"].shape == (1, 2)
    assert out["attention_mask"].shape == (1, 8)
    assert out["position_ids"].shape == (1, 8)
