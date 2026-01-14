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


def _culen_from_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
    """
    position_ids: [1, T], where a new subsequence starts whenever position_ids resets to 0.
    Returns cu_seqlens: [num_seq+1], int32.
    Example: [0,1,2,0,1] -> [0,3,5]
    """
    pos = position_ids.view(-1).tolist()
    starts = [0]
    for i in range(1, len(pos)):
        if pos[i] == 0:
            starts.append(i)
    starts.append(len(pos))
    return torch.tensor(starts, dtype=torch.int32, device=position_ids.device)


def test_data_collator_packing_values_and_calls_add_fa(monkeypatch, features_two_samples):
    import veomni.data.data_collator as m

    # sp disabled -> should call add_flash_attention_kwargs_from_position_ids
    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=False))

    called = {"add_fa": 0}

    def fake_add_fa(batch):
        called["add_fa"] += 1
        # Validate inputs to the helper are correct (stronger than "called once")
        exp_pos = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long)
        assert torch.equal(batch["position_ids"], exp_pos)
        cu = _culen_from_position_ids(batch["position_ids"])
        # mimic return signature: cu_q, cu_k, max_q, max_k
        max_len = int((cu[1:] - cu[:-1]).max().item())
        return cu, cu, max_len, max_len

    monkeypatch.setattr(m, "add_flash_attention_kwargs_from_position_ids", fake_add_fa)

    collator = m.DataCollatorWithPositionIDs(mask_boundary_labels=False)
    out = collator(features_two_samples)

    exp_input_ids = torch.tensor([[11, 12, 13, 21, 22]], dtype=torch.long)
    exp_attn = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long)
    exp_pos = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long)
    exp_labels = torch.tensor([[2, 1]], dtype=torch.long)

    assert torch.equal(out["input_ids"], exp_input_ids)
    assert torch.equal(out["attention_mask"], exp_attn)
    assert torch.equal(out["position_ids"], exp_pos)
    assert torch.equal(out["labels"], exp_labels)
    assert called["add_fa"] == 1


def test_data_collator_sp_enabled_values_and_calls_prepare_fa(monkeypatch, features_two_samples):
    import veomni.data.data_collator as m

    # sp enabled -> should call prepare_fa_kwargs_from_position_ids
    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=True))

    called = {"prep": 0}

    def fake_prepare(position_ids):
        called["prep"] += 1
        exp_pos = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long)
        assert torch.equal(position_ids, exp_pos)

        cu = _culen_from_position_ids(position_ids)
        max_len = int((cu[1:] - cu[:-1]).max().item())
        # returns ((cu_q, max_q), (cu_k, max_k)) to match your collator unpacking
        return (cu, max_len), (cu, max_len)

    monkeypatch.setattr(m, "prepare_fa_kwargs_from_position_ids", fake_prepare)

    collator = m.DataCollatorWithPositionIDs(mask_boundary_labels=False)
    out = collator(features_two_samples)

    exp_input_ids = torch.tensor([[11, 12, 13, 21, 22]], dtype=torch.long)
    exp_attn = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long)
    exp_pos = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long)
    exp_labels = torch.tensor([[2, 1]], dtype=torch.long)

    assert torch.equal(out["input_ids"], exp_input_ids)
    assert torch.equal(out["attention_mask"], exp_attn)
    assert torch.equal(out["position_ids"], exp_pos)
    assert torch.equal(out["labels"], exp_labels)
    assert called["prep"] == 1


def test_seqcls_text_sequence_shard_collator_no_shift_no_mask_values(monkeypatch):
    import veomni.data.data_collator as m

    # sp_size=2, rank=0
    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=True, sp_size=2, sp_rank=0))

    T = 5
    input_ids = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long)
    attention_mask = torch.ones((1, T), dtype=torch.long)
    position_ids = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long)

    # token-level labels: class id sits on last token of each subsequence
    labels = torch.full((1, T), IGNORE_INDEX, dtype=torch.long)
    labels[0, 2] = 3
    labels[0, 4] = 1

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "labels": labels,
    }

    def fake_add_fa(b):
        b["_fa_called"] = True

    monkeypatch.setattr(m, "add_flash_attention_kwargs_from_position_ids", fake_add_fa)

    collator = m.TextSequenceShardCollator(
        rmpad=False,
        rmpad_with_pos_ids=False,
        pad_token_id=0,
        shift_labels=False,
        mask_boundary_labels=False,
    )
    out = collator(batch)

    # seq_len=5, sp_size=2 => chunk=3, pad to 6 (pad_length=1), rank0 gets indices [0,1,2]
    exp_input_ids = torch.tensor([[10, 11, 12]], dtype=torch.long)
    exp_labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3]], dtype=torch.long)

    # attention_mask is NOT sliced; it is padded with 0 when rmpad_with_pos_ids=False
    exp_attn = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.long)

    # position_ids is NOT sliced; sequential padding appends [0] for pad_length=1
    exp_pos = torch.tensor([[0, 1, 2, 0, 1, 0]], dtype=torch.long)

    assert torch.equal(out["input_ids"], exp_input_ids)
    assert torch.equal(out["labels"], exp_labels)
    assert torch.equal(out["attention_mask"], exp_attn)
    assert torch.equal(out["position_ids"], exp_pos)
    assert out.get("_fa_called", False) is True


def test_seqcls_text_sequence_shard_collator_padding_and_rank_slice_values(monkeypatch):
    import veomni.data.data_collator as m

    # sp_size=4, rank=2
    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=True, sp_size=4, sp_rank=2))

    T = 7
    input_ids = torch.arange(T, dtype=torch.long).view(1, T)  # [0..6]
    labels = torch.full((1, T), IGNORE_INDEX, dtype=torch.long)
    labels[0, T - 1] = 2
    attention_mask = torch.ones((1, T), dtype=torch.long)
    position_ids = torch.arange(T, dtype=torch.long).view(1, T)

    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    def fake_add_fa(b):
        b["_fa_called"] = True

    monkeypatch.setattr(m, "add_flash_attention_kwargs_from_position_ids", fake_add_fa)

    collator = m.TextSequenceShardCollator(
        rmpad=False,
        rmpad_with_pos_ids=False,
        pad_token_id=0,
        shift_labels=False,
        mask_boundary_labels=False,
    )
    out = collator(batch)

    # seq_len=7, sp_size=4 => chunk=2, pad to 8 (pad_length=1)
    # padded input_ids: [0,1,2,3,4,5,6,0], rank2 slice indices [4,5] => [4,5]
    exp_input_ids = torch.tensor([[4, 5]], dtype=torch.long)

    # padded labels: [-100,...,-100,2,-100], rank2 indices [4,5] => [-100,-100]
    exp_labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX]], dtype=torch.long)

    # attention_mask padded with 0 to length 8, NOT sliced
    exp_attn = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.long)

    # position_ids sequential padding appends [0], NOT sliced
    exp_pos = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 0]], dtype=torch.long)

    assert torch.equal(out["input_ids"], exp_input_ids)
    assert torch.equal(out["labels"], exp_labels)
    assert torch.equal(out["attention_mask"], exp_attn)
    assert torch.equal(out["position_ids"], exp_pos)
    assert out.get("_fa_called", False) is True
