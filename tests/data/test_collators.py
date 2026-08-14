import types

import pytest
import torch

from veomni.utils.constants import IGNORE_INDEX
from veomni.utils.device import IS_NPU_AVAILABLE


def _fake_ps(
    sp_enabled: bool,
    sp_size: int = 1,
    sp_rank: int = 0,
    *,
    cp_size: int = 1,
    cp_rank: int = 0,
    ulysses_size: int | None = None,
    ulysses_rank: int | None = None,
    gdn_context_parallel_implementation: str = "disabled",
):
    return types.SimpleNamespace(
        sp_enabled=sp_enabled,
        sp_size=sp_size,
        sp_rank=sp_rank,
        cp_size=cp_size,
        cp_rank=cp_rank,
        ulysses_size=sp_size if ulysses_size is None else ulysses_size,
        ulysses_rank=sp_rank if ulysses_rank is None else ulysses_rank,
        gdn_context_parallel_implementation=gdn_context_parallel_implementation,
    )


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


def test_text_collator_builds_hybrid_cp_u_partition_and_host_cu(monkeypatch, features_two_samples):
    import veomni.data.data_collator as m

    monkeypatch.setattr(
        m,
        "get_parallel_state",
        lambda: _fake_ps(
            sp_enabled=True,
            sp_size=4,
            sp_rank=2,
            cp_size=2,
            cp_rank=1,
            ulysses_size=2,
            ulysses_rank=0,
            gdn_context_parallel_implementation="state_passing_lossless",
        ),
    )
    token_labels = [
        {**features_two_samples[0], "labels": torch.tensor([2, 3, 4], dtype=torch.long)},
        {**features_two_samples[1], "labels": torch.tensor([1, 2], dtype=torch.long)},
    ]

    out = m.MainCollator()(token_labels)

    # Each sample is independently padded to 2*CP*U=8, then CP1 owns the
    # middle zigzag pair and U0 owns the first half of that local shard.
    assert torch.equal(out["input_ids"], torch.tensor([[13, 0, 0, 0]], dtype=torch.long))
    assert torch.equal(out["position_ids"], torch.tensor([[2, 0, 0, 0]], dtype=torch.long))
    assert out["cu_seqlens_list_q"] == [0, 2, 4]
    assert out["linear_attn_cu_seqlens_list_q"] == [0, 3, 5]
    assert torch.equal(out["cu_seq_lens_q"], torch.tensor([0, 2, 4], dtype=torch.int32))
    assert torch.equal(out["linear_attn_cu_seq_lens_q"], torch.tensor([0, 3, 5], dtype=torch.int32))
    assert out["attention_mask"].shape[-1] == 16
    assert torch.equal(out["router_attention_mask"], torch.tensor([[1, 0, 0, 0]], dtype=torch.long))


def test_text_collator_refuses_unselected_context_parallel_sharding(monkeypatch):
    import veomni.data.data_collator as m

    monkeypatch.setattr(
        m,
        "get_parallel_state",
        lambda: _fake_ps(sp_enabled=True, sp_size=2, cp_size=2, cp_rank=0),
    )
    with pytest.raises(RuntimeError, match="Refusing to context-shard tokens"):
        m.MainCollator()


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
