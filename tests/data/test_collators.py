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
    # Route A: pad position_ids with arange(pad_len) so the pad region collapses
    # to a single segment in cu_seq_lens (avoids per-step cu_seqlens.shape churn
    # that triggers flash_qla prepare_chunk_offsets recompiles).
    exp_pos = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2]], dtype=torch.long)
    exp_labels = torch.tensor([[2, 3, 4, IGNORE_INDEX, 2, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]], dtype=torch.long)
    exp_cu_seq_lens = torch.tensor([0, 3, 5, 8], dtype=torch.int32)
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
    # attention mask is not sliced.
    # Route A: full position_ids before SP slice = [0,1,2,0,1,0,1,2]
    # → rank 0 sees first half [0,1,2,0]; rank 1 sees [1,0,1,2].
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
    assert out["max_length_q"] == exp_max_length
    assert out["max_length_k"] == exp_max_length

    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=True, sp_size=sp_size, sp_rank=1))
    collator = m.MainCollator(pad_to_length=pad_to_length)
    out = collator(token_labels)
    # SP slicing; lengths stay at pad_to_length // sp_size.
    # attention mask is not sliced
    exp_input_ids = torch.tensor([[22, 0, 0, 0]], dtype=torch.long)
    exp_attn = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long)
    exp_pos = torch.tensor([[1, 0, 1, 2]], dtype=torch.long)
    exp_labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]], dtype=torch.long)
    exp_cu_seq_lens = torch.tensor([0, 3, 5, 8], dtype=torch.int32)
    exp_max_length = 3

    assert torch.equal(out["input_ids"], exp_input_ids)
    assert torch.equal(out["attention_mask"], exp_attn)
    assert torch.equal(out["position_ids"], exp_pos)
    assert torch.equal(out["labels"], exp_labels)
    assert torch.equal(out["cu_seq_lens_q"], exp_cu_seq_lens)
    assert torch.equal(out["cu_seq_lens_k"], exp_cu_seq_lens)
    assert out["max_length_q"] == exp_max_length
    assert out["max_length_k"] == exp_max_length


def test_vlm_collator_pad_to_length_sp_disabled(monkeypatch):
    """Cover VLM-specific seq_len-keyed fields under pad_to_length:
    3D mrope ``position_ids``, ``image_mask`` / ``video_mask`` and
    ``mm_token_type_ids`` (transformers v5 VLM)."""
    if IS_NPU_AVAILABLE:
        pytest.skip("NPU does not support this padding test yet.")
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=False))

    # Sample 1: text token, image token, text token (image at pos 1).
    f1 = {
        "input_ids": torch.tensor([11, 0, 13], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
        "labels": torch.tensor([2, 3, 4], dtype=torch.long),
        "image_mask": torch.tensor([False, True, False]),
        "video_mask": torch.tensor([False, False, False]),
        "mm_token_type_ids": torch.tensor([0, 1, 0], dtype=torch.long),
        # 3D mrope: (mrope_dim=3, seq_len). Squeeze pattern matches data_transform.
        "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=torch.long),
    }
    f2 = {
        "input_ids": torch.tensor([21, 22], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1], dtype=torch.long),
        "labels": torch.tensor([1, 2], dtype=torch.long),
        "image_mask": torch.tensor([False, False]),
        "video_mask": torch.tensor([False, False]),
        "mm_token_type_ids": torch.tensor([0, 0], dtype=torch.long),
        "position_ids": torch.tensor([[0, 1], [0, 1], [0, 1]], dtype=torch.long),
    }

    collator = m.MainCollator(pad_to_length=8)
    out = collator([f1, f2])

    exp_input_ids = torch.tensor([[11, 0, 13, 21, 22, 0, 0, 0]], dtype=torch.long)
    exp_attn = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long)
    # 3D mrope: route A pads the seq dim with arange(pad_len) broadcast across
    # mrope rows, so the pad region forms one trailing segment.
    exp_pos = torch.tensor(
        [[[0, 1, 2, 0, 1, 0, 1, 2], [0, 1, 2, 0, 1, 0, 1, 2], [0, 1, 2, 0, 1, 0, 1, 2]]],
        dtype=torch.long,
    )
    exp_image_mask = torch.tensor([[False, True, False, False, False, False, False, False]])
    exp_video_mask = torch.tensor([[False, False, False, False, False, False, False, False]])
    exp_mm_tt = torch.tensor([[0, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.long)
    exp_labels = torch.tensor([[2, 3, 4, IGNORE_INDEX, 2, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]], dtype=torch.long)

    assert torch.equal(out["input_ids"], exp_input_ids)
    assert torch.equal(out["attention_mask"], exp_attn)
    assert torch.equal(out["position_ids"], exp_pos)
    assert torch.equal(out["labels"], exp_labels)
    assert torch.equal(out["image_mask"], exp_image_mask)
    assert torch.equal(out["video_mask"], exp_video_mask)
    assert torch.equal(out["mm_token_type_ids"], exp_mm_tt)


def test_pad_to_length_strips_tail_segment_in_seqlens(monkeypatch, features_two_samples):
    """Regression: under route-A pad_to_length, the trailing arange-padded
    region forms a single cu_seq_lens segment of length pad_len. Downstream
    seqlen consumers (PostCollator, EnvironMeter) must strip it so that
    DPO's chosen/rejected indexing on log_probs.split(seq_lens) doesn't
    treat padding as a real sample."""
    if IS_NPU_AVAILABLE:
        pytest.skip("NPU does not support this padding test yet.")
    import veomni.data.data_collator as m
    from veomni.utils.helper import _compute_seqlens

    pad_to_length = 8
    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=False))
    collator = m.MainCollator(pad_to_length=pad_to_length)
    out = collator(features_two_samples)

    # cu_seq_lens_q = [0, 3, 5, 8] → last segment is the pad of length 3.
    assert torch.equal(out["cu_seq_lens_q"], torch.tensor([0, 3, 5, 8], dtype=torch.int32))
    assert int(out["_tail_pad_len"]) == 3

    # SeqlensComputePostCollator must strip the tail pad segment.
    assert m.SeqlensComputePostCollator()(out) == [3, 2]
    # EnvironMeter helper must do the same.
    assert _compute_seqlens(out) == [3, 2]


def test_pad_to_length_asserts_sp_divisibility(monkeypatch, features_two_samples):
    """SequenceParallelCollator constant-pads position_ids, which would split
    the route-A arange tail into many length-1 segments under non-divisible
    pad_to_length. PackingCollator must reject that configuration upfront."""
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=True, sp_size=4, sp_rank=0))
    with pytest.raises(AssertionError, match="must be divisible by sp_size"):
        m.MainCollator(pad_to_length=10)


def test_pad_seq_to_multiple_of_sp_disabled(monkeypatch, features_two_samples):
    if IS_NPU_AVAILABLE:
        pytest.skip("NPU does not support this padding test yet.")
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=False))
    token_labels = [
        {**features_two_samples[0], "labels": torch.tensor([2, 3, 4], dtype=torch.long)},
        {**features_two_samples[1], "labels": torch.tensor([1, 2], dtype=torch.long)},
    ]
    # packed length = 3 + 2 = 5 -> next multiple of 4 is 8 -> pad_len = 3
    collator = m.MainCollator(pad_seq_to_multiple_of=4)
    out = collator(token_labels)

    exp_input_ids = torch.tensor([[11, 12, 13, 21, 22, 0, 0, 0]], dtype=torch.long)
    exp_attn = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long)
    # pad region is ONE self-contained segment: position ids restart at 0 (arange(3))
    exp_pos = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2]], dtype=torch.long)
    exp_labels = torch.tensor([[2, 3, 4, IGNORE_INDEX, 2, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]], dtype=torch.long)
    # three segments: real (3), real (2), padding (3)
    exp_cu_seq_lens = torch.tensor([0, 3, 5, 8], dtype=torch.int32)
    exp_max_length = 3

    assert torch.equal(out["input_ids"], exp_input_ids)
    assert torch.equal(out["attention_mask"], exp_attn)
    assert torch.equal(out["position_ids"], exp_pos)
    assert torch.equal(out["labels"], exp_labels)
    assert torch.equal(out["cu_seq_lens_q"], exp_cu_seq_lens)
    assert torch.equal(out["cu_seq_lens_k"], exp_cu_seq_lens)
    assert out["max_length_q"] == exp_max_length
    assert out["max_length_k"] == exp_max_length


def test_pad_seq_to_multiple_of_noop_when_already_multiple(monkeypatch, features_two_samples):
    if IS_NPU_AVAILABLE:
        pytest.skip("NPU does not support this padding test yet.")
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=False))
    token_labels = [
        {**features_two_samples[0], "labels": torch.tensor([2, 3, 4], dtype=torch.long)},
        {**features_two_samples[1], "labels": torch.tensor([1, 2], dtype=torch.long)},
    ]
    # packed length = 5 is already a multiple of 5 -> no padding
    collator = m.MainCollator(pad_seq_to_multiple_of=5)
    out = collator(token_labels)
    assert out["input_ids"].shape == (1, 5)
    assert torch.equal(out["input_ids"], torch.tensor([[11, 12, 13, 21, 22]], dtype=torch.long))
    assert torch.equal(out["cu_seq_lens_q"], torch.tensor([0, 3, 5], dtype=torch.int32))


def test_pad_to_length_takes_priority_over_multiple_of(monkeypatch, features_two_samples):
    if IS_NPU_AVAILABLE:
        pytest.skip("NPU does not support this padding test yet.")
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=False))
    token_labels = [
        {**features_two_samples[0], "labels": torch.tensor([2, 3, 4], dtype=torch.long)},
        {**features_two_samples[1], "labels": torch.tensor([1, 2], dtype=torch.long)},
    ]
    # pad_to_length=10 wins over pad_seq_to_multiple_of=4 -> pad to 10. pad_to_length
    # uses route-A arange position_ids, so the tail is one segment of length pad_len.
    collator = m.MainCollator(pad_to_length=10, pad_seq_to_multiple_of=4)
    out = collator(token_labels)
    assert out["input_ids"].shape == (1, 10)
    assert torch.equal(out["input_ids"], torch.tensor([[11, 12, 13, 21, 22, 0, 0, 0, 0, 0]], dtype=torch.long))
    assert torch.equal(out["position_ids"], torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2, 3, 4]], dtype=torch.long))


def test_pad_seq_to_multiple_of_3d_position_ids(monkeypatch):
    if IS_NPU_AVAILABLE:
        pytest.skip("NPU does not support this padding test yet.")
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=False))
    # mrope-style 3-D position_ids per sample: shape [3, len]
    f1 = {
        "input_ids": torch.tensor([11, 12, 13], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
        "labels": torch.tensor([2, 3, 4], dtype=torch.long),
        "position_ids": torch.stack([torch.arange(3, dtype=torch.long)] * 3),  # [3, 3]
    }
    f2 = {
        "input_ids": torch.tensor([21, 22], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1], dtype=torch.long),
        "labels": torch.tensor([1, 2], dtype=torch.long),
        "position_ids": torch.stack([torch.arange(2, dtype=torch.long)] * 3),  # [3, 2]
    }
    collator = m.MainCollator(pad_seq_to_multiple_of=4)
    out = collator([f1, f2])
    # packed -> [1, 3, 5] -> padded to [1, 3, 8], pad region arange(3) on every channel
    assert out["position_ids"].shape == (1, 3, 8)
    for c in range(3):
        assert torch.equal(out["position_ids"][0, c], torch.tensor([0, 1, 2, 0, 1, 0, 1, 2], dtype=torch.long))
    assert torch.equal(out["input_ids"], torch.tensor([[11, 12, 13, 21, 22, 0, 0, 0]], dtype=torch.long))
    # cu_seqlens derived from channel 0 of position_ids
    assert torch.equal(out["cu_seq_lens_q"], torch.tensor([0, 3, 5, 8], dtype=torch.int32))


# TODO: add omni data ci test
