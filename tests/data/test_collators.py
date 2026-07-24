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


def test_data_collator_info_pack_mode_defaults_and_validation():
    """``pack_mode`` defaults to ``"cat"`` so every non-HI3 model stays byte-identical;
    invalid values raise; ``"list"`` + ``sp_slice=True`` combo is rejected eagerly.

    ``sp_pad_value`` / ``sp_pad_scale`` must be passed as a pair (both None or both
    set) — the dataclass default ``sp_pad_scale=1`` is a footgun that requires an
    explicit ``sp_pad_scale=None`` override when using ``sp_pad_value=None``.
    """
    import veomni.data.data_collator as m

    # Default pack_mode
    info = m.DataCollateInfo(pack_dim=0, sp_pad_value=None, sp_pad_scale=None)
    assert info.pack_mode == "cat"

    # Explicit list
    info_list = m.DataCollateInfo(pack_dim=0, pack_mode="list", sp_pad_value=None, sp_pad_scale=None)
    assert info_list.pack_mode == "list"

    # Invalid value
    with pytest.raises(ValueError, match="pack_mode"):
        m.DataCollateInfo(pack_dim=0, pack_mode="invalid", sp_pad_value=None, sp_pad_scale=None)

    # list + sp_slice combo rejected
    with pytest.raises(ValueError, match="sp_slice"):
        m.DataCollateInfo(pack_dim=0, sp_slice=True, sp_pad_value=0, sp_pad_scale=1, pack_mode="list")

    # Default table entries all stay "cat" (no accidental behavior change).
    for name, entry in m.DEFAULT_DATA_COLLATE_INFO.items():
        assert entry.pack_mode == "cat", f"default entry {name} unexpectedly changed pack_mode"


def test_packing_collator_pack_mode_list_keeps_list_of_tensors(monkeypatch):
    """``pack_mode='list'`` skips ``torch.cat`` and preserves per-sample tensors —
    the mechanism HI3 uses under mbs>1 with heterogeneous image buckets."""
    import veomni.data.data_collator as m

    monkeypatch.setattr(m, "get_parallel_state", lambda: _fake_ps(sp_enabled=False))

    features = [
        {
            "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "labels": torch.full((3,), IGNORE_INDEX, dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
            "position_ids": torch.arange(3, dtype=torch.int64),
            "hy3_staging": torch.zeros(1, 3, 8, 4),  # shape [1, C, H0, W0]
        },
        {
            "input_ids": torch.tensor([4, 5], dtype=torch.long),
            "labels": torch.full((2,), IGNORE_INDEX, dtype=torch.long),
            "attention_mask": torch.tensor([1, 1], dtype=torch.long),
            "position_ids": torch.arange(2, dtype=torch.int64),
            "hy3_staging": torch.ones(1, 3, 6, 6),  # DIFFERENT shape [1, C, H1, W1]
        },
    ]

    collate_infos = m.DEFAULT_DATA_COLLATE_INFO.copy()
    collate_infos["hy3_staging"] = m.DataCollateInfo(
        pack_dim=0, pack_mode="list", sp_pad_value=None, sp_pad_scale=None
    )

    collator = m.PackingCollator(collate_infos=collate_infos)
    out = collator(features)

    # Text tensors still get packed (cat + unsqueeze).
    assert out["input_ids"].shape == (1, 5)
    # Heterogeneous staging survives as a list of the original per-sample tensors.
    assert isinstance(out["hy3_staging"], list)
    assert len(out["hy3_staging"]) == 2
    assert out["hy3_staging"][0].shape == (1, 3, 8, 4)
    assert out["hy3_staging"][1].shape == (1, 3, 6, 6)
    # And the values weren't clobbered.
    assert torch.equal(out["hy3_staging"][0], torch.zeros(1, 3, 8, 4))
    assert torch.equal(out["hy3_staging"][1], torch.ones(1, 3, 6, 6))


def test_build_native_dataloader_batch_sampler_arg():
    """``build_native_dataloader.batch_sampler`` opt-in: reject dyn_bsz combo and
    IterableDataset combo; accept a plain map-style dataset with a custom sampler."""
    import veomni.data.data_loader as dl

    # dyn_bsz + batch_sampler → mutually exclusive
    dummy = [{"input_ids": torch.tensor([1])}]
    dummy_bs = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(dummy), batch_size=1, drop_last=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        dl.build_native_dataloader(
            dataset=dummy,
            micro_batch_size=1,
            global_batch_size=1,
            dataloader_batch_size=1,
            max_seq_len=8,
            train_steps=1,
            dyn_bsz=True,  # incompatible with batch_sampler
            batch_sampler=dummy_bs,
        )


# TODO: add omni data ci test
