"""BAGEL Qwen2-MoT mask materializers: Flex, Magi, and SDPA share one metadata."""

from __future__ import annotations

import pytest
import torch
from torch.nn.attention.flex_attention import BlockMask

from veomni.models.seed_omni.modules.bagel.qwen2_mot.accelerated.accelerated import BagelQwen2MoTAttentionAccelerated
from veomni.models.seed_omni.modules.bagel.qwen2_mot.masking import (
    build_mot_attention_metadata,
    build_mot_magi_mask,
    build_mot_sdpa_mask,
    pad_mot_attention_metadata,
)
from veomni.ops.kernels.attention import MagiAttentionMask


def _expand_magi_mask(mask: MagiAttentionMask, sequence_length: int) -> torch.Tensor:
    visible = torch.zeros(sequence_length, sequence_length, dtype=torch.bool, device=mask.q_ranges.device)
    attn_types = mask.attn_type_map
    if attn_types is None:
        attn_types = torch.zeros(mask.q_ranges.shape[0], dtype=torch.int32, device=mask.q_ranges.device)
    for range_index in range(mask.q_ranges.shape[0]):
        query_start, query_end = (int(mask.q_ranges[range_index, 0]), int(mask.q_ranges[range_index, 1]))
        key_start, key_end = (int(mask.k_ranges[range_index, 0]), int(mask.k_ranges[range_index, 1]))
        attn_type = int(attn_types[range_index])
        if attn_type == 0:
            visible[query_start:query_end, key_start:key_end] = True
            continue
        if attn_type != 1:
            raise AssertionError(f"Unexpected Magi attn type {attn_type} in BAGEL mask tests.")
        query_idx = torch.arange(query_start, query_end, device=visible.device)
        key_idx = torch.arange(key_start, key_end, device=visible.device)
        visible[query_start:query_end, key_start:key_end] |= query_idx[:, None] >= key_idx[None, :]
    return visible


@pytest.mark.parametrize(
    ("sample_splits", "sample_attn_modes"),
    [
        ([[8]], [["causal"]]),
        ([[4, 8]], [["full", "causal"]]),
        ([[8, 6]], [["causal", "noise"]]),
        ([[4, 8, 6]], [["full", "causal", "noise"]]),
        ([[3, 4, 5]], [["causal", "noise", "causal"]]),
        ([[4], [5]], [["causal"], ["causal"]]),
        ([[2, 3], [4, 2]], [["causal", "full"], ["causal", "noise"]]),
    ],
)
def test_magi_mask_matches_sdpa_visibility(
    sample_splits: list[list[int]],
    sample_attn_modes: list[list[str]],
) -> None:
    metadata = build_mot_attention_metadata(sample_splits, sample_attn_modes, device=torch.device("cpu"))
    sdpa = build_mot_sdpa_mask(metadata)
    magi = build_mot_magi_mask(metadata)
    visible = _expand_magi_mask(magi, int(metadata.shape[1]))
    torch.testing.assert_close(visible[None, None], sdpa)


def test_magi_mask_keeps_padding_as_isolated_full_document() -> None:
    metadata = build_mot_attention_metadata([[4]], [["causal"]], device=torch.device("cpu"))
    padded = pad_mot_attention_metadata(metadata, padded_length=8)
    sdpa = build_mot_sdpa_mask(padded)
    magi = build_mot_magi_mask(padded)
    visible = _expand_magi_mask(magi, 8)
    torch.testing.assert_close(visible[None, None], sdpa)
    assert not bool(visible[:4, 4:].any())
    assert bool(visible[4:, 4:].all())


def test_accelerated_build_attention_mask_dispatches_flex_and_magi() -> None:
    metadata = build_mot_attention_metadata([[4, 4]], [["full", "causal"]], device=torch.device("cpu"))
    flex_mask = BagelQwen2MoTAttentionAccelerated.build_attention_mask(
        metadata,
        attn_implementation="veomni_flex_attention_with_sp",
    )
    magi_mask = BagelQwen2MoTAttentionAccelerated.build_attention_mask(
        metadata,
        attn_implementation="veomni_magi_attention_with_sp",
    )
    assert isinstance(flex_mask, BlockMask)
    assert isinstance(magi_mask, MagiAttentionMask)
    default_mask = BagelQwen2MoTAttentionAccelerated.build_attention_mask(metadata)
    assert isinstance(default_mask, BlockMask)


def test_accelerated_build_attention_mask_rejects_unsupported_backend() -> None:
    metadata = build_mot_attention_metadata([[2]], [["causal"]], device=torch.device("cpu"))
    with pytest.raises(ValueError, match="packed fused attention"):
        BagelQwen2MoTAttentionAccelerated.build_attention_mask(metadata, attn_implementation="sdpa")
