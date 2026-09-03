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
# See the License for the specific language governing limitations
# under the License.

"""HF mask builders registered on ``ALL_MASK_ATTENTION_FUNCTIONS``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.nn.attention.flex_attention import BlockMask
from transformers import PreTrainedConfig
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, create_causal_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from tests.kernels.attention_cases import (
    bagel_dense_mask,
    bagel_flex_mask,
    bagel_magi_mask,
    flex_visible,
    materialize_magi_mask,
)
from veomni.kernels._kernels.attention.install import _VEOMNI_HF_PATCHES
from veomni.kernels._kernels.attention.mask import flex as flex_mask
from veomni.kernels._kernels.attention.mask import magi as magi_mask
from veomni.kernels._kernels.attention.mask import sdpa as sdpa_mask
from veomni.kernels.mask import (
    MagiAttentionMask,
    flash_attention_mask_builder,
    flex_attention_mask_builder,
    magi_attention_mask_builder,
    sdpa_attention_mask_builder,
)


def test_veomni_hf_patches_pair_attention_and_mask():
    for name, forward, mask_builder in _VEOMNI_HF_PATCHES:
        assert ALL_ATTENTION_FUNCTIONS[name] is forward
        assert ALL_MASK_ATTENTION_FUNCTIONS[name] is mask_builder


def test_mask_builder_return_types():
    assert flash_attention_mask_builder(1, 4, 4) is None
    sdpa = sdpa_attention_mask_builder(1, 4, 4, device="cpu", allow_is_causal_skip=False)
    assert isinstance(sdpa, torch.Tensor)
    assert sdpa.dtype == torch.bool
    assert sdpa.shape == (1, 1, 4, 4)
    flex = flex_attention_mask_builder(1, 4, 4, device="cpu")
    assert isinstance(flex, BlockMask)
    magi = magi_attention_mask_builder(1, 4, 4, device="cpu")
    assert isinstance(magi, MagiAttentionMask)
    assert magi.q_ranges.dtype == torch.int32
    assert magi.q_ranges.shape == (1, 2)


def test_hf_create_causal_mask_uses_registered_builders():
    embeds = torch.randn(1, 8, 16)
    position_ids = torch.arange(8).unsqueeze(0)

    flex_config = PreTrainedConfig()
    flex_config._attn_implementation = "veomni_flex_attention"
    flex = create_causal_mask(flex_config, embeds, None, None, position_ids)
    assert isinstance(flex, BlockMask)
    assert flex.shape == (1, 1, 8, 8)

    magi_config = PreTrainedConfig()
    magi_config._attn_implementation = "veomni_magi_attention"
    magi = create_causal_mask(magi_config, embeds, None, None, position_ids)
    assert isinstance(magi, MagiAttentionMask)
    assert magi.q_ranges.tolist() == [[0, 8]]
    assert magi.k_ranges.tolist() == [[0, 8]]


def test_flex_and_sdpa_2d_padding_masks_align():
    attention_2d = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    flex = flex_attention_mask_builder(1, 4, 4, attention_mask=attention_2d, device="cpu")
    sdpa = sdpa_attention_mask_builder(
        1,
        4,
        4,
        attention_mask=attention_2d,
        device="cpu",
        allow_is_causal_skip=False,
    )
    assert sdpa is not None
    torch.testing.assert_close(flex_visible(flex, 4, 4), sdpa[0, 0])
    assert not bool(sdpa[0, 0, :, -1].any())


def test_magi_2d_mask_length_aligns_with_full_kv():
    attention_2d = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    mask = magi_attention_mask_builder(1, 4, 4, attention_mask=attention_2d, device="cpu")
    assert mask.q_ranges.tolist() == [[0, 4]]
    assert mask.k_ranges.tolist() == [[0, 4]]
    with pytest.raises(ValueError, match="full post-Ulysses key sequence"):
        magi_attention_mask_builder(1, 4, 4, attention_mask=torch.ones(1, 3, dtype=torch.bool), device="cpu")


def _sync_ulysses_state(*, size: int = 2) -> SimpleNamespace:
    return SimpleNamespace(ulysses_size=size, async_enabled=False)


def _patch_mask_ulysses(monkeypatch, *modules, apply: bool, size: int = 2) -> None:
    state = _sync_ulysses_state(size=size)
    for module in modules:
        monkeypatch.setattr(module, "should_apply_ulysses", lambda *, skip_ulysses=False: apply and not skip_ulysses)
        monkeypatch.setattr(module, "get_parallel_state", lambda: state)


def test_flex_and_magi_ulysses_2d_mask_lengths_align(monkeypatch):
    _patch_mask_ulysses(monkeypatch, flex_mask, magi_mask, apply=True)
    full_2d = torch.ones(1, 8, dtype=torch.bool)
    flex = flex_attention_mask_builder(1, 4, 4, attention_mask=full_2d, device="cpu")
    magi = magi_attention_mask_builder(1, 4, 4, attention_mask=full_2d, device="cpu")
    assert tuple(flex.shape[-2:]) == (8, 8)
    assert magi.q_ranges.tolist() == [[0, 8]]
    assert magi.k_ranges.tolist() == [[0, 8]]


@pytest.mark.parametrize(
    ("builder", "module", "attention_mask", "match"),
    [
        (flex_attention_mask_builder, flex_mask, None, "full-sequence 2D attention mask"),
        (sdpa_attention_mask_builder, sdpa_mask, None, "full-sequence 2D attention mask"),
        (
            flex_attention_mask_builder,
            flex_mask,
            torch.ones(1, 4, dtype=torch.bool),
            "local q_length \\* ulysses_size",
        ),
        (
            sdpa_attention_mask_builder,
            sdpa_mask,
            torch.ones(1, 4, dtype=torch.bool),
            "local q_length \\* ulysses_size",
        ),
    ],
)
def test_flex_and_sdpa_ulysses_reject_incomplete_2d_mask(monkeypatch, builder, module, attention_mask, match):
    _patch_mask_ulysses(monkeypatch, module, apply=True)
    with pytest.raises(ValueError, match=match):
        builder(1, 4, 4, attention_mask=attention_mask, device="cpu")


def test_magi_hf_builder_expands_ulysses_local_lengths(monkeypatch):
    _patch_mask_ulysses(monkeypatch, magi_mask, apply=True)
    mask = magi_attention_mask_builder(1, 4, 4, device="cpu")
    assert mask.q_ranges.tolist() == [[0, 8]]
    assert mask.k_ranges.tolist() == [[0, 8]]


@pytest.mark.parametrize(
    ("builder", "module", "kwargs"),
    [
        (flex_attention_mask_builder, flex_mask, {"attention_mask": torch.ones(1, 4, dtype=torch.bool)}),
        (
            sdpa_attention_mask_builder,
            sdpa_mask,
            {"attention_mask": torch.ones(1, 4, dtype=torch.bool), "allow_is_causal_skip": False},
        ),
        (magi_attention_mask_builder, magi_mask, {}),
    ],
)
def test_mask_builders_keep_local_lengths_when_async_or_skipped(monkeypatch, builder, module, kwargs):
    _patch_mask_ulysses(monkeypatch, module, apply=False)
    local = builder(1, 4, 4, device="cpu", **kwargs)
    _patch_mask_ulysses(monkeypatch, module, apply=True)
    skipped = builder(1, 4, 4, device="cpu", skip_ulysses=True, **kwargs)
    if builder is magi_attention_mask_builder:
        assert local.q_ranges.tolist() == [[0, 4]]
        assert local.k_ranges.tolist() == [[0, 4]]
        assert skipped.q_ranges.tolist() == [[0, 4]]
        assert skipped.k_ranges.tolist() == [[0, 4]]
    elif builder is flex_attention_mask_builder:
        assert tuple(local.shape[-2:]) == (4, 4)
        assert tuple(skipped.shape[-2:]) == (4, 4)
    else:
        assert local.shape[-2:] == (4, 4)
        assert skipped.shape[-2:] == (4, 4)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"batch_size": 2}, "physical batch size 1"),
        ({"q_offset": 1}, "does not support KV-cache offsets"),
        ({"mask_function": lambda *args: True}, "canonical causal or bidirectional"),
    ],
)
def test_magi_hf_builder_rejects_unsupported_registry_inputs(override, match):
    arguments = {"batch_size": 1, "q_length": 4, "kv_length": 4, "device": "cpu", **override}
    with pytest.raises(ValueError, match=match):
        magi_attention_mask_builder(**arguments)


def test_magi_hf_builder_does_not_recover_packed_visibility_from_position_ids():
    config = PreTrainedConfig()
    config._attn_implementation = "veomni_magi_attention"
    embeds = torch.randn(1, 8, 16)
    attention_mask = torch.ones(1, 8, dtype=torch.long)
    packed_position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]])
    mask = create_causal_mask(config, embeds, attention_mask, None, packed_position_ids)
    assert isinstance(mask, MagiAttentionMask)
    assert mask.q_ranges.tolist() == [[0, 8]]
    assert mask.k_ranges.tolist() == [[0, 8]]


def test_magi_from_ranges_casts_ffa_contract():
    mask = MagiAttentionMask.from_ranges(
        torch.tensor([[0, 4]]),
        torch.tensor([[0, 4]]),
        torch.tensor([1]),
    )
    assert mask.q_ranges.dtype == torch.int32
    assert mask.q_ranges.tolist() == [[0, 4]]
    assert mask.attn_type_map is not None
    assert mask.attn_type_map.tolist() == [1]


def test_magi_from_ranges_preserves_mixed_visibility():
    q_ranges = torch.tensor([[0, 4], [4, 8], [4, 8]], dtype=torch.int32)
    k_ranges = torch.tensor([[0, 4], [0, 4], [4, 8]], dtype=torch.int32)
    attn_type_map = torch.tensor([1, 0, 0], dtype=torch.int32)
    mask = MagiAttentionMask.from_ranges(q_ranges, k_ranges, attn_type_map)
    torch.testing.assert_close(mask.q_ranges, q_ranges)
    torch.testing.assert_close(mask.k_ranges, k_ranges)
    torch.testing.assert_close(mask.attn_type_map, attn_type_map)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"q_ranges": torch.tensor([[0, 8]], dtype=torch.int64)}, "dtype int32"),
        ({"q_ranges": torch.tensor([0, 8], dtype=torch.int32)}, "shape \\[num_ranges, 2\\]"),
        ({"q_ranges": torch.empty(0, 2, dtype=torch.int32)}, "shape \\[num_ranges, 2\\]"),
        ({"q_ranges": torch.tensor([[-1, 8]], dtype=torch.int32)}, "0 <= start < end"),
        ({"k_ranges": torch.tensor([[4, 4]], dtype=torch.int32)}, "0 <= start < end"),
        ({"attn_type_map": torch.tensor([4], dtype=torch.int32)}, "attn_type_map values must be in \\[0, 3\\]"),
    ],
)
def test_magi_mask_rejects_invalid_metadata(overrides, match):
    values = {
        "q_ranges": torch.tensor([[0, 8]], dtype=torch.int32),
        "k_ranges": torch.tensor([[0, 8]], dtype=torch.int32),
        "attn_type_map": None,
        **overrides,
    }
    with pytest.raises(ValueError, match=match):
        MagiAttentionMask(**values)


def test_bagel_mixed_masks_match_dense_visibility():
    sequence_length = 16
    device = torch.device("cpu")
    dense = bagel_dense_mask(sequence_length, device)
    magi = bagel_magi_mask(sequence_length, device)
    flex = bagel_flex_mask(sequence_length, device)
    torch.testing.assert_close(materialize_magi_mask(magi, sequence_length), dense)
    torch.testing.assert_close(flex_visible(flex, sequence_length, sequence_length), dense[0, 0])
