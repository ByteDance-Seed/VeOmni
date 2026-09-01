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

"""Shape mask APIs: ``causal_mask`` / ``sliding_window_mask`` / ``packed_causal_mask``."""

from __future__ import annotations

import pytest
import torch

from tests.kernels.attention_cases import flex_visible
from veomni.kernels.mask import (
    MagiAttentionMask,
    causal_mask,
    flex_attention_mask_builder,
    magi_attention_mask_builder,
    packed_causal_mask,
    sdpa_attention_mask_builder,
    sliding_window_mask,
)


def test_flash_shapes_are_none():
    assert causal_mask(8, 8, impl="veomni_flash_attention_2", device="cpu") is None
    assert sliding_window_mask(8, 8, impl="flash_attention_2", device="cpu", sliding_window=4) is None
    assert (
        packed_causal_mask(8, 8, impl="veomni_flash_attention_3", device="cpu", cu_seqlens=torch.tensor([0, 8]))
        is None
    )


@pytest.mark.parametrize("impl", ("sdpa", "veomni_sdpa", "eager"))
def test_sdpa_causal_aligns_with_hf_builder(impl):
    built = sdpa_attention_mask_builder(1, 4, 4, device="cpu", allow_is_causal_skip=False)
    shaped = causal_mask(4, 4, impl=impl, device="cpu")
    torch.testing.assert_close(shaped, built)
    torch.testing.assert_close(shaped[0, 0], torch.tril(torch.ones(4, 4, dtype=torch.bool)))


@pytest.mark.parametrize("impl", ("sdpa", "veomni_sdpa"))
def test_sdpa_cached_causal_aligns_with_hf_builder(impl):
    built = sdpa_attention_mask_builder(1, 2, 4, q_offset=2, device="cpu")
    shaped = causal_mask(2, 4, impl=impl, device="cpu")
    assert built is not None
    torch.testing.assert_close(shaped, built)
    torch.testing.assert_close(shaped[0, 0], torch.ones(2, 4, dtype=torch.bool).tril(diagonal=2))


@pytest.mark.parametrize("impl", ("sdpa", "eager"))
def test_sdpa_sliding_aligns_with_hf_builder(impl):
    built = sdpa_attention_mask_builder(1, 4, 4, device="cpu", sliding_window=2, allow_is_causal_skip=False)
    shaped = sliding_window_mask(4, 4, impl=impl, device="cpu", sliding_window=2)
    torch.testing.assert_close(shaped, built)
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [False, True, True, False],
            [False, False, True, True],
        ]
    )
    torch.testing.assert_close(shaped[0, 0], expected)


def test_sdpa_packed_aligns_with_hf_builder():
    cu_seqlens = torch.tensor([0, 2, 4])
    built = sdpa_attention_mask_builder(
        1,
        4,
        4,
        device="cpu",
        cu_seqlens=cu_seqlens,
        allow_is_causal_skip=False,
    )
    shaped = packed_causal_mask(4, 4, impl="sdpa", device="cpu", cu_seqlens=cu_seqlens)
    torch.testing.assert_close(shaped, built)
    expected = torch.zeros(4, 4, dtype=torch.bool)
    expected[:2, :2] = torch.tril(torch.ones(2, 2, dtype=torch.bool))
    expected[2:, 2:] = torch.tril(torch.ones(2, 2, dtype=torch.bool))
    torch.testing.assert_close(shaped[0, 0], expected)


def test_sdpa_packed_cached_aligns_with_hf_builder():
    built = sdpa_attention_mask_builder(
        1,
        2,
        4,
        q_offset=2,
        device="cpu",
        cu_seqlens=torch.tensor([0, 2]),
        cu_seqlens_k=torch.tensor([0, 4]),
    )
    shaped = packed_causal_mask(
        2,
        4,
        impl="sdpa",
        device="cpu",
        cu_seqlens=torch.tensor([0, 2]),
        cu_seq_lens_k=torch.tensor([0, 4]),
    )
    assert built is not None
    torch.testing.assert_close(shaped, built)


@pytest.mark.parametrize("impl", ("flex_attention", "veomni_flex_attention"))
def test_flex_causal_aligns_with_hf_builder(impl):
    built = flex_attention_mask_builder(1, 4, 4, device="cpu")
    shaped = causal_mask(4, 4, impl=impl, device="cpu")
    torch.testing.assert_close(flex_visible(shaped, 4, 4), flex_visible(built, 4, 4))


@pytest.mark.parametrize("impl", ("magi_attention", "veomni_magi_attention"))
def test_magi_causal_aligns_with_hf_builder(impl):
    built = magi_attention_mask_builder(1, 4, 4, device="cpu")
    shaped = causal_mask(4, 4, impl=impl, device="cpu")
    torch.testing.assert_close(shaped.q_ranges, built.q_ranges)
    torch.testing.assert_close(shaped.k_ranges, built.k_ranges)
    torch.testing.assert_close(shaped.attn_type_map, built.attn_type_map)


def test_magi_packed_aligns_with_from_cu_seqlens():
    cu_seqlens = torch.tensor([0, 2, 4])
    built = MagiAttentionMask.from_cu_seqlens(cu_seqlens)
    shaped = packed_causal_mask(4, 4, impl="magi_attention", device="cpu", cu_seqlens=cu_seqlens)
    torch.testing.assert_close(shaped.q_ranges, built.q_ranges)
    torch.testing.assert_close(shaped.k_ranges, built.k_ranges)
    torch.testing.assert_close(shaped.attn_type_map, built.attn_type_map)


def test_magi_sliding_window_is_unsupported():
    with pytest.raises(ValueError, match="sliding windows in ranges"):
        sliding_window_mask(4, 4, impl="magi_attention", device="cpu", sliding_window=2)
