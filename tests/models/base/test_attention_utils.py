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
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for ``veomni.models.utils.mask_utils``."""

from __future__ import annotations

import pytest
import torch
from transformers import PreTrainedConfig

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.models.utils.mask_utils import create_causal_mask, create_sliding_window_causal_mask


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _mask_kept(mask: torch.Tensor, query: int, key: int) -> bool:
    value = mask[0, 0, query, key] if mask.ndim == 4 else mask[0, query, key]
    if mask.dtype == torch.bool:
        return bool(value)
    return float(value) >= 0


def test_create_causal_mask_builds_packed_mask_for_multi_segment():
    config = PreTrainedConfig()
    config._attn_implementation = "sdpa"
    hidden = torch.randn(1, 8, 4)
    attention_mask = torch.ones(1, 8, dtype=torch.long)
    packed_mask = create_causal_mask(
        config=config,
        inputs_embeds=hidden,
        attention_mask=attention_mask,
        past_key_values=None,
        cu_seq_lens_q=torch.tensor([0, 4, 8], dtype=torch.int32),
    )
    assert isinstance(packed_mask, torch.Tensor)
    assert packed_mask.shape[-2:] == (8, 8)
    # Token 4 (start of sample 1) must not see token 0 (sample 0).
    assert not _mask_kept(packed_mask, 4, 0)
    assert _mask_kept(packed_mask, 5, 4)


def test_create_causal_mask_delegates_single_segment_to_hf():
    config = PreTrainedConfig()
    config._attn_implementation = "sdpa"
    hidden = torch.randn(1, 8, 4)
    attention_mask = torch.ones(1, 8, dtype=torch.long)
    mask = create_causal_mask(
        config=config,
        inputs_embeds=hidden,
        attention_mask=attention_mask,
        past_key_values=None,
        cu_seq_lens_q=torch.tensor([0, 8], dtype=torch.int32),
    )
    assert mask is None


def test_create_causal_mask_keeps_2d_padding_inside_packed_sample():
    config = PreTrainedConfig()
    config._attn_implementation = "sdpa"
    hidden = torch.randn(1, 8, 4)
    attention_mask = torch.ones(1, 8, dtype=torch.long)
    attention_mask[0, 1] = 0
    packed_mask = create_causal_mask(
        config=config,
        inputs_embeds=hidden,
        attention_mask=attention_mask,
        past_key_values=None,
        cu_seq_lens_q=torch.tensor([0, 4, 8], dtype=torch.int32),
    )
    assert isinstance(packed_mask, torch.Tensor)
    assert not _mask_kept(packed_mask, 2, 1)
    assert not _mask_kept(packed_mask, 4, 0)
    assert _mask_kept(packed_mask, 2, 0)


@pytest.mark.parametrize("impl", ["sdpa", "eager"])
def test_create_causal_mask_merges_4d_padding_with_packed_isolation(impl):
    config = PreTrainedConfig()
    config._attn_implementation = impl
    hidden = torch.randn(1, 8, 4)
    query = torch.arange(8)[:, None]
    key = torch.arange(8)[None, :]
    existing = (key <= query).view(1, 1, 8, 8).clone()
    existing[..., :, 1] = False
    packed_mask = create_causal_mask(
        config=config,
        inputs_embeds=hidden,
        attention_mask=existing,
        past_key_values=None,
        cu_seq_lens_q=torch.tensor([0, 4, 8], dtype=torch.int32),
    )
    assert isinstance(packed_mask, torch.Tensor)
    if impl == "eager":
        assert packed_mask.dtype.is_floating_point
        assert torch.isneginf(packed_mask[0, 0, 4, 0])
    else:
        assert packed_mask.dtype == torch.bool
    assert not _mask_kept(packed_mask, 4, 0)
    assert not _mask_kept(packed_mask, 2, 1)
    assert _mask_kept(packed_mask, 2, 0)
    assert _mask_kept(packed_mask, 5, 4)


@pytest.mark.parametrize("impl", ["sdpa", "eager"])
def test_create_causal_mask_keeps_positive_additive_bias(impl):
    config = PreTrainedConfig()
    config._attn_implementation = impl
    hidden = torch.randn(1, 4, 4)
    query = torch.arange(4, dtype=torch.long)[:, None]
    key = torch.arange(4, dtype=torch.long)[None, :]
    allowed = key <= query
    existing = torch.where(allowed, torch.zeros((), dtype=torch.float32), torch.finfo(torch.float32).min)
    existing = existing.view(1, 1, 4, 4).clone()
    existing[0, 0, 1, 0] = 2.0
    packed_mask = create_causal_mask(
        config=config,
        inputs_embeds=hidden,
        attention_mask=existing,
        past_key_values=None,
        cu_seq_lens_q=torch.tensor([0, 2, 4], dtype=torch.int32),
    )
    assert isinstance(packed_mask, torch.Tensor)
    assert packed_mask.dtype.is_floating_point
    torch.testing.assert_close(packed_mask[0, 0, 1, 0], existing.new_tensor(2.0))
    assert not _mask_kept(packed_mask, 2, 0)
    assert _mask_kept(packed_mask, 3, 2)


def _eager_attn(query: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scores = torch.matmul(query, query.transpose(-2, -1)) + mask
    return torch.matmul(torch.softmax(scores, dim=-1), values)


@pytest.mark.parametrize("mask_kind", ["bool", "additive"])
def test_packed_eager_fully_masked_row_isolates_other_sample(mask_kind):
    config = PreTrainedConfig()
    config._attn_implementation = "eager"
    sample_start = 2
    query = torch.zeros(1, 1, 4, 1)
    values = torch.tensor([0.0, 1.0, 10.0, 20.0]).view(1, 1, 4, 1).requires_grad_()
    allowed = torch.ones(4, 4, dtype=torch.bool).tril()
    if mask_kind == "bool":
        existing = allowed.view(1, 1, 4, 4).clone()
        existing[:, :, sample_start, sample_start : sample_start + 2] = False
    else:
        existing = torch.zeros(1, 1, 4, 4).masked_fill(~allowed, torch.finfo(torch.float32).min)
        existing[:, :, sample_start, sample_start : sample_start + 2] = torch.finfo(torch.float32).min
    mask = create_causal_mask(
        config=config,
        inputs_embeds=torch.zeros(1, 4, 1),
        attention_mask=existing,
        past_key_values=None,
        cu_seq_lens_q=torch.tensor([0, 2, 4], dtype=torch.int32),
    )
    assert isinstance(mask, torch.Tensor)
    other_start = 0
    assert torch.isneginf(mask[0, 0, sample_start, other_start : other_start + 2]).all()
    actual = _eager_attn(query, values, mask)
    actual[:, :, sample_start : sample_start + 2].sum().backward()
    other_grad = values.grad[:, :, other_start : other_start + 2]
    torch.testing.assert_close(other_grad, torch.zeros_like(other_grad))
    perturbed = values.detach().clone()
    perturbed[:, :, other_start : other_start + 2] += 100
    actual_perturbed = _eager_attn(query, perturbed, mask)
    torch.testing.assert_close(
        actual.detach()[:, :, sample_start : sample_start + 2],
        actual_perturbed[:, :, sample_start : sample_start + 2],
    )


def test_packed_eager_fully_padded_sample_uses_neg_inf_isolation():
    config = PreTrainedConfig()
    config._attn_implementation = "eager"
    hidden = torch.zeros(1, 4, 1)
    attention_mask = torch.ones(1, 4, dtype=torch.long)
    attention_mask[0, :2] = 0
    mask = create_causal_mask(
        config=config,
        inputs_embeds=hidden,
        attention_mask=attention_mask,
        past_key_values=None,
        cu_seq_lens_q=torch.tensor([0, 2, 4], dtype=torch.int32),
    )
    assert isinstance(mask, torch.Tensor)
    assert torch.isneginf(mask[0, 0, 0, 2:4]).all()
    assert torch.isneginf(mask[0, 0, 2, 0:2]).all()
    assert not _mask_kept(mask, 0, 0)
    assert _mask_kept(mask, 3, 2)


def test_packed_sdpa_fully_masked_row_matches_separate_outputs_and_grads():
    config = PreTrainedConfig()
    config._attn_implementation = "sdpa"
    sample_start = 2
    query = torch.zeros(1, 1, 4, 1)
    values = torch.tensor([0.0, 1.0, 10.0, 20.0]).view(1, 1, 4, 1).requires_grad_()
    separate_values = values.detach().clone().requires_grad_()
    existing = torch.zeros(1, 1, 4, 4).masked_fill(~torch.ones(4, 4, dtype=torch.bool).tril(), float("-inf"))
    # Mask this query's own sample; earlier samples remain visible in the
    # incoming triangle and must be blocked by packed isolation itself.
    existing[:, :, sample_start, sample_start : sample_start + 2] = float("-inf")
    mask = create_causal_mask(
        config=config,
        inputs_embeds=torch.zeros(1, 4, 1),
        attention_mask=existing,
        past_key_values=None,
        cu_seq_lens_q=torch.tensor([0, 2, 4], dtype=torch.int32),
    )
    assert isinstance(mask, torch.Tensor)
    actual = torch.nn.functional.scaled_dot_product_attention(query, query, values, attn_mask=mask)
    expected = torch.cat(
        [
            torch.nn.functional.scaled_dot_product_attention(
                query[:, :, start : start + 2],
                query[:, :, start : start + 2],
                separate_values[:, :, start : start + 2],
                attn_mask=existing[:, :, start : start + 2, start : start + 2],
            )
            for start in (0, 2)
        ],
        dim=2,
    )
    torch.testing.assert_close(actual, expected)
    assert actual[0, 0, sample_start, 0].item() == 0.0
    # A loss on either sample must not backpropagate into the other sample.
    actual[:, :, sample_start : sample_start + 2].sum().backward()
    expected[:, :, sample_start : sample_start + 2].sum().backward()
    torch.testing.assert_close(values.grad, separate_values.grad)
    other_start = 2 - sample_start
    other_grad = values.grad[:, :, other_start : other_start + 2]
    torch.testing.assert_close(other_grad, torch.zeros_like(other_grad))


def test_create_causal_mask_rejects_cached_packed_sequences():
    config = PreTrainedConfig()
    config._attn_implementation = "sdpa"
    hidden = torch.randn(1, 8, 4)
    attention_mask = torch.ones(1, 1, 8, 16)
    with pytest.raises(ValueError, match="does not support cached sequences"):
        create_causal_mask(
            config=config,
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            past_key_values=None,
            cu_seq_lens_q=torch.tensor([0, 4, 8], dtype=torch.int32),
        )


def test_create_causal_mask_with_ulysses_uses_post_ulysses_lengths_without_double_expansion(monkeypatch):
    """Under Ulysses SP, rank-local inputs_embeds (4 tokens) with a global 8x8 mask or 2D mask
    must resolve to the post-Ulysses 8x8 mask without raising or double-expanding to 16x16."""
    from types import SimpleNamespace

    from veomni.ops.kernels.attention import ulysses as ulysses_mod

    monkeypatch.setattr(ulysses_mod, "get_parallel_state", lambda: SimpleNamespace(ulysses_size=2))
    config = PreTrainedConfig()
    config._attn_implementation = "veomni_sdpa"
    local_hidden = torch.randn(1, 4, 8)
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)

    # Case 1: existing 4D global mask (1, 1, 8, 8)
    query = torch.arange(8)[:, None]
    key = torch.arange(8)[None, :]
    existing_4d = (key <= query).view(1, 1, 8, 8).clone()
    existing_4d[..., :, 7] = False  # last token padded
    mask_4d = create_causal_mask(
        config=config,
        inputs_embeds=local_hidden,
        attention_mask=existing_4d,
        past_key_values=None,
        cu_seq_lens_q=cu_seqlens,
    )
    assert isinstance(mask_4d, torch.Tensor) and mask_4d.shape == (1, 1, 8, 8)
    assert not _mask_kept(mask_4d, 3, 0)  # sample 1 isolated from sample 0
    assert _mask_kept(mask_4d, 4, 3)  # within sample 1 allowed
    assert not _mask_kept(mask_4d, 7, 7)  # padding preserved

    # Case 2: 2D full-sequence padding mask (1, 8)
    padding_2d = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.long)
    mask_2d = create_causal_mask(
        config=config,
        inputs_embeds=local_hidden,
        attention_mask=padding_2d,
        past_key_values=None,
        cu_seq_lens_q=cu_seqlens,
    )
    assert isinstance(mask_2d, torch.Tensor) and mask_2d.shape == (1, 1, 8, 8)
    torch.testing.assert_close(mask_2d, mask_4d)

    # Case 3: skip_ulysses=True keeps local 4x4 dimensions
    local_cu = torch.tensor([0, 2, 4], dtype=torch.int32)
    mask_skip = create_causal_mask(
        config=config,
        inputs_embeds=local_hidden,
        attention_mask=None,
        past_key_values=None,
        cu_seq_lens_q=local_cu,
        skip_ulysses=True,
    )
    assert isinstance(mask_skip, torch.Tensor) and mask_skip.shape == (1, 1, 4, 4)
    assert not _mask_kept(mask_skip, 2, 0)


@pytest.mark.parametrize("impl", ["eager", "sdpa", "veomni_sdpa", "veomni_flex_attention", "veomni_magi_attention"])
def test_mask_utils_create_causal_mask_isolates_packed_samples_with_2d_all_ones_mask(impl):
    """mask_utils.create_causal_mask routes packed cu_seq_lens_q through veomni.ops.mask even when
    attention_mask is a 2D all-ones tensor."""
    from torch.nn.attention.flex_attention import BlockMask

    from veomni.ops.mask import MagiAttentionMask

    config = PreTrainedConfig()
    config._attn_implementation = impl
    inputs_embeds = torch.randn(1, 8, 16)
    attention_mask = torch.ones(1, 8, dtype=torch.long)
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]])
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)

    mask = create_causal_mask(
        config=config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=position_ids,
        cu_seq_lens_q=cu_seqlens,
        cu_seq_lens_k=cu_seqlens,
    )

    if impl == "veomni_flex_attention":
        assert isinstance(mask, BlockMask)
        assert mask.shape == (1, 1, 8, 8)
        zero = torch.tensor(0)
        assert not mask.mask_mod(zero, zero, torch.tensor(3), torch.tensor(2))
        assert mask.mask_mod(zero, zero, torch.tensor(4), torch.tensor(3))
    elif impl == "veomni_magi_attention":
        assert isinstance(mask, MagiAttentionMask)
        assert mask.q_ranges.tolist() == [[0, 3], [3, 8]]
        assert mask.k_ranges.tolist() == [[0, 3], [3, 8]]
    else:
        assert isinstance(mask, torch.Tensor)
        assert mask.shape == (1, 1, 8, 8)
        assert not _mask_kept(mask, 3, 2)
        assert _mask_kept(mask, 4, 3)


def test_mask_utils_create_sliding_window_causal_mask_isolates_packed_samples():
    from torch.nn.attention.flex_attention import BlockMask

    config = PreTrainedConfig()
    config._attn_implementation = "veomni_flex_attention"
    config.sliding_window = 3
    inputs_embeds = torch.randn(1, 8, 16)
    attention_mask = torch.ones(1, 8, dtype=torch.long)
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]])
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)

    mask = create_sliding_window_causal_mask(
        config=config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=position_ids,
        cu_seq_lens_q=cu_seqlens,
    )
    assert isinstance(mask, BlockMask)
    zero = torch.tensor(0)
    assert not mask.mask_mod(zero, zero, torch.tensor(3), torch.tensor(2))
    assert not mask.mask_mod(zero, zero, torch.tensor(7), torch.tensor(3))
    assert mask.mask_mod(zero, zero, torch.tensor(7), torch.tensor(5))


@pytest.mark.parametrize("impl", ["sage_attention", "veomni_sage_attention"])
def test_mask_utils_keeps_sage_attention_on_flash_mask_path(impl):
    config = PreTrainedConfig()
    config._attn_implementation = impl
    inputs_embeds = torch.randn(1, 8, 16)
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)

    all_ones = torch.ones(1, 8, dtype=torch.long)
    mask_valid = create_causal_mask(
        config=config,
        inputs_embeds=inputs_embeds,
        attention_mask=all_ones,
        past_key_values=None,
        cu_seq_lens_q=cu_seqlens,
    )
    assert mask_valid is None

    padded = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.long)
    mask_padded = create_causal_mask(
        config=config,
        inputs_embeds=inputs_embeds,
        attention_mask=padded,
        past_key_values=None,
        cu_seq_lens_q=cu_seqlens,
    )
    assert isinstance(mask_padded, torch.Tensor) and mask_padded.shape == (1, 8)
