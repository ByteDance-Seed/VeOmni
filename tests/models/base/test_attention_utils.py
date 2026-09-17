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

"""CPU tests for packed SDPA/eager attention helpers."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.parametrize("impl", ["eager", "sdpa", "veomni_sdpa"])
def test_drop_packed_attention_metadata_strips_gdn_keys_for_sdpa_and_eager(impl):
    from veomni.models.utils.attention_utils import drop_packed_attention_metadata

    kwargs = {
        "cu_seq_lens_q": torch.tensor([0, 32], dtype=torch.int32),
        "max_length_q": 32,
        "keep": 1,
    }
    filtered = drop_packed_attention_metadata(kwargs, impl=impl)
    assert "cu_seq_lens_q" not in filtered
    assert "max_length_q" not in filtered
    assert filtered["keep"] == 1
    assert "cu_seq_lens_q" in kwargs


def test_prepare_dense_attention_inputs_builds_packed_mask_for_multi_segment():
    from veomni.models.utils.attention_utils import prepare_dense_attention_inputs

    hidden = torch.randn(1, 8, 4)
    kwargs = {"cu_seq_lens_q": torch.tensor([0, 4, 8], dtype=torch.int32), "keep": 1}
    attention_mask = torch.ones(1, 8, dtype=torch.long)
    filtered, packed_mask = prepare_dense_attention_inputs(
        kwargs, impl="veomni_sdpa", attention_mask=attention_mask, hidden_states=hidden
    )
    assert "cu_seq_lens_q" not in filtered
    assert filtered["keep"] == 1
    assert packed_mask is not None
    assert packed_mask.shape[-2:] == (8, 8)
    # Token 4 (start of sample 1) must not see token 0 (sample 0).
    q4_k0 = packed_mask[0, 0, 4, 0] if packed_mask.dtype == torch.bool else packed_mask[0, 0, 4, 0]
    if packed_mask.dtype == torch.bool:
        assert not bool(q4_k0)
    else:
        assert q4_k0 < 0


def _mask_kept(mask: torch.Tensor, query: int, key: int) -> bool:
    value = mask[0, 0, query, key] if mask.ndim == 4 else mask[0, query, key]
    if mask.dtype == torch.bool:
        return bool(value)
    return float(value) >= 0


def test_prepare_dense_attention_inputs_strips_single_segment_without_replacing_mask():
    from veomni.models.utils.attention_utils import prepare_dense_attention_inputs

    hidden = torch.randn(1, 8, 4)
    kwargs = {"cu_seq_lens_q": torch.tensor([0, 8], dtype=torch.int32)}
    attention_mask = torch.ones(1, 8, dtype=torch.long)
    filtered, mask = prepare_dense_attention_inputs(
        kwargs, impl="sdpa", attention_mask=attention_mask, hidden_states=hidden
    )
    assert "cu_seq_lens_q" not in filtered
    assert mask is attention_mask


def test_prepare_dense_attention_inputs_keeps_2d_padding_inside_packed_sample():
    from veomni.models.utils.attention_utils import prepare_dense_attention_inputs

    hidden = torch.randn(1, 8, 4)
    kwargs = {"cu_seq_lens_q": torch.tensor([0, 4, 8], dtype=torch.int32)}
    attention_mask = torch.ones(1, 8, dtype=torch.long)
    attention_mask[0, 1] = 0
    _, packed_mask = prepare_dense_attention_inputs(
        kwargs, impl="sdpa", attention_mask=attention_mask, hidden_states=hidden
    )
    assert packed_mask is not None
    assert not _mask_kept(packed_mask, 2, 1)
    assert not _mask_kept(packed_mask, 4, 0)
    assert _mask_kept(packed_mask, 2, 0)


@pytest.mark.parametrize("impl", ["sdpa", "eager", "veomni_sdpa"])
def test_prepare_dense_attention_inputs_merges_4d_padding_with_packed_isolation(impl):
    from veomni.models.utils.attention_utils import prepare_dense_attention_inputs

    hidden = torch.randn(1, 8, 4)
    kwargs = {"cu_seq_lens_q": torch.tensor([0, 4, 8], dtype=torch.int32)}
    query = torch.arange(8)[:, None]
    key = torch.arange(8)[None, :]
    existing = (key <= query).view(1, 1, 8, 8).clone()
    existing[..., :, 1] = False
    _, packed_mask = prepare_dense_attention_inputs(kwargs, impl=impl, attention_mask=existing, hidden_states=hidden)
    assert packed_mask is not None
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
def test_prepare_dense_attention_inputs_keeps_positive_additive_bias(impl):
    from veomni.models.utils.attention_utils import prepare_dense_attention_inputs

    hidden = torch.randn(1, 4, 4)
    kwargs = {"cu_seq_lens_q": torch.tensor([0, 2, 4], dtype=torch.int32)}
    query = torch.arange(4, dtype=torch.long)[:, None]
    key = torch.arange(4, dtype=torch.long)[None, :]
    allowed = key <= query
    existing = torch.where(allowed, torch.zeros((), dtype=torch.float32), torch.finfo(torch.float32).min)
    existing = existing.view(1, 1, 4, 4).clone()
    existing[0, 0, 1, 0] = 2.0
    _, packed_mask = prepare_dense_attention_inputs(kwargs, impl=impl, attention_mask=existing, hidden_states=hidden)
    assert packed_mask is not None
    assert packed_mask.dtype.is_floating_point
    torch.testing.assert_close(packed_mask[0, 0, 1, 0], existing.new_tensor(2.0))
    assert not _mask_kept(packed_mask, 2, 0)
    assert _mask_kept(packed_mask, 3, 2)


def _eager_attn(query: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scores = torch.matmul(query, query.transpose(-2, -1)) + mask
    return torch.matmul(torch.softmax(scores, dim=-1), values)


@pytest.mark.parametrize("sample_start", [0, 2])
@pytest.mark.parametrize("mask_kind", ["bool", "additive"])
def test_packed_eager_fully_masked_row_isolates_other_sample(sample_start, mask_kind):
    from veomni.models.utils.attention_utils import prepare_dense_attention_inputs

    query = torch.zeros(1, 1, 4, 1)
    values = torch.tensor([0.0, 1.0, 10.0, 20.0]).view(1, 1, 4, 1).requires_grad_()
    allowed = torch.ones(4, 4, dtype=torch.bool).tril()
    if mask_kind == "bool":
        existing = allowed.view(1, 1, 4, 4).clone()
        existing[:, :, sample_start, sample_start : sample_start + 2] = False
    else:
        existing = torch.zeros(1, 1, 4, 4).masked_fill(~allowed, torch.finfo(torch.float32).min)
        existing[:, :, sample_start, sample_start : sample_start + 2] = torch.finfo(torch.float32).min
    _, mask = prepare_dense_attention_inputs(
        {"cu_seq_lens_q": torch.tensor([0, 2, 4], dtype=torch.int32)},
        impl="eager",
        attention_mask=existing,
        hidden_states=torch.zeros(1, 4, 1),
    )
    other_start = 2 - sample_start
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
    from veomni.models.utils.attention_utils import prepare_dense_attention_inputs

    hidden = torch.zeros(1, 4, 1)
    kwargs = {"cu_seq_lens_q": torch.tensor([0, 2, 4], dtype=torch.int32)}
    attention_mask = torch.ones(1, 4, dtype=torch.long)
    attention_mask[0, :2] = 0
    _, mask = prepare_dense_attention_inputs(kwargs, impl="eager", attention_mask=attention_mask, hidden_states=hidden)
    assert mask is not None
    assert torch.isneginf(mask[0, 0, 0, 2:4]).all()
    assert not _mask_kept(mask, 0, 0)
    assert _mask_kept(mask, 3, 2)


@pytest.mark.parametrize("impl", ["sdpa", "veomni_sdpa"])
@pytest.mark.parametrize("sample_start", [0, 2])
def test_packed_sdpa_fully_masked_row_matches_separate_outputs_and_grads(impl, sample_start):
    from veomni.models.utils.attention_utils import prepare_dense_attention_inputs

    query = torch.zeros(1, 1, 4, 1)
    values = torch.tensor([0.0, 1.0, 10.0, 20.0]).view(1, 1, 4, 1).requires_grad_()
    separate_values = values.detach().clone().requires_grad_()
    existing = torch.zeros(1, 1, 4, 4).masked_fill(~torch.ones(4, 4, dtype=torch.bool).tril(), float("-inf"))
    # Mask this query's own sample; earlier samples remain visible in the
    # incoming triangle and must be blocked by packed isolation itself.
    existing[:, :, sample_start, sample_start : sample_start + 2] = float("-inf")
    _, mask = prepare_dense_attention_inputs(
        {"cu_seq_lens_q": torch.tensor([0, 2, 4], dtype=torch.int32)},
        impl=impl,
        attention_mask=existing,
        hidden_states=torch.zeros(1, 4, 1),
    )
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


def test_prepare_dense_attention_inputs_rejects_cached_packed_sequences():
    from veomni.models.utils.attention_utils import prepare_dense_attention_inputs

    hidden = torch.randn(1, 8, 4)
    kwargs = {"cu_seq_lens_q": torch.tensor([0, 4, 8], dtype=torch.int32)}
    attention_mask = torch.ones(1, 1, 8, 16)
    with pytest.raises(ValueError, match="does not support cached sequences"):
        prepare_dense_attention_inputs(kwargs, impl="sdpa", attention_mask=attention_mask, hidden_states=hidden)


def test_drop_packed_attention_metadata_keeps_keys_for_flash():
    from veomni.models.utils.attention_utils import drop_packed_attention_metadata

    kwargs = {"cu_seq_lens_q": torch.tensor([0, 32], dtype=torch.int32)}
    assert drop_packed_attention_metadata(kwargs, impl="flash_attention_2") is kwargs
