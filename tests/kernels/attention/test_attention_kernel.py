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

"""Install and registry contract for ``veomni_*`` attention names."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers.integrations.flex_attention import flex_attention_forward as hf_flex_attention_forward
from transformers.integrations.sdpa_attention import sdpa_attention_forward as hf_sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from veomni.kernels import VeomniKernel, resolve_kernel
from veomni.kernels._kernels.attention import lookup
from veomni.kernels._kernels.attention import ulysses as ulysses_backend
from veomni.kernels._kernels.attention.install import apply_veomni_attention_patch
from veomni.kernels._kernels.attention.standard.flash import flash_attention_forward
from veomni.kernels._kernels.attention.standard.flex import flex_attention_forward
from veomni.kernels._kernels.attention.standard.magi import magi_attention_forward
from veomni.kernels._kernels.attention.standard.sage import sage_attention_forward
from veomni.kernels._kernels.attention.standard.sdpa import sdpa_attention_forward
from veomni.kernels._kernels.attention.ulysses import should_apply_ulysses


_VEOMNI_FORWARDS = {
    "veomni_flash_attention_2": flash_attention_forward,
    "veomni_flash_attention_3": flash_attention_forward,
    "veomni_flash_attention_4": flash_attention_forward,
    "veomni_flex_attention": flex_attention_forward,
    "veomni_magi_attention": magi_attention_forward,
    "veomni_sage_attention": sage_attention_forward,
    "veomni_sdpa": sdpa_attention_forward,
}


def test_standard_rows_are_registered():
    for impl in (
        "eager",
        "sdpa",
        "flash_attention_2",
        *_VEOMNI_FORWARDS,
    ):
        entry = resolve_kernel("attention", "standard", impl)
        assert entry.wrapper is not None


def test_veomni_names_register_on_hf_dict_without_overwriting_stock():
    for name, forward in _VEOMNI_FORWARDS.items():
        assert ALL_ATTENTION_FUNCTIONS[name] is forward
    assert ALL_ATTENTION_FUNCTIONS["flex_attention"] is hf_flex_attention_forward
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is hf_sdpa_attention_forward
    kernel = VeomniKernel("attention", "standard", "veomni_flash_attention_2")
    assert kernel.impl == "veomni_flash_attention_2"


def test_apply_veomni_attention_patch_is_idempotent():
    apply_veomni_attention_patch()
    apply_veomni_attention_patch()
    for name, forward in _VEOMNI_FORWARDS.items():
        assert ALL_ATTENTION_FUNCTIONS[name] is forward


def test_apply_kernel_patch_registers_kernels_names():
    from veomni.kernels import apply_kernel_patch

    apply_kernel_patch()
    for name, forward in _VEOMNI_FORWARDS.items():
        assert ALL_ATTENTION_FUNCTIONS[name] is forward


def eager_attention_forward(module, query, key, value, attention_mask, **kwargs):
    del module, key, value, attention_mask, kwargs
    return query.transpose(1, 2), "eager-local"


class _EagerAttentionModule(torch.nn.Module):
    """Lookup resolves ``eager_attention_forward`` from this test module."""


def test_lookup_eager_uses_module_local_forward():
    module = _EagerAttentionModule()
    query = torch.randn(2, 4, 3, 8)
    output, metadata = lookup("eager")(module, query, query, query, None, dropout=0.0, scaling=0.5)
    torch.testing.assert_close(output, query.transpose(1, 2))
    assert metadata == "eager-local"

    kernel = VeomniKernel("attention", "standard", "eager")
    kernel_output, kernel_metadata = kernel(module, query, query, query, None, dropout=0.0, scaling=0.5)
    torch.testing.assert_close(kernel_output, query.transpose(1, 2))
    assert kernel_metadata == "eager-local"


def test_lookup_dispatches_through_hf_dict(monkeypatch):
    captured = {}

    def replacement(module, query, key, value, attention_mask, **kwargs):
        captured.update(module=module, query=query, kwargs=kwargs)
        return query.transpose(1, 2) + 1, "attention-metadata"

    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS._global_mapping, "veomni_sdpa", replacement)
    module = SimpleNamespace(is_causal=True)
    query = torch.randn(2, 4, 3, 8)
    output, metadata = lookup("veomni_sdpa")(module, query, query, query, None, dropout=0.0, scaling=0.5)

    assert captured["module"] is module
    assert captured["query"] is query
    assert captured["kwargs"]["scaling"] == 0.5
    torch.testing.assert_close(output, query.transpose(1, 2) + 1)
    assert metadata == "attention-metadata"


def test_should_apply_ulysses_follows_parallel_state(monkeypatch):
    def _set_state(*, ulysses_size: int, async_enabled: bool) -> None:
        monkeypatch.setattr(
            ulysses_backend,
            "get_parallel_state",
            lambda: SimpleNamespace(ulysses_size=ulysses_size, async_enabled=async_enabled),
        )

    _set_state(ulysses_size=1, async_enabled=False)
    assert not should_apply_ulysses()
    _set_state(ulysses_size=2, async_enabled=False)
    assert should_apply_ulysses()
    assert not should_apply_ulysses(skip_ulysses=True)
    _set_state(ulysses_size=2, async_enabled=True)
    assert not should_apply_ulysses()


def test_ulysses_helpers_preserve_layout(monkeypatch):
    exchanges = []

    def fake_gather_seq(tensor, *, seq_dim, head_dim, group):
        exchanges.append(("prepare", tensor.shape, seq_dim, head_dim, group))
        return tensor

    def fake_gather_heads(tensor, *, seq_dim, head_dim, group):
        exchanges.append(("restore", tensor.shape, seq_dim, head_dim, group))
        return tensor

    monkeypatch.setattr(ulysses_backend, "gather_seq_scatter_heads", fake_gather_seq)
    monkeypatch.setattr(ulysses_backend, "gather_heads_scatter_seq", fake_gather_heads)
    group = object()
    query = torch.randn(2, 5, 4, 8)
    key = torch.randn(2, 5, 1, 8)
    value = torch.randn(2, 5, 1, 8)

    prepared_query, prepared_key, prepared_value, query_heads = ulysses_backend.prepare_ulysses_qkv(
        query, key, value, group=group, ulysses_size=2
    )
    restored = ulysses_backend.restore_ulysses_output(prepared_query[:, :, :2], group=group)

    assert query_heads == 4
    torch.testing.assert_close(prepared_key, key.repeat_interleave(2, dim=2))
    torch.testing.assert_close(prepared_value, value.repeat_interleave(2, dim=2))
    assert [item[0] for item in exchanges] == ["prepare", "prepare", "prepare", "restore"]
    assert restored.shape == (2, 5, 2, 8)


def test_ulysses_helpers_reject_nondivisible_key_value_heads():
    query = torch.randn(1, 5, 8, 8)
    key = torch.randn(1, 5, 3, 8)
    value = torch.randn(1, 5, 3, 8)
    with pytest.raises(AssertionError, match="must be divisible"):
        ulysses_backend.prepare_ulysses_qkv(query, key, value, group=object(), ulysses_size=4)


def test_ulysses_head_auxiliary_slices_global_vector_by_rank(monkeypatch):
    monkeypatch.setattr(ulysses_backend.dist, "get_rank", lambda group: 1)
    sliced = ulysses_backend.slice_ulysses_head_auxiliary(
        torch.arange(4),
        query_head_count=4,
        local_query_head_count=2,
        group=object(),
    )
    torch.testing.assert_close(sliced, torch.tensor([2, 3]))
