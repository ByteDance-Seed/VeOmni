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

import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from veomni.ops.kernels import attention as veomni_attention
from veomni.ops.kernels.attention import flash as flash_backend
from veomni.ops.kernels.attention import ulysses as attention_ulysses


_FLASH_IMPLEMENTATIONS = (
    ("veomni_flash_attention_2_with_sp", "flash_attention_2"),
    ("veomni_flash_attention_3_with_sp", "flash_attention_3"),
    ("veomni_flash_attention_4_with_sp", "veomni_flash_attention_4_with_sp"),
)


class _FakeAttentionModule(nn.Module):
    def __init__(self, implementation: str):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation=implementation)
        self.is_causal = True
        self.layer_idx = 7
        self.proj = nn.Linear(4, 4)


def test_flash_attention_forward_public_signature_is_stable():
    signature = inspect.signature(veomni_attention.flash_attention_forward)

    assert list(signature.parameters) == [
        "module",
        "query",
        "key",
        "value",
        "attention_mask",
        "dropout",
        "scaling",
        "sliding_window",
        "softcap",
        "skip_ulysses",
        "kwargs",
    ]
    assert signature.parameters["attention_mask"].default is inspect.Parameter.empty
    assert signature.parameters["dropout"].default == 0.0
    assert signature.parameters["scaling"].default is None
    assert signature.parameters["sliding_window"].default is None
    assert signature.parameters["softcap"].default is None
    assert signature.parameters["skip_ulysses"].default is False
    assert signature.parameters["kwargs"].kind is inspect.Parameter.VAR_KEYWORD


@pytest.mark.parametrize(("implementation", "expected_backend"), _FLASH_IMPLEMENTATIONS)
def test_flash_attention_forward_preserves_layout_and_backend_contract(
    monkeypatch,
    implementation,
    expected_backend,
):
    captured = {}

    def replacement_backend(query, key, value, attention_mask, **kwargs):
        captured.update(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            kwargs=kwargs,
        )
        return query + 1

    monkeypatch.setattr(flash_backend, "_flash_attention_forward", replacement_backend)
    monkeypatch.setattr(
        flash_backend,
        "get_parallel_state",
        lambda: SimpleNamespace(ulysses_enabled=False),
    )

    module = _FakeAttentionModule(implementation)
    query = torch.randn(2, 4, 3, 4, dtype=torch.float16)
    key = torch.randn(2, 2, 3, 4, dtype=torch.float16)
    value = torch.randn(2, 2, 3, 4, dtype=torch.float16)
    attention_mask = torch.ones(2, 1, 3, 3, dtype=torch.bool)
    marker = object()

    output, attention_weights = veomni_attention.flash_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=0.25,
        scaling=0.5,
        sliding_window=16,
        softcap=30.0,
        is_causal=False,
        contract_marker=marker,
    )

    torch.testing.assert_close(captured["query"], query.transpose(1, 2))
    torch.testing.assert_close(captured["key"], key.transpose(1, 2))
    torch.testing.assert_close(captured["value"], value.transpose(1, 2))
    assert captured["attention_mask"] is attention_mask

    backend_kwargs = captured["kwargs"]
    assert backend_kwargs["query_length"] == query.shape[2]
    assert backend_kwargs["is_causal"] is False
    assert backend_kwargs["dropout"] == 0.25
    assert backend_kwargs["softmax_scale"] == 0.5
    assert backend_kwargs["sliding_window"] == 16
    assert backend_kwargs["softcap"] == 30.0
    assert backend_kwargs["use_top_left_mask"] is False
    assert backend_kwargs["target_dtype"] is None
    assert backend_kwargs["attn_implementation"] == expected_backend
    assert backend_kwargs["layer_idx"] == module.layer_idx
    assert backend_kwargs["contract_marker"] is marker

    assert output.shape == (2, 3, 4, 4)
    torch.testing.assert_close(output, query.transpose(1, 2) + 1)
    assert attention_weights is None


@pytest.mark.parametrize(
    ("batch_size", "seq_dim", "head_dim"),
    [(1, 0, 1), (2, 1, 2)],
)
def test_ulysses_helpers_preserve_flash_layout(monkeypatch, batch_size, seq_dim, head_dim):
    exchanges = []

    def fake_gather_seq(tensor, *, seq_dim, head_dim, group):
        exchanges.append(("prepare", tensor.shape, seq_dim, head_dim, group))
        return tensor

    def fake_gather_heads(tensor, *, seq_dim, head_dim, group):
        exchanges.append(("restore", tensor.shape, seq_dim, head_dim, group))
        return tensor

    monkeypatch.setattr(attention_ulysses, "gather_seq_scatter_heads", fake_gather_seq)
    monkeypatch.setattr(attention_ulysses, "gather_heads_scatter_seq", fake_gather_heads)
    group = object()
    query = torch.randn(batch_size, 5, 4, 8)
    key = torch.randn(batch_size, 5, 1, 8)
    value = torch.randn(batch_size, 5, 1, 8)

    prepared_query, prepared_key, prepared_value, query_heads = attention_ulysses.prepare_ulysses_qkv(
        query, key, value, group=group, ulysses_size=2
    )
    restored = attention_ulysses.restore_ulysses_output(prepared_query[:, :, :2], group=group)

    assert query_heads == 4
    torch.testing.assert_close(prepared_key, key.repeat_interleave(2, dim=2))
    torch.testing.assert_close(prepared_value, value.repeat_interleave(2, dim=2))
    assert [item[0] for item in exchanges] == ["prepare", "prepare", "prepare", "restore"]
    assert all(item[2:] == (seq_dim, head_dim, group) for item in exchanges)
    assert restored.shape == (batch_size, 5, 2, 8)


def test_ulysses_head_auxiliary_slices_global_vector_by_rank(monkeypatch):
    monkeypatch.setattr(attention_ulysses.dist, "get_rank", lambda group: 1)
    auxiliary = torch.arange(4)

    sliced = attention_ulysses.slice_ulysses_head_auxiliary(
        auxiliary,
        query_head_count=4,
        local_query_head_count=2,
        group=object(),
    )

    torch.testing.assert_close(sliced, torch.tensor([2, 3]))


def test_flash_attention_delegates_active_ulysses_to_shared_helpers(monkeypatch):
    group = object()
    state = SimpleNamespace(ulysses_enabled=True, ulysses_group=group, ulysses_size=2)
    calls = []

    def fake_prepare(query, key, value, *, group, ulysses_size):
        calls.append(("prepare", query, key, value, group, ulysses_size))
        return query, key, value, 4

    def fake_slice(auxiliary, *, query_head_count, local_query_head_count, group):
        calls.append(("slice", auxiliary, query_head_count, local_query_head_count, group))
        return auxiliary[:local_query_head_count]

    def fake_restore(output, *, group):
        calls.append(("restore", output, group))
        return output

    def fake_flash(query, key, value, attention_mask, **kwargs):
        calls.append(("backend", query, key, value, attention_mask, kwargs))
        return query

    monkeypatch.setattr(flash_backend, "get_parallel_state", lambda: state)
    monkeypatch.setattr(flash_backend, "prepare_ulysses_qkv", fake_prepare)
    monkeypatch.setattr(flash_backend, "slice_ulysses_head_auxiliary", fake_slice)
    monkeypatch.setattr(flash_backend, "restore_ulysses_output", fake_restore)
    monkeypatch.setattr(flash_backend, "_flash_attention_forward", fake_flash)
    query = torch.randn(1, 4, 5, 8, dtype=torch.float16)
    key = torch.randn(1, 2, 5, 8, dtype=torch.float16)
    value = torch.randn(1, 2, 5, 8, dtype=torch.float16)
    auxiliary = torch.arange(4, dtype=torch.float16)

    output, _ = flash_backend.flash_attention_forward(
        _FakeAttentionModule("veomni_flash_attention_2_with_sp"),
        query,
        key,
        value,
        attention_mask=None,
        s_aux=auxiliary,
    )

    assert [call[0] for call in calls] == ["prepare", "slice", "backend", "restore"]
    assert calls[0][1].shape == (1, 5, 4, 8)
    assert calls[0][2].shape == (1, 5, 2, 8)
    assert calls[0][4:] == (group, 2)
    torch.testing.assert_close(calls[2][-1]["s_aux"], auxiliary)
    assert output.shape == (1, 5, 4, 8)
