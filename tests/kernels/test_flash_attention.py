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

"""Flash attention adapter contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from veomni.kernels._kernels.attention.standard import flash as flash_backend


class _FakeAttentionModule(nn.Module):
    def __init__(self, implementation: str):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation=implementation)
        self.is_causal = True
        self.layer_idx = 7
        self.proj = nn.Linear(4, 4)


@pytest.mark.parametrize(
    ("implementation", "expected_backend"),
    [
        ("veomni_flash_attention_2", "flash_attention_2"),
        ("veomni_flash_attention_3", "flash_attention_3"),
        ("veomni_flash_attention_4", "veomni_flash_attention_4"),
    ],
)
def test_flash_attention_preserves_layout_and_backend_contract(monkeypatch, implementation, expected_backend):
    captured = {}

    def replacement_backend(query, key, value, attention_mask, **kwargs):
        captured.update(query=query, key=key, value=value, attention_mask=attention_mask, kwargs=kwargs)
        return query + 1

    monkeypatch.setattr(flash_backend, "_flash_attention_forward", replacement_backend)
    monkeypatch.setattr(flash_backend, "should_apply_ulysses", lambda: False)

    module = _FakeAttentionModule(implementation)
    query = torch.randn(2, 4, 3, 4, dtype=torch.float16)
    key = torch.randn(2, 2, 3, 4, dtype=torch.float16)
    value = torch.randn(2, 2, 3, 4, dtype=torch.float16)
    attention_mask = torch.ones(2, 1, 3, 3, dtype=torch.bool)
    marker = object()

    output, attention_weights = flash_backend.flash_attention_forward(
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
    assert backend_kwargs["attn_implementation"] == expected_backend
    assert backend_kwargs["layer_idx"] == module.layer_idx
    assert backend_kwargs["contract_marker"] is marker
    assert output.shape == (2, 3, 4, 4)
    torch.testing.assert_close(output, query.transpose(1, 2) + 1)
    assert attention_weights is None


def test_flash_attention_delegates_active_ulysses_to_shared_helpers(monkeypatch):
    group = object()
    state = SimpleNamespace(ulysses_group=group, ulysses_size=2)
    calls = []

    def fake_prepare(query, key, value, *, group, ulysses_size):
        calls.append(("prepare", query, key, value, group, ulysses_size))
        return query[:, :, :2], key[:, :, :1], value[:, :, :1], 4

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
    monkeypatch.setattr(flash_backend, "should_apply_ulysses", lambda: True)
    monkeypatch.setattr(flash_backend, "prepare_ulysses_qkv", fake_prepare)
    monkeypatch.setattr(flash_backend, "slice_ulysses_head_auxiliary", fake_slice)
    monkeypatch.setattr(flash_backend, "restore_ulysses_output", fake_restore)
    monkeypatch.setattr(flash_backend, "_flash_attention_forward", fake_flash)
    query = torch.randn(1, 4, 5, 8, dtype=torch.float16)
    auxiliary = torch.arange(4, dtype=torch.float16)

    output, _ = flash_backend.flash_attention_forward(
        _FakeAttentionModule("veomni_flash_attention_2"),
        query,
        query[:, :2],
        query[:, :2],
        attention_mask=None,
        s_aux=auxiliary,
    )

    assert [call[0] for call in calls] == ["prepare", "slice", "backend", "restore"]
    assert calls[0][1].shape == (1, 5, 4, 8)
    assert calls[0][4:] == (group, 2)
    torch.testing.assert_close(calls[2][-1]["s_aux"], auxiliary[:2])
    assert output.shape == (1, 5, 2, 8)
