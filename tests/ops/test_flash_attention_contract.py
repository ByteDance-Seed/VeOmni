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
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from veomni.ops.kernels import attention as veomni_attention
from veomni.ops.kernels.attention import flash as flash_backend


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


def test_fused_attention_forward_matches_the_public_attention_signature():
    flash_signature = inspect.signature(veomni_attention.flash_attention_forward)
    fused_signature = inspect.signature(veomni_attention.fused_attention_forward)

    assert list(fused_signature.parameters) == list(flash_signature.parameters)
    for name, flash_parameter in flash_signature.parameters.items():
        fused_parameter = fused_signature.parameters[name]
        assert fused_parameter.kind is flash_parameter.kind
        assert fused_parameter.default == flash_parameter.default


def test_apply_veomni_attention_patch_registers_all_flash_implementations(monkeypatch):
    patch_calls = []
    monkeypatch.setattr(
        veomni_attention,
        "patch_transformers_hub_kernel_loader_for_veomni",
        lambda: patch_calls.append(True),
    )

    veomni_attention.apply_veomni_attention_patch()

    assert patch_calls == [True]
    for implementation, _ in _FLASH_IMPLEMENTATIONS:
        assert ALL_ATTENTION_FUNCTIONS[implementation] is veomni_attention.fused_attention_forward


@pytest.mark.parametrize("implementation", [item[0] for item in _FLASH_IMPLEMENTATIONS])
def test_fused_attention_forward_dispatches_to_the_selected_flash_adapter(monkeypatch, implementation):
    captured = {}

    def replacement_adapter(module, query, key, value, attention_mask, **kwargs):
        captured.update(
            module=module,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            kwargs=kwargs,
        )
        return query.transpose(1, 2) + 1, "attention-metadata"

    monkeypatch.setitem(veomni_attention._ATTENTION_FORWARD_DISPATCH, implementation, replacement_adapter)

    module = _FakeAttentionModule(implementation)
    query = torch.randn(2, 4, 3, 4, dtype=torch.float16)
    key = torch.randn(2, 2, 3, 4, dtype=torch.float16)
    value = torch.randn(2, 2, 3, 4, dtype=torch.float16)
    attention_mask = torch.ones(2, 1, 3, 3, dtype=torch.bool)
    marker = object()

    output, attention_metadata = veomni_attention.fused_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=0.25,
        scaling=0.5,
        sliding_window=16,
        softcap=30.0,
        skip_ulysses=True,
        contract_marker=marker,
    )

    assert captured["module"] is module
    assert captured["query"] is query
    assert captured["key"] is key
    assert captured["value"] is value
    assert captured["attention_mask"] is attention_mask
    assert captured["kwargs"] == {
        "dropout": 0.25,
        "scaling": 0.5,
        "sliding_window": 16,
        "softcap": 30.0,
        "skip_ulysses": True,
        "contract_marker": marker,
    }
    torch.testing.assert_close(output, query.transpose(1, 2) + 1)
    assert attention_metadata == "attention-metadata"


def test_fused_attention_forward_rejects_an_unregistered_implementation():
    module = _FakeAttentionModule("unregistered_attention")
    query = torch.randn(1, 4, 3, 4, dtype=torch.float16)

    with pytest.raises(
        ValueError, match="Unsupported VeOmni fused attention implementation: 'unregistered_attention'"
    ):
        veomni_attention.fused_attention_forward(
            module,
            query,
            query,
            query,
            attention_mask=None,
        )


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


def test_flash_module_compute_slot_is_the_called_backend(monkeypatch):
    replacement_calls = []

    def replacement_backend(query, key, value, attention_mask, **kwargs):
        replacement_calls.append((query, key, value, attention_mask, kwargs))
        return torch.zeros_like(query)

    monkeypatch.setattr(flash_backend, "_flash_attention_forward", replacement_backend)
    monkeypatch.setattr(
        flash_backend,
        "get_parallel_state",
        lambda: SimpleNamespace(ulysses_enabled=False),
    )

    module = _FakeAttentionModule("veomni_flash_attention_2_with_sp")
    query = torch.randn(1, 4, 3, 4, dtype=torch.float16)
    key = torch.randn(1, 2, 3, 4, dtype=torch.float16)
    value = torch.randn(1, 2, 3, 4, dtype=torch.float16)

    output, attention_weights = veomni_attention.flash_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask=None,
    )

    assert len(replacement_calls) == 1
    assert output.shape == (1, 3, 4, 4)
    assert attention_weights is None
