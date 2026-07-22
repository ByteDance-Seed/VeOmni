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
from torch.nn.attention.flex_attention import BlockMask
from transformers import PreTrainedConfig
from transformers.integrations.flex_attention import flex_attention_forward as hf_flex_attention_forward
from transformers.masking_utils import (
    ALL_MASK_ATTENTION_FUNCTIONS,
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.ops.kernels import attention as veomni_attention


_FLASH_IMPLEMENTATIONS = (
    "veomni_flash_attention_2_with_sp",
    "veomni_flash_attention_3_with_sp",
    "veomni_flash_attention_4_with_sp",
)
_FLEX_IMPLEMENTATION = "veomni_flex_attention_with_sp"


class _FakeAttentionModule(nn.Module):
    def __init__(self, implementation: str):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation=implementation)


@pytest.mark.parametrize(
    "backend",
    [veomni_attention.flash_attention_forward, veomni_attention.flex_attention_forward],
)
def test_fused_attention_forward_matches_backend_public_signatures(backend):
    fused_signature = inspect.signature(veomni_attention.fused_attention_forward)
    backend_signature = inspect.signature(backend)

    assert list(fused_signature.parameters) == list(backend_signature.parameters)
    for name, backend_parameter in backend_signature.parameters.items():
        fused_parameter = fused_signature.parameters[name]
        assert fused_parameter.kind is backend_parameter.kind
        assert fused_parameter.default == backend_parameter.default


def test_apply_veomni_attention_patch_registers_custom_facade_names(monkeypatch):
    patch_calls = []
    monkeypatch.setattr(
        veomni_attention,
        "patch_transformers_hub_kernel_loader_for_veomni",
        lambda: patch_calls.append("hub_kernel_loader"),
    )
    monkeypatch.setattr(
        veomni_attention,
        "register_veomni_flex_attention_mask_builder",
        lambda: patch_calls.append("flex_mask_builder"),
    )

    veomni_attention.apply_veomni_attention_patch()

    assert patch_calls == ["hub_kernel_loader", "flex_mask_builder"]
    for implementation in (*_FLASH_IMPLEMENTATIONS, _FLEX_IMPLEMENTATION):
        assert ALL_ATTENTION_FUNCTIONS[implementation] is veomni_attention.fused_attention_forward
    assert ALL_ATTENTION_FUNCTIONS["flex_attention"] is hf_flex_attention_forward


def test_register_veomni_flex_attention_mask_builder_reuses_transformers_flex_builder(monkeypatch):
    mask_mapping = ALL_MASK_ATTENTION_FUNCTIONS._global_mapping
    monkeypatch.setitem(mask_mapping, _FLEX_IMPLEMENTATION, object())

    veomni_attention.register_veomni_flex_attention_mask_builder()
    veomni_attention.register_veomni_flex_attention_mask_builder()

    assert ALL_MASK_ATTENTION_FUNCTIONS[_FLEX_IMPLEMENTATION] is ALL_MASK_ATTENTION_FUNCTIONS["flex_attention"]

    config = PreTrainedConfig()
    config._attn_implementation = _FLEX_IMPLEMENTATION
    config.sliding_window = 4
    inputs_embeds = torch.randn(1, 8, 16)
    position_ids = torch.arange(8).unsqueeze(0)

    causal_mask = create_causal_mask(config, inputs_embeds, None, None, position_ids)
    sliding_window_mask = create_sliding_window_causal_mask(config, inputs_embeds, None, None, position_ids)

    assert isinstance(causal_mask, BlockMask)
    assert isinstance(sliding_window_mask, BlockMask)
    assert causal_mask.shape == sliding_window_mask.shape == (1, 1, 8, 8)


@pytest.mark.parametrize("implementation", [*_FLASH_IMPLEMENTATIONS, _FLEX_IMPLEMENTATION])
def test_fused_attention_forward_dispatches_to_selected_adapter(monkeypatch, implementation):
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


def test_fused_attention_forward_rejects_unregistered_implementation():
    module = _FakeAttentionModule("unregistered_attention")
    query = torch.randn(1, 4, 3, 4, dtype=torch.float16)

    with pytest.raises(
        ValueError, match="Unsupported VeOmni fused attention implementation: 'unregistered_attention'"
    ):
        veomni_attention.fused_attention_forward(module, query, query, query, attention_mask=None)


def test_veomni_ops_config_rewrites_flex_to_sp_aware_registration(monkeypatch):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")

    config = OpsImplementationConfig(attn_implementation="flex_attention")

    assert config.attn_implementation == _FLEX_IMPLEMENTATION
