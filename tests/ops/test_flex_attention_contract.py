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
from torch.nn.attention.flex_attention import create_block_mask
from transformers.integrations.flex_attention import flex_attention_forward as hf_flex_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from veomni.ops import build_ALL_OPS
from veomni.ops.kernels import attention as veomni_attention
from veomni.ops.kernels.attention import flex as flex_backend


class _FakeAttentionModule(nn.Module):
    def __init__(self, *, training: bool = False):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="flex_attention")
        self.train(training)


def _causal_block_mask(sequence_length: int, device: torch.device):
    return create_block_mask(
        lambda batch_idx, head_idx, query_idx, key_idx: query_idx >= key_idx,
        B=None,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=device,
        BLOCK_SIZE=128,
    )


def test_flex_attention_forward_matches_the_public_attention_signature():
    flash_signature = inspect.signature(veomni_attention.flash_attention_forward)
    flex_signature = inspect.signature(veomni_attention.flex_attention_forward)

    assert list(flex_signature.parameters) == list(flash_signature.parameters)
    for name, flash_parameter in flash_signature.parameters.items():
        flex_parameter = flex_signature.parameters[name]
        assert flex_parameter.kind is flash_parameter.kind
        assert flex_parameter.default == flash_parameter.default


def test_apply_veomni_attention_patch_preserves_the_transformers_flex_registration():
    veomni_attention.apply_veomni_attention_patch()

    assert ALL_ATTENTION_FUNCTIONS["flex_attention"] is hf_flex_attention_forward


def test_flex_module_compute_slot_preserves_the_hf_adapter_contract_and_diagnostics(monkeypatch):
    captured = {}

    def replacement_backend(module, query, key, value, attention_mask, **kwargs):
        captured.update(
            module=module,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            kwargs=kwargs,
        )
        return query.transpose(1, 2) + 1, torch.ones(query.shape[:-1])

    monkeypatch.setattr(flex_backend, "_flex_attention_forward", replacement_backend)
    module = _FakeAttentionModule()
    query = torch.randn(2, 4, 8, 8)
    key = torch.randn(2, 2, 8, 8)
    value = torch.randn(2, 2, 8, 12)
    block_mask = _causal_block_mask(8, query.device)
    kernel_options = {"BACKEND": "TRITON", "BLOCKS_ARE_CONTIGUOUS": True}

    output, auxiliary = veomni_attention.flex_attention_forward(
        module,
        query,
        key,
        value,
        block_mask,
        scaling=0.19,
        softcap=30.0,
        kernel_options=kernel_options,
        output_attentions=False,
    )

    assert captured["module"] is module
    assert captured["query"] is query
    assert captured["key"] is key
    assert captured["value"] is value
    assert captured["attention_mask"] is block_mask
    assert captured["kwargs"] == {
        "dropout": 0.0,
        "scaling": 0.19,
        "softcap": 30.0,
        "kernel_options": kernel_options,
        "output_attentions": False,
    }
    torch.testing.assert_close(output, query.transpose(1, 2) + 1)
    assert auxiliary is not None
    assert dict(build_ALL_OPS())["_flex_attention_forward"] is replacement_backend


def test_flex_attention_cpu_forward_uses_native_block_mask_and_hf_layout():
    sequence_length = 17
    query = torch.randn(2, 4, sequence_length, 8)
    key = torch.randn(2, 2, sequence_length, 8)
    value = torch.randn(2, 2, sequence_length, 8)
    block_mask = _causal_block_mask(sequence_length, query.device)

    output, auxiliary = veomni_attention.flex_attention_forward(
        _FakeAttentionModule(),
        query,
        key,
        value,
        block_mask,
        scaling=0.25,
    )

    assert output.shape == (2, sequence_length, 4, 8)
    assert output.dtype == torch.float32
    assert auxiliary is None
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("query_heads", "kv_heads", "expected_message"),
    [
        (3, 2, "GQA requires query heads"),
        (4, 0, "does not support query/key/value tensors with zero dimensions"),
    ],
)
def test_flex_attention_rejects_invalid_gqa(query_heads, kv_heads, expected_message):
    query = torch.randn(1, query_heads, 8, 8)
    key = torch.randn(1, kv_heads, 8, 8)
    value = torch.randn(1, kv_heads, 8, 8)

    with pytest.raises(ValueError, match=expected_message):
        veomni_attention.flex_attention_forward(
            _FakeAttentionModule(),
            query,
            key,
            value,
            _causal_block_mask(8, query.device),
        )


def test_flex_attention_rejects_non_block_mask_and_standalone_sliding_window():
    query = torch.randn(1, 4, 8, 8)
    block_mask = _causal_block_mask(8, query.device)
    module = _FakeAttentionModule()

    with pytest.raises(TypeError, match="requires a BlockMask"):
        veomni_attention.flex_attention_forward(
            module,
            query,
            query,
            query,
            torch.ones(1, 1, 8, 8, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="must be encoded in the supplied BlockMask"):
        veomni_attention.flex_attention_forward(module, query, query, query, block_mask, sliding_window=4)


def test_flex_attention_rejects_active_ulysses_until_distributed_support_is_enabled(monkeypatch):
    monkeypatch.setattr(
        flex_backend,
        "get_parallel_state",
        lambda: SimpleNamespace(ulysses_enabled=True),
    )
    query = torch.randn(1, 4, 8, 8)
    block_mask = _causal_block_mask(8, query.device)

    with pytest.raises(RuntimeError, match="sequence parallelism is not enabled"):
        veomni_attention.flex_attention_forward(
            _FakeAttentionModule(),
            query,
            query,
            query,
            block_mask,
        )
