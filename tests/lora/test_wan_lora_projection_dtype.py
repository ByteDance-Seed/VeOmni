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

"""CPU regression tests for Wan attention projection dtype resolution under LoRA.

Wan reads the projection dtype off the ``to_*`` modules in three places. A
LoRA-wrapped projection is a ``LoraLinear`` whose frozen weight lives at
``base_layer``, so reading ``.weight`` directly raised ``AttributeError`` on the
first attention forward of any Wan LoRA run. These tests pin both layouts.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from veomni.lora.layers import LoraLinear
from veomni.models.diffusers.wan_t2v.wan_transformer.modeling_wan_transformer import (
    WanAttentionKernelModule,
    _assert_wan_flash_attention_bf16,
    _projection_weight,
)


HIDDEN = 8
DTYPE = torch.bfloat16


class _StubWanAttention(nn.Module):
    """The attribute surface the three dtype call sites touch, nothing more."""

    def __init__(self, lora: bool):
        super().__init__()

        def projection() -> nn.Module:
            linear = nn.Linear(HIDDEN, HIDDEN, bias=False, dtype=DTYPE)
            if not lora:
                return linear
            return LoraLinear(linear, "default", r=4, lora_alpha=8)

        self.to_q = projection()
        self.to_k = projection()
        self.to_v = projection()
        self.to_out = nn.ModuleList([projection(), nn.Identity()])


def _bf16_qkv() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(torch.zeros(1, 2, 3, HIDDEN, dtype=DTYPE) for _ in range(3))


def test_lora_linear_has_no_weight_attribute():
    """The failure this module guards against, pinned so the guard is not 'simplified' away."""
    attn = _StubWanAttention(lora=True)

    assert isinstance(attn.to_q, LoraLinear)
    with pytest.raises(AttributeError):
        _ = attn.to_q.weight


@pytest.mark.parametrize("lora", [False, True])
def test_projection_weight_resolves_to_the_base_weight(lora):
    attn = _StubWanAttention(lora=lora)
    expected = attn.to_q.base_layer.weight if lora else attn.to_q.weight

    assert _projection_weight(attn.to_q) is expected
    assert _projection_weight(attn.to_q).dtype == DTYPE


@pytest.mark.parametrize("lora", [False, True])
def test_kernel_module_resolves_pre_quantization_dtype(lora):
    attn = _StubWanAttention(lora=lora)
    config = SimpleNamespace(_attn_implementation="veomni_flash_attention_2_with_sp")

    kernel_module = WanAttentionKernelModule(config, attn)

    assert kernel_module.config._pre_quantization_dtype == torch.bfloat16
    assert kernel_module.config._attn_implementation == config._attn_implementation


@pytest.mark.parametrize("lora", [False, True])
def test_flash_attention_bf16_assert_accepts_both_layouts(lora):
    """The assert reads all of to_q/to_k/to_v, so it fails on a layout the kernel module misses."""
    attn = _StubWanAttention(lora=lora)

    _assert_wan_flash_attention_bf16(*_bf16_qkv(), attn)


@pytest.mark.parametrize("lora", [False, True])
def test_fp32_projections_are_promoted_to_bf16(lora):
    """fp32 weights resolve through the same helper and still land on the bf16 kernel dtype."""
    attn = _StubWanAttention(lora=lora)
    attn.to(torch.float32)
    config = SimpleNamespace(_attn_implementation="veomni_flash_attention_2_with_sp")

    assert _projection_weight(attn.to_q).dtype == torch.float32
    assert WanAttentionKernelModule(config, attn).config._pre_quantization_dtype == torch.bfloat16


@pytest.mark.parametrize("lora", [False, True])
def test_type_as_target_is_the_projection_weight(lora):
    """`WanSPAttnProcessor` casts the attention output with this weight before `to_out[0]`."""
    attn = _StubWanAttention(lora=lora)
    hidden_states = torch.zeros(1, 3, HIDDEN, dtype=torch.float32)

    assert hidden_states.type_as(_projection_weight(attn.to_out[0])).dtype == DTYPE
