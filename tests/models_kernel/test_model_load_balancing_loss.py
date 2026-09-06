# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Model-facing load-balancing-loss input policy and gradient coverage."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    load_balancing_loss_func as hf_load_balancing_loss,
)

from tests.kernels.tol import EAGER_ATOL, EAGER_GRAD_ATOL, EAGER_GRAD_RTOL, EAGER_RTOL
from veomni.kernels import VeomniKernel
from veomni.models_kernel.loss_utils import load_balancing_loss


class _RecordingKernel:
    def __init__(self) -> None:
        self.gate_logits: Tensor | None = None
        self.attention_mask: Tensor | None = None
        self.top_k: int | None = None

    def __call__(self, gate_logits: Tensor, attention_mask: Tensor, *, top_k: int) -> Tensor:
        self.gate_logits = gate_logits
        self.attention_mask = attention_mask
        self.top_k = top_k
        return gate_logits.sum()


@pytest.mark.parametrize("gate_logits", [None, torch.randn(4, 8)])
def test_none_or_non_tuple_returns_zero_without_calling_kernel(gate_logits):
    class _FailKernel:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("kernel must not be called")

    assert load_balancing_loss(gate_logits, 8, 2, kernel=_FailKernel()) == 0


def test_concatenates_layers_flattens_leading_dims_and_uses_empty_mask():
    first = torch.randn(2, 3, 4, requires_grad=True)
    second = torch.randn(2, 3, 4, requires_grad=True)
    kernel = _RecordingKernel()

    output = load_balancing_loss((first, second), 4, 2, kernel=kernel)

    assert kernel.gate_logits is not None
    assert kernel.gate_logits.shape == (12, 4)
    assert kernel.gate_logits.is_contiguous()
    assert kernel.attention_mask is not None
    assert kernel.attention_mask.numel() == 0
    assert kernel.attention_mask.device == first.device
    assert kernel.top_k == 2

    output.backward()
    torch.testing.assert_close(first.grad, torch.ones_like(first))
    torch.testing.assert_close(second.grad, torch.ones_like(second))


def test_forwards_attention_mask_without_changing_it():
    gate_logits = (torch.randn(8, 4),)
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]])
    kernel = _RecordingKernel()

    load_balancing_loss(gate_logits, 4, 2, attention_mask, kernel=kernel)

    assert kernel.attention_mask is attention_mask


def test_rejects_num_experts_mismatch():
    with pytest.raises(ValueError, match="last dim .* != num_experts"):
        load_balancing_loss((torch.randn(8, 4),), 8, 2, kernel=_RecordingKernel())


def test_eager_helper_matches_hf_with_mask_and_layer_grads():
    torch.manual_seed(0)
    num_layers, batch, seq_len, num_experts, top_k = 2, 2, 8, 4, 2
    base = torch.randn(num_layers, batch * seq_len, num_experts)
    attention_mask = torch.ones(batch, seq_len)
    attention_mask[:, seq_len // 2 :] = 0
    layers_hf = tuple(base[i].detach().requires_grad_(True) for i in range(num_layers))
    layers_kernel = tuple(base[i].detach().requires_grad_(True) for i in range(num_layers))

    expected = hf_load_balancing_loss(layers_hf, num_experts, top_k, attention_mask)
    actual = load_balancing_loss(
        layers_kernel,
        num_experts,
        top_k,
        attention_mask,
        kernel=VeomniKernel("load_balancing_loss", "standard", "eager"),
    )
    torch.testing.assert_close(actual, expected, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    expected.backward()
    actual.backward()
    for actual_layer, expected_layer in zip(layers_kernel, layers_hf):
        torch.testing.assert_close(
            actual_layer.grad,
            expected_layer.grad,
            atol=EAGER_GRAD_ATOL,
            rtol=EAGER_GRAD_RTOL,
        )
