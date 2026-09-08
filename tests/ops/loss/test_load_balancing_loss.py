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

"""Load-balancing loss eager vs HF, and fused impls vs eager."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    load_balancing_loss_func as hf_load_balancing_loss,
)

from tests.ops.tol import (
    EAGER_ATOL,
    EAGER_GRAD_ATOL,
    EAGER_GRAD_RTOL,
    EAGER_RTOL,
    LB_FUSED_ATOL,
    LB_FUSED_GRAD_ATOL,
    LB_FUSED_GRAD_RTOL,
    LB_FUSED_RTOL,
)
from veomni.ops import OP_REGISTRY, resolve_op
from veomni.utils.device import IS_CUDA_AVAILABLE


_CONFIGS = [
    (8, 2, 1, 4, 128),
    (32, 4, 2, 2, 256),
    (60, 8, 4, 4, 512),
    (60, 8, 28, 2, 4096),
    (128, 4, 32, 1, 8192),
]


def _empty_mask(device: torch.device | str) -> Tensor:
    return torch.empty(0, device=device, dtype=torch.float32)


def _concat_layers(base: Tensor) -> Tensor:
    """``[num_layers, tokens, E]`` -> ops-style ``[N, E]``."""
    return base.reshape(-1, base.shape[-1]).detach().requires_grad_(True)


def test_registered_impls():
    assert OP_REGISTRY.list_registered("load_balancing_loss", "standard") == ["eager", "triton"]


def test_eager_uniform_distribution_is_well_conditioned():
    torch.manual_seed(0)
    gate_logits = torch.randn(3 * 1024, 8)
    output = resolve_op("load_balancing_loss", "standard", "eager").wrapper(
        gate_logits, _empty_mask(gate_logits.device), top_k=2
    )
    assert 0 < output.item() < 5.0


def test_eager_uniform_probabilities_return_top_k():
    gate_logits = torch.zeros(64, 4)
    output = resolve_op("load_balancing_loss", "standard", "eager").wrapper(
        gate_logits, _empty_mask(gate_logits.device), top_k=2
    )
    torch.testing.assert_close(output, torch.tensor(2.0), atol=1e-5, rtol=0)


def test_eager_one_hot_probabilities_return_num_experts():
    gate_logits = torch.full((64, 4), -1e9)
    gate_logits[:, 0] = 1e9
    output = resolve_op("load_balancing_loss", "standard", "eager").wrapper(
        gate_logits, _empty_mask(gate_logits.device), top_k=2
    )
    torch.testing.assert_close(output, torch.tensor(4.0), atol=1e-3, rtol=0)


def test_eager_attention_mask_matches_kept_tokens():
    torch.manual_seed(1)
    gate_logits = torch.randn(64, 4)
    attention_mask = torch.tensor([[1.0] * 32 + [0.0] * 32])
    eager = resolve_op("load_balancing_loss", "standard", "eager").wrapper
    masked = eager(gate_logits, attention_mask, top_k=2)
    kept = eager(gate_logits[:32], _empty_mask(gate_logits.device), top_k=2)
    torch.testing.assert_close(masked, kept, atol=1e-5, rtol=0)


def test_eager_repeated_layers_are_invariant():
    torch.manual_seed(2)
    layer = torch.randn(64, 4)
    eager = resolve_op("load_balancing_loss", "standard", "eager").wrapper
    once = eager(layer, _empty_mask(layer.device), top_k=2)
    twice = eager(torch.cat((layer, layer), dim=0), _empty_mask(layer.device), top_k=2)
    torch.testing.assert_close(once, twice, atol=1e-4, rtol=0)


@pytest.mark.parametrize(
    "num_experts,top_k,num_layers,batch_size,seq_len",
    [(8, 2, 2, 4, 128), (60, 8, 4, 2, 512)],
)
def test_eager_is_deterministic(num_experts, top_k, num_layers, batch_size, seq_len):
    torch.manual_seed(42)
    gate_logits = torch.randn(num_layers * batch_size * seq_len, num_experts)
    eager = resolve_op("load_balancing_loss", "standard", "eager").wrapper
    empty_mask = _empty_mask(gate_logits.device)

    outputs = [eager(gate_logits, empty_mask, top_k=top_k) for _ in range(5)]

    for output in outputs[1:]:
        assert torch.equal(outputs[0], output)


def test_eager_all_masked_returns_zero_with_zero_grad():
    gate_logits = torch.randn(8, 4, requires_grad=True)
    attention_mask = torch.zeros(2, 4)
    output = resolve_op("load_balancing_loss", "standard", "eager").wrapper(gate_logits, attention_mask, top_k=2)
    assert output.item() == 0.0
    output.backward()
    assert torch.count_nonzero(gate_logits.grad) == 0


def test_eager_matches_hf():
    torch.manual_seed(0)
    num_layers, batch, seq_len, num_experts, top_k = 2, 2, 16, 8, 2
    base = torch.randn(num_layers, batch * seq_len, num_experts, dtype=torch.float32)
    layers_h = tuple(base[i].detach().requires_grad_(True) for i in range(num_layers))
    concat_e = _concat_layers(base)

    out_h = hf_load_balancing_loss(layers_h, num_experts, top_k)
    out_e = resolve_op("load_balancing_loss", "standard", "eager").wrapper(
        concat_e, _empty_mask(base.device), top_k=top_k
    )
    assert torch.allclose(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    out_h.backward()
    out_e.backward()
    grad_h = torch.cat([layer.grad for layer in layers_h], dim=0)
    assert torch.allclose(concat_e.grad, grad_h, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_eager_matches_hf_with_mask():
    torch.manual_seed(1)
    num_layers, batch, seq_len, num_experts, top_k = 2, 2, 16, 8, 2
    base = torch.randn(num_layers, batch * seq_len, num_experts, dtype=torch.float32)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.float32)
    attention_mask[:, seq_len // 2 :] = 0
    layers_h = tuple(base[i].detach().requires_grad_(True) for i in range(num_layers))
    concat_e = _concat_layers(base)

    out_h = hf_load_balancing_loss(layers_h, num_experts, top_k, attention_mask)
    out_e = resolve_op("load_balancing_loss", "standard", "eager").wrapper(concat_e, attention_mask, top_k=top_k)
    assert torch.allclose(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    out_h.backward()
    out_e.backward()
    grad_h = torch.cat([layer.grad for layer in layers_h], dim=0)
    assert torch.allclose(concat_e.grad, grad_h, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


@pytest.mark.parametrize("num_experts,top_k,num_layers,batch,seq_len", _CONFIGS)
@pytest.mark.parametrize("use_mask", [False, True])
def test_eager_forward_matrix_matches_hf(num_experts, top_k, num_layers, batch, seq_len, use_mask):
    torch.manual_seed(2)
    base = torch.randn(num_layers, batch * seq_len, num_experts)
    layers = tuple(base[i] for i in range(num_layers))
    concatenated = base.reshape(-1, num_experts)
    if use_mask:
        attention_mask = torch.ones(batch, seq_len)
        attention_mask[:, seq_len // 2 :] = 0
        hf_mask = attention_mask
    else:
        attention_mask = _empty_mask(base.device)
        hf_mask = None

    expected = hf_load_balancing_loss(layers, num_experts, top_k, hf_mask)
    actual = resolve_op("load_balancing_loss", "standard", "eager").wrapper(concatenated, attention_mask, top_k=top_k)

    torch.testing.assert_close(actual, expected, atol=EAGER_ATOL, rtol=EAGER_RTOL)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="triton load-balancing loss needs CUDA")
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("num_experts,top_k,num_layers,batch,seq_len", _CONFIGS)
def test_triton_matches_eager(use_mask: bool, num_experts, top_k, num_layers, batch, seq_len):
    pytest.importorskip("triton")
    eager = resolve_op("load_balancing_loss", "standard", "eager").wrapper
    other = resolve_op("load_balancing_loss", "standard", "triton").wrapper
    torch.manual_seed(0)
    base = torch.randn(num_layers, batch * seq_len, num_experts, device="cuda", dtype=torch.float32)
    if use_mask:
        attention_mask = torch.ones(batch, seq_len, device="cuda", dtype=torch.float32)
        attention_mask[:, seq_len // 2 :] = 0
    else:
        attention_mask = _empty_mask("cuda")

    concat_e = _concat_layers(base)
    concat_o = _concat_layers(base)
    out_e = eager(concat_e, attention_mask, top_k=top_k)
    out_o = other(concat_o, attention_mask, top_k=top_k)
    assert torch.allclose(out_e, out_o, atol=LB_FUSED_ATOL, rtol=LB_FUSED_RTOL)

    out_e.backward()
    out_o.backward()
    assert torch.allclose(concat_e.grad, concat_o.grad, atol=LB_FUSED_GRAD_ATOL, rtol=LB_FUSED_GRAD_RTOL)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="triton load-balancing loss needs CUDA")
def test_triton_all_masked_returns_zero_with_zero_grad():
    pytest.importorskip("triton")
    gate_logits = torch.randn(8, 4, device="cuda", requires_grad=True)
    attention_mask = torch.zeros(2, 4, device="cuda")
    output = resolve_op("load_balancing_loss", "standard", "triton").wrapper(gate_logits, attention_mask, top_k=2)
    assert output.item() == 0.0
    output.backward()
    assert torch.count_nonzero(gate_logits.grad) == 0


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="triton load-balancing loss needs CUDA")
@pytest.mark.parametrize(
    "num_experts,top_k,num_layers,batch_size,seq_len",
    [(8, 2, 2, 4, 128), (60, 8, 4, 2, 512)],
)
@pytest.mark.parametrize("use_mask", [False, True])
def test_triton_is_deterministic(num_experts, top_k, num_layers, batch_size, seq_len, use_mask):
    pytest.importorskip("triton")
    torch.manual_seed(42)
    gate_logits = torch.randn(num_layers * batch_size * seq_len, num_experts, device="cuda")
    if use_mask:
        attention_mask = torch.ones(batch_size, seq_len, device="cuda")
        attention_mask[:, seq_len // 2 :] = 0
    else:
        attention_mask = _empty_mask("cuda")
    triton = resolve_op("load_balancing_loss", "standard", "triton").wrapper

    outputs = [triton(gate_logits, attention_mask, top_k=top_k) for _ in range(5)]

    for output in outputs[1:]:
        assert torch.equal(outputs[0], output)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="triton load-balancing loss needs CUDA")
@pytest.mark.parametrize("num_experts,top_k,num_layers,batch_size,seq_len", _CONFIGS)
def test_triton_uses_less_peak_memory_than_hf(num_experts, top_k, num_layers, batch_size, seq_len):
    pytest.importorskip("triton")
    torch.manual_seed(0)
    base = torch.randn(num_layers, batch_size * seq_len, num_experts, device="cuda")
    layers = tuple(base[i] for i in range(num_layers))
    concatenated = base.reshape(-1, num_experts)
    triton = resolve_op("load_balancing_loss", "standard", "triton").wrapper
    empty_mask = _empty_mask("cuda")

    triton(concatenated[:32], empty_mask, top_k=top_k)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    hf_load_balancing_loss(layers, num_experts, top_k)
    torch.cuda.synchronize()
    hf_peak = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    triton(concatenated, empty_mask, top_k=top_k)
    torch.cuda.synchronize()
    triton_peak = torch.cuda.max_memory_allocated()

    assert triton_peak < hf_peak
