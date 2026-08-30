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

from tests.kernels.tol import (
    EAGER_ATOL,
    EAGER_GRAD_ATOL,
    EAGER_GRAD_RTOL,
    EAGER_RTOL,
    LB_FUSED_ATOL,
    LB_FUSED_GRAD_ATOL,
    LB_FUSED_GRAD_RTOL,
    LB_FUSED_RTOL,
)
from veomni.kernels import resolve_kernel
from veomni.utils.device import IS_CUDA_AVAILABLE


def _empty_mask(device: torch.device | str) -> Tensor:
    return torch.empty(0, device=device, dtype=torch.float32)


def _concat_layers(base: Tensor) -> Tensor:
    """``[num_layers, tokens, E]`` -> ops-style ``[N, E]``."""
    return base.reshape(-1, base.shape[-1]).detach().requires_grad_(True)


def test_eager_matches_hf():
    torch.manual_seed(0)
    num_layers, batch, seq_len, num_experts, top_k = 2, 2, 16, 8, 2
    base = torch.randn(num_layers, batch * seq_len, num_experts, dtype=torch.float32)
    layers_h = tuple(base[i].detach().requires_grad_(True) for i in range(num_layers))
    concat_e = _concat_layers(base)

    out_h = hf_load_balancing_loss(layers_h, num_experts, top_k)
    out_e = resolve_kernel("load_balancing_loss", "standard", "eager").wrapper(
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
    out_e = resolve_kernel("load_balancing_loss", "standard", "eager").wrapper(concat_e, attention_mask, top_k=top_k)
    assert torch.allclose(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    out_h.backward()
    out_e.backward()
    grad_h = torch.cat([layer.grad for layer in layers_h], dim=0)
    assert torch.allclose(concat_e.grad, grad_h, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="triton load-balancing loss needs CUDA")
@pytest.mark.parametrize("use_mask", [False, True])
def test_triton_matches_eager(use_mask: bool):
    pytest.importorskip("triton")
    eager = resolve_kernel("load_balancing_loss", "standard", "eager").wrapper
    other = resolve_kernel("load_balancing_loss", "standard", "triton").wrapper
    torch.manual_seed(0)
    num_layers, batch, seq_len, num_experts, top_k = 2, 2, 32, 8, 2
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
