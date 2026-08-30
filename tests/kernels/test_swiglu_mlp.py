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

"""SwiGLU eager vs HF, and fused impls vs eager."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from tests.kernels.tol import (
    EAGER_ATOL,
    EAGER_GRAD_ATOL,
    EAGER_GRAD_RTOL,
    EAGER_RTOL,
    SWIGLU_FUSED_ATOL,
    SWIGLU_FUSED_GRAD_ATOL,
    SWIGLU_FUSED_GRAD_RTOL,
    SWIGLU_FUSED_RTOL,
)
from veomni.kernels import resolve_kernel
from veomni.utils.device import IS_CUDA_AVAILABLE


def _clone_pair(gate: Tensor, up: Tensor) -> tuple[Tensor, Tensor]:
    return gate.detach().requires_grad_(True), up.detach().requires_grad_(True)


def _tiny_qwen3_mlp() -> Qwen3MLP:
    config = Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
    )
    return Qwen3MLP(config)


def test_eager_matches_hf():
    torch.manual_seed(0)
    mlp_h = _tiny_qwen3_mlp()
    mlp_e = _tiny_qwen3_mlp()
    mlp_e.load_state_dict(mlp_h.state_dict())
    x = torch.randn(2, 16, mlp_h.hidden_size, dtype=torch.float32)

    x_h = x.detach().requires_grad_(True)
    out_h = mlp_h(x_h)

    x_e = x.detach().requires_grad_(True)
    wrapper = resolve_kernel("swiglu_mlp", "standard", "eager").wrapper
    out_e = mlp_e.down_proj(wrapper(mlp_e.gate_proj(x_e), mlp_e.up_proj(x_e)))
    assert torch.allclose(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = torch.randn_like(out_e)
    out_h.backward(go)
    out_e.backward(go)
    assert torch.allclose(x_e.grad, x_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    for param_e, param_h in zip(mlp_e.parameters(), mlp_h.parameters(), strict=True):
        assert torch.allclose(param_e.grad, param_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_eager_matches_hf_activation():
    torch.manual_seed(0)
    mlp = _tiny_qwen3_mlp()
    hidden = torch.randn(2, 16, mlp.hidden_size, dtype=torch.float32)
    gate = mlp.gate_proj(hidden).detach().requires_grad_(True)
    up = mlp.up_proj(hidden).detach().requires_grad_(True)

    gate_h, up_h = _clone_pair(gate, up)
    out_h = mlp.act_fn(gate_h) * up_h

    gate_e, up_e = _clone_pair(gate, up)
    out_e = resolve_kernel("swiglu_mlp", "standard", "eager").wrapper(gate_e, up_e)
    assert torch.allclose(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = torch.randn_like(out_e)
    out_h.backward(go)
    out_e.backward(go)
    assert torch.allclose(gate_e.grad, gate_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(up_e.grad, up_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="liger SwiGLU needs CUDA")
def test_liger_matches_eager():
    pytest.importorskip("liger_kernel")
    eager = resolve_kernel("swiglu_mlp", "standard", "eager").wrapper
    other = resolve_kernel("swiglu_mlp", "standard", "liger_kernel").wrapper
    torch.manual_seed(0)
    base_gate = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
    base_up = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)

    gate_e, up_e = _clone_pair(base_gate, base_up)
    gate_o, up_o = _clone_pair(base_gate, base_up)
    out_e = eager(gate_e, up_e)
    out_o = other(gate_o, up_o)
    assert torch.allclose(out_e, out_o, atol=SWIGLU_FUSED_ATOL, rtol=SWIGLU_FUSED_RTOL)

    go = torch.randn_like(out_e)
    out_e.backward(go)
    out_o.backward(go)
    assert torch.allclose(gate_e.grad, gate_o.grad, atol=SWIGLU_FUSED_GRAD_ATOL, rtol=SWIGLU_FUSED_GRAD_RTOL)
    assert torch.allclose(up_e.grad, up_o.grad, atol=SWIGLU_FUSED_GRAD_ATOL, rtol=SWIGLU_FUSED_GRAD_RTOL)
