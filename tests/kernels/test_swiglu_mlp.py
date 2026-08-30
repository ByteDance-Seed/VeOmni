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

"""SwiGLU MLP eager vs HF, and fused impls vs eager."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn
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


def _empty_bias(weight: Tensor) -> Tensor:
    return weight.new_empty(0)


def _bias(linear: nn.Linear) -> Tensor:
    return linear.bias if linear.bias is not None else _empty_bias(linear.weight)


def _mlp_args(mlp: nn.Module, x: Tensor) -> tuple[Tensor, ...]:
    return (
        x,
        mlp.gate_proj.weight,
        _bias(mlp.gate_proj),
        mlp.up_proj.weight,
        _bias(mlp.up_proj),
        mlp.down_proj.weight,
        _bias(mlp.down_proj),
    )


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


def _copy_linear(src: nn.Linear) -> nn.Linear:
    dst = nn.Linear(src.in_features, src.out_features, bias=src.bias is not None)
    dst.load_state_dict(src.state_dict())
    return dst


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
    out_e = wrapper(*_mlp_args(mlp_e, x_e))
    assert torch.allclose(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = torch.randn_like(out_e)
    out_h.backward(go)
    out_e.backward(go)
    assert torch.allclose(x_e.grad, x_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    for param_e, param_h in zip(mlp_e.parameters(), mlp_h.parameters(), strict=True):
        assert torch.allclose(param_e.grad, param_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_eager_matches_biased_linears():
    torch.manual_seed(1)
    hidden, intermediate = 64, 128
    gate_h = nn.Linear(hidden, intermediate, bias=True)
    up_h = nn.Linear(hidden, intermediate, bias=True)
    down_h = nn.Linear(intermediate, hidden, bias=True)
    gate_e = _copy_linear(gate_h)
    up_e = _copy_linear(up_h)
    down_e = _copy_linear(down_h)
    x = torch.randn(2, 16, hidden, dtype=torch.float32)

    x_h = x.detach().requires_grad_(True)
    out_h = down_h(F.silu(gate_h(x_h)) * up_h(x_h))

    x_e = x.detach().requires_grad_(True)
    wrapper = resolve_kernel("swiglu_mlp", "standard", "eager").wrapper
    out_e = wrapper(x_e, gate_e.weight, gate_e.bias, up_e.weight, up_e.bias, down_e.weight, down_e.bias)
    assert torch.allclose(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = torch.randn_like(out_e)
    out_h.backward(go)
    out_e.backward(go)
    assert torch.allclose(x_e.grad, x_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(gate_e.weight.grad, gate_h.weight.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(gate_e.bias.grad, gate_h.bias.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(up_e.weight.grad, up_h.weight.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(up_e.bias.grad, up_h.bias.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(down_e.weight.grad, down_h.weight.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(down_e.bias.grad, down_h.bias.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_eager_matches_swiglu_limit():
    torch.manual_seed(2)
    hidden, intermediate, limit = 64, 128, 7.0
    gate_h = nn.Linear(hidden, intermediate, bias=False)
    up_h = nn.Linear(hidden, intermediate, bias=False)
    down_h = nn.Linear(intermediate, hidden, bias=False)
    gate_e = _copy_linear(gate_h)
    up_e = _copy_linear(up_h)
    down_e = _copy_linear(down_h)
    x = torch.randn(2, 16, hidden, dtype=torch.float32)

    x_h = x.detach().requires_grad_(True)
    gate_act = gate_h(x_h).float().clamp(max=limit)
    up_act = up_h(x_h).float().clamp(min=-limit, max=limit)
    out_h = down_h((F.silu(gate_act) * up_act).to(dtype=x_h.dtype))

    x_e = x.detach().requires_grad_(True)
    wrapper = resolve_kernel("swiglu_mlp", "standard", "eager").wrapper
    out_e = wrapper(
        x_e,
        gate_e.weight,
        _empty_bias(gate_e.weight),
        up_e.weight,
        _empty_bias(up_e.weight),
        down_e.weight,
        _empty_bias(down_e.weight),
        swiglu_limit=limit,
    )
    assert torch.allclose(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = torch.randn_like(out_e)
    out_h.backward(go)
    out_e.backward(go)
    assert torch.allclose(x_e.grad, x_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(gate_e.weight.grad, gate_h.weight.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(up_e.weight.grad, up_h.weight.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(down_e.weight.grad, down_h.weight.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="liger SwiGLU needs CUDA")
def test_liger_matches_eager():
    pytest.importorskip("liger_kernel")
    eager = resolve_kernel("swiglu_mlp", "standard", "eager").wrapper
    other = resolve_kernel("swiglu_mlp", "standard", "liger_kernel").wrapper
    torch.manual_seed(0)
    mlp = _tiny_qwen3_mlp().to(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(2, 16, mlp.hidden_size, device="cuda", dtype=torch.bfloat16)

    x_e = x.detach().requires_grad_(True)
    x_o = x.detach().requires_grad_(True)
    mlp_e = _tiny_qwen3_mlp().to(device="cuda", dtype=torch.bfloat16)
    mlp_o = _tiny_qwen3_mlp().to(device="cuda", dtype=torch.bfloat16)
    mlp_e.load_state_dict(mlp.state_dict())
    mlp_o.load_state_dict(mlp.state_dict())
    out_e = eager(*_mlp_args(mlp_e, x_e))
    out_o = other(*_mlp_args(mlp_o, x_o))
    assert torch.allclose(out_e, out_o, atol=SWIGLU_FUSED_ATOL, rtol=SWIGLU_FUSED_RTOL)

    go = torch.randn_like(out_e)
    out_e.backward(go)
    out_o.backward(go)
    assert torch.allclose(x_e.grad, x_o.grad, atol=SWIGLU_FUSED_GRAD_ATOL, rtol=SWIGLU_FUSED_GRAD_RTOL)
    for param_e, param_o in zip(mlp_e.parameters(), mlp_o.parameters(), strict=True):
        assert torch.allclose(param_e.grad, param_o.grad, atol=SWIGLU_FUSED_GRAD_ATOL, rtol=SWIGLU_FUSED_GRAD_RTOL)
