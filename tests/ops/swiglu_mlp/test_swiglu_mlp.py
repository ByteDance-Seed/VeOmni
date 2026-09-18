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
from transformers import DeepseekV4Config, Gemma3TextConfig, Qwen3Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts
from transformers.models.gemma3.modeling_gemma3 import Gemma3MLP
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from tests.ops.tol import (
    EAGER_ATOL,
    EAGER_GRAD_ATOL,
    EAGER_GRAD_RTOL,
    EAGER_RTOL,
    SWIGLU_FUSED_ATOL,
    SWIGLU_FUSED_GRAD_ATOL,
    SWIGLU_FUSED_GRAD_RTOL,
    SWIGLU_FUSED_RTOL,
)
from veomni.ops import resolve_op
from veomni.ops.kernels.swiglu_mlp.geglu import liger_kernel as geglu_liger
from veomni.ops.kernels.swiglu_mlp.standard import liger_kernel as standard_liger
from veomni.ops.registry import OpEntry
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


def _tiny_qwen3_mlp(hidden_size: int = 64, intermediate_size: int = 128) -> Qwen3MLP:
    config = Qwen3Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
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
    wrapper = resolve_op("swiglu_mlp", "standard", "eager").wrapper
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
    wrapper = resolve_op("swiglu_mlp", "standard", "eager").wrapper
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


@pytest.mark.parametrize(
    ("impl", "device"),
    (
        ("eager", "cpu"),
        pytest.param("eager", "cuda", marks=pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="needs CUDA")),
        pytest.param(
            "liger_kernel",
            "cuda",
            marks=pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Liger SwiGLU needs CUDA"),
        ),
    ),
)
@pytest.mark.parametrize("swiglu_limit", (None, 1.0), ids=("plain", "clamped"))
def test_autocast_forward_backward_matches_pytorch(impl: str, device: str, swiglu_limit: float | None):
    if impl == "liger_kernel":
        pytest.importorskip("liger_kernel")

    torch.manual_seed(12)
    hidden, intermediate = 64, 128
    tensors = (
        torch.randn(2, 8, hidden, device=device) * 0.1,
        torch.randn(intermediate, hidden, device=device) * 0.1,
        torch.randn(intermediate, device=device) * 0.1,
        torch.randn(intermediate, hidden, device=device) * 0.1,
        torch.randn(intermediate, device=device) * 0.1,
        torch.randn(hidden, intermediate, device=device) * 0.1,
        torch.randn(hidden, device=device) * 0.1,
    )
    reference_args = tuple(tensor.detach().clone().requires_grad_(True) for tensor in tensors)
    actual_args = tuple(tensor.detach().clone().requires_grad_(True) for tensor in tensors)

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        x, gate_w, gate_b, up_w, up_b, down_w, down_b = reference_args
        gate = F.linear(x, gate_w, gate_b)
        up = F.linear(x, up_w, up_b)
        if swiglu_limit is not None:
            gate = gate.float().clamp(max=swiglu_limit)
            up = up.float().clamp(min=-swiglu_limit, max=swiglu_limit)
        hidden_state = F.silu(gate) * up
        if hidden_state.dtype != x.dtype:
            hidden_state = hidden_state.to(dtype=x.dtype)
        expected = F.linear(hidden_state, down_w, down_b)
        actual = resolve_op("swiglu_mlp", "standard", impl).wrapper(
            *actual_args,
            swiglu_limit=swiglu_limit,
        )

    grad_output = torch.randn_like(expected)
    expected.backward(grad_output)
    actual.backward(grad_output)

    torch.testing.assert_close(actual, expected, atol=SWIGLU_FUSED_ATOL, rtol=SWIGLU_FUSED_RTOL)
    for actual_arg, reference_arg in zip(actual_args, reference_args, strict=True):
        torch.testing.assert_close(
            actual_arg.grad,
            reference_arg.grad,
            atol=SWIGLU_FUSED_GRAD_ATOL,
            rtol=SWIGLU_FUSED_GRAD_RTOL,
        )


def test_eager_matches_swiglu_limit():
    """``swiglu_limit`` matches HF ``DeepseekV4Experts._apply_gate``.

    Installed class: ``transformers.models.deepseek_v4.modeling_deepseek_v4.DeepseekV4Experts``.

    One expert, every token routed to it with weight 1, so the expert MLP
    is the same as ``swiglu_mlp`` with the DSV4 clamp.
    """
    torch.manual_seed(2)
    hidden, intermediate, limit = 64, 128, 7.0
    tokens = 2 * 16
    config = DeepseekV4Config(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_local_experts=1,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        swiglu_limit=limit,
    )
    config._experts_implementation = "eager"
    experts = DeepseekV4Experts(config)
    nn.init.normal_(experts.gate_up_proj, std=0.5)
    nn.init.normal_(experts.down_proj, std=0.1)
    x = torch.randn(2, 16, hidden, dtype=torch.float32)
    selected = torch.zeros(tokens, 1, dtype=torch.long)
    routing = torch.ones(tokens, 1, dtype=torch.float32)

    gate_w, up_w = experts.gate_up_proj[0].detach().chunk(2, dim=0)
    gate = F.linear(x, gate_w)
    up = F.linear(x, up_w)
    assert torch.any(gate > limit)
    assert torch.any(up.abs() > limit)

    x_h = x.detach().requires_grad_(True)
    out_h = experts(x_h.reshape(tokens, hidden), selected, routing).reshape_as(x)

    x_e = x.detach().requires_grad_(True)
    down_w = experts.down_proj[0].detach()
    gate_e = nn.Parameter(gate_w.clone())
    up_e = nn.Parameter(up_w.clone())
    down_e = nn.Parameter(down_w.clone())
    wrapper = resolve_op("swiglu_mlp", "standard", "eager").wrapper
    out_e = wrapper(
        x_e,
        gate_e,
        _empty_bias(gate_e),
        up_e,
        _empty_bias(up_e),
        down_e,
        _empty_bias(down_e),
        swiglu_limit=limit,
    )
    assert torch.allclose(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = torch.randn_like(out_e)
    out_h.backward(go)
    out_e.backward(go)
    assert torch.allclose(x_e.grad, x_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    gate_h_grad, up_h_grad = experts.gate_up_proj.grad[0].chunk(2, dim=0)
    assert torch.allclose(gate_e.grad, gate_h_grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(up_e.grad, up_h_grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(down_e.grad, experts.down_proj.grad[0], atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


@pytest.mark.parametrize("with_bias", (False, True), ids=("no-bias", "bias"))
def test_liger_empty_input_falls_back_to_eager_backward(with_bias: bool):
    hidden, intermediate = 4, 8
    x = torch.empty(2, 0, hidden, requires_grad=True)
    gate_w = torch.randn(intermediate, hidden, requires_grad=True)
    up_w = torch.randn(intermediate, hidden, requires_grad=True)
    down_w = torch.randn(hidden, intermediate, requires_grad=True)
    if with_bias:
        gate_b = torch.randn(intermediate, requires_grad=True)
        up_b = torch.randn(intermediate, requires_grad=True)
        down_b = torch.randn(hidden, requires_grad=True)
    else:
        gate_b = torch.empty(0, requires_grad=True)
        up_b = torch.empty(0, requires_grad=True)
        down_b = torch.empty(0, requires_grad=True)

    entry = OpEntry(
        "swiglu_mlp",
        "standard",
        "liger-empty-test",
        description="Test-only Liger SwiGLU entry",
        forward=standard_liger.forward,
        backward=standard_liger.backward,
    )
    assert entry.wrapper is not None
    output = entry.wrapper(x, gate_w, gate_b, up_w, up_b, down_w, down_b, swiglu_limit=1.0)

    assert output.shape == x.shape
    output.sum().backward()
    for tensor in (x, gate_w, up_w, down_w):
        torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor))
    for bias in (gate_b, up_b, down_b):
        if with_bias:
            torch.testing.assert_close(bias.grad, torch.zeros_like(bias))
        else:
            assert bias.grad is None


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="liger SwiGLU needs a GPU")
@pytest.mark.parametrize("hidden, intermediate", [(64, 128), (128, 256)])
@pytest.mark.parametrize("swiglu_limit", [None, 1.0])
def test_liger_matches_eager(hidden: int, intermediate: int, swiglu_limit: float | None):
    pytest.importorskip("liger_kernel")
    eager = resolve_op("swiglu_mlp", "standard", "eager").wrapper
    other = resolve_op("swiglu_mlp", "standard", "liger_kernel").wrapper
    torch.manual_seed(0)
    mlp = _tiny_qwen3_mlp(hidden, intermediate).to(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(2, 16, mlp.hidden_size, device="cuda", dtype=torch.bfloat16)

    x_e = x.detach().requires_grad_(True)
    x_o = x.detach().requires_grad_(True)
    mlp_e = _tiny_qwen3_mlp(hidden, intermediate).to(device="cuda", dtype=torch.bfloat16)
    mlp_o = _tiny_qwen3_mlp(hidden, intermediate).to(device="cuda", dtype=torch.bfloat16)
    mlp_e.load_state_dict(mlp.state_dict())
    mlp_o.load_state_dict(mlp.state_dict())
    out_e = eager(*_mlp_args(mlp_e, x_e), swiglu_limit=swiglu_limit)
    out_o = other(*_mlp_args(mlp_o, x_o), swiglu_limit=swiglu_limit)
    assert torch.allclose(out_e, out_o, atol=SWIGLU_FUSED_ATOL, rtol=SWIGLU_FUSED_RTOL)

    go = torch.randn_like(out_e)
    out_e.backward(go)
    out_o.backward(go)
    assert torch.allclose(x_e.grad, x_o.grad, atol=SWIGLU_FUSED_GRAD_ATOL, rtol=SWIGLU_FUSED_GRAD_RTOL)
    for param_e, param_o in zip(mlp_e.parameters(), mlp_o.parameters(), strict=True):
        assert torch.allclose(param_e.grad, param_o.grad, atol=SWIGLU_FUSED_GRAD_ATOL, rtol=SWIGLU_FUSED_GRAD_RTOL)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="liger SwiGLU needs a GPU")
def test_liger_swiglu_limit_matches_fp32_activation_contract():
    """Catch BF16 rounding between the clamp and fused SiLU multiplication."""
    pytest.importorskip("liger_kernel")
    hidden, intermediate = 64, 128
    x = torch.zeros(2, 16, hidden, device="cuda", dtype=torch.bfloat16)
    x[..., 0] = 1
    gate_w = torch.zeros(intermediate, hidden, device="cuda", dtype=torch.bfloat16)
    up_w = torch.zeros_like(gate_w)
    gate_w[:, 0] = -0.5
    up_w[:, 0] = 0.9921875
    down_w = torch.zeros(hidden, intermediate, device="cuda", dtype=torch.bfloat16)
    down_w[:, :hidden] = torch.eye(hidden, device="cuda", dtype=torch.bfloat16)
    tensors = (x, gate_w, up_w, down_w)

    eager_args = [tensor.detach().clone().requires_grad_(True) for tensor in tensors]
    liger_args = [tensor.detach().clone().requires_grad_(True) for tensor in tensors]
    empty = x.new_empty(0)
    eager = resolve_op("swiglu_mlp", "standard", "eager").wrapper
    liger = resolve_op("swiglu_mlp", "standard", "liger_kernel").wrapper
    output_eager = eager(
        eager_args[0],
        eager_args[1],
        empty,
        eager_args[2],
        empty,
        eager_args[3],
        empty,
        swiglu_limit=1.0,
    )
    output_liger = liger(
        liger_args[0],
        liger_args[1],
        empty,
        liger_args[2],
        empty,
        liger_args[3],
        empty,
        swiglu_limit=1.0,
    )

    torch.testing.assert_close(output_liger, output_eager, rtol=0, atol=5e-4)
    grad_output = torch.randn_like(output_eager)
    output_eager.backward(grad_output)
    output_liger.backward(grad_output)
    for actual, expected in zip(liger_args, eager_args, strict=True):
        torch.testing.assert_close(
            actual.grad,
            expected.grad,
            atol=SWIGLU_FUSED_GRAD_ATOL,
            rtol=SWIGLU_FUSED_GRAD_RTOL,
        )


def _tiny_gemma3_mlp(hidden_size: int = 64, intermediate_size: int = 128) -> Gemma3MLP:
    config = Gemma3TextConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_activation="gelu_pytorch_tanh",
    )
    return Gemma3MLP(config)


def test_geglu_eager_matches_hf():
    torch.manual_seed(0)
    mlp_h = _tiny_gemma3_mlp()
    mlp_e = _tiny_gemma3_mlp()
    mlp_e.load_state_dict(mlp_h.state_dict())
    x = torch.randn(2, 16, mlp_h.hidden_size, dtype=torch.float32)

    x_h = x.detach().requires_grad_(True)
    out_h = mlp_h(x_h)

    x_e = x.detach().requires_grad_(True)
    wrapper = resolve_op("swiglu_mlp", "geglu", "eager").wrapper
    out_e = wrapper(*_mlp_args(mlp_e, x_e))
    assert torch.allclose(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = torch.randn_like(out_e)
    out_h.backward(go)
    out_e.backward(go)
    assert torch.allclose(x_e.grad, x_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    for param_e, param_h in zip(mlp_e.parameters(), mlp_h.parameters(), strict=True):
        assert torch.allclose(param_e.grad, param_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_geglu_eager_matches_biased_linears():
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
    out_h = down_h(F.gelu(gate_h(x_h), approximate="tanh") * up_h(x_h))

    x_e = x.detach().requires_grad_(True)
    wrapper = resolve_op("swiglu_mlp", "geglu", "eager").wrapper
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


@pytest.mark.parametrize(
    ("impl", "device"),
    (
        ("eager", "cpu"),
        pytest.param("eager", "cuda", marks=pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="needs CUDA")),
        pytest.param(
            "liger_kernel",
            "cuda",
            marks=pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Liger GeGLU needs CUDA"),
        ),
    ),
)
def test_geglu_autocast_forward_backward_matches_pytorch(impl: str, device: str):
    if impl == "liger_kernel":
        pytest.importorskip("liger_kernel")

    torch.manual_seed(12)
    hidden, intermediate = 64, 128
    tensors = (
        torch.randn(2, 8, hidden, device=device) * 0.1,
        torch.randn(intermediate, hidden, device=device) * 0.1,
        torch.randn(intermediate, device=device) * 0.1,
        torch.randn(intermediate, hidden, device=device) * 0.1,
        torch.randn(intermediate, device=device) * 0.1,
        torch.randn(hidden, intermediate, device=device) * 0.1,
        torch.randn(hidden, device=device) * 0.1,
    )
    reference_args = tuple(tensor.detach().clone().requires_grad_(True) for tensor in tensors)
    actual_args = tuple(tensor.detach().clone().requires_grad_(True) for tensor in tensors)

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        x, gate_w, gate_b, up_w, up_b, down_w, down_b = reference_args
        expected = F.linear(
            F.gelu(F.linear(x, gate_w, gate_b), approximate="tanh") * F.linear(x, up_w, up_b), down_w, down_b
        )
        actual = resolve_op("swiglu_mlp", "geglu", impl).wrapper(*actual_args)

    grad_output = torch.randn_like(expected)
    expected.backward(grad_output)
    actual.backward(grad_output)

    torch.testing.assert_close(actual, expected, atol=SWIGLU_FUSED_ATOL, rtol=SWIGLU_FUSED_RTOL)
    for actual_arg, reference_arg in zip(actual_args, reference_args, strict=True):
        torch.testing.assert_close(
            actual_arg.grad,
            reference_arg.grad,
            atol=SWIGLU_FUSED_GRAD_ATOL,
            rtol=SWIGLU_FUSED_GRAD_RTOL,
        )


@pytest.mark.parametrize("with_bias", (False, True), ids=("no-bias", "bias"))
def test_geglu_liger_empty_input_falls_back_to_eager_backward(with_bias: bool):
    hidden, intermediate = 4, 8
    x = torch.empty(2, 0, hidden, requires_grad=True)
    gate_w = torch.randn(intermediate, hidden, requires_grad=True)
    up_w = torch.randn(intermediate, hidden, requires_grad=True)
    down_w = torch.randn(hidden, intermediate, requires_grad=True)
    if with_bias:
        gate_b = torch.randn(intermediate, requires_grad=True)
        up_b = torch.randn(intermediate, requires_grad=True)
        down_b = torch.randn(hidden, requires_grad=True)
    else:
        gate_b = torch.empty(0, requires_grad=True)
        up_b = torch.empty(0, requires_grad=True)
        down_b = torch.empty(0, requires_grad=True)

    entry = OpEntry(
        "swiglu_mlp",
        "geglu",
        "liger-empty-test",
        description="Test-only Liger GeGLU entry",
        forward=geglu_liger.forward,
        backward=geglu_liger.backward,
    )
    assert entry.wrapper is not None
    output = entry.wrapper(x, gate_w, gate_b, up_w, up_b, down_w, down_b)

    assert output.shape == x.shape
    output.sum().backward()
    for tensor in (x, gate_w, up_w, down_w):
        torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor))
    for bias in (gate_b, up_b, down_b):
        if with_bias:
            torch.testing.assert_close(bias.grad, torch.zeros_like(bias))
        else:
            assert bias.grad is None


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="liger GeGLU needs a GPU")
@pytest.mark.parametrize("hidden, intermediate", [(64, 128), (128, 256)])
def test_geglu_liger_matches_eager(hidden: int, intermediate: int):
    pytest.importorskip("liger_kernel")
    eager = resolve_op("swiglu_mlp", "geglu", "eager").wrapper
    other = resolve_op("swiglu_mlp", "geglu", "liger_kernel").wrapper
    torch.manual_seed(0)
    mlp = _tiny_gemma3_mlp(hidden, intermediate).to(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(2, 16, mlp.hidden_size, device="cuda", dtype=torch.bfloat16)

    x_e = x.detach().requires_grad_(True)
    x_o = x.detach().requires_grad_(True)
    mlp_e = _tiny_gemma3_mlp(hidden, intermediate).to(device="cuda", dtype=torch.bfloat16)
    mlp_o = _tiny_gemma3_mlp(hidden, intermediate).to(device="cuda", dtype=torch.bfloat16)
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
