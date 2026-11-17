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

"""Gated MLP eager vs HF, and fused impls vs eager.

Every row takes the MLP module and returns the block's output, so the calls below
pass ``(mlp, x)`` -- the same shape a model's patched forward uses.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import DeepseekV4Config, Gemma3TextConfig, Qwen3Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts
from transformers.models.gemma3.modeling_gemma3 import Gemma3MLP
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from scripts.ci.accelerator.launch_plan import LaunchSpec
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
from veomni.ops.activation import supported_activations
from veomni.ops.kernels.gated_mlp.geglu import liger_kernel as geglu_liger
from veomni.ops.kernels.gated_mlp.geglu.eager import gelu_tanh_mul_backward
from veomni.ops.kernels.gated_mlp.standard import liger_kernel as standard_liger
from veomni.utils.device import IS_CUDA_AVAILABLE
from veomni.utils.import_utils import is_package_available


#: Nothing in this file launches a worker, so a case that never asks for a GPU
#: should not be charged one. ``default_plan`` assumes ``accelerator``, so an
#: unmarked case books a card it would leave idle.
_cpu = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _empty_bias(weight: Tensor) -> Tensor:
    return weight.new_empty(0)


def _bias(linear: nn.Linear) -> Tensor:
    return linear.bias if linear.bias is not None else _empty_bias(linear.weight)


def _mlp_args(mlp: nn.Module, x: Tensor) -> tuple[nn.Module, Tensor]:
    return mlp, x


def _mlp_of(gate: nn.Linear, up: nn.Linear, down: nn.Linear) -> nn.Module:
    """A stand-in MLP block around three projections."""
    mlp = nn.Module()
    mlp.gate_proj, mlp.up_proj, mlp.down_proj = gate, up, down
    return mlp


def _mlp_from_tensors(
    gate_w: Tensor,
    gate_b: Tensor,
    up_w: Tensor,
    up_b: Tensor,
    down_w: Tensor,
    down_b: Tensor,
) -> nn.Module:
    """A stand-in MLP block whose projections are the given tensors.

    An empty bias means the projection has none, which is exactly how the rows read
    it: ``nn.Linear(bias=False)``, so ``.grad`` stays ``None`` for a bias the model
    does not have.
    """
    layers = []
    for weight, bias in ((gate_w, gate_b), (up_w, up_b), (down_w, down_b)):
        layer = nn.Linear(weight.shape[1], weight.shape[0], bias=bias.numel() > 0)
        layer.weight = nn.Parameter(weight.detach().clone())
        layer.bias = nn.Parameter(bias.detach().clone()) if bias.numel() > 0 else None
        layers.append(layer)
    return _mlp_of(*layers)


def _reference(mlp: nn.Module, x: Tensor, *, tanh_gelu: bool = False, swiglu_limit: float | None = None):
    """The plain-PyTorch MLP the rows are compared against."""
    gate, up = mlp.gate_proj(x), mlp.up_proj(x)
    if swiglu_limit is not None:
        gate = gate.float().clamp(max=swiglu_limit)
        up = up.float().clamp(min=-swiglu_limit, max=swiglu_limit)
    activation = F.gelu(gate, approximate="tanh") if tanh_gelu else F.silu(gate)
    hidden_state = activation * up
    if hidden_state.dtype != x.dtype:
        hidden_state = hidden_state.to(dtype=x.dtype)
    return mlp.down_proj(hidden_state)


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
    wrapper = resolve_op("gated_mlp", "standard", "eager").wrapper
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
    wrapper = resolve_op("gated_mlp", "standard", "eager").wrapper
    out_e = wrapper(_mlp_of(gate_e, up_e, down_e), x_e)
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
    weights = (
        torch.randn(intermediate, hidden, device=device) * 0.1,
        torch.randn(intermediate, device=device) * 0.1,
        torch.randn(intermediate, hidden, device=device) * 0.1,
        torch.randn(intermediate, device=device) * 0.1,
        torch.randn(hidden, intermediate, device=device) * 0.1,
        torch.randn(hidden, device=device) * 0.1,
    )
    reference_mlp = _mlp_from_tensors(*weights)
    actual_mlp = _mlp_from_tensors(*weights)
    x = torch.randn(2, 8, hidden, device=device) * 0.1
    x_ref = x.detach().requires_grad_(True)
    x_act = x.detach().requires_grad_(True)

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        expected = _reference(reference_mlp, x_ref, swiglu_limit=swiglu_limit)
        actual = resolve_op("gated_mlp", "standard", impl).wrapper(
            actual_mlp,
            x_act,
            swiglu_limit=swiglu_limit,
        )

    grad_output = torch.randn_like(expected)
    expected.backward(grad_output)
    actual.backward(grad_output)

    torch.testing.assert_close(actual, expected, atol=SWIGLU_FUSED_ATOL, rtol=SWIGLU_FUSED_RTOL)
    for actual_arg, reference_arg in zip(
        (x_act, *actual_mlp.parameters()), (x_ref, *reference_mlp.parameters()), strict=True
    ):
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
    is the same as ``gated_mlp`` with the DSV4 clamp.
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
    mlp_e = _mlp_from_tensors(
        gate_w,
        gate_w.new_empty(0),
        up_w,
        up_w.new_empty(0),
        experts.down_proj[0].detach(),
        gate_w.new_empty(0),
    )
    wrapper = resolve_op("gated_mlp", "standard", "eager").wrapper
    out_e = wrapper(mlp_e, x_e, swiglu_limit=limit)
    assert torch.allclose(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = torch.randn_like(out_e)
    out_h.backward(go)
    out_e.backward(go)
    assert torch.allclose(x_e.grad, x_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    gate_h_grad, up_h_grad = experts.gate_up_proj.grad[0].chunk(2, dim=0)
    assert torch.allclose(mlp_e.gate_proj.weight.grad, gate_h_grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(mlp_e.up_proj.weight.grad, up_h_grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(
        mlp_e.down_proj.weight.grad, experts.down_proj.grad[0], atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL
    )


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

    mlp = _mlp_from_tensors(gate_w, gate_b, up_w, up_b, down_w, down_b)
    output = standard_liger.wrapper(mlp, x, swiglu_limit=1.0)

    assert output.shape == x.shape
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.zeros_like(x))
    for projection in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        torch.testing.assert_close(projection.weight.grad, torch.zeros_like(projection.weight))
        if with_bias:
            torch.testing.assert_close(projection.bias.grad, torch.zeros_like(projection.bias))
        else:
            assert projection.bias is None


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="liger SwiGLU needs a GPU")
@pytest.mark.parametrize("hidden, intermediate", [(64, 128), (128, 256)])
@pytest.mark.parametrize("swiglu_limit", [None, 1.0])
def test_liger_matches_eager(hidden: int, intermediate: int, swiglu_limit: float | None):
    pytest.importorskip("liger_kernel")
    eager = resolve_op("gated_mlp", "standard", "eager").wrapper
    other = resolve_op("gated_mlp", "standard", "liger_kernel").wrapper
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
    empty = x.new_empty(0)
    eager = resolve_op("gated_mlp", "standard", "eager").wrapper
    liger = resolve_op("gated_mlp", "standard", "liger_kernel").wrapper
    # Fresh copies per row: the two backward passes must not share parameters.
    mlp_eager = _mlp_from_tensors(gate_w, empty, up_w, empty, down_w, empty)
    mlp_liger = _mlp_from_tensors(gate_w, empty, up_w, empty, down_w, empty)
    x_eager = x.detach().clone().requires_grad_(True)
    x_liger = x.detach().clone().requires_grad_(True)

    output_eager = eager(mlp_eager, x_eager, swiglu_limit=1.0)
    output_liger = liger(mlp_liger, x_liger, swiglu_limit=1.0)

    torch.testing.assert_close(output_liger, output_eager, rtol=0, atol=5e-4)
    grad_output = torch.randn_like(output_eager)
    output_eager.backward(grad_output)
    output_liger.backward(grad_output)
    for actual, expected in zip((x_liger, *mlp_liger.parameters()), (x_eager, *mlp_eager.parameters()), strict=True):
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


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", marks=pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))),
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.launch_plan(provider=LaunchSpec()),
                pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="GeGLU CUDA rounding needs a GPU"),
            ],
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("transposed", [False, True])
def test_geglu_backward_matches_autograd_rounding(dtype, transposed, device):
    """Preserve autograd's multiply-then-GELU-backward rounding, including views."""
    torch.manual_seed(31)
    gate = (torch.randn(32, 64, device=device) * 2).to(dtype)
    up = torch.randn_like(gate)
    grad_output = torch.randn_like(gate)
    if transposed:
        gate, up, grad_output = (value.transpose(0, 1) for value in (gate, up, grad_output))
    gate.requires_grad_(True)
    up.requires_grad_(True)
    expected = torch.autograd.grad(F.gelu(gate, approximate="tanh") * up, (gate, up), grad_output)
    actual = gelu_tanh_mul_backward(grad_output, gate, up)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_geglu_eager_matches_hf():
    torch.manual_seed(0)
    mlp_h = _tiny_gemma3_mlp()
    mlp_e = _tiny_gemma3_mlp()
    mlp_e.load_state_dict(mlp_h.state_dict())
    x = torch.randn(2, 16, mlp_h.hidden_size, dtype=torch.float32)

    x_h = x.detach().requires_grad_(True)
    out_h = mlp_h(x_h)

    x_e = x.detach().requires_grad_(True)
    wrapper = resolve_op("gated_mlp", "geglu", "eager").wrapper
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
    wrapper = resolve_op("gated_mlp", "geglu", "eager").wrapper
    out_e = wrapper(_mlp_of(gate_e, up_e, down_e), x_e)
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
    weights = (
        torch.randn(intermediate, hidden, device=device) * 0.1,
        torch.randn(intermediate, device=device) * 0.1,
        torch.randn(intermediate, hidden, device=device) * 0.1,
        torch.randn(intermediate, device=device) * 0.1,
        torch.randn(hidden, intermediate, device=device) * 0.1,
        torch.randn(hidden, device=device) * 0.1,
    )
    reference_mlp = _mlp_from_tensors(*weights)
    actual_mlp = _mlp_from_tensors(*weights)
    x = torch.randn(2, 8, hidden, device=device) * 0.1
    x_ref = x.detach().requires_grad_(True)
    x_act = x.detach().requires_grad_(True)

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        expected = _reference(reference_mlp, x_ref, tanh_gelu=True)
        actual = resolve_op("gated_mlp", "geglu", impl).wrapper(actual_mlp, x_act)

    grad_output = torch.randn_like(expected)
    expected.backward(grad_output)
    actual.backward(grad_output)

    torch.testing.assert_close(actual, expected, atol=SWIGLU_FUSED_ATOL, rtol=SWIGLU_FUSED_RTOL)
    for actual_arg, reference_arg in zip(
        (x_act, *actual_mlp.parameters()), (x_ref, *reference_mlp.parameters()), strict=True
    ):
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

    mlp = _mlp_from_tensors(gate_w, gate_b, up_w, up_b, down_w, down_b)
    output = geglu_liger.wrapper(mlp, x)

    assert output.shape == x.shape
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.zeros_like(x))
    for projection in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        torch.testing.assert_close(projection.weight.grad, torch.zeros_like(projection.weight))
        if with_bias:
            torch.testing.assert_close(projection.bias.grad, torch.zeros_like(projection.bias))
        else:
            assert projection.bias is None


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="liger GeGLU needs a GPU")
@pytest.mark.parametrize("hidden, intermediate", [(64, 128), (128, 256)])
def test_geglu_liger_matches_eager(hidden: int, intermediate: int):
    pytest.importorskip("liger_kernel")
    eager = resolve_op("gated_mlp", "geglu", "eager").wrapper
    other = resolve_op("gated_mlp", "geglu", "liger_kernel").wrapper
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


def test_qat_row_keeps_the_clamp_in_the_model_dtype(monkeypatch):
    """A deployment fused kernel clamps BF16 projection outputs, not FP32 ones.

    Widening the clamp to FP32 (what ``eager`` does) changes the trained function
    for values around ``swiglu_limit``, so the QAT row must not inherit that. The
    quantizers are stubbed to identity here so the assertion isolates the clamp;
    the real ones need an SM90 GPU.
    """
    import veomni.ops.kernels.gated_mlp.standard.qat as qat_impl

    monkeypatch.setattr(qat_impl, "fp8_fake_quant_act", lambda t, block_size=128: t)
    monkeypatch.setattr(qat_impl, "fp8_fake_quant_weight", lambda t, block_size=128: t)

    limit = 10.0
    torch.manual_seed(0)
    # Identity gate/up weights make the projection a known, invertible passthrough
    # of the input, so the expected value is computable without the op.
    hidden_dim = 64
    eye = torch.eye(hidden_dim, dtype=torch.bfloat16)
    down = torch.eye(hidden_dim, dtype=torch.bfloat16)
    x = torch.randn(1, hidden_dim, dtype=torch.bfloat16) * 40

    mlp = _mlp_from_tensors(eye, eye.new_empty(0), eye, eye.new_empty(0), down, down.new_empty(0))
    actual = qat_impl.wrapper(mlp, x, swiglu_limit=limit)

    clamped = x.clamp(max=limit)
    expected = F.silu(clamped) * x.clamp(min=-limit, max=limit)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # The FP32 clamp disagrees, so the assertion above is not vacuous.
    widened = (F.silu(x.float().clamp(max=limit)) * x.float().clamp(min=-limit, max=limit)).bfloat16()
    assert not torch.equal(expected, widened)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("activation", sorted(supported_activations()))
def test_the_eager_row_evaluates_every_activation_it_claims(activation, dtype):
    """The row takes the activation as an argument, so a non-SiLU model can quantize.

    The oracle is ``ACT2FN[activation]``, not a formula restated here. That is the
    point: ``gelu_new`` (``NewGELUActivation``) stages ``torch.pow(x, 3.0)`` while
    ``gelu_pytorch_tanh`` (``PytorchGELUTanh``) writes ``x * x * x``, and the two
    round differently in bf16. A hand-written tanh formula would have accepted the
    wrong function for one of them.

    Parametrized over ``supported_activations()`` rather than a list here, so the
    test and the implementation cannot disagree about what is supported.

    Goes through the row's ``wrapper`` -- the only entry point a model can reach --
    rather than the raw ``forward`` it uses internally.
    """
    import torch.nn.functional as F
    from transformers.activations import ACT2FN

    from veomni.ops.kernels.gated_mlp.standard import eager

    torch.manual_seed(0)
    x = torch.randn(3, 16, dtype=dtype)
    mlp = _mlp_from_tensors(
        torch.randn(32, 16, dtype=dtype) * 0.2,
        torch.empty(0),
        torch.randn(32, 16, dtype=dtype) * 0.2,
        torch.empty(0),
        torch.randn(16, 32, dtype=dtype) * 0.2,
        torch.empty(0),
    )

    out = eager.wrapper(mlp, x, activation=activation)

    expected = F.linear(
        ACT2FN[activation](F.linear(x, mlp.gate_proj.weight)) * F.linear(x, mlp.up_proj.weight),
        mlp.down_proj.weight,
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("activation", sorted(supported_activations()))
def test_the_eager_row_backward_matches_autograd_for_every_activation(activation):
    """The derivative comes from autograd inside the custom Function, and must match.

    A wrong derivative leaves the forward bit-exact while every gradient is off --
    the shape of bug a forward-only test cannot see. It happened once here: an
    early draft dropped the ``up`` factor from ``d/dgate``, and the forward matched
    perfectly.

    Differentiates the row's ``wrapper`` output, which is the graph a model trains.
    """
    import torch.nn.functional as F
    from transformers.activations import ACT2FN

    from veomni.ops.kernels.gated_mlp.standard import eager

    torch.manual_seed(0)
    x = torch.randn(3, 16, dtype=torch.float32, requires_grad=True)
    mlp = _mlp_from_tensors(
        torch.randn(32, 16, dtype=torch.float32) * 0.2,
        torch.empty(0),
        torch.randn(32, 16, dtype=torch.float32) * 0.2,
        torch.empty(0),
        torch.randn(16, 32, dtype=torch.float32) * 0.2,
        torch.empty(0),
    )
    weights = (mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight)

    out = eager.wrapper(mlp, x, activation=activation)
    grad_out = torch.randn_like(out)
    grad_x, *grad_weights = torch.autograd.grad(out, (x, *weights), grad_out)

    reference = F.linear(ACT2FN[activation](F.linear(x, weights[0])) * F.linear(x, weights[1]), weights[2])
    expected = torch.autograd.grad(reference, (x, *weights), grad_out)

    torch.testing.assert_close(grad_x, expected[0], rtol=1e-5, atol=1e-6)
    for got, want in zip(grad_weights, expected[1:], strict=True):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


@_cpu
@pytest.mark.parametrize(("variant", "impl"), [("standard", "liger_kernel"), ("geglu", "liger_kernel")])
def test_every_row_reaches_the_projection_gradients(variant, impl):
    """A row's output must be differentiable w.r.t. the projections it read by weight.

    This is the check a ``forward``-only test cannot make, and its absence hid a real
    bug: these rows call a bare Triton kernel (``swiglu_forward`` / ``geglu_forward``)
    inside a raw ``forward`` that returns ``(output, SavedState)``. ``SavedState`` is
    what ``_make_autograd_fn`` turns into an ``autograd.Function``'s saved tensors, so
    a wrapper that called ``forward`` directly and dropped it returned a tensor with
    **no** gradient path back to ``gate_proj`` / ``up_proj`` -- and to ``x``. The GPU
    default row silently trained only the down projection.

    Liger's kernels are replaced by differentiable stand-ins with the same contract,
    so this runs on CPU and tests the wiring rather than the kernel.
    """
    import sys
    import types

    import torch.nn.functional as F

    # The stand-in forward runs under ``no_grad``, which is what makes a Triton
    # kernel's output carry no ``grad_fn``. A differentiable stand-in would let a
    # gradient leak through the math itself and hide exactly the bug under test.
    def swiglu_forward(a, b):
        with torch.no_grad():
            return a, b, F.silu(a) * b

    def geglu_forward(a, b):
        with torch.no_grad():
            return a, b, F.gelu(a, approximate="tanh") * b

    module = types.ModuleType("liger_kernel.ops.swiglu")
    module.swiglu_forward = swiglu_forward
    module.swiglu_backward = lambda a, b, g: (torch.ops.aten.silu_backward(g * b, a), g * F.silu(a))
    geglu = types.ModuleType("liger_kernel.ops.geglu")
    geglu.geglu_forward = geglu_forward
    geglu.geglu_backward = lambda a, b, g: (
        torch.ops.aten.gelu_backward(g * b, a, approximate="tanh"),
        g * F.gelu(a, approximate="tanh"),
    )
    monkeypatched = {
        "liger_kernel.ops.swiglu": module,
        "liger_kernel.ops.geglu": geglu,
    }
    saved = {name: sys.modules.get(name) for name in monkeypatched}
    sys.modules.update(monkeypatched)
    try:
        from veomni.ops.kernels.gated_mlp.geglu import liger_kernel as geglu_liger
        from veomni.ops.kernels.gated_mlp.standard import liger_kernel as standard_liger

        row = standard_liger if variant == "standard" else geglu_liger
        if row is standard_liger:
            # The row imports the kernel lazily, by name, on each call.
            row.forward.__globals__.pop("swiglu_forward", None)

        mlp = _tiny_qwen3_mlp(16, 32) if variant == "standard" else _tiny_gemma3_mlp(16, 32)
        x = torch.randn(2, 4, 16, requires_grad=True)
        row.wrapper(mlp, x).sum().backward()
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    assert x.grad is not None and torch.any(x.grad != 0), "input gradient is missing"
    for name in ("gate_proj", "up_proj", "down_proj"):
        weight = getattr(mlp, name).weight
        assert weight.grad is not None and torch.any(weight.grad != 0), f"{name} got no gradient"


@_cpu
def test_the_supported_set_is_the_bounded_whitelist():
    """Five names, chosen for what the repository runs plus the ones worth pinning.

    The set is explicit rather than derived from ``ACT2FN`` on purpose: deriving
    means every activation transformers ships becomes a capability claimed here,
    including ones whose backward cannot be reproduced (``mish``). See
    ``veomni/ops/activation.py`` for the reasoning.

    Everything outside it takes the model's projection fallback, which calls its own
    ``act_fn`` and is exact -- so this test also pins that the *fallback* is what a
    narrower set buys, not a loss of correctness.
    """
    from veomni.ops.activation import SUPPORTED_ACTIVATIONS, normalize_activation, supports

    assert set(SUPPORTED_ACTIVATIONS) == {"silu", "swish", "gelu", "gelu_new", "gelu_pytorch_tanh"}

    for name in SUPPORTED_ACTIVATIONS:
        assert supports(name)
        assert normalize_activation(name)

    # Names transformers has but this repository does not claim to run through a row.
    for name in ("relu", "mish", "laplace", "gelu_10", "quick_gelu", "sigmoid"):
        assert not supports(name)
        with pytest.raises(ValueError, match="Unsupported MLP activation"):
            normalize_activation(name)


@_cpu
def test_a_linear_subclass_projection_runs_its_own_forward():
    """Only an exact ``nn.Linear`` may be read by weight; a subclass runs its own forward.

    torchao's ``Float8Linear`` subclasses ``nn.Linear`` and does its float8 GEMM in
    ``forward``. An ``isinstance`` check would let the fused row read its weight and
    silently skip float8, so the check must be on the exact type. A quantizing row
    on such a projection is refused instead of falling back.
    """
    from veomni.ops.kernels.gated_mlp._module import render

    class _SubclassLinear(nn.Linear):
        calls = 0

        def forward(self, x):
            type(self).calls += 1
            return super().forward(x)

    mlp = _tiny_qwen3_mlp()
    assert render(mlp, impl="eager", activation="silu").projections is None

    mlp.up_proj = _SubclassLinear(mlp.hidden_size, mlp.intermediate_size, bias=False)
    assert render(mlp, impl="eager", activation="silu").projections is not None

    # The row routes through the projection modules, so the subclass sees the call.
    _SubclassLinear.calls = 0
    resolve_op("gated_mlp", "standard", "eager").wrapper(mlp, torch.randn(2, 4, mlp.hidden_size))
    assert _SubclassLinear.calls == 1

    # A quantizing row would have to fake-quantize a weight this projection does not
    # compute with, so it refuses rather than silently training at full precision.
    with pytest.raises(ValueError, match="nn.Linear subclass"):
        render(mlp, impl="qat", activation="silu")
    with pytest.raises(ValueError, match="nn.Linear subclass"):
        render(mlp, impl="torchao_nvfp4", activation="silu")


@_cpu
def test_render_routes_each_row_by_shape_and_activation():
    """Every row decides weights-or-projections the same way, and only that once.

    The decision lives in ``gated_mlp._module.render`` rather than in modeling, so a
    model's MLP forward is one handle call with no branch. What it decides has to
    hold for each row: ``swish`` is HF's alias of the SiLU a row implements (so it
    takes the weights path, not the fallback), and any activation a row does not
    implement -- or any projection it cannot read -- takes the projection modules.
    """
    from veomni.ops.kernels.gated_mlp._module import render

    mlp = _tiny_qwen3_mlp()

    for impl in ("eager", "qat", "torchao_nvfp4", "liger_kernel"):
        for activation in ("silu", "swish"):
            assert render(mlp, impl=impl, activation=activation).projections is None, (impl, activation)

    # Liger's swiglu kernels are SiLU only, and ``render`` is the only place that
    # knows it: a ``gelu`` model never enters the row, so there is no second guard
    # inside ``liger_kernel.forward`` to keep in sync (or to test separately).
    assert render(mlp, impl="liger_kernel", activation="gelu").projections is not None
    for impl in ("eager", "qat", "torchao_nvfp4"):
        assert render(mlp, impl=impl, activation="gelu").projections is None

    # An activation outside the set is never served by a row.
    for impl in ("eager", "liger_kernel"):
        assert render(mlp, impl=impl, activation="mish").projections is not None


@_cpu
@pytest.mark.parametrize("activation", ["relu", "mish", "quick_gelu", "gelu_10"])
def test_the_eager_row_runs_an_activation_outside_the_whitelist_exactly(activation):
    """The reference row keeps every config that worked before this family existed.

    ``veomni/ops/activation.py`` serves five names, and a config outside them cannot
    be evaluated by a row. The *reference* row must still run it: the block owns an
    ``act_fn``, which is exactly what the model called before these rows existed and
    is correct for any activation. Before this test the row called
    ``apply_activation`` there and raised, so a ``relu`` model that trained on
    ``main`` stopped loading -- the fallback the docs promised did not exist.

    A quantizing row still raises for those names, because falling back would train
    at full precision under a config that said otherwise.
    """
    from torch import nn
    from transformers.activations import ACT2FN

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(16, 32, bias=False)
            self.up_proj = nn.Linear(16, 32, bias=False)
            self.down_proj = nn.Linear(32, 16, bias=False)
            self.act_fn = ACT2FN[activation]

    mlp = _MLP()
    x = torch.randn(2, 4, 16)

    out = resolve_op("gated_mlp", "standard", "eager").wrapper(mlp, x, activation=activation)
    expected = mlp.down_proj(mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x))
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@_cpu
def test_render_refuses_a_quantizing_row_it_cannot_serve():
    """A quantizing row raises rather than fall back, and says which knob to turn.

    Falling back would train at full precision under a config that claimed
    quantization -- a silent correctness bug, not a graceful degradation.
    """
    from veomni.ops.kernels.gated_mlp._module import render

    mlp = _tiny_qwen3_mlp()

    with pytest.raises(ValueError, match="no quantizing row evaluates that activation"):
        render(mlp, impl="qat", activation="mish")

    class _SubclassLinear(nn.Linear):
        pass

    mlp.up_proj = _SubclassLinear(mlp.hidden_size, mlp.intermediate_size, bias=False)
    with pytest.raises(ValueError, match="nn.Linear subclass"):
        render(mlp, impl="qat", activation="silu")

    # Both messages name `eager` as the way to keep the projection path.
    for impl, activation in (("qat", "mish"), ("torchao_nvfp4", "mish")):
        with pytest.raises(ValueError, match="gated_mlp_implementation='eager'"):
            render(mlp, impl=impl, activation=activation)


@_cpu
@pytest.mark.parametrize("activation", ["silu", "gelu", "gelu_pytorch_tanh", "gelu_new"])
def test_the_eager_row_compiles_fullgraph(activation):
    """A compiled forward must not regress against the plain projection path.

    A model whose ``hidden_act`` is not SiLU compiled and trained before these rows
    existed, so failing to compile afterwards would regress exactly the models this
    change serves. ``fullgraph=True`` is the strict form: it rejects any graph break,
    including the ones a hand-written backward used to cause -- rebuilding a
    derivative with ``requires_grad_(True)`` fails with
    ``Unsupported: Unsupported Tensor.requires_grad_() call``, and ``torch.func.vjp``
    refuses to run inside ``saved_tensors_hooks``, which is how activation offload
    works. The row is an opaque wrapper now, so autograd builds the backward and
    Dynamo traces it; this pins that.

    ``backend="eager"`` keeps it to the graph capture itself.
    """
    import torch

    from veomni.ops.kernels.gated_mlp.standard import eager

    torch.manual_seed(0)
    mlp = _tiny_qwen3_mlp(16, 32)
    x = torch.randn(2, 4, 16, requires_grad=True)

    def step(module, inp):
        return eager.wrapper(module, inp, activation=activation)

    expected = step(mlp, x)
    actual = torch.compile(step, fullgraph=True, backend="eager")(mlp, x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("activation", ["silu", "gelu", "gelu_new"])
def test_the_eager_row_backward_survives_activation_offload(activation):
    """The row must run inside ``saved_tensors_hooks``, which is how offload works.

    ``BaseTrainer``/``OmniTrainer`` build a ``custom_save_on_cpu`` context from
    ``veomni/distributed/offloading.py`` and run backward inside it. The eager row
    used to be a raw ``forward``/``backward`` pair, and its backward called
    ``torch.func.vjp`` for a non-SiLU activation -- which refuses to run there:

        RuntimeError: torch.func.{grad, vjp, jacrev, hessian} don't yet support
        saved tensor hooks

    So a ``gelu`` model raised under offload, where it trained fine before this
    work. The row is an opaque wrapper now; autograd builds the backward, so the
    hooks see individual ops.
    """
    import torch
    from torch.autograd.graph import saved_tensors_hooks

    from veomni.ops import VeomniOp

    op = VeomniOp("gated_mlp", "standard", "eager")

    # Fixed inputs: the comparison is between two backward passes over the same
    # graph, so both must see the same tensors.
    torch.manual_seed(0)
    x = torch.randn(4, 16, dtype=torch.bfloat16)
    weight = torch.randn(32, 16, dtype=torch.bfloat16) * 0.1
    down = torch.randn(16, 32, dtype=torch.bfloat16) * 0.1
    empty = weight.new_empty(0)

    def step():
        mlp = _mlp_from_tensors(weight, empty, weight, empty, down, empty)
        leaf = x.detach().requires_grad_(True)
        out = op(mlp, leaf, activation=activation)
        return torch.autograd.grad(out.sum(), leaf)[0]

    plain = step()
    # Identity pack/unpack: the point is that the context is entered at all.
    with saved_tensors_hooks(lambda t: t, lambda t: t):
        offloaded = step()

    torch.testing.assert_close(offloaded, plain, rtol=0, atol=0)


_requires_torchao = pytest.mark.skipif(not is_package_available("torchao"), reason="torchao_nvfp4 row needs torchao")


@_cpu
@_requires_torchao
@pytest.mark.parametrize("swiglu_limit", [None, 1.0])
def test_torchao_nvfp4_matches_manual_round_trip(swiglu_limit: float | None):
    """The fused row equals the same math with operands quantized by hand, clamp in the model dtype."""
    from veomni.ops.qat.torchao_nvfp4 import nvfp4_fake_quant

    impl = resolve_op("gated_mlp", "standard", "torchao_nvfp4").wrapper
    torch.manual_seed(0)
    mlp = _tiny_qwen3_mlp(64, 128).to(dtype=torch.bfloat16)
    x = torch.randn(2, 16, 64, dtype=torch.bfloat16)

    out = impl(*_mlp_args(mlp, x), swiglu_limit=swiglu_limit)

    x_q = nvfp4_fake_quant(x)
    gate = F.linear(x_q, nvfp4_fake_quant(mlp.gate_proj.weight), None)
    up = F.linear(x_q, nvfp4_fake_quant(mlp.up_proj.weight), None)
    if swiglu_limit is not None:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    hidden_states = F.silu(gate) * up
    expected = F.linear(nvfp4_fake_quant(hidden_states), nvfp4_fake_quant(mlp.down_proj.weight), None)

    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert not torch.equal(out, mlp(x))


@_cpu
@_requires_torchao
def test_torchao_nvfp4_backward_produces_finite_gradients():
    """Straight-through gradients reach the input and every projection weight."""
    impl = resolve_op("gated_mlp", "standard", "torchao_nvfp4").wrapper
    torch.manual_seed(0)
    mlp = _tiny_qwen3_mlp().to(dtype=torch.bfloat16)
    x = torch.randn(2, 16, mlp.hidden_size, dtype=torch.bfloat16, requires_grad=True)

    impl(*_mlp_args(mlp, x)).float().pow(2).sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in mlp.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


@_cpu
@_requires_torchao
def test_torchao_nvfp4_handles_empty_input():
    """Zero-token inputs keep their shape and propagate zero gradients to the MLP weights."""
    impl = resolve_op("gated_mlp", "standard", "torchao_nvfp4").wrapper
    mlp = _tiny_qwen3_mlp().to(dtype=torch.bfloat16)
    x = torch.randn(0, 8, mlp.hidden_size, dtype=torch.bfloat16, requires_grad=True)

    out = impl(*_mlp_args(mlp, x))
    assert out.shape == (0, 8, mlp.hidden_size)
    out.sum().backward()
    for parameter in mlp.parameters():
        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))
