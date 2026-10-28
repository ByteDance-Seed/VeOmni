from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.ops.kernels.moe_experts.shared.swiglu import (
    swiglu_activation,
    swiglu_activation_backward,
    swiglu_oai,
    swiglu_oai_backward,
)


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_swiglu_oai_moe_variant_registers_eager_gpu_and_npu_rows():
    from veomni.ops.registry import OP_REGISTRY

    impls = {entry.impl for entry in OP_REGISTRY.list_entries("moe_experts", "swiglu_oai")}

    # Only the merged-fc1 paths implement SwiGLU-OAI, so e.g. fused_quack must not resolve.
    assert impls == {"eager", "fused_triton", "fused_npu"}


def test_swiglu_oai_manual_backward_matches_autograd():
    torch.manual_seed(0)
    alpha = 1.702
    gate = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    up = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    grad_output = torch.randn_like(gate)

    output = swiglu_oai(gate, up, alpha)
    expected_grad_gate, expected_grad_up = torch.autograd.grad(output, (gate, up), grad_output)
    actual_grad_gate, actual_grad_up = swiglu_oai_backward(grad_output, gate.detach(), up.detach(), alpha)

    torch.testing.assert_close(actual_grad_gate, expected_grad_gate)
    torch.testing.assert_close(actual_grad_up, expected_grad_up)


def test_swiglu_oai_is_not_standard_swiglu():
    gate = torch.tensor([[0.5, -0.5]])
    up = torch.tensor([[0.25, -0.25]])

    output = swiglu_oai(gate, up, 1.702)
    standard = torch.nn.functional.silu(gate) * up

    assert not torch.equal(output, standard)


def test_swiglu_oai_clamp_backward_masks_saturated_gate_and_up_values():
    limit = 0.7
    alpha = 1.702
    raw_gate = torch.tensor([[-1.0, 0.2, 0.7, 1.0]], dtype=torch.float64, requires_grad=True)
    raw_up = torch.tensor([[-1.0, -0.7, 0.7, 1.0]], dtype=torch.float64, requires_grad=True)
    grad_output = torch.tensor([[0.3, -0.4, 0.5, -0.6]], dtype=torch.float64)

    gate = raw_gate.clamp(max=limit)
    up = raw_up.clamp(min=-limit, max=limit)
    output = swiglu_oai(gate, up, alpha)
    expected_grad_gate, expected_grad_up = torch.autograd.grad(output, (raw_gate, raw_up), grad_output)

    actual_grad_gate, actual_grad_up = swiglu_oai_backward(grad_output, gate.detach(), up.detach(), alpha)
    actual_grad_gate.masked_fill_(raw_gate.detach() > limit, 0)
    actual_grad_up.masked_fill_((raw_up.detach() < -limit) | (raw_up.detach() > limit), 0)

    torch.testing.assert_close(actual_grad_gate, expected_grad_gate)
    torch.testing.assert_close(actual_grad_up, expected_grad_up)


def test_swiglu_oai_matches_minimax_eager_gate_bitwise_in_bf16():
    """Every bf16 op rounds, so the fused activation must use the eager op order exactly."""
    modeling = pytest.importorskip("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")
    limit, alpha = 7.0, 1.702
    experts = modeling.MiniMaxM3VLExperts.__new__(modeling.MiniMaxM3VLExperts)
    experts.__dict__.update(vars(SimpleNamespace(swiglu_limit=limit, swiglu_alpha=alpha)))
    torch.manual_seed(0)
    gate_up = (4 * torch.randn(64, 2 * 256)).to(torch.bfloat16)

    gate, up = gate_up.chunk(2, dim=-1)
    output = swiglu_oai(gate.clamp(max=limit), up.clamp(min=-limit, max=limit), alpha)

    torch.testing.assert_close(output, experts._apply_gate(gate_up), rtol=0, atol=0)


def test_swiglu_activation_without_alpha_is_bitwise_the_standard_swiglu():
    """``swiglu_alpha=None`` must keep every existing fused-MoE model's exact bf16 op order."""
    torch.manual_seed(0)
    gate, up, grad = (torch.randn(64, 32).to(torch.bfloat16) for _ in range(3))

    activation = swiglu_activation(gate, up, None)
    grad_gate, grad_up = swiglu_activation_backward(grad, gate, up, None)
    # The EP backward recomputes the activation once and reuses silu(gate).
    recomputed, gate_activation = swiglu_activation(gate, up, None, return_gate_activation=True)
    reused_grad_gate, reused_grad_up = swiglu_activation_backward(grad, gate, up, None, gate_activation)
    assert torch.equal(recomputed, activation)
    assert torch.equal(reused_grad_gate, grad_gate) and torch.equal(reused_grad_up, grad_up)

    silu_gate = torch.ops.aten.silu(gate)
    assert torch.equal(activation, silu_gate * up)
    assert torch.equal(grad_gate, torch.ops.aten.silu_backward(grad * up, gate))
    assert torch.equal(grad_up, silu_gate * grad)


def test_swiglu_activation_with_alpha_is_swiglu_oai():
    torch.manual_seed(0)
    gate, up, grad = (torch.randn(64, 32) for _ in range(3))
    alpha = 1.702

    assert torch.equal(swiglu_activation(gate, up, alpha), swiglu_oai(gate, up, alpha))
    for actual, expected in zip(
        swiglu_activation_backward(grad, gate, up, alpha), swiglu_oai_backward(grad, gate, up, alpha)
    ):
        assert torch.equal(actual, expected)
