"""SwiGLU ops coverage for the BAGEL Qwen2-MoT backbone."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP as TransformersQwen2MLP

import veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling as modeling
from veomni.ops.dispatch import OpSlot
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


def _config(hidden_act: str = "silu") -> SimpleNamespace:
    return SimpleNamespace(hidden_size=128, intermediate_size=256, hidden_act=hidden_act)


def test_qwen2_mot_swiglu_dispatches_configured_kernel(monkeypatch):
    captured = {}

    class RecordingSlot:
        use_non_eager_impl = True

        def __call__(self, module, hidden_states):
            captured["module"] = module
            captured["hidden_states"] = hidden_states
            return hidden_states.clone()

    monkeypatch.setattr(modeling, "veomni_swiglu_mlp", RecordingSlot())

    mlp = modeling.Qwen2MLP(_config())
    hidden_states = torch.randn(2, 3, 128)
    output = mlp(hidden_states)

    torch.testing.assert_close(output, hidden_states)
    assert captured["module"] is mlp
    assert captured["hidden_states"] is hidden_states


def test_qwen2_mot_swiglu_rejects_unsupported_fused_activation(monkeypatch):
    class UnexpectedSlot:
        use_non_eager_impl = True

        def __call__(self, *args, **kwargs):
            raise AssertionError("unsupported activation must fail before kernel dispatch")

    monkeypatch.setattr(modeling, "veomni_swiglu_mlp", UnexpectedSlot())

    mlp = modeling.Qwen2MLP(_config(hidden_act="gelu"))
    with pytest.raises(
        ValueError,
        match="Set model.ops_implementation.swiglu_mlp_implementation='eager'",
    ):
        mlp(torch.randn(2, 128))


def test_qwen2_mot_swiglu_empty_input_skips_fused_kernel(monkeypatch):
    class FailingSlot:
        use_non_eager_impl = True

        def __call__(self, *args, **kwargs):
            raise AssertionError("fused SwiGLU must not receive an empty input")

    monkeypatch.setattr(modeling, "veomni_swiglu_mlp", FailingSlot())

    mlp = modeling.Qwen2MLP(_config())
    hidden_states = torch.empty(0, 128, requires_grad=True)
    output = mlp(hidden_states)
    output.sum().backward()

    assert output.shape == hidden_states.shape
    assert hidden_states.grad is not None
    for parameter in mlp.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad).item() == 0


def test_qwen2_mot_swiglu_preserves_state_dict_contract():
    source = TransformersQwen2MLP(_config())
    target = modeling.Qwen2MLP(_config())
    for parameter in source.parameters():
        parameter.data.normal_()

    assert source.state_dict().keys() == target.state_dict().keys()
    target.load_state_dict(source.state_dict())
    for target_parameter, source_parameter in zip(target.parameters(), source.parameters(), strict=True):
        torch.testing.assert_close(target_parameter, source_parameter)


def _run_forward_backward(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    output = module(hidden_states)
    output.backward(grad_output)
    assert hidden_states.grad is not None
    parameter_grads = {}
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None
        parameter_grads[name] = parameter.grad.detach().clone()
    return output.detach(), hidden_states.grad.detach().clone(), parameter_grads


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Liger SwiGLU requires CUDA")
def test_qwen2_mot_liger_swiglu_matches_eager(monkeypatch):
    pytest.importorskip("liger_kernel")
    torch.manual_seed(2026)
    device = get_device_type()

    eager_mlp = modeling.Qwen2MLP(_config()).to(device=device, dtype=torch.bfloat16)
    liger_mlp = copy.deepcopy(eager_mlp)
    eager_input = torch.randn(7, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    liger_input = eager_input.detach().clone().requires_grad_()
    grad_output = torch.randn_like(eager_input)

    monkeypatch.setattr(modeling, "veomni_swiglu_mlp", OpSlot("swiglu_mlp", "standard"))
    eager_output, eager_input_grad, eager_parameter_grads = _run_forward_backward(eager_mlp, eager_input, grad_output)

    liger_slot = OpSlot("swiglu_mlp", "standard")
    liger_slot.bind("liger_kernel")
    monkeypatch.setattr(modeling, "veomni_swiglu_mlp", liger_slot)
    liger_output, liger_input_grad, liger_parameter_grads = _run_forward_backward(liger_mlp, liger_input, grad_output)

    torch.testing.assert_close(liger_output, eager_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(liger_input_grad, eager_input_grad, atol=2e-2, rtol=2e-2)
    assert liger_parameter_grads.keys() == eager_parameter_grads.keys()
    for name in eager_parameter_grads:
        torch.testing.assert_close(
            liger_parameter_grads[name],
            eager_parameter_grads[name],
            atol=2e-2,
            rtol=2e-2,
        )
