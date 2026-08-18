"""RMSNorm ops coverage for the BAGEL Qwen2-MoT backbone."""

from __future__ import annotations

import copy

import pytest
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm as TransformersQwen2RMSNorm

import veomni.models.seed_omni.modules.bagel.qwen2_mot.accelerated as accelerated
from veomni.ops.dispatch import OpSlot
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


def test_qwen2_mot_rms_norm_dispatches_configured_kernel(monkeypatch):
    captured = {}

    class RecordingSlot:
        use_non_eager_impl = True

        def __call__(self, hidden_states, weight, eps):
            captured["hidden_states"] = hidden_states
            captured["weight"] = weight
            captured["eps"] = eps
            return hidden_states.clone()

    monkeypatch.setattr(accelerated, "veomni_rms_norm", RecordingSlot())

    norm = accelerated.Qwen2RMSNormAccelerated(8, eps=1e-5)
    hidden_states = torch.randn(2, 3, 8)
    output = norm(hidden_states)

    torch.testing.assert_close(output, hidden_states)
    assert captured["hidden_states"] is hidden_states
    assert captured["weight"] is norm.weight
    assert captured["eps"] == norm.variance_epsilon


def test_qwen2_mot_rms_norm_empty_input_skips_fused_kernel(monkeypatch):
    class FailingSlot:
        use_non_eager_impl = True

        def __call__(self, *args, **kwargs):
            raise AssertionError("fused RMSNorm must not receive an empty input")

    monkeypatch.setattr(accelerated, "veomni_rms_norm", FailingSlot())

    norm = accelerated.Qwen2RMSNormAccelerated(128)
    hidden_states = torch.empty(0, 28, 128, requires_grad=True)
    output = norm(hidden_states)
    output.sum().backward()

    assert output.shape == hidden_states.shape
    assert hidden_states.grad is not None
    assert norm.weight.grad is not None
    assert torch.count_nonzero(norm.weight.grad).item() == 0


def test_qwen2_mot_rms_norm_preserves_state_dict_contract():
    source = TransformersQwen2RMSNorm(128)
    target = TransformersQwen2RMSNorm(128)
    source.weight.data.normal_()

    assert source.state_dict().keys() == target.state_dict().keys()
    target.load_state_dict(source.state_dict())
    torch.testing.assert_close(target.weight, source.weight)


def _run_forward_backward(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output = module(hidden_states)
    output.backward(grad_output)
    assert hidden_states.grad is not None
    assert module.weight.grad is not None
    return output.detach(), hidden_states.grad.detach().clone(), module.weight.grad.detach().clone()


@pytest.mark.parametrize(
    "shape",
    [
        (7, 28, 128),
        (7, 4, 128),
        (7, 3584),
    ],
)
@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Liger RMSNorm requires CUDA")
def test_qwen2_mot_liger_rms_norm_matches_eager(monkeypatch, shape):
    pytest.importorskip("liger_kernel")
    torch.manual_seed(2025)
    device = get_device_type()

    eager_norm = TransformersQwen2RMSNorm(shape[-1]).to(device=device, dtype=torch.bfloat16)
    liger_norm = accelerated.Qwen2RMSNormAccelerated(shape[-1]).to(device=device, dtype=torch.bfloat16)
    liger_norm.load_state_dict(copy.deepcopy(eager_norm.state_dict()))
    eager_input = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
    liger_input = eager_input.detach().clone().requires_grad_()
    grad_output = torch.randn_like(eager_input)

    eager_result = _run_forward_backward(eager_norm, eager_input, grad_output)

    liger_slot = OpSlot("rms_norm", "standard")
    liger_slot.bind("liger_kernel")
    monkeypatch.setattr(accelerated, "veomni_rms_norm", liger_slot)
    liger_result = _run_forward_backward(liger_norm, liger_input, grad_output)

    for eager_value, liger_value in zip(eager_result, liger_result, strict=True):
        torch.testing.assert_close(liger_value, eager_value, atol=2e-2, rtol=2e-2)
