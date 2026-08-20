"""OpSlot dispatch and Liger kernel parity for BAGEL Qwen2-MoT."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP as TransformersQwen2MLP
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm as TransformersQwen2RMSNorm

import veomni.models.seed_omni.modules.bagel.qwen2_mot.accelerated as accelerated
import veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling as modeling
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


def _run_rms_norm_forward_backward(
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

    eager_result = _run_rms_norm_forward_backward(eager_norm, eager_input, grad_output)

    liger_slot = OpSlot("rms_norm", "standard")
    liger_slot.bind("liger_kernel")
    monkeypatch.setattr(accelerated, "veomni_rms_norm", liger_slot)
    liger_result = _run_rms_norm_forward_backward(liger_norm, liger_input, grad_output)

    for eager_value, liger_value in zip(eager_result, liger_result, strict=True):
        torch.testing.assert_close(liger_value, eager_value, atol=2e-2, rtol=2e-2)


def _cos_sin(
    num_tokens: int,
    head_dim: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    angles = torch.randn(num_tokens, head_dim // 2, device=device, dtype=torch.float32)
    cos_half = angles.cos().to(dtype)
    sin_half = angles.sin().to(dtype)
    return torch.cat((cos_half, cos_half), dim=-1), torch.cat((sin_half, sin_half), dim=-1)


def test_qwen2_mot_rope_dispatches_packed_inputs(monkeypatch):
    captured = {}

    class RecordingSlot:
        use_non_eager_impl = True

        def __call__(self, q, k, cos, sin, **kwargs):
            captured.update(q=q, k=k, cos=cos, sin=sin, kwargs=kwargs)
            return q + 1, k + 2

    monkeypatch.setattr(accelerated, "veomni_apply_rotary_pos_emb", RecordingSlot())

    q = torch.randn(7, 28, 128)
    k = torch.randn(7, 4, 128)
    cos, sin = _cos_sin(7, 128)
    q_output, k_output = accelerated._apply_rotary_pos_emb(q, k, cos, sin)

    assert captured["q"].shape == (1, 28, 7, 128)
    assert captured["k"].shape == (1, 4, 7, 128)
    assert captured["cos"].shape == (1, 7, 128)
    assert captured["sin"].shape == (1, 7, 128)
    assert captured["kwargs"] == {"unsqueeze_dim": 1}
    torch.testing.assert_close(q_output, q + 1)
    torch.testing.assert_close(k_output, k + 2)
    assert q_output.is_contiguous()
    assert k_output.is_contiguous()


def test_qwen2_mot_rope_empty_input_skips_fused_kernel(monkeypatch):
    class FailingSlot:
        use_non_eager_impl = True

        def __call__(self, *args, **kwargs):
            raise AssertionError("fused RoPE must not receive an empty input")

    monkeypatch.setattr(accelerated, "veomni_apply_rotary_pos_emb", FailingSlot())

    q = torch.empty(0, 28, 128, requires_grad=True)
    k = torch.empty(0, 4, 128, requires_grad=True)
    cos = torch.empty(0, 128)
    sin = torch.empty(0, 128)
    q_output, k_output = accelerated._apply_rotary_pos_emb(q, k, cos, sin)
    torch.autograd.backward((q_output.sum(), k_output.sum()))

    assert q_output.shape == q.shape
    assert k_output.shape == k.shape
    assert q.grad is not None
    assert k.grad is not None


def test_qwen2_mot_rope_rejects_partial_fused_dimensions(monkeypatch):
    class UnexpectedSlot:
        use_non_eager_impl = True

        def __call__(self, *args, **kwargs):
            raise AssertionError("partial RoPE must fail before kernel dispatch")

    monkeypatch.setattr(accelerated, "veomni_apply_rotary_pos_emb", UnexpectedSlot())

    q = torch.randn(7, 28, 128)
    k = torch.randn(7, 4, 128)
    cos, sin = _cos_sin(7, 64)
    with pytest.raises(NotImplementedError, match="does not support partial rotary dimensions"):
        accelerated._apply_rotary_pos_emb(q, k, cos, sin)


def test_qwen2_mot_rope_rejects_unsupported_unsqueeze_dimension(monkeypatch):
    class UnexpectedSlot:
        use_non_eager_impl = True

        def __call__(self, *args, **kwargs):
            raise AssertionError("unsupported layout must fail before kernel dispatch")

    monkeypatch.setattr(accelerated, "veomni_apply_rotary_pos_emb", UnexpectedSlot())

    q = torch.randn(7, 28, 128)
    k = torch.randn(7, 4, 128)
    cos, sin = _cos_sin(7, 128)
    with pytest.raises(NotImplementedError, match="requires unsqueeze_dim=1"):
        accelerated._apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=0)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Liger RoPE requires CUDA")
def test_qwen2_mot_liger_rope_matches_eager_forward_backward(monkeypatch):
    pytest.importorskip("liger_kernel")
    torch.manual_seed(2027)
    device = get_device_type()
    dtype = torch.bfloat16

    eager_q_leaf = torch.randn(17, 28, 128, device=device, dtype=dtype, requires_grad=True)
    eager_k_leaf = torch.randn(17, 4, 128, device=device, dtype=dtype, requires_grad=True)
    liger_q_leaf = eager_q_leaf.detach().clone().requires_grad_()
    liger_k_leaf = eager_k_leaf.detach().clone().requires_grad_()
    cos, sin = _cos_sin(17, 128, device=device, dtype=dtype)
    q_grad_output = torch.randn_like(eager_q_leaf)
    k_grad_output = torch.randn_like(eager_k_leaf)

    eager_q_output, eager_k_output = modeling._apply_rotary_pos_emb(
        eager_q_leaf * 1,
        eager_k_leaf * 1,
        cos,
        sin,
    )
    torch.autograd.backward((eager_q_output, eager_k_output), (q_grad_output, k_grad_output))

    liger_slot = OpSlot("rotary_pos_emb", "full")
    liger_slot.bind("liger_kernel")
    monkeypatch.setattr(accelerated, "veomni_apply_rotary_pos_emb", liger_slot)
    liger_q_output, liger_k_output = accelerated._apply_rotary_pos_emb(
        liger_q_leaf * 1,
        liger_k_leaf * 1,
        cos,
        sin,
    )
    torch.autograd.backward((liger_q_output, liger_k_output), (q_grad_output, k_grad_output))

    torch.testing.assert_close(liger_q_output, eager_q_output, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(liger_k_output, eager_k_output, atol=1e-2, rtol=1e-2)
    assert eager_q_leaf.grad is not None
    assert eager_k_leaf.grad is not None
    assert liger_q_leaf.grad is not None
    assert liger_k_leaf.grad is not None
    torch.testing.assert_close(liger_q_leaf.grad, eager_q_leaf.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(liger_k_leaf.grad, eager_k_leaf.grad, atol=2e-2, rtol=2e-2)


def test_rotary_inv_freq_survives_to_bfloat16() -> None:
    from veomni.models.seed_omni.modules.bagel.qwen2_mot.configuration import BagelQwen2MoTConfig
    from veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling import BagelQwen2RotaryEmbedding

    rope = BagelQwen2RotaryEmbedding(BagelQwen2MoTConfig())
    original = rope.inv_freq.detach().clone()
    rope = rope.to(dtype=torch.bfloat16)
    assert rope.inv_freq.dtype == torch.float32
    assert torch.equal(rope.inv_freq.cpu(), original.cpu())


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

    monkeypatch.setattr(accelerated, "veomni_swiglu_mlp", RecordingSlot())

    mlp = accelerated.Qwen2MLPAccelerated(_config())
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

    monkeypatch.setattr(accelerated, "veomni_swiglu_mlp", UnexpectedSlot())

    mlp = accelerated.Qwen2MLPAccelerated(_config(hidden_act="gelu"))
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

    monkeypatch.setattr(accelerated, "veomni_swiglu_mlp", FailingSlot())

    mlp = accelerated.Qwen2MLPAccelerated(_config())
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
    target = TransformersQwen2MLP(_config())
    for parameter in source.parameters():
        parameter.data.normal_()

    assert source.state_dict().keys() == target.state_dict().keys()
    target.load_state_dict(source.state_dict())
    for target_parameter, source_parameter in zip(target.parameters(), source.parameters(), strict=True):
        torch.testing.assert_close(target_parameter, source_parameter)


def _run_swiglu_forward_backward(
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

    eager_mlp = TransformersQwen2MLP(_config()).to(device=device, dtype=torch.bfloat16)
    liger_mlp = accelerated.Qwen2MLPAccelerated(_config()).to(device=device, dtype=torch.bfloat16)
    liger_mlp.load_state_dict(copy.deepcopy(eager_mlp.state_dict()))
    eager_input = torch.randn(7, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    liger_input = eager_input.detach().clone().requires_grad_()
    grad_output = torch.randn_like(eager_input)

    eager_output, eager_input_grad, eager_parameter_grads = _run_swiglu_forward_backward(
        eager_mlp, eager_input, grad_output
    )

    liger_slot = OpSlot("swiglu_mlp", "standard")
    liger_slot.bind("liger_kernel")
    monkeypatch.setattr(accelerated, "veomni_swiglu_mlp", liger_slot)
    liger_output, liger_input_grad, liger_parameter_grads = _run_swiglu_forward_backward(
        liger_mlp, liger_input, grad_output
    )

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
