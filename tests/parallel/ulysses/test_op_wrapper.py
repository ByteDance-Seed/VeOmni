"""Numerical coverage for async Ulysses op wrappers."""

from __future__ import annotations

import sys
import types

import pytest
import torch

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.distributed.sequence_parallel import op_wrappers as op_wrappers_mod
from veomni.distributed.sequence_parallel.op_wrappers import (
    _EagerRMSNorm,
    _EagerRotary,
    _LigerRMSNorm,
    _LigerRotary,
    _NpuRMSNorm,
    get_op_wrapper,
    pack_saved_state,
    unpack_saved_state,
)
from veomni.ops.config.singleton import get_ops_config, set_ops_config
from veomni.ops.dispatch import OpSlot
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


def _clear_wrapper_cache() -> None:
    op_wrappers_mod._WRAPPERS.clear()


@pytest.fixture(autouse=True)
def _isolate_ops_config():
    previous = get_ops_config()
    _clear_wrapper_cache()
    set_ops_config(None)
    try:
        yield
    finally:
        _clear_wrapper_cache()
        set_ops_config(previous)


def _set_ops(*, rms_norm: str = "eager", rotary_pos_emb: str = "eager") -> None:
    _clear_wrapper_cache()
    set_ops_config(
        OpsImplementationConfig(
            rms_norm_implementation=rms_norm,
            rotary_pos_emb_implementation=rotary_pos_emb,
        )
    )


def _eager_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, *, offset: float = 0.0) -> torch.Tensor:
    x_f = x.float()
    rstd = torch.rsqrt(x_f.square().mean(dim=-1, keepdim=True) + eps)
    normed = x_f * rstd
    scale = offset + weight
    if offset == 0.0:
        return scale * normed.to(x.dtype)
    return (scale.float() * normed).to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _eager_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos_u = cos.unsqueeze(1)
    sin_u = sin.unsqueeze(1)
    return (x * cos_u) + (_rotate_half(x) * sin_u)


def _cos_sin(num_tokens: int, head_dim: int, *, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    angles = torch.randn(num_tokens, head_dim // 2, device=device, dtype=torch.float32)
    return (
        torch.cat((angles.cos(), angles.cos()), dim=-1).to(dtype),
        torch.cat((angles.sin(), angles.sin()), dim=-1).to(dtype),
    )


@pytest.mark.parametrize("shape", [(5, 3, 16), (7, 16), (0, 4, 8)])
def test_eager_rms_norm_matches_autograd(shape: tuple[int, ...]) -> None:
    torch.manual_seed(5101)
    eps = 1e-6
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    weight = (torch.randn(shape[-1], dtype=torch.float64) * 0.1 + 1).requires_grad_()
    grad_output = torch.randn_like(x)

    reference = _eager_rms_norm(x, weight, eps)
    expected_x, expected_weight = torch.autograd.grad(reference, (x, weight), grad_output)

    wrapper = _EagerRMSNorm()
    actual, saved = wrapper.forward(x.detach(), weight.detach(), eps)
    actual_x, actual_weight = wrapper.backward(grad_output, saved)

    torch.testing.assert_close(actual, reference.detach())
    torch.testing.assert_close(actual_x, expected_x)
    torch.testing.assert_close(actual_weight, expected_weight)


def test_eager_rms_norm_qwen3_5_matches_autograd() -> None:
    torch.manual_seed(5102)
    eps = 1e-6
    x = torch.randn(4, 2, 16, dtype=torch.float64, requires_grad=True)
    weight = (0.01 * torch.randn(16, dtype=torch.float64)).requires_grad_()
    grad_output = torch.randn_like(x)

    reference = _eager_rms_norm(x, weight, eps, offset=1.0)
    expected_x, expected_weight = torch.autograd.grad(reference, (x, weight), grad_output)

    wrapper = _EagerRMSNorm(offset=1.0)
    actual, saved = wrapper.forward(x.detach(), weight.detach(), eps)
    actual_x, actual_weight = wrapper.backward(grad_output, saved)

    torch.testing.assert_close(actual, reference.detach())
    torch.testing.assert_close(actual_x, expected_x)
    torch.testing.assert_close(actual_weight, expected_weight)


def test_wrapper_state_uses_autograd_saved_tensor_hooks() -> None:
    wrapper = _EagerRMSNorm()
    packed: list[torch.Tensor] = []

    class _CompoundRMSNorm(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
            output, state = wrapper.forward(x, weight, eps)
            saved: list[torch.Tensor | None] = []
            ctx.state_ref = pack_saved_state(saved, state)
            ctx.save_for_backward(*saved)
            return output

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            state = unpack_saved_state(ctx.saved_tensors, ctx.state_ref)
            grad_x, grad_weight = wrapper.backward(grad_output, state)
            return grad_x, grad_weight, None

    x = torch.randn(3, 8, requires_grad=True)
    weight = torch.randn(8, requires_grad=True)

    def pack_hook(tensor: torch.Tensor) -> torch.Tensor:
        packed.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda tensor: tensor):
        _CompoundRMSNorm.apply(x, weight, 1e-6).sum().backward()

    assert len(packed) == 3
    assert x.grad is not None
    assert weight.grad is not None


def test_pack_saved_state_reuses_tensor_identity() -> None:
    wrapper = _EagerRotary()
    x = torch.randn(3, 2, 8)
    cos, sin = _cos_sin(3, 8, device=x.device, dtype=x.dtype)
    _, q_state = wrapper.forward(x, cos, sin)
    _, k_state = wrapper.forward(x, cos, sin)
    saved: list[torch.Tensor | None] = []

    q_ref = pack_saved_state(saved, q_state)
    k_ref = pack_saved_state(saved, k_state)

    assert len(saved) == 2
    assert q_ref.indices == k_ref.indices
    restored = unpack_saved_state(tuple(saved), q_ref).tensors
    assert restored[0] is cos
    assert restored[1] is sin


def test_npu_qwen3_5_rms_norm_uses_shifted_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch_npu = types.ModuleType("torch_npu")
    observed_scales: list[torch.Tensor] = []

    def npu_rms_norm(x: torch.Tensor, scale: torch.Tensor, eps: float):
        observed_scales.append(scale.detach().clone())
        x_f = x.float()
        rstd = torch.rsqrt(x_f.square().mean(dim=-1, keepdim=True) + eps)
        return (scale.float() * x_f * rstd).to(x.dtype), rstd

    def npu_rms_norm_backward(
        grad_output: torch.Tensor,
        x: torch.Tensor,
        scale: torch.Tensor,
        rstd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_f = x.float()
        scaled_grad = grad_output.float() * scale.float()
        grad_x = rstd * scaled_grad - (rstd.pow(3) / x.shape[-1]) * x_f * (scaled_grad * x_f).sum(dim=-1, keepdim=True)
        grad_scale = (grad_output.float() * x_f * rstd).sum(dim=tuple(range(x.ndim - 1)))
        return grad_x.to(x.dtype), grad_scale.to(scale.dtype)

    fake_torch_npu.npu_rms_norm = npu_rms_norm
    fake_torch_npu.npu_rms_norm_backward = npu_rms_norm_backward
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)
    monkeypatch.setattr(op_wrappers_mod, "IS_NPU_AVAILABLE", True)

    torch.manual_seed(5106)
    eps = 1e-6
    x = torch.randn(4, 2, 16, requires_grad=True)
    weight = (0.01 * torch.randn(16)).requires_grad_()
    grad_output = torch.randn_like(x)
    reference = _eager_rms_norm(x, weight, eps, offset=1.0)
    expected_x, expected_weight = torch.autograd.grad(reference, (x, weight), grad_output)

    wrapper = op_wrappers_mod._build_rms_norm("qwen3_5", "npu")
    assert isinstance(wrapper, _NpuRMSNorm)
    actual, saved = wrapper.forward(x.detach(), weight.detach(), eps)
    actual_x, actual_weight = wrapper.backward(grad_output, saved)

    torch.testing.assert_close(observed_scales[0], 1.0 + weight.detach())
    torch.testing.assert_close(actual, reference.detach())
    torch.testing.assert_close(actual_x, expected_x)
    torch.testing.assert_close(actual_weight, expected_weight)


@pytest.mark.parametrize("shape", [(7, 3, 8), (0, 4, 8)])
def test_eager_rotary_matches_autograd(shape: tuple[int, ...]) -> None:
    torch.manual_seed(5103)
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    cos, sin = _cos_sin(shape[0], shape[-1], device=x.device, dtype=torch.float64)
    grad_output = torch.randn_like(x)

    reference = _eager_rotary(x, cos, sin)
    (expected_x,) = torch.autograd.grad(reference, x, grad_output)

    wrapper = _EagerRotary()
    actual, saved = wrapper.forward(x.detach(), cos, sin)
    (actual_x,) = wrapper.backward(grad_output, saved)

    torch.testing.assert_close(actual, reference.detach())
    torch.testing.assert_close(actual_x, expected_x)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Liger RMSNorm requires CUDA")
def test_liger_rms_norm_matches_eager_and_opslot() -> None:
    pytest.importorskip("liger_kernel")
    torch.manual_seed(5104)
    device = get_device_type()
    dtype = torch.bfloat16
    eps = 1e-6
    x = torch.randn(5, 3, 16, device=device, dtype=dtype)
    weight = (torch.randn(16, device=device, dtype=dtype) * 0.1 + 1).contiguous()
    grad_output = torch.randn_like(x)

    slot = OpSlot("rms_norm", "standard")
    slot.bind("liger_kernel")
    slot_x = x.detach().clone().requires_grad_()
    slot_weight = weight.detach().clone().requires_grad_()
    slot_output = slot(slot_x, slot_weight, eps)
    slot_output.backward(grad_output)

    wrapper = _LigerRMSNorm(offset=0.0, casting_mode="llama")
    actual, saved = wrapper.forward(x, weight, eps)
    actual_x, actual_weight = wrapper.backward(grad_output, saved)
    eager = _eager_rms_norm(x, weight, eps)

    torch.testing.assert_close(actual, slot_output.detach(), atol=0, rtol=0)
    torch.testing.assert_close(actual_x, slot_x.grad, atol=0, rtol=0)
    torch.testing.assert_close(actual_weight, slot_weight.grad, atol=0, rtol=0)
    torch.testing.assert_close(actual, eager, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Liger RoPE requires CUDA")
def test_liger_rotary_matches_eager_forward_backward() -> None:
    pytest.importorskip("liger_kernel")
    torch.manual_seed(5105)
    device = get_device_type()
    dtype = torch.bfloat16
    x = torch.randn(17, 28, 128, device=device, dtype=dtype)
    cos, sin = _cos_sin(17, 128, device=device, dtype=dtype)
    grad_output = torch.randn_like(x)

    leaf = x.detach().clone().requires_grad_()
    reference = _eager_rotary(leaf, cos, sin)
    reference.backward(grad_output)

    wrapper = _LigerRotary()
    actual, saved = wrapper.forward(x, cos, sin)
    (actual_x,) = wrapper.backward(grad_output, saved)

    torch.testing.assert_close(actual, reference.detach(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual_x, leaf.grad, atol=2e-2, rtol=2e-2)


def test_get_op_wrapper_defaults_to_eager_without_ops_config() -> None:
    rms = get_op_wrapper("rms_norm")
    rope = get_op_wrapper("rotary_pos_emb")
    assert isinstance(rms, _EagerRMSNorm)
    assert isinstance(rope, _EagerRotary)


def test_get_op_wrapper_follows_ops_config() -> None:
    _set_ops(rms_norm="eager", rotary_pos_emb="eager")
    assert isinstance(get_op_wrapper("rms_norm"), _EagerRMSNorm)

    if IS_CUDA_AVAILABLE:
        pytest.importorskip("liger_kernel")
        _set_ops(rms_norm="liger_kernel", rotary_pos_emb="liger_kernel")
        assert isinstance(get_op_wrapper("rms_norm"), _LigerRMSNorm)
        assert isinstance(get_op_wrapper("rotary_pos_emb"), _LigerRotary)


def test_get_op_wrapper_rejects_unknown_op() -> None:
    with pytest.raises(KeyError, match="Unsupported async Ulysses op"):
        get_op_wrapper("swiglu_mlp")


def test_unknown_implementation_raises() -> None:
    with pytest.raises(KeyError, match="No async RMSNorm wrapper"):
        _set_ops(rms_norm="not_a_backend")
        get_op_wrapper("rms_norm")
