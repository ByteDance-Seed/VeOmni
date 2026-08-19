"""RoPE ops coverage for the BAGEL Qwen2-MoT backbone."""

from __future__ import annotations

import pytest
import torch

import veomni.models.seed_omni.modules.bagel.qwen2_mot.accelerated as accelerated
import veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling as modeling
from veomni.ops.dispatch import OpSlot
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


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
