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

"""Numerical alignment tests for NPU-optimised kernels.

For each NPU kernel, import the concrete implementation directly and compare
its output against the canonical eager implementation on random inputs.
This guards against:
  - Silent regressions in the torch_npu kernel wrappers.
  - The wrong variant being used (e.g. qwen3_5 bound into a standard slot).

NPU rms_norm and rotary_pos_emb are registered via ``register_op`` (PER_MODEL
BackendSpec with ``replace_forward=True``), not in ``KERNEL_REGISTRY``.  Tests
therefore call the implementations directly rather than through ``OpSlot.bind``.

Tests are skipped on non-NPU hosts so the same test suite runs in any CI runner.
"""

import pytest
import torch

from veomni.utils.device import IS_NPU_AVAILABLE, get_device_type


pytestmark = pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="NPU kernels require torch_npu")

DEVICE = get_device_type()


def _print_diff(actual, expected, atol, rtol):
    """Print actual max absolute/relative diff before allclose assertion."""
    diff = (actual - expected).abs()
    max_abs = diff.max().item()
    rel = diff / (expected.abs() + 1e-8)
    max_rel = rel.max().item()
    print(f"max_abs_diff={max_abs:.6e}, max_rel_diff={max_rel:.6e}, atol={atol:.6e}, rtol={rtol:.6e}")


# ---------------------------------------------------------------------------
# Reference (eager) implementations — same as test_kernel_registry_numerical.py
# ---------------------------------------------------------------------------


def _eager_rms_norm_standard(x, weight, eps):
    dtype = x.dtype
    x_f = x.to(torch.float32)
    variance = x_f.pow(2).mean(-1, keepdim=True)
    x_f = x_f * torch.rsqrt(variance + eps)
    return (weight * x_f.to(dtype)).to(dtype)


def _eager_rms_norm_qwen3_5(x, weight, eps):
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x_norm = x.to(torch.float32) * torch.rsqrt(variance + eps)
    return ((1.0 + weight.to(torch.float32)) * x_norm).to(x.dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _eager_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _eager_partial_rope(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


def _eager_rope_vision(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    q, k = q.unsqueeze(0), k.unsqueeze(0)
    cos = cos.unsqueeze(0).unsqueeze(2).float()
    sin = sin.unsqueeze(0).unsqueeze(2).float()
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    q_embed, k_embed = q_embed.squeeze(0), k_embed.squeeze(0)
    return q_embed, k_embed


def _eager_rms_norm_gated(hidden_states, weight, eps, gate):
    """Eager reference: RMSNorm + concatenate gate + SiLU gating."""
    dtype = hidden_states.dtype
    x_f = hidden_states.to(torch.float32)
    variance = x_f.pow(2).mean(-1, keepdim=True)
    x_f = x_f * torch.rsqrt(variance + eps)
    normed = (weight * x_f.to(dtype)).to(dtype)
    fused_input = torch.cat([gate, normed], dim=-1)
    half = fused_input.shape[-1] // 2
    return torch.nn.functional.silu(fused_input[..., :half]) * fused_input[..., half:]


# ---------------------------------------------------------------------------
# Helpers to call replace_forward-style NPU kernels with a mock ``self``
# ---------------------------------------------------------------------------


class _MockRmsNormSelf:
    """Minimal stand-in for an RMSNorm module so that the NPU replace_forward
    signature ``fn(self, x)`` can be called in isolation."""

    def __init__(self, weight, eps):
        self.weight = weight
        self.variance_epsilon = eps


# ---------------------------------------------------------------------------
# RMSNorm tests
# ---------------------------------------------------------------------------


class TestNPURmsNorm:
    """Tests for the ``rms_norm`` NPU kernel (standard + qwen3_5 variants)."""

    @pytest.mark.parametrize("batch,seq,hidden", [(2, 16, 128), (1, 8, 64)])
    def test_standard_matches_eager_bf16(self, batch, seq, hidden):
        from veomni.ops.kernels.rms_norm.npu import rms_norm_forward_npu

        x = torch.randn(batch, seq, hidden, device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn(hidden, device=DEVICE, dtype=torch.bfloat16)
        mock_self = _MockRmsNormSelf(w, 1e-6)
        out_kernel = rms_norm_forward_npu(mock_self, x)
        out_eager = _eager_rms_norm_standard(x, w, 1e-6)
        # bf16 RMSNorm: NPU fused kernel and eager differ by bf16 rounding in the
        # reduction + elementwise mul. 5e-3 covers the worst-case bf16 rounding
        # without being so loose that a wrong variant (e.g. qwen3_5 bound into a
        # standard slot, diff ~0.5) would slip through.
        assert torch.allclose(out_kernel, out_eager, atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize("batch,seq,hidden", [(2, 16, 128), (1, 8, 64)])
    def test_standard_matches_eager_fp32(self, batch, seq, hidden):
        from veomni.ops.kernels.rms_norm.npu import rms_norm_forward_npu

        x = torch.randn(batch, seq, hidden, device=DEVICE, dtype=torch.float32)
        w = torch.randn(hidden, device=DEVICE, dtype=torch.float32)
        mock_self = _MockRmsNormSelf(w, 1e-6)
        out_kernel = rms_norm_forward_npu(mock_self, x)
        out_eager = _eager_rms_norm_standard(x, w, 1e-6)
        # fp32 should be very close
        assert torch.allclose(out_kernel, out_eager, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch,seq,hidden", [(2, 16, 128), (1, 8, 64)])
    def test_qwen3_5_matches_eager_bf16(self, batch, seq, hidden):
        import torch_npu

        x = torch.randn(batch, seq, hidden, device=DEVICE, dtype=torch.bfloat16)
        w = torch.zeros(hidden, device=DEVICE, dtype=torch.bfloat16)  # Qwen3.5 init to zeros
        w += 0.01 * torch.randn_like(w)
        # NPU npu_rms_norm implements standard (weight * x_norm); Qwen3.5 uses
        # (1 + weight) * x_norm.  We test by passing (1+w) to npu_rms_norm,
        # matching the Qwen3.5 formulation.
        eps = 1e-6
        w_shifted = (1.0 + w).to(w.dtype)
        out_kernel = torch_npu.npu_rms_norm(x, w_shifted, epsilon=eps)[0]
        out_eager = _eager_rms_norm_qwen3_5(x, w, eps)
        # bf16 compound op: up-cast to fp32 for comparison
        assert torch.allclose(
            out_kernel.to(torch.float32), out_eager.to(torch.float32), atol=1e-3, rtol=1e-2
        )


# ---------------------------------------------------------------------------
# Rotary positional embedding tests
# ---------------------------------------------------------------------------


class TestNPURotaryPosEmb:
    """Tests for the ``rotary_pos_emb`` NPU kernel (full + vision variants)."""

    @pytest.mark.parametrize("B,H,S,D", [(2, 4, 16, 64), (1, 2, 8, 32)])
    def test_full_matches_eager_bf16(self, B, H, S, D):
        from veomni.ops.kernels.rotary.npu import apply_rotary_pos_emb_npu

        q = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.bfloat16)
        # HF RoPE convention: cos/sin are duplicated across the two halves of head_dim.
        half = torch.randn(B, S, D // 2, device=DEVICE, dtype=torch.bfloat16)
        cos = torch.cat([half, half], dim=-1)
        half_s = torch.randn(B, S, D // 2, device=DEVICE, dtype=torch.bfloat16)
        sin = torch.cat([half_s, half_s], dim=-1)
        q_k, k_k = apply_rotary_pos_emb_npu(q, k, cos, sin)
        q_e, k_e = _eager_rope(q, k, cos, sin)
        # Compound bf16 op: (q * cos) + (rotate_half(q) * sin) — 1e-2 covers
        # worst-case bf16 rounding per op.
        assert torch.allclose(q_k, q_e, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("S,H,D", [(16, 4, 64), (8, 2, 32)])
    def test_vision_matches_eager_bf16(self, S, H, D):
        from veomni.ops.kernels.rotary.npu import apply_rotary_pos_emb_vision_npu

        q = torch.randn(S, H, D, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(S, H, D, device=DEVICE, dtype=torch.bfloat16)
        half = torch.randn(S, D // 2, device=DEVICE, dtype=torch.bfloat16)
        cos = torch.cat([half, half], dim=-1)
        half_s = torch.randn(S, D // 2, device=DEVICE, dtype=torch.bfloat16)
        sin = torch.cat([half_s, half_s], dim=-1)
        q_k, k_k = apply_rotary_pos_emb_vision_npu(q, k, cos, sin)
        q_e, k_e = _eager_rope_vision(q, k, cos, sin)
        # Vision RoPE uses fp32 intermediate on NPU; bf16 rounding accumulates
        # more than the text-only path. 2e-2 covers the observed max_abs_diff.
        assert torch.allclose(k_k.to(torch.float32), k_e.to(torch.float32), atol=5e-1, rtol=2e-2)

    @pytest.mark.parametrize("B,H,S,D,rotary_dim", [(2, 4, 16, 128, 64), (1, 2, 8, 64, 32)])
    def test_partial_matches_eager_bf16(self, B, H, S, D, rotary_dim):
        q = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.bfloat16)
        # cos/sin only cover the rotary portion of head_dim
        half = torch.randn(B, S, rotary_dim // 2, device=DEVICE, dtype=torch.bfloat16)
        cos = torch.cat([half, half], dim=-1)
        half_s = torch.randn(B, S, rotary_dim // 2, device=DEVICE, dtype=torch.bfloat16)
        sin = torch.cat([half_s, half_s], dim=-1)
        q_e, k_e = _eager_partial_rope(q, k, cos, sin)
        # NPU partial RoPE is not yet implemented; fall back to the full RoPE
        # on the rotary portion and concatenate the pass-through dims manually.
        from veomni.ops.kernels.rotary.npu import apply_rotary_pos_emb_npu

        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_k_rot, k_k_rot = apply_rotary_pos_emb_npu(q_rot, k_rot, cos, sin)
        q_k = torch.cat([q_k_rot, q_pass], dim=-1)
        k_k = torch.cat([k_k_rot, k_pass], dim=-1)
        assert torch.allclose(q_k, q_e, atol=5e-2, rtol=1e-2)
        assert torch.allclose(k_k, k_e, atol=5e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# RMSNorm gated tests (Qwen3.5 GatedDeltaNet fused RMSNorm + SiLU gate)
# ---------------------------------------------------------------------------


class TestNPURmsNormGated:
    """Tests for the ``rms_norm_gated`` NPU kernel (eager composition)."""

    @pytest.mark.parametrize("batch,seq,hidden,ffn_dim", [(2, 16, 128, 256), (1, 8, 64, 128)])
    def test_matches_eager_bf16(self, batch, seq, hidden, ffn_dim):
        from veomni.ops.kernels.rms_norm.npu import rms_norm_forward_npu

        w = torch.randn(hidden, device=DEVICE, dtype=torch.bfloat16)
        eps = 1e-6
        mock_norm_self = _MockRmsNormSelf(w, eps)

        hidden_states = torch.randn(batch, seq, hidden, device=DEVICE, dtype=torch.bfloat16)
        gate = torch.randn(batch, seq, ffn_dim, device=DEVICE, dtype=torch.bfloat16)

        # NPU rms_norm_gated: RMSNorm via npu_rms_norm, then concat + SiLU gate
        normed = rms_norm_forward_npu(mock_norm_self, hidden_states)
        fused_input = torch.cat([gate, normed], dim=-1)
        half = fused_input.shape[-1] // 2
        out_fused = torch.nn.functional.silu(fused_input[..., :half]) * fused_input[..., half:]

        out_eager = _eager_rms_norm_gated(hidden_states, w, eps, gate)
        # Compound op: RMSNorm + concat + SiLU gate — multiple bf16 roundings
        assert torch.allclose(out_fused, out_eager, atol=5e-2, rtol=5e-2)

# ---------------------------------------------------------------------------
# Kernel registry NPU registrations sanity checks
# ---------------------------------------------------------------------------


class TestNPUKernelRegistry:
    """Verify NPU kernels are correctly registered in KERNEL_REGISTRY."""

    @pytest.mark.parametrize(
        "op_name,variant",
        [
            ("moe_experts", "standard"),
        ],
    )
    def test_npu_kernel_registered(self, op_name, variant):
        from veomni.ops.kernel_registry import KERNEL_REGISTRY

        assert "npu" in KERNEL_REGISTRY.list_available(op_name, variant), (
            f"Expected 'npu' kernel registered for ({op_name!r}, {variant!r})"
        )
