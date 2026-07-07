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

For each NPU kernel registered in KERNEL_REGISTRY, bind an OpSlot to the ``npu``
implementation and compare its output against the canonical eager implementation
on random inputs. This guards against:
  - The wrong variant being bound into a slot (e.g. standard bound into qwen3_5).
  - Silent regressions in the torch_npu kernel wrappers.

Tests are skipped on non-NPU hosts so the same test suite runs in any CI runner.
"""

import pytest
import torch

import veomni.ops  # noqa: F401 — trigger KERNEL_REGISTRY registrations
from veomni.ops.dispatch import OpSlot
from veomni.utils.device import IS_NPU_AVAILABLE, get_device_type


pytestmark = pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="NPU kernels require torch_npu")

DEVICE = get_device_type()


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
# RMSNorm tests
# ---------------------------------------------------------------------------


class TestNPURmsNorm:
    """Tests for the ``rms_norm`` NPU kernel (standard + qwen3_5 variants)."""

    @pytest.mark.parametrize("batch,seq,hidden", [(2, 16, 128), (1, 8, 64)])
    def test_standard_matches_eager_bf16(self, batch, seq, hidden):
        slot = OpSlot("rms_norm", "standard")
        slot.bind("npu")
        x = torch.randn(batch, seq, hidden, device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn(hidden, device=DEVICE, dtype=torch.bfloat16)
        out_kernel = slot(x, w, 1e-6)
        out_eager = _eager_rms_norm_standard(x, w, 1e-6)
        # bf16 RMSNorm on Ascend 910 drifts by 1-2 bf16 ULPs from the eager
        # reference due to rounding in the final elementwise mul.  The ULP at
        # value 4.0 is 0.031, so 1e-2 atol + 1e-2 rtol covers the worst case
        # (1 ULP at any value <= 4) while still being 50x smaller than the 0.5
        # gap a wrong kernel variant (e.g. qwen3_5 bound into a standard slot)
        # would produce.
        atol = 1e-2
        rtol = 1e-2
        assert torch.allclose(out_kernel, out_eager, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("batch,seq,hidden", [(2, 16, 128), (1, 8, 64)])
    def test_standard_matches_eager_fp32(self, batch, seq, hidden):
        slot = OpSlot("rms_norm", "standard")
        slot.bind("npu")
        x = torch.randn(batch, seq, hidden, device=DEVICE, dtype=torch.float32)
        w = torch.randn(hidden, device=DEVICE, dtype=torch.float32)
        out_kernel = slot(x, w, 1e-6)
        out_eager = _eager_rms_norm_standard(x, w, 1e-6)
        # fp32 should be very close
        assert torch.allclose(out_kernel, out_eager, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch,seq,hidden", [(2, 16, 128), (1, 8, 64)])
    def test_qwen3_5_matches_eager_bf16(self, batch, seq, hidden):
        slot = OpSlot("rms_norm", "qwen3_5")
        slot.bind("npu")
        x = torch.randn(batch, seq, hidden, device=DEVICE, dtype=torch.bfloat16)
        w = torch.zeros(hidden, device=DEVICE, dtype=torch.bfloat16)  # Qwen3.5 init to zeros
        w += 0.01 * torch.randn_like(w)
        # bf16 RMSNorm on Ascend 910 drifts by 1-2 bf16 ULPs from the eager
        # reference even after up-casting to fp32, because the rounding happens
        # in the bf16 multiply before the cast.  1e-2 atol + 1e-2 rtol covers
        # 1 ULP at any normalized value while staying well below the gap a
        # wrong kernel variant would produce.
        out_kernel = slot(x, w, 1e-6).to(torch.float32)
        out_eager = _eager_rms_norm_qwen3_5(x, w, 1e-6).to(torch.float32)
        assert torch.allclose(out_kernel, out_eager, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Rotary positional embedding tests
# ---------------------------------------------------------------------------


class TestNPURotaryPosEmb:
    """Tests for the ``rotary_pos_emb`` NPU kernel (full, vision, partial variants)."""

    @pytest.mark.parametrize("B,H,S,D", [(2, 4, 16, 64), (1, 2, 8, 32)])
    def test_full_matches_eager_bf16(self, B, H, S, D):
        slot = OpSlot("rotary_pos_emb", "full")
        slot.bind("npu")
        q = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.bfloat16)
        # HF RoPE convention: cos/sin are duplicated across the two halves of head_dim.
        half = torch.randn(B, S, D // 2, device=DEVICE, dtype=torch.bfloat16)
        cos = torch.cat([half, half], dim=-1)
        half_s = torch.randn(B, S, D // 2, device=DEVICE, dtype=torch.bfloat16)
        sin = torch.cat([half_s, half_s], dim=-1)
        q_k, k_k = slot(q, k, cos, sin)
        q_e, k_e = _eager_rope(q, k, cos, sin)
        # Cast to fp32 because torch.allclose lowers to aclnnIsClose on NPU,
        # which only supports DT_FLOAT (raises EZ1001 on bf16 inputs).
        # bf16 RoPE compounds two rounds (q*cos) + (rotate_half(q)*sin); on
        # larger shapes (e.g. (2,4,16,64)) the bf16 ULP drift stacks to ~2e-2.
        assert torch.allclose(q_k.float(), q_e.float(), atol=2e-2, rtol=2e-2)
        assert torch.allclose(k_k.float(), k_e.float(), atol=2e-2, rtol=2e-2)

    @pytest.mark.parametrize("S,H,D", [(16, 4, 64), (8, 2, 32)])
    def test_vision_matches_eager_bf16(self, S, H, D):
        slot = OpSlot("rotary_pos_emb_vision", "full")
        slot.bind("npu")
        q = torch.randn(S, H, D, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(S, H, D, device=DEVICE, dtype=torch.bfloat16)
        half = torch.randn(S, D // 2, device=DEVICE, dtype=torch.bfloat16)
        cos = torch.cat([half, half], dim=-1)
        half_s = torch.randn(S, D // 2, device=DEVICE, dtype=torch.bfloat16)
        sin = torch.cat([half_s, half_s], dim=-1)
        q_k, k_k = slot(q, k, cos, sin)
        q_e, k_e = _eager_rope_vision(q, k, cos, sin)
        # Cast to fp32 + 2e-2 tolerance (see comment in test_full_matches_eager_bf16).
        assert torch.allclose(q_k.float(), q_e.float(), atol=2e-2, rtol=2e-2)
        assert torch.allclose(k_k.float(), k_e.float(), atol=2e-2, rtol=2e-2)

    @pytest.mark.parametrize("B,H,S,D,rotary_dim", [(2, 4, 16, 128, 64), (1, 2, 8, 64, 32)])
    def test_partial_matches_eager_bf16(self, B, H, S, D, rotary_dim):
        slot = OpSlot("rotary_pos_emb", "partial")
        slot.bind("npu")
        q = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.bfloat16)
        # cos/sin only cover the rotary portion of head_dim
        half = torch.randn(B, S, rotary_dim // 2, device=DEVICE, dtype=torch.bfloat16)
        cos = torch.cat([half, half], dim=-1)
        half_s = torch.randn(B, S, rotary_dim // 2, device=DEVICE, dtype=torch.bfloat16)
        sin = torch.cat([half_s, half_s], dim=-1)
        q_k, k_k = slot(q, k, cos, sin)
        q_e, k_e = _eager_partial_rope(q, k, cos, sin)
        # Cast to fp32 + 2e-2 tolerance (see comment in test_full_matches_eager_bf16).
        assert torch.allclose(q_k.float(), q_e.float(), atol=2e-2, rtol=2e-2)
        assert torch.allclose(k_k.float(), k_e.float(), atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# RMSNorm gated tests (Qwen3.5 GatedDeltaNet fused RMSNorm + SiLU gate)
# ---------------------------------------------------------------------------


class TestNPURmsNormGated:
    """Tests for the ``rms_norm_gated`` NPU kernel (NPUFusedRMSNormGated)."""

    @pytest.mark.parametrize("batch,seq,hidden,ffn_dim", [(2, 16, 128, 256), (1, 8, 64, 128)])
    def test_matches_eager_bf16(self, batch, seq, hidden, ffn_dim):
        slot = OpSlot("rms_norm_gated", "standard")
        slot.bind("npu")
        # The bound kernel is the NPUFusedRMSNormGated class; instantiate it.
        # (OpSlot exposes bound_kernel(); resolve() is on the KERNEL_REGISTRY.)
        fused_cls = slot.bound_kernel()
        fused_module = fused_cls(hidden_size=hidden, eps=1e-6).to(device=DEVICE, dtype=torch.bfloat16)

        hidden_states = torch.randn(batch, seq, hidden, device=DEVICE, dtype=torch.bfloat16)
        gate = torch.randn(batch, seq, ffn_dim, device=DEVICE, dtype=torch.bfloat16)

        out_fused = fused_module(hidden_states, gate=gate)
        out_eager = _eager_rms_norm_gated(hidden_states, fused_module.weight, fused_module.variance_epsilon, gate)
        # Compound op: RMSNorm + concat + SiLU gate — multiple bf16 roundings.
        # 1e-2 atol+rtol covers 1-2 bf16 ULPs at typical normalized values.
        assert torch.allclose(out_fused.float(), out_eager.float(), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# HCCL premul_sum patch tests (mock-based, no distributed required)
# ---------------------------------------------------------------------------


class TestHcclPremulSum:
    """Tests for the HCCL PREMUL_SUM compatibility wrapper.

    Verifies that the wrapper correctly decomposes PREMUL_SUM into SUM +
    scalar multiplication, and that non-PREMUL_SUM operations pass through
    unchanged. Uses mock to avoid requiring a real distributed environment.
    """

    def test_premul_sum_decomposes_to_sum_plus_mul(self):
        """PREMUL_SUM should be converted to SUM followed by scalar multiplication."""
        from torch.distributed import ReduceOp

        from veomni.ops.platform.npu.hccl_premul_sum import hccl_premul_sum_wrapper

        factor = 0.5

        # Build the mock as a real class so that ``op == ReduceOp.PREMUL_SUM``
        # dispatches to our ``__eq__`` (Python only consults class-level
        # ``__eq__`` for the ``==`` operator; an attribute set on the instance
        # is ignored).  ``__getstate__`` returns the tuple shape the wrapper
        # expects: ``("PREMUL_SUM", factor)`` and the wrapper reads ``[1]``.
        class MockPremulSum:
            def __eq__(self, other):
                return other is ReduceOp.PREMUL_SUM

            def __getstate__(self):
                return ("PREMUL_SUM", factor)

        mock_op = MockPremulSum()

        # Track calls to the original op
        calls = []
        original_output = torch.tensor([2.0, 4.0, 6.0])

        def mock_op_fn(*args, **kwargs):
            calls.append(("op_fn", args, kwargs.copy()))
            return None  # synchronous op returns None

        # The first positional arg is the output tensor
        output_tensor = original_output.clone()
        wrapper = hccl_premul_sum_wrapper(mock_op_fn, "tensor")

        wrapper(output_tensor, op=mock_op)

        # Verify SUM was called (op was changed from PREMUL_SUM to SUM)
        assert len(calls) == 1
        assert calls[0][2]["op"] is not mock_op  # op was replaced

        # Verify the output was multiplied by the factor
        expected = original_output * factor
        assert torch.allclose(output_tensor, expected)

    def test_non_premul_sum_passes_through(self):
        """Non-PREMUL_SUM operations should pass through unchanged."""
        from torch.distributed import ReduceOp

        from veomni.ops.platform.npu.hccl_premul_sum import hccl_premul_sum_wrapper

        calls = []

        def mock_op_fn(*args, **kwargs):
            calls.append(("op_fn", args, kwargs.copy()))
            return None

        wrapper = hccl_premul_sum_wrapper(mock_op_fn, "tensor")
        tensor = torch.tensor([1.0, 2.0, 3.0])
        original_data = tensor.clone()

        wrapper(tensor, op=ReduceOp.SUM)

        # Verify the op was passed through unchanged
        assert len(calls) == 1
        assert calls[0][2]["op"] == ReduceOp.SUM
        # Verify tensor was NOT modified (no multiplication for non-PREMUL_SUM)
        assert torch.equal(tensor, original_data)

    def test_apply_hccl_premul_sum_patch_patches_dist(self):
        """apply_hccl_premul_sum_patch should monkey-patch torch.distributed functions."""
        import torch.distributed as dist

        from veomni.ops.platform.npu.hccl_premul_sum import apply_hccl_premul_sum_patch

        # Save originals
        orig_all_reduce = dist.all_reduce
        orig_reduce_scatter = dist.reduce_scatter
        orig_reduce_scatter_tensor = dist.reduce_scatter_tensor

        try:
            apply_hccl_premul_sum_patch()
            # Verify the functions were replaced with wrappers
            assert dist.all_reduce is not orig_all_reduce
            assert dist.reduce_scatter is not orig_reduce_scatter
            assert dist.reduce_scatter_tensor is not orig_reduce_scatter_tensor
        finally:
            # Restore originals
            dist.all_reduce = orig_all_reduce
            dist.reduce_scatter = orig_reduce_scatter
            dist.reduce_scatter_tensor = orig_reduce_scatter_tensor


# ---------------------------------------------------------------------------
# Kernel registry NPU registrations sanity checks
# ---------------------------------------------------------------------------


class TestNPUKernelRegistry:
    """Verify NPU kernels are correctly registered in KERNEL_REGISTRY."""

    @pytest.mark.parametrize(
        "op_name,variant",
        [
            ("rms_norm", "standard"),
            ("rms_norm", "qwen3_5"),
            ("rotary_pos_emb", "full"),
            ("rotary_pos_emb_vision", "full"),
            ("rotary_pos_emb", "partial"),
            ("rms_norm_gated", "standard"),
            ("moe_experts", "standard"),
        ],
    )
    def test_npu_kernel_registered(self, op_name, variant):
        from veomni.ops.kernel_registry import KERNEL_REGISTRY

        assert "npu" in KERNEL_REGISTRY.list_available(op_name, variant), (
            f"Expected 'npu' kernel registered for ({op_name!r}, {variant!r})"
        )
