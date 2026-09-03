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
            ("rms_norm_gated", "standard"),
        ],
    )
    def test_npu_kernel_registered(self, op_name, variant):
        from veomni.ops.kernel_registry import KERNEL_REGISTRY

        assert "npu" in KERNEL_REGISTRY.list_available(op_name, variant), (
            f"Expected 'npu' kernel registered for ({op_name!r}, {variant!r})"
        )
