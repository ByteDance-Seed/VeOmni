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

"""NPU clamped-SwiGLU and HCCL premul-sum tests."""

import pytest
import torch

from veomni.utils.device import IS_NPU_AVAILABLE, get_device_type


pytestmark = pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="NPU kernels require torch_npu")

DEVICE = get_device_type()


def _eager_clamped_swiglu(x, limit):
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return torch.nn.functional.silu(gate) * up


class TestNPUClampedSwiGLU:
    @pytest.fixture(autouse=True)
    def require_triton_ascend(self):
        try:
            from triton._C import libtriton
        except ImportError:
            pytest.skip("clamped SwiGLU requires triton-ascend")
        if not hasattr(libtriton, "ascend"):
            pytest.skip("clamped SwiGLU requires the Triton Ascend backend")

    @pytest.mark.parametrize("shape", [(3, 34), (2, 3, 256)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward_backward_matches_eager(self, shape, dtype):
        from veomni.kernels._kernels.moe_experts.shared.npu_clamped_swiglu import npu_triton_clamped_swiglu

        torch.manual_seed(17)
        source = torch.randn(shape, dtype=torch.float32).mul_(12)
        actual_input = source.to(device=DEVICE, dtype=dtype).requires_grad_()
        expected_input = source.to(device=DEVICE, dtype=dtype).requires_grad_()

        actual = npu_triton_clamped_swiglu(actual_input, 7.0)
        expected = _eager_clamped_swiglu(expected_input, 7.0)
        grad_output = torch.randn_like(actual)
        actual.backward(grad_output)
        expected.backward(grad_output)

        # Triton evaluates in FP32 before storing, while the production eager fallback
        # keeps the input dtype. BF16 can therefore differ by a few low-precision ULPs;
        # FP16 stays closer. Exact zero-gradient masks below guard the clamp semantics
        # independently of this numerical allowance.
        tolerance = 5e-3 if dtype == torch.float16 else 2e-2
        torch.testing.assert_close(actual.float(), expected.float(), rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(
            actual_input.grad.float(), expected_input.grad.float(), rtol=tolerance, atol=tolerance
        )

        hidden_size = actual_input.shape[-1] // 2
        gate, up = actual_input.detach().split(hidden_size, dim=-1)
        gate_grad, up_grad = actual_input.grad.split(hidden_size, dim=-1)
        assert torch.count_nonzero(gate_grad[gate > 7.0]) == 0
        assert torch.count_nonzero(up_grad[(up < -7.0) | (up > 7.0)]) == 0

    def test_clamp_boundaries_match_eager_gradients(self):
        from veomni.kernels._kernels.moe_experts.shared.npu_clamped_swiglu import npu_triton_clamped_swiglu

        values = torch.tensor([[-8.0, -7.0, 7.0, 8.0, -8.0, -7.0, 7.0, 8.0]], device=DEVICE)
        actual_input = values.clone().requires_grad_()
        expected_input = values.clone().requires_grad_()

        actual = npu_triton_clamped_swiglu(actual_input, 7.0)
        expected = _eager_clamped_swiglu(expected_input, 7.0)
        actual.sum().backward()
        expected.sum().backward()

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=1e-5, atol=1e-5)

    def test_empty_batch_preserves_shape_and_backward(self):
        from veomni.kernels._kernels.moe_experts.shared.npu_clamped_swiglu import npu_triton_clamped_swiglu

        x = torch.empty((0, 30), device=DEVICE, dtype=torch.bfloat16, requires_grad=True)

        output = npu_triton_clamped_swiglu(x, 7.0)
        output.sum().backward()

        assert output.shape == (0, 15)
        assert x.grad.shape == x.shape

    def test_rejects_odd_last_dimension(self):
        from veomni.kernels._kernels.moe_experts.shared.npu_clamped_swiglu import npu_triton_clamped_swiglu

        x = torch.empty((2, 15), device=DEVICE, dtype=torch.bfloat16)

        with pytest.raises(ValueError, match="even last dimension"):
            npu_triton_clamped_swiglu(x, 7.0)


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
