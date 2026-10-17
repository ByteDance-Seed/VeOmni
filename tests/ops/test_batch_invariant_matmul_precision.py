"""Batch-invariant GEMM must retain FP32 operand precision."""

import pytest
import torch

from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Triton GEMM requires CUDA")
@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("shape", [(37, 67, 53), (129, 129, 131)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_invariant_matmul_precision_and_batch_partition(seed, shape, dtype, with_bias):
    from veomni.ops.batch_invariant_ops.batch_invariant_ops import matmul_persistent

    device = get_device_type()
    generator = torch.Generator(device=device).manual_seed(seed)
    m, k, n = shape
    a = torch.randn((k, m), device=device, dtype=dtype, generator=generator).t()
    b = torch.randn((n, k), device=device, dtype=dtype, generator=generator).t()
    bias = torch.randn((n,), device=device, dtype=dtype, generator=generator) if with_bias else None
    reference = a.double() @ b.double()
    if bias is not None:
        reference = reference + bias.double()
    actual = matmul_persistent(a, b, bias)
    assert actual.dtype == dtype and bool(torch.isfinite(actual).all())
    error = actual.double() - reference
    eps = torch.finfo(dtype).eps
    # FP64 math over the actual rounded operands is independent of Triton.
    assert error.norm() <= 16 * eps * reference.norm()
    assert error.abs().max() <= 32 * eps * reference.abs().max()
    split = torch.cat([matmul_persistent(a[:7], b, bias), matmul_persistent(a[7:], b, bias)])
    assert torch.equal(actual, split), "GEMM changed with batch partition"
