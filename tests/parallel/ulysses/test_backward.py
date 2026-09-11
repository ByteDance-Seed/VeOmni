"""Numerical coverage for reusable async Ulysses backward units.

These tests exercise ``veomni.distributed.sequence_parallel.backward`` directly.
Dense ``Function.apply`` contracts with mocked collectives live in
``test_async_ulysses_grad.py``. Four-GPU async vs sync parity lives in
``test_async_ulysses.py``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from veomni.distributed.sequence_parallel.backward import (
    layer_norm_backward,
    linear_backward,
    linear_input_backward,
    linear_parameter_backward,
    reduce_repeated_kv_gradient,
)
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("input_shape", [(2, 3, 5), (0, 5)])
def test_linear_backward_units_match_autograd(has_bias: bool, input_shape: tuple[int, ...]) -> None:
    torch.manual_seed(4101)
    input_tensor = torch.randn(input_shape, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(7, dtype=torch.float64, requires_grad=True) if has_bias else None
    grad_output = torch.randn((*input_shape[:-1], 7), dtype=torch.float64)

    output = F.linear(input_tensor, weight, bias)
    reference_grads = torch.autograd.grad(
        output, (input_tensor, weight, *(() if bias is None else (bias,))), grad_output
    )
    reference_input, reference_weight, *reference_bias = reference_grads

    actual_input = linear_input_backward(grad_output, input_tensor, weight)
    actual_weight, actual_bias = linear_parameter_backward(
        grad_output,
        input_tensor,
        weight,
        has_bias=has_bias,
    )
    combined_input, combined_weight, combined_bias = linear_backward(
        grad_output,
        input_tensor,
        weight,
        has_bias=has_bias,
    )

    torch.testing.assert_close(actual_input, reference_input)
    torch.testing.assert_close(actual_weight, reference_weight)
    torch.testing.assert_close(combined_input, reference_input)
    torch.testing.assert_close(combined_weight, reference_weight)
    if has_bias:
        torch.testing.assert_close(actual_bias, reference_bias[0])
        torch.testing.assert_close(combined_bias, reference_bias[0])
    else:
        assert actual_bias is None
        assert combined_bias is None


def test_linear_backward_rejects_mismatched_rows() -> None:
    with pytest.raises(ValueError, match="matching input and output row counts"):
        linear_backward(
            torch.randn(5, 7),
            torch.randn(4, 5),
            torch.randn(7, 5),
            has_bias=False,
        )


@pytest.mark.parametrize("head_dimension", [1, -2])
def test_reduce_repeated_kv_gradient_matches_autograd(head_dimension: int) -> None:
    torch.manual_seed(4103)
    input_tensor = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    repeated = torch.repeat_interleave(input_tensor, repeats=4, dim=head_dimension)
    grad_output = torch.randn_like(repeated)
    (reference_grad,) = torch.autograd.grad(repeated, input_tensor, grad_output)

    actual_grad = reduce_repeated_kv_gradient(
        grad_output,
        original_num_heads=3,
        repeats=4,
        head_dimension=head_dimension,
    )

    torch.testing.assert_close(actual_grad, reference_grad)


def test_reduce_repeated_kv_gradient_validates_repeat_contract() -> None:
    grad_output = torch.randn(2, 6, 5)
    assert reduce_repeated_kv_gradient(grad_output, 6, 1, head_dimension=1) is grad_output
    with pytest.raises(ValueError, match="repeats must be positive"):
        reduce_repeated_kv_gradient(grad_output, 3, 0, head_dimension=1)
    with pytest.raises(ValueError, match="head dimension must have size 9"):
        reduce_repeated_kv_gradient(grad_output, 3, 3, head_dimension=1)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="fused LayerNorm backward requires CUDA")
def test_layer_norm_backward_matches_eager_autograd() -> None:
    fused_layer_norm_cuda = pytest.importorskip("fused_layer_norm_cuda")
    torch.manual_seed(4105)
    device = get_device_type()
    normalized_shape = torch.Size((16,))
    eps = 1e-5
    input_tensor = torch.randn(5, 3, 16, device=device, dtype=torch.float32)
    weight = torch.randn(16, device=device, dtype=torch.float32)
    bias = torch.randn(16, device=device, dtype=torch.float32)
    grad_output = torch.randn_like(input_tensor)
    _, mean, invvar = fused_layer_norm_cuda.forward_affine(
        input_tensor,
        normalized_shape,
        weight,
        bias,
        eps,
    )

    reference_input = input_tensor.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    reference_bias = bias.detach().clone().requires_grad_()
    reference_output = F.layer_norm(
        reference_input,
        normalized_shape,
        reference_weight,
        reference_bias,
        eps,
    )
    expected = torch.autograd.grad(
        reference_output,
        (reference_input, reference_weight, reference_bias),
        grad_output,
    )

    actual = layer_norm_backward(
        grad_output,
        input_tensor,
        mean,
        invvar,
        weight,
        bias,
        normalized_shape,
        eps,
    )

    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value, atol=2e-6, rtol=2e-5)
