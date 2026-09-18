# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Affine LayerNorm eager vs ``F.layer_norm``, and Apex vs eager."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tests.ops.tol import EAGER_ATOL, EAGER_GRAD_ATOL, EAGER_GRAD_RTOL, EAGER_RTOL
from tests.ops.utils import make_grad_leaves
from veomni.ops import resolve_op
from veomni.utils.device import IS_CUDA_AVAILABLE
from veomni.utils.import_utils import is_package_available


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("shape", "seed"),
    (
        pytest.param((2, 16, 64), 0, id="rank-three"),
        pytest.param((64,), 9, id="rank-one"),
    ),
)
def test_eager_matches_layer_norm(dtype: torch.dtype, shape: tuple[int, ...], seed: int) -> None:
    """Eager affine LayerNorm matches ``F.layer_norm`` for batched and vector inputs."""
    torch.manual_seed(seed)
    eps = 1e-5
    x = torch.randn(shape, dtype=dtype)
    weight = torch.randn(shape[-1], dtype=dtype)
    bias = torch.randn(shape[-1], dtype=dtype)

    x_ref, weight_ref, bias_ref = make_grad_leaves(x, weight, bias)
    out_ref = F.layer_norm(x_ref, (shape[-1],), weight_ref, bias_ref, eps)

    x_eager, weight_eager, bias_eager = make_grad_leaves(x, weight, bias)
    out_eager = resolve_op("layer_norm", "standard", "eager").wrapper(
        x_eager,
        weight_eager,
        bias_eager,
        normalized_shape=shape[-1],
        eps=eps,
    )
    torch.testing.assert_close(out_eager.float(), out_ref.float(), atol=EAGER_ATOL, rtol=EAGER_RTOL)

    grad_output = torch.randn_like(out_eager)
    reference_grads = torch.autograd.grad(out_ref, (x_ref, weight_ref, bias_ref), grad_outputs=grad_output)
    eager_grads = torch.autograd.grad(out_eager, (x_eager, weight_eager, bias_eager), grad_outputs=grad_output)
    for actual, expected, expected_shape in zip(
        eager_grads, reference_grads, (x.shape, weight.shape, bias.shape), strict=True
    ):
        assert actual.shape == expected_shape
        torch.testing.assert_close(actual.float(), expected.float(), atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_eager_empty_tensor_skips_reduction() -> None:
    """Empty activations keep the affine scale without a mean/var reduction."""
    x = torch.randn(0, 8, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(8, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(8, dtype=torch.float32, requires_grad=True)
    out = resolve_op("layer_norm", "standard", "eager").wrapper(x, weight, bias, eps=1e-5)
    torch.testing.assert_close(out, x * weight + bias)
    out.sum().backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.zeros_like(x))


@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or not is_package_available("fused_layer_norm_cuda"),
    reason="Apex fused LayerNorm needs CUDA and fused_layer_norm_cuda",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_apex_matches_eager(dtype: torch.dtype) -> None:
    """Apex fused LayerNorm stays close to the eager contract on CUDA."""
    torch.manual_seed(0)
    hidden = 64
    eps = 1e-5
    x = torch.randn(2, 16, hidden, device="cuda", dtype=dtype)
    weight = torch.randn(hidden, device="cuda", dtype=dtype)
    bias = torch.randn(hidden, device="cuda", dtype=dtype)

    eager = resolve_op("layer_norm", "standard", "eager").wrapper
    apex = resolve_op("layer_norm", "standard", "apex").wrapper
    x_e, w_e, b_e = make_grad_leaves(x, weight, bias)
    x_a, w_a, b_a = make_grad_leaves(x, weight, bias)
    out_e = eager(x_e, w_e, b_e, normalized_shape=hidden, eps=eps)
    out_a = apex(x_a, w_a, b_a, normalized_shape=hidden, eps=eps)
    torch.testing.assert_close(out_a.float(), out_e.float(), atol=2e-3, rtol=2e-3)

    grad_output = torch.randn_like(out_e)
    out_e.backward(grad_output)
    out_a.backward(grad_output)
    torch.testing.assert_close(x_a.grad.float(), x_e.grad.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(w_a.grad.float(), w_e.grad.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(b_a.grad.float(), b_e.grad.float(), atol=2e-2, rtol=2e-2)
