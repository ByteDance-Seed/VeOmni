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

"""mHC eager vs HuggingFace DeepSeek-V4, and TileLang / CANN vs eager."""

from __future__ import annotations

import importlib.util

import pytest
import torch
from torch import Tensor
from transformers import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperConnection, DeepseekV4HyperHead

from tests.ops.tol import (
    EAGER_ATOL,
    EAGER_GRAD_ATOL,
    EAGER_GRAD_RTOL,
    EAGER_RTOL,
    MHC_CANN_ATOL,
    MHC_CANN_RTOL,
    MHC_FUSED_ATOL,
    MHC_FUSED_GRAD_COSINE,
    MHC_FUSED_RTOL,
)
from tests.ops.utils import assert_gradient_direction_and_scale, is_nvidia_cuda_available, make_grad_leaves
from veomni.ops import OP_REGISTRY, resolve_op
from veomni.ops.kernels.mhc.pre import cann as pre_cann
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE, get_device_type


# Installed Transformers classes used as the mHC eager reference.

_TILELANG_AVAILABLE = is_nvidia_cuda_available(min_cc=90) and importlib.util.find_spec("tile_kernels") is not None
_TRITON_ROCM_AVAILABLE = (
    IS_CUDA_AVAILABLE and torch.version.hip is not None and importlib.util.find_spec("triton") is not None
)
_CANN_AVAILABLE = IS_NPU_AVAILABLE and importlib.util.find_spec("cann_ops_transformer") is not None


def _tiny_dsv4_config(
    *,
    hidden_size: int,
    hc_mult: int,
    sinkhorn_iters: int,
    hc_eps: float,
    rms_eps: float,
) -> DeepseekV4Config:
    return DeepseekV4Config(
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        hc_sinkhorn_iters=sinkhorn_iters,
        hc_eps=hc_eps,
        rms_norm_eps=rms_eps,
        num_hidden_layers=1,
        num_attention_heads=2,
    )


def _hf_decoder_layer_post(output: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
    """HuggingFace decoder-layer residual mix. There is no standalone post module.

    This is the residual-mixing expression used by the installed
    ``DeepseekV4DecoderLayer.forward`` implementation.
    """
    dtype = residual.dtype
    return post.to(dtype).unsqueeze(-1) * output.unsqueeze(-2) + torch.matmul(
        comb.to(dtype).transpose(-1, -2), residual
    )


def test_mhc_pre_eager_matches_hf():
    torch.manual_seed(0)
    batch, seq_len, hc_mult, hidden = 2, 8, 4, 32
    norm_eps, hc_eps, sinkhorn_iters = 1e-6, 1e-6, 4
    mix = (2 + hc_mult) * hc_mult
    x = torch.randn(batch, seq_len, hc_mult, hidden)
    fn = torch.randn(mix, hc_mult * hidden) * 0.01
    scale = torch.randn(3) * 0.01
    base = torch.randn(mix) * 0.01

    hc = DeepseekV4HyperConnection(
        _tiny_dsv4_config(
            hidden_size=hidden,
            hc_mult=hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            hc_eps=hc_eps,
            rms_eps=norm_eps,
        )
    )
    with torch.no_grad():
        hc.fn.copy_(fn)
        hc.base.copy_(base)
        hc.scale.copy_(scale)

    x_h = x.detach().requires_grad_(True)
    post_h, comb_h, collapsed_h = hc(x_h)

    x_e, fn_e, scale_e, base_e = make_grad_leaves(x, fn, scale, base)
    post_e, comb_e, collapsed_e = resolve_op("mhc", "pre", "eager").wrapper(
        x_e, fn_e, scale_e, base_e, norm_eps, hc_mult, sinkhorn_iters, hc_eps
    )
    assert torch.allclose(post_e, post_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)
    assert torch.allclose(comb_e, comb_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)
    assert torch.allclose(collapsed_e, collapsed_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    grad_outputs = tuple(torch.randn_like(output) for output in (post_e, comb_e, collapsed_e))
    hf_grads = torch.autograd.grad(
        (post_h, comb_h, collapsed_h),
        (x_h, hc.fn, hc.scale, hc.base),
        grad_outputs=grad_outputs,
    )
    eager_grads = torch.autograd.grad(
        (post_e, comb_e, collapsed_e),
        (x_e, fn_e, scale_e, base_e),
        grad_outputs=grad_outputs,
    )
    for actual, expected in zip(eager_grads, hf_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_mhc_post_eager_matches_hf():
    torch.manual_seed(1)
    batch, seq_len, hc_mult, hidden = 2, 8, 4, 32
    residual = torch.randn(batch, seq_len, hc_mult, hidden)
    output = torch.randn(batch, seq_len, hidden)
    post = torch.randn(batch, seq_len, hc_mult)
    comb = torch.randn(batch, seq_len, hc_mult, hc_mult)

    out_h, res_h, post_h, comb_h = make_grad_leaves(output, residual, post, comb)
    y_h = _hf_decoder_layer_post(out_h, res_h, post_h, comb_h)
    out_e, res_e, post_e, comb_e = make_grad_leaves(output, residual, post, comb)
    y_e = resolve_op("mhc", "post", "eager").wrapper(out_e, res_e, post_e, comb_e)
    assert torch.allclose(y_e, y_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = torch.randn_like(y_e)
    y_h.backward(go)
    y_e.backward(go)
    assert torch.allclose(out_e.grad, out_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(res_e.grad, res_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(post_e.grad, post_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(comb_e.grad, comb_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_mhc_head_eager_matches_hf():
    torch.manual_seed(2)
    batch, seq_len, hc_mult, hidden = 2, 8, 4, 32
    norm_eps, hc_eps = 1e-6, 1e-6
    x = torch.randn(batch, seq_len, hc_mult, hidden)
    fn = torch.randn(hc_mult, hc_mult * hidden) * 0.01
    scale = torch.randn(1) * 0.01
    base = torch.randn(hc_mult) * 0.01

    head = DeepseekV4HyperHead(
        _tiny_dsv4_config(
            hidden_size=hidden,
            hc_mult=hc_mult,
            sinkhorn_iters=1,
            hc_eps=hc_eps,
            rms_eps=norm_eps,
        )
    )
    with torch.no_grad():
        head.hc_fn.copy_(fn)
        head.hc_base.copy_(base)
        head.hc_scale.copy_(scale)

    x_h = x.detach().requires_grad_(True)
    y_h = head(x_h)
    x_e, fn_e, scale_e, base_e = make_grad_leaves(x, fn, scale, base)
    y_e = resolve_op("mhc", "head", "eager").wrapper(x_e, fn_e, scale_e, base_e, norm_eps, hc_mult, hc_eps)
    assert torch.allclose(y_e, y_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = torch.randn_like(y_e)
    y_h.backward(go)
    y_e.backward(go)
    assert torch.allclose(x_e.grad, x_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(fn_e.grad, head.hc_fn.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(scale_e.grad, head.hc_scale.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(base_e.grad, head.hc_base.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="TileKernels mHC requires an SM90+ NVIDIA CUDA GPU")
def test_mhc_tilelang_pre_post_matches_eager():
    torch.manual_seed(17)
    device = torch.device("cuda")
    batch, seq_len, hc_mult, hidden = 1, 32, 4, 256
    norm_eps, hc_eps, sinkhorn_iters = 1e-6, 1e-6, 20
    mix = (2 + hc_mult) * hc_mult
    x = torch.randn(batch, seq_len, hc_mult, hidden, device=device, dtype=torch.bfloat16)
    fn = torch.randn(mix, hc_mult * hidden, device=device, dtype=torch.float32) * 0.01
    scale = torch.randn(3, device=device, dtype=torch.float32) * 0.01
    base = torch.randn(mix, device=device, dtype=torch.float32) * 0.01
    eager_pre = resolve_op("mhc", "pre", "eager").wrapper
    other_pre = resolve_op("mhc", "pre", "tilelang").wrapper
    eager_post = resolve_op("mhc", "post", "eager").wrapper
    other_post = resolve_op("mhc", "post", "tilelang").wrapper

    with torch.no_grad():
        inference = other_pre(x, fn, scale, base, norm_eps, hc_mult, sinkhorn_iters, hc_eps)
        eager_inference = eager_pre(x, fn, scale, base, norm_eps, hc_mult, sinkhorn_iters, hc_eps)
    for actual, expected in zip(inference, eager_inference, strict=True):
        torch.testing.assert_close(actual, expected, rtol=MHC_FUSED_RTOL, atol=MHC_FUSED_ATOL)

    x_o, fn_o, scale_o, base_o = make_grad_leaves(x, fn, scale, base)
    x_e, fn_e, scale_e, base_e = make_grad_leaves(x, fn, scale, base)
    post_o, comb_o, collapsed_o = other_pre(x_o, fn_o, scale_o, base_o, norm_eps, hc_mult, sinkhorn_iters, hc_eps)
    post_e, comb_e, collapsed_e = eager_pre(x_e, fn_e, scale_e, base_e, norm_eps, hc_mult, sinkhorn_iters, hc_eps)
    y_o = other_post(collapsed_o * 0.75, x_o, post_o, comb_o)
    y_e = eager_post(collapsed_e * 0.75, x_e, post_e, comb_e)

    torch.testing.assert_close(post_o, post_e, rtol=MHC_FUSED_RTOL, atol=MHC_FUSED_ATOL)
    torch.testing.assert_close(comb_o, comb_e, rtol=MHC_FUSED_RTOL, atol=MHC_FUSED_ATOL)
    torch.testing.assert_close(collapsed_o, collapsed_e, rtol=MHC_FUSED_RTOL, atol=MHC_FUSED_ATOL)
    torch.testing.assert_close(y_o, y_e, rtol=MHC_FUSED_RTOL, atol=MHC_FUSED_ATOL)

    grad = torch.randn_like(y_o)
    other_grads = torch.autograd.grad((y_o * grad).sum(), (x_o, fn_o, scale_o, base_o))
    eager_grads = torch.autograd.grad((y_e * grad).sum(), (x_e, fn_e, scale_e, base_e))
    for actual, expected in zip(other_grads, eager_grads, strict=True):
        assert_gradient_direction_and_scale(
            actual,
            expected,
            min_cosine=MHC_FUSED_GRAD_COSINE,
            norm_rtol=0.25,
        )


@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="TileKernels mHC requires an SM90+ NVIDIA CUDA GPU")
def test_mhc_tilelang_head_matches_eager():
    torch.manual_seed(23)
    device = torch.device("cuda")
    batch, seq_len, hc_mult, hidden = 1, 32, 4, 256
    norm_eps, hc_eps = 1e-6, 1e-6
    x = torch.randn(batch, seq_len, hc_mult, hidden, device=device, dtype=torch.bfloat16)
    fn = torch.randn(hc_mult, hc_mult * hidden, device=device, dtype=torch.float32) * 0.01
    scale = torch.randn(1, device=device, dtype=torch.float32) * 0.01
    base = torch.randn(hc_mult, device=device, dtype=torch.float32) * 0.01
    eager = resolve_op("mhc", "head", "eager").wrapper
    other = resolve_op("mhc", "head", "tilelang").wrapper

    x_o, fn_o, scale_o, base_o = make_grad_leaves(x, fn, scale, base)
    x_e, fn_e, scale_e, base_e = make_grad_leaves(x, fn, scale, base)
    y_o = other(x_o, fn_o, scale_o, base_o, norm_eps, hc_mult, hc_eps)
    y_e = eager(x_e, fn_e, scale_e, base_e, norm_eps, hc_mult, hc_eps)
    torch.testing.assert_close(y_o, y_e, rtol=MHC_FUSED_RTOL, atol=MHC_FUSED_ATOL)

    grad = torch.randn_like(y_o)
    other_grads = torch.autograd.grad((y_o * grad).sum(), (x_o, fn_o, scale_o, base_o))
    eager_grads = torch.autograd.grad((y_e * grad).sum(), (x_e, fn_e, scale_e, base_e))
    for actual, expected in zip(other_grads, eager_grads, strict=True):
        assert_gradient_direction_and_scale(
            actual,
            expected,
            min_cosine=MHC_FUSED_GRAD_COSINE,
            norm_rtol=0.25,
        )


def _mhc_pre_inputs(batch: int, seq_len: int, hc_mult: int, hidden: int, seed: int):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    mix = (2 + hc_mult) * hc_mult
    x = torch.randn(batch, seq_len, hc_mult, hidden, device=device, dtype=torch.bfloat16)
    fn = torch.randn(mix, hc_mult * hidden, device=device, dtype=torch.float32) * 0.01
    scale = torch.randn(3, device=device, dtype=torch.float32) * 0.5
    base = torch.randn(mix, device=device, dtype=torch.float32) * 0.5
    return x, fn, scale, base


def _noncontiguous_last_dim(shape: tuple[int, ...], *, dtype: torch.dtype, seed: int) -> Tensor:
    torch.manual_seed(seed)
    storage = torch.randn(*shape[:-1], shape[-1] * 2, device="cuda", dtype=dtype)
    tensor = storage[..., ::2]
    assert tensor.shape == shape and not tensor.is_contiguous()
    return tensor


@pytest.mark.skipif(not _TRITON_ROCM_AVAILABLE, reason="Triton mHC requires an AMD ROCm GPU")
@pytest.mark.parametrize(("batch", "seq_len", "hidden"), [(1, 32, 256), (2, 37, 320)])
def test_mhc_triton_pre_post_matches_eager(batch: int, seq_len: int, hidden: int):
    hc_mult = 4
    norm_eps, hc_eps, sinkhorn_iters = 1e-6, 1e-6, 20
    x, fn, scale, base = _mhc_pre_inputs(batch, seq_len, hc_mult, hidden, seed=17)
    eager_pre = resolve_op("mhc", "pre", "eager").wrapper
    triton_pre = resolve_op("mhc", "pre", "triton").wrapper
    eager_post = resolve_op("mhc", "post", "eager").wrapper
    triton_post = resolve_op("mhc", "post", "triton").wrapper

    with torch.no_grad():
        inference = triton_pre(x, fn, scale, base, norm_eps, hc_mult, sinkhorn_iters, hc_eps)
        eager_inference = eager_pre(x, fn, scale, base, norm_eps, hc_mult, sinkhorn_iters, hc_eps)
    for actual, expected in zip(inference, eager_inference, strict=True):
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=MHC_FUSED_RTOL, atol=MHC_FUSED_ATOL)

    x_t, fn_t, scale_t, base_t = make_grad_leaves(x, fn, scale, base)
    x_e, fn_e, scale_e, base_e = make_grad_leaves(x, fn, scale, base)
    post_t, comb_t, collapsed_t = triton_pre(x_t, fn_t, scale_t, base_t, norm_eps, hc_mult, sinkhorn_iters, hc_eps)
    post_e, comb_e, collapsed_e = eager_pre(x_e, fn_e, scale_e, base_e, norm_eps, hc_mult, sinkhorn_iters, hc_eps)
    # The FP32 mixes only differ from eager by reduction order.
    torch.testing.assert_close(post_t, post_e, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(comb_t, comb_e, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(collapsed_t, collapsed_e, rtol=MHC_FUSED_RTOL, atol=MHC_FUSED_ATOL)

    y_t = triton_post(collapsed_t * 0.75, x_t, post_t, comb_t)
    y_e = eager_post(collapsed_e * 0.75, x_e, post_e, comb_e)
    torch.testing.assert_close(y_t, y_e, rtol=MHC_FUSED_RTOL, atol=MHC_FUSED_ATOL)

    grad = torch.randn_like(y_t)
    triton_grads = torch.autograd.grad((y_t * grad).sum(), (x_t, fn_t, scale_t, base_t))
    eager_grads = torch.autograd.grad((y_e * grad).sum(), (x_e, fn_e, scale_e, base_e))
    for actual, expected in zip(triton_grads, eager_grads, strict=True):
        assert actual.dtype == expected.dtype
        assert_gradient_direction_and_scale(actual, expected, min_cosine=0.999, norm_rtol=0.01)


@pytest.mark.skipif(not _TRITON_ROCM_AVAILABLE, reason="Triton mHC requires an AMD ROCm GPU")
@pytest.mark.parametrize("sinkhorn_iters", [0, 1, 3])
def test_mhc_triton_pre_matches_eager_for_short_sinkhorn(sinkhorn_iters: int):
    hc_mult = 4
    x, fn, scale, base = _mhc_pre_inputs(1, 16, hc_mult, 128, seed=29)
    args = (1e-6, hc_mult, sinkhorn_iters, 1e-6)
    x_t, fn_t, scale_t, base_t = make_grad_leaves(x, fn, scale, base)
    x_e, fn_e, scale_e, base_e = make_grad_leaves(x, fn, scale, base)
    _, comb_t, _ = resolve_op("mhc", "pre", "triton").wrapper(x_t, fn_t, scale_t, base_t, *args)
    _, comb_e, _ = resolve_op("mhc", "pre", "eager").wrapper(x_e, fn_e, scale_e, base_e, *args)
    torch.testing.assert_close(comb_t, comb_e, rtol=1e-5, atol=1e-6)

    grad = torch.randn_like(comb_t)
    triton_grads = torch.autograd.grad((comb_t * grad).sum(), (fn_t, scale_t, base_t))
    eager_grads = torch.autograd.grad((comb_e * grad).sum(), (fn_e, scale_e, base_e))
    for actual, expected in zip(triton_grads, eager_grads, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(not _TRITON_ROCM_AVAILABLE, reason="Triton mHC requires an AMD ROCm GPU")
def test_mhc_triton_head_matches_eager():
    torch.manual_seed(23)
    device = torch.device("cuda")
    batch, seq_len, hc_mult, hidden = 2, 37, 4, 320
    norm_eps, hc_eps = 1e-6, 1e-6
    x = torch.randn(batch, seq_len, hc_mult, hidden, device=device, dtype=torch.bfloat16)
    fn = torch.randn(hc_mult, hc_mult * hidden, device=device, dtype=torch.float32) * 0.01
    scale = torch.randn(1, device=device, dtype=torch.float32) * 0.5
    base = torch.randn(hc_mult, device=device, dtype=torch.float32) * 0.5
    eager = resolve_op("mhc", "head", "eager").wrapper
    triton_head = resolve_op("mhc", "head", "triton").wrapper

    x_t, fn_t, scale_t, base_t = make_grad_leaves(x, fn, scale, base)
    x_e, fn_e, scale_e, base_e = make_grad_leaves(x, fn, scale, base)
    y_t = triton_head(x_t, fn_t, scale_t, base_t, norm_eps, hc_mult, hc_eps)
    y_e = eager(x_e, fn_e, scale_e, base_e, norm_eps, hc_mult, hc_eps)
    torch.testing.assert_close(y_t, y_e, rtol=MHC_FUSED_RTOL, atol=MHC_FUSED_ATOL)

    grad = torch.randn_like(y_t)
    triton_grads = torch.autograd.grad((y_t * grad).sum(), (x_t, fn_t, scale_t, base_t))
    eager_grads = torch.autograd.grad((y_e * grad).sum(), (x_e, fn_e, scale_e, base_e))
    for actual, expected in zip(triton_grads, eager_grads, strict=True):
        assert_gradient_direction_and_scale(actual, expected, min_cosine=0.999, norm_rtol=0.01)


@pytest.mark.skipif(not _TRITON_ROCM_AVAILABLE, reason="Triton mHC requires an AMD ROCm GPU")
@pytest.mark.parametrize("hc_mult", [1, 2, 4, 8, 16])
def test_mhc_triton_supported_stream_counts(hc_mult: int):
    hidden = 33
    x, fn, scale, base = _mhc_pre_inputs(1, 3, hc_mult, hidden, seed=100 + hc_mult)
    args = (1e-6, hc_mult, 3, 1e-6)
    x_t, fn_t, scale_t, base_t = make_grad_leaves(x, fn, scale, base)
    x_e, fn_e, scale_e, base_e = make_grad_leaves(x, fn, scale, base)
    values_t = resolve_op("mhc", "pre", "triton").wrapper(x_t, fn_t, scale_t, base_t, *args)
    values_e = resolve_op("mhc", "pre", "eager").wrapper(x_e, fn_e, scale_e, base_e, *args)
    for actual, expected in zip(values_t, values_e, strict=True):
        torch.testing.assert_close(actual, expected, rtol=MHC_FUSED_RTOL, atol=MHC_FUSED_ATOL)

    grad_outputs = tuple(torch.randn_like(value) for value in values_t)
    grads_t = torch.autograd.grad(values_t, (x_t, fn_t, scale_t, base_t), grad_outputs=grad_outputs)
    grads_e = torch.autograd.grad(values_e, (x_e, fn_e, scale_e, base_e), grad_outputs=grad_outputs)
    for actual, expected in zip(grads_t, grads_e, strict=True):
        assert_gradient_direction_and_scale(actual, expected, min_cosine=0.999, norm_rtol=0.01)


@pytest.mark.skipif(not _TRITON_ROCM_AVAILABLE, reason="Triton mHC requires an AMD ROCm GPU")
def test_mhc_triton_empty_sequence_backward():
    hc_mult, hidden = 4, 64
    x, fn, scale, base = _mhc_pre_inputs(1, 0, hc_mult, hidden, seed=37)
    leaves = make_grad_leaves(x, fn, scale, base)
    outputs = resolve_op("mhc", "pre", "triton").wrapper(*leaves, 1e-6, hc_mult, 20, 1e-6)
    sum(output.sum() for output in outputs).backward()
    for tensor in leaves:
        assert tensor.grad is not None
        assert tensor.grad.shape == tensor.shape


@pytest.mark.skipif(not _TRITON_ROCM_AVAILABLE, reason="Triton mHC requires an AMD ROCm GPU")
def test_mhc_triton_post_noncontiguous_direct_gradients():
    batch, seq_len, hc_mult, hidden = 1, 5, 4, 129
    residual = _noncontiguous_last_dim((batch, seq_len, hc_mult, hidden), dtype=torch.bfloat16, seed=41)
    output = _noncontiguous_last_dim((batch, seq_len, hidden), dtype=torch.bfloat16, seed=42)
    post = _noncontiguous_last_dim((batch, seq_len, hc_mult), dtype=torch.float32, seed=43)
    comb = _noncontiguous_last_dim((batch, seq_len, hc_mult, hc_mult), dtype=torch.float32, seed=44)
    grad = _noncontiguous_last_dim(tuple(residual.shape), dtype=torch.bfloat16, seed=45)
    inputs_t = make_grad_leaves(output, residual, post, comb)
    inputs_e = make_grad_leaves(*(tensor.contiguous() for tensor in (output, residual, post, comb)))
    y_t = resolve_op("mhc", "post", "triton").wrapper(*inputs_t)
    y_e = resolve_op("mhc", "post", "eager").wrapper(*inputs_e)
    torch.testing.assert_close(y_t, y_e, rtol=MHC_FUSED_RTOL, atol=MHC_FUSED_ATOL)

    grads_t = torch.autograd.grad(y_t, inputs_t, grad_outputs=grad)
    grads_e = torch.autograd.grad(y_e, inputs_e, grad_outputs=grad.contiguous())
    for actual, expected in zip(grads_t, grads_e, strict=True):
        assert_gradient_direction_and_scale(actual, expected, min_cosine=0.999, norm_rtol=0.01)


@pytest.mark.skipif(not _TRITON_ROCM_AVAILABLE, reason="Triton mHC requires an AMD ROCm GPU")
def test_mhc_triton_pre_production_width():
    hc_mult, hidden = 4, 7168
    x, fn, scale, base = _mhc_pre_inputs(1, 3, hc_mult, hidden, seed=47)
    args = (1e-6, hc_mult, 3, 1e-6)
    inputs_t = make_grad_leaves(x, fn, scale, base)
    inputs_e = make_grad_leaves(x, fn, scale, base)
    values_t = resolve_op("mhc", "pre", "triton").wrapper(*inputs_t, *args)
    values_e = resolve_op("mhc", "pre", "eager").wrapper(*inputs_e, *args)
    grad_outputs = tuple(torch.randn_like(value) for value in values_t)
    grads_t = torch.autograd.grad(values_t, inputs_t, grad_outputs=grad_outputs)
    grads_e = torch.autograd.grad(values_e, inputs_e, grad_outputs=grad_outputs)
    for actual, expected in zip(grads_t, grads_e, strict=True):
        assert_gradient_direction_and_scale(actual, expected, min_cosine=0.999, norm_rtol=0.01)


@pytest.mark.skipif(not _TRITON_ROCM_AVAILABLE, reason="Triton mHC requires an AMD ROCm GPU")
def test_mhc_triton_rejects_invalid_contracts():
    hc_mult, hidden = 4, 64
    x, fn, scale, base = _mhc_pre_inputs(1, 8, hc_mult, hidden, seed=53)
    pre = resolve_op("mhc", "pre", "triton").wrapper
    post_op = resolve_op("mhc", "post", "triton").wrapper
    with pytest.raises(ValueError, match="scale shape"):
        pre(x, fn, scale.view(1, 3), base, 1e-6, hc_mult, 20, 1e-6)
    with pytest.raises(ValueError, match="base shape"):
        pre(x, fn, scale, base.view(1, -1), 1e-6, hc_mult, 20, 1e-6)
    with pytest.raises(ValueError, match="nonzero hidden"):
        pre(x[..., :0], fn[:, :0], scale, base, 1e-6, hc_mult, 20, 1e-6)

    output = torch.randn(1, 8, hidden, device="cuda", dtype=torch.float32)
    post = torch.randn(1, 8, hc_mult, device="cuda")
    comb = torch.randn(1, 8, hc_mult, hc_mult, device="cuda")
    with pytest.raises(TypeError, match="share a dtype"):
        post_op(output, x, post, comb)
    with pytest.raises(TypeError, match="float16, bfloat16, or float32"):
        post_op(output.to(torch.int32), x.to(torch.int32), post, comb)


@pytest.mark.skipif(not _TRITON_ROCM_AVAILABLE, reason="Triton mHC requires an AMD ROCm GPU")
def test_mhc_triton_rejects_unsupported_stream_count():
    x, fn, scale, base = _mhc_pre_inputs(1, 8, 3, 64, seed=31)
    with pytest.raises(ValueError, match="hc_mult"):
        resolve_op("mhc", "pre", "triton").wrapper(x, fn, scale, base, 1e-6, 3, 20, 1e-6)


def test_mhc_cann_pre_passes_fp32_parameters_in_kernel_order(monkeypatch: pytest.MonkeyPatch):
    calls = []
    monkeypatch.setattr(pre_cann, "npu_mhc_pre_sinkhorn", lambda *args: calls.append(args) or args[:3])
    hc_mult, hidden = 4, 8
    mix = (2 + hc_mult) * hc_mult
    x = torch.randn(1, 3, hc_mult, hidden, dtype=torch.bfloat16)
    fn = torch.randn(hc_mult * hidden, mix, dtype=torch.bfloat16).t()
    scale = torch.randn(3, dtype=torch.bfloat16)
    base = torch.randn(mix, dtype=torch.bfloat16)

    pre_cann.wrapper(x, fn, scale, base, 1e-6, hc_mult, 20, 2e-3)

    ((kernel_x, kernel_fn, kernel_scale, kernel_base, *scalars),) = calls
    assert kernel_x is x
    assert all(tensor.dtype == torch.float32 for tensor in (kernel_fn, kernel_scale, kernel_base))
    assert kernel_fn.is_contiguous()
    # The kernel takes hc_eps before norm_eps, the reverse of the op contract.
    assert scalars == [hc_mult, 20, 2e-3, 1e-6]


@pytest.mark.skipif(not _CANN_AVAILABLE, reason="CANN mHC requires Ascend NPU and cann_ops_transformer")
def test_mhc_cann_pre_post_matches_eager():
    torch.manual_seed(29)
    device = get_device_type()
    batch, seq_len, hc_mult, hidden = 1, 8, 4, 128
    norm_eps, hc_eps, sinkhorn_iters = 1e-6, 1e-6, 20
    mix = (2 + hc_mult) * hc_mult
    x = torch.randn(batch, seq_len, hc_mult, hidden, device=device, dtype=torch.bfloat16)
    fn = torch.randn(mix, hc_mult * hidden, device=device, dtype=torch.float32) * 0.01
    scale = torch.randn(3, device=device, dtype=torch.float32) * 0.01
    base = torch.randn(mix, device=device, dtype=torch.float32) * 0.01
    eager_pre = resolve_op("mhc", "pre", "eager").wrapper
    other_pre = resolve_op("mhc", "pre", "cann").wrapper
    eager_post = resolve_op("mhc", "post", "eager").wrapper
    other_post = resolve_op("mhc", "post", "cann").wrapper

    x_o, fn_o, scale_o, base_o = make_grad_leaves(x, fn, scale, base)
    x_e, fn_e, scale_e, base_e = make_grad_leaves(x, fn, scale, base)
    post_o, comb_o, collapsed_o = other_pre(x_o, fn_o, scale_o, base_o, norm_eps, hc_mult, sinkhorn_iters, hc_eps)
    post_e, comb_e, collapsed_e = eager_pre(x_e, fn_e, scale_e, base_e, norm_eps, hc_mult, sinkhorn_iters, hc_eps)
    y_o = other_post(collapsed_o * 0.75, x_o, post_o, comb_o)
    y_e = eager_post(collapsed_e * 0.75, x_e, post_e, comb_e)

    for actual, expected in ((post_o, post_e), (comb_o, comb_e), (collapsed_o, collapsed_e), (y_o, y_e)):
        torch.testing.assert_close(actual, expected, rtol=MHC_CANN_RTOL, atol=MHC_CANN_ATOL)

    grad = torch.randn_like(y_o)
    other_grads = torch.autograd.grad((y_o * grad).sum(), (x_o, fn_o, scale_o, base_o))
    eager_grads = torch.autograd.grad((y_e * grad).sum(), (x_e, fn_e, scale_e, base_e))
    for actual, expected in zip(other_grads, eager_grads, strict=True):
        assert_gradient_direction_and_scale(
            actual,
            expected,
            min_cosine=MHC_FUSED_GRAD_COSINE,
            norm_rtol=0.25,
        )


def test_mhc_cann_head_is_the_eager_math():
    cann_head = OP_REGISTRY._entries["mhc", "head", "cann", "npu"]
    assert cann_head.wrapper is resolve_op("mhc", "head", "eager").wrapper
