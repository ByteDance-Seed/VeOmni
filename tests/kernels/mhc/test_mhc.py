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

"""mHC eager vs HuggingFace DeepSeek-V4, and TileLang vs eager."""

from __future__ import annotations

import importlib.util

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperConnection, DeepseekV4HyperHead

from tests.kernels.tol import (
    EAGER_ATOL,
    EAGER_GRAD_ATOL,
    EAGER_GRAD_RTOL,
    EAGER_RTOL,
    MHC_FUSED_ATOL,
    MHC_FUSED_GRAD_COSINE,
    MHC_FUSED_RTOL,
)
from veomni.kernels import KERNEL_REGISTRY, resolve_kernel
from veomni.utils.device import IS_CUDA_AVAILABLE, get_gpu_compute_capability


# Installed classes: DeepseekV4HyperConnection / DeepseekV4HyperHead /
# DeepseekV4DecoderLayer in transformers 5.9.0.
# https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py

_TILELANG_AVAILABLE = (
    IS_CUDA_AVAILABLE and get_gpu_compute_capability() >= 90 and importlib.util.find_spec("tile_kernels") is not None
)


def _clone(*tensors: Tensor) -> tuple[Tensor, ...]:
    return tuple(t.detach().requires_grad_(True) for t in tensors)


def _cosine(actual: Tensor, expected: Tensor) -> float:
    return F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0).item()


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

    Copied from ``DeepseekV4DecoderLayer.forward`` (transformers v5.9.0):
    https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py
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

    x_e, fn_e, scale_e, base_e = _clone(x, fn, scale, base)
    post_e, comb_e, collapsed_e = resolve_kernel("mhc", "pre", "eager").wrapper(
        x_e, fn_e, scale_e, base_e, norm_eps, hc_mult, sinkhorn_iters, hc_eps
    )
    assert torch.allclose(post_e, post_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)
    assert torch.allclose(comb_e, comb_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)
    assert torch.allclose(collapsed_e, collapsed_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = torch.randn_like(collapsed_e)
    collapsed_h.backward(go)
    collapsed_e.backward(go)
    assert torch.allclose(x_e.grad, x_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(fn_e.grad, hc.fn.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(scale_e.grad, hc.scale.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(base_e.grad, hc.base.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_mhc_post_eager_matches_hf():
    torch.manual_seed(1)
    batch, seq_len, hc_mult, hidden = 2, 8, 4, 32
    residual = torch.randn(batch, seq_len, hc_mult, hidden)
    output = torch.randn(batch, seq_len, hidden)
    post = torch.randn(batch, seq_len, hc_mult)
    comb = torch.randn(batch, seq_len, hc_mult, hc_mult)

    out_h, res_h, post_h, comb_h = _clone(output, residual, post, comb)
    y_h = _hf_decoder_layer_post(out_h, res_h, post_h, comb_h)
    out_e, res_e, post_e, comb_e = _clone(output, residual, post, comb)
    y_e = resolve_kernel("mhc", "post", "eager").wrapper(out_e, res_e, post_e, comb_e)
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
    x_e, fn_e, scale_e, base_e = _clone(x, fn, scale, base)
    y_e = resolve_kernel("mhc", "head", "eager").wrapper(x_e, fn_e, scale_e, base_e, norm_eps, hc_mult, hc_eps)
    assert torch.allclose(y_e, y_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = torch.randn_like(y_e)
    y_h.backward(go)
    y_e.backward(go)
    assert torch.allclose(x_e.grad, x_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(fn_e.grad, head.hc_fn.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(scale_e.grad, head.hc_scale.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(base_e.grad, head.hc_base.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_mhc_eager_is_registered():
    for variant in ("pre", "post", "head"):
        assert "eager" in KERNEL_REGISTRY.list_available("mhc", variant)
        assert "eager" in KERNEL_REGISTRY.list_registered("mhc", variant)
        assert "tilelang" in KERNEL_REGISTRY.list_registered("mhc", variant)


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
    eager_pre = resolve_kernel("mhc", "pre", "eager").wrapper
    other_pre = resolve_kernel("mhc", "pre", "tilelang").wrapper
    eager_post = resolve_kernel("mhc", "post", "eager").wrapper
    other_post = resolve_kernel("mhc", "post", "tilelang").wrapper

    with torch.no_grad():
        inference = other_pre(x, fn, scale, base, norm_eps, hc_mult, sinkhorn_iters, hc_eps)
        eager_inference = eager_pre(x, fn, scale, base, norm_eps, hc_mult, sinkhorn_iters, hc_eps)
    for actual, expected in zip(inference, eager_inference, strict=True):
        torch.testing.assert_close(actual, expected, rtol=MHC_FUSED_RTOL, atol=MHC_FUSED_ATOL)

    x_o, fn_o, scale_o, base_o = _clone(x, fn, scale, base)
    x_e, fn_e, scale_e, base_e = _clone(x, fn, scale, base)
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
        assert torch.isfinite(actual).all()
        assert _cosine(actual, expected) > MHC_FUSED_GRAD_COSINE


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
    eager = resolve_kernel("mhc", "head", "eager").wrapper
    other = resolve_kernel("mhc", "head", "tilelang").wrapper

    x_o, fn_o, scale_o, base_o = _clone(x, fn, scale, base)
    x_e, fn_e, scale_e, base_e = _clone(x, fn, scale, base)
    y_o = other(x_o, fn_o, scale_o, base_o, norm_eps, hc_mult, hc_eps)
    y_e = eager(x_e, fn_e, scale_e, base_e, norm_eps, hc_mult, hc_eps)
    torch.testing.assert_close(y_o, y_e, rtol=MHC_FUSED_RTOL, atol=MHC_FUSED_ATOL)

    grad = torch.randn_like(y_o)
    other_grads = torch.autograd.grad((y_o * grad).sum(), (x_o, fn_o, scale_o, base_o))
    eager_grads = torch.autograd.grad((y_e * grad).sum(), (x_e, fn_e, scale_e, base_e))
    for actual, expected in zip(other_grads, eager_grads, strict=True):
        assert torch.isfinite(actual).all()
        assert _cosine(actual, expected) > MHC_FUSED_GRAD_COSINE
