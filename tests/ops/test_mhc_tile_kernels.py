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

import importlib.util
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from veomni.ops.dispatch import OpSlot
from veomni.ops.kernel_registry import KERNEL_REGISTRY
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_gpu_compute_capability


_REGISTRY_MODULE = "veomni.ops.kernel_registry"
DEVICE = get_device_type()
_MHC_GPU_AVAILABLE = (
    IS_CUDA_AVAILABLE and get_gpu_compute_capability() >= 90 and importlib.util.find_spec("tile_kernels") is not None
)


def _eager_pre(x, fn, scale, base, norm_eps, hc_mult, sinkhorn_iters, hc_eps):
    flat = x.flatten(start_dim=2).float()
    flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    pre_w, post_w, comb_w = F.linear(flat, fn.float()).split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    pre_b, post_b, comb_b = base.split([hc_mult, hc_mult, hc_mult * hc_mult])
    pre_scale, post_scale, comb_scale = scale.unbind(0)
    pre = torch.sigmoid(pre_w * pre_scale + pre_b) + hc_eps
    post = 2 * torch.sigmoid(post_w * post_scale + post_b)
    comb_logits = comb_w.view(*comb_w.shape[:-1], hc_mult, hc_mult) * comb_scale + comb_b.view(hc_mult, hc_mult)
    comb = torch.softmax(comb_logits, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    collapsed = (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)
    return post, comb, collapsed


def _eager_post(output, residual, post, comb):
    dtype = residual.dtype
    return post.to(dtype).unsqueeze(-1) * output.unsqueeze(-2) + torch.matmul(
        comb.to(dtype).transpose(-1, -2), residual
    )


def _eager_head(x, fn, scale, base, norm_eps, hc_eps):
    flat = x.flatten(2).float()
    flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(flat, fn.float())
    pre = torch.sigmoid(mixes * scale.float() + base.float()) + hc_eps
    return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)


def _clone_with_grad(tensor):
    return tensor.detach().clone().requires_grad_(True)


def _cosine(actual, expected):
    return F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0).item()


@pytest.mark.skipif(not _MHC_GPU_AVAILABLE, reason="TileKernels mHC requires an SM90+ NVIDIA CUDA GPU")
def test_tile_kernels_mhc_pre_post_forward_backward_matches_eager():
    from veomni.ops.kernels.mhc.tile_kernels import mhc_post_tile_kernels, mhc_pre_tile_kernels

    torch.manual_seed(17)
    batch, seq_len, hc_mult, hidden = 1, 32, 4, 256
    norm_eps, hc_eps, sinkhorn_iters = 1e-6, 1e-6, 20
    mix = (2 + hc_mult) * hc_mult
    x = torch.randn(batch, seq_len, hc_mult, hidden, device=DEVICE, dtype=torch.bfloat16)
    fn = torch.randn(mix, hc_mult * hidden, device=DEVICE, dtype=torch.float32) * 0.01
    scale = torch.randn(3, device=DEVICE, dtype=torch.float32) * 0.01
    base = torch.randn(mix, device=DEVICE, dtype=torch.float32) * 0.01

    with torch.no_grad():
        inference_outputs = mhc_pre_tile_kernels(x, fn, scale, base, norm_eps, hc_mult, sinkhorn_iters, hc_eps)
        eager_inference_outputs = _eager_pre(x, fn, scale, base, norm_eps, hc_mult, sinkhorn_iters, hc_eps)
    for actual, expected in zip(inference_outputs, eager_inference_outputs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    kernel_inputs = tuple(_clone_with_grad(tensor) for tensor in (x, fn, scale, base))
    eager_inputs = tuple(_clone_with_grad(tensor) for tensor in (x, fn, scale, base))

    kernel_post, kernel_comb, kernel_collapsed = mhc_pre_tile_kernels(
        *kernel_inputs, norm_eps, hc_mult, sinkhorn_iters, hc_eps
    )
    eager_post, eager_comb, eager_collapsed = _eager_pre(*eager_inputs, norm_eps, hc_mult, sinkhorn_iters, hc_eps)
    kernel_output = mhc_post_tile_kernels(kernel_collapsed * 0.75, kernel_inputs[0], kernel_post, kernel_comb)
    eager_output = _eager_post(eager_collapsed * 0.75, eager_inputs[0], eager_post, eager_comb)

    torch.testing.assert_close(kernel_post, eager_post, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(kernel_comb, eager_comb, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(kernel_collapsed, eager_collapsed, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(kernel_output, eager_output, rtol=2e-2, atol=2e-2)

    grad = torch.randn_like(kernel_output)
    kernel_grads = torch.autograd.grad((kernel_output * grad).sum(), kernel_inputs)
    eager_grads = torch.autograd.grad((eager_output * grad).sum(), eager_inputs)
    for actual, expected in zip(kernel_grads, eager_grads, strict=True):
        assert torch.isfinite(actual).all()
        assert _cosine(actual, expected) > 0.98


@pytest.mark.skipif(not _MHC_GPU_AVAILABLE, reason="TileKernels mHC requires an SM90+ NVIDIA CUDA GPU")
def test_tile_kernels_mhc_head_forward_backward_matches_eager():
    from veomni.ops.kernels.mhc.tile_kernels import mhc_head_tile_kernels

    torch.manual_seed(23)
    batch, seq_len, hc_mult, hidden = 1, 32, 4, 256
    norm_eps, hc_eps = 1e-6, 1e-6
    x = torch.randn(batch, seq_len, hc_mult, hidden, device=DEVICE, dtype=torch.bfloat16)
    fn = torch.randn(hc_mult, hc_mult * hidden, device=DEVICE, dtype=torch.float32) * 0.01
    scale = torch.randn(1, device=DEVICE, dtype=torch.float32) * 0.01
    base = torch.randn(hc_mult, device=DEVICE, dtype=torch.float32) * 0.01
    kernel_inputs = tuple(_clone_with_grad(tensor) for tensor in (x, fn, scale, base))
    eager_inputs = tuple(_clone_with_grad(tensor) for tensor in (x, fn, scale, base))

    kernel_output = mhc_head_tile_kernels(*kernel_inputs, norm_eps, hc_mult, hc_eps)
    eager_output = _eager_head(*eager_inputs, norm_eps, hc_eps)
    torch.testing.assert_close(kernel_output, eager_output, rtol=2e-2, atol=2e-2)

    grad = torch.randn_like(kernel_output)
    kernel_grads = torch.autograd.grad((kernel_output * grad).sum(), kernel_inputs)
    eager_grads = torch.autograd.grad((eager_output * grad).sum(), eager_inputs)
    for actual, expected in zip(kernel_grads, eager_grads, strict=True):
        assert torch.isfinite(actual).all()
        assert _cosine(actual, expected) > 0.98


@pytest.mark.parametrize("variant", ["pre", "post", "head"])
def test_tile_kernels_mhc_registry_entries(variant):
    assert "tilelang" in KERNEL_REGISTRY.list_available("mhc", variant)


@patch(f"{_REGISTRY_MODULE}.IS_CUDA_AVAILABLE", True)
@patch(f"{_REGISTRY_MODULE}.IS_NPU_AVAILABLE", False)
@patch(f"{_REGISTRY_MODULE}.get_gpu_compute_capability", return_value=90)
def test_bind_veomni_ops_uses_mhc_implementation(_mock_cc):
    from veomni.arguments.arguments_types import OpsImplementationConfig
    from veomni.models.auto import _bind_veomni_ops

    slots = [OpSlot("mhc", variant) for variant in ("pre", "post", "head")]
    fake_module = SimpleNamespace(
        veomni_mhc_pre=slots[0],
        veomni_mhc_post=slots[1],
        veomni_mhc_head=slots[2],
    )
    ops_config = OpsImplementationConfig(
        attn_implementation="eager",
        moe_implementation="eager",
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
        load_balancing_loss_implementation="eager",
        rms_norm_gated_implementation="eager",
        causal_conv1d_implementation="eager",
        chunk_gated_delta_rule_implementation="eager",
        mhc_implementation="tilelang",
    )

    assert _bind_veomni_ops(fake_module, ops_config)
    assert all(slot.use_non_eager_impl for slot in slots)
