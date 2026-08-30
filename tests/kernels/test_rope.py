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

"""RoPE eager vs HF, and fused impls vs eager."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from transformers.models.deepseek_v4.modeling_deepseek_v4 import apply_rotary_pos_emb as hf_dsv4_rope
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb as hf_full_rope
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb as hf_partial_rope
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb_vision as hf_vision_rope

from tests.kernels.tol import (
    EAGER_ATOL,
    EAGER_GRAD_ATOL,
    EAGER_GRAD_RTOL,
    EAGER_RTOL,
    ROPE_FUSED_ATOL,
    ROPE_FUSED_GRAD_ATOL,
    ROPE_FUSED_GRAD_RTOL,
    ROPE_FUSED_RTOL,
    ROPE_NPU_ATOL,
    ROPE_NPU_RTOL,
)
from veomni.kernels import resolve_kernel
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE


def _wan_official_rope_apply(x: Tensor, freqs: Tensor, head_dim: int) -> Tensor:
    """Wan2.1 official RoPE. transformers has no Wan.

    Copied from Wan-Video/Wan2.1 ``wan/modules/model.py`` ``rope_apply``:
    ``view_as_complex(x.float64.reshape(..., 2)) * freqs`` then ``view_as_real``.
    Upstream already receives ``[S, N, D]`` per sample. This only unpacks VeOmni's
    packed ``[B, S, N*D]``.
    """
    x = x.reshape(*x.shape[:2], -1, head_dim)
    x_c = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:3], -1, 2))
    return torch.view_as_real(x_c * freqs).flatten(2).to(x.dtype)


def _clone_qk(q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
    return q.detach().requires_grad_(True), k.detach().requires_grad_(True)


def _assert_pair(left: tuple[Tensor, Tensor], right: tuple[Tensor, Tensor], *, atol: float, rtol: float) -> None:
    assert torch.allclose(left[0], right[0], atol=atol, rtol=rtol)
    assert torch.allclose(left[1], right[1], atol=atol, rtol=rtol)


def test_full_eager_matches_hf():
    torch.manual_seed(0)
    q = torch.randn(2, 8, 16, 64, dtype=torch.float32, requires_grad=True)
    k = torch.randn(2, 4, 16, 64, dtype=torch.float32, requires_grad=True)
    cos = torch.randn(2, 16, 64, dtype=torch.float32)
    sin = torch.randn(2, 16, 64, dtype=torch.float32)

    q_h, k_h = _clone_qk(q, k)
    out_h = hf_full_rope(q_h, k_h, cos, sin, unsqueeze_dim=1)

    q_e, k_e = _clone_qk(q, k)
    out_e = resolve_kernel("rope", "full", "eager").wrapper(q_e, k_e, cos, sin, unsqueeze_dim=1)
    _assert_pair(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = (torch.randn_like(out_e[0]), torch.randn_like(out_e[1]))
    torch.autograd.backward(out_h, go)
    torch.autograd.backward(out_e, go)
    assert torch.allclose(q_e.grad, q_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(k_e.grad, k_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_partial_eager_matches_hf():
    torch.manual_seed(0)
    q = torch.randn(2, 8, 16, 128, dtype=torch.float32, requires_grad=True)
    k = torch.randn(2, 4, 16, 128, dtype=torch.float32, requires_grad=True)
    cos = torch.randn(2, 16, 64, dtype=torch.float32)
    sin = torch.randn(2, 16, 64, dtype=torch.float32)

    q_h, k_h = _clone_qk(q, k)
    out_h = hf_partial_rope(q_h, k_h, cos, sin, unsqueeze_dim=1)

    q_e, k_e = _clone_qk(q, k)
    out_e = resolve_kernel("rope", "partial", "eager").wrapper(q_e, k_e, cos, sin, unsqueeze_dim=1)
    _assert_pair(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = (torch.randn_like(out_e[0]), torch.randn_like(out_e[1]))
    torch.autograd.backward(out_h, go)
    torch.autograd.backward(out_e, go)
    assert torch.allclose(q_e.grad, q_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(k_e.grad, k_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_vision_eager_matches_hf():
    torch.manual_seed(0)
    q = torch.randn(16, 8, 64, dtype=torch.float32, requires_grad=True)
    k = torch.randn(16, 8, 64, dtype=torch.float32, requires_grad=True)
    cos = torch.randn(16, 64, dtype=torch.float32)
    sin = torch.randn(16, 64, dtype=torch.float32)

    q_h, k_h = _clone_qk(q, k)
    out_h = hf_vision_rope(q_h, k_h, cos, sin)

    q_e, k_e = _clone_qk(q, k)
    out_e = resolve_kernel("rope_vision", "full", "eager").wrapper(q_e, k_e, cos, sin)
    _assert_pair(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = (torch.randn_like(out_e[0]), torch.randn_like(out_e[1]))
    torch.autograd.backward(out_h, go)
    torch.autograd.backward(out_e, go)
    assert torch.allclose(q_e.grad, q_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(k_e.grad, k_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="liger RoPE needs CUDA")
def test_full_liger_matches_eager():
    pytest.importorskip("liger_kernel")
    eager = resolve_kernel("rope", "full", "eager").wrapper
    other = resolve_kernel("rope", "full", "liger_kernel").wrapper
    torch.manual_seed(0)
    q = torch.randn(2, 8, 16, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 4, 16, 64, device="cuda", dtype=torch.bfloat16)
    # Llama-style tables duplicate the first half. Liger only reads that half.
    cos_half = torch.randn(2, 16, 32, device="cuda", dtype=torch.bfloat16)
    sin_half = torch.randn(2, 16, 32, device="cuda", dtype=torch.bfloat16)
    cos = torch.cat((cos_half, cos_half), dim=-1)
    sin = torch.cat((sin_half, sin_half), dim=-1)

    q_e, k_e = _clone_qk(q, k)
    q_o, k_o = _clone_qk(q, k)
    out_e = eager(q_e, k_e, cos, sin, unsqueeze_dim=1)
    out_o = other(q_o, k_o, cos, sin, unsqueeze_dim=1)
    _assert_pair(out_e, out_o, atol=ROPE_FUSED_ATOL, rtol=ROPE_FUSED_RTOL)

    go = (torch.randn_like(out_e[0]), torch.randn_like(out_e[1]))
    torch.autograd.backward(out_e, go)
    torch.autograd.backward(out_o, go)
    assert torch.allclose(q_e.grad, q_o.grad, atol=ROPE_FUSED_GRAD_ATOL, rtol=ROPE_FUSED_GRAD_RTOL)
    assert torch.allclose(k_e.grad, k_o.grad, atol=ROPE_FUSED_GRAD_ATOL, rtol=ROPE_FUSED_GRAD_RTOL)


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="NPU RoPE needs NPU")
@pytest.mark.parametrize("variant", ["full", "partial"])
def test_rope_npu_matches_eager(variant: str):
    eager = resolve_kernel("rope", variant, "eager").wrapper
    other = resolve_kernel("rope", variant, "npu").wrapper
    torch.manual_seed(0)
    head_dim = 64 if variant == "full" else 128
    rotary_dim = 64
    q = torch.randn(2, 8, 16, head_dim, device="npu", dtype=torch.bfloat16)
    k = torch.randn(2, 4, 16, head_dim, device="npu", dtype=torch.bfloat16)
    cos = torch.randn(2, 16, rotary_dim, device="npu", dtype=torch.bfloat16)
    sin = torch.randn(2, 16, rotary_dim, device="npu", dtype=torch.bfloat16)

    q_e, k_e = _clone_qk(q, k)
    q_o, k_o = _clone_qk(q, k)
    out_e = eager(q_e, k_e, cos, sin, unsqueeze_dim=1)
    out_o = other(q_o, k_o, cos, sin, unsqueeze_dim=1)
    _assert_pair(
        (out_e[0].float(), out_e[1].float()),
        (out_o[0].float(), out_o[1].float()),
        atol=ROPE_NPU_ATOL,
        rtol=ROPE_NPU_RTOL,
    )

    go = (torch.randn_like(out_e[0]), torch.randn_like(out_e[1]))
    torch.autograd.backward(out_e, go)
    torch.autograd.backward(out_o, go)
    assert torch.allclose(q_e.grad.float(), q_o.grad.float(), atol=ROPE_NPU_ATOL, rtol=ROPE_NPU_RTOL)
    assert torch.allclose(k_e.grad.float(), k_o.grad.float(), atol=ROPE_NPU_ATOL, rtol=ROPE_NPU_RTOL)


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="NPU vision RoPE needs NPU")
def test_vision_npu_matches_eager():
    eager = resolve_kernel("rope_vision", "full", "eager").wrapper
    other = resolve_kernel("rope_vision", "full", "npu").wrapper
    torch.manual_seed(0)
    q = torch.randn(16, 8, 64, device="npu", dtype=torch.bfloat16)
    k = torch.randn(16, 8, 64, device="npu", dtype=torch.bfloat16)
    cos = torch.randn(16, 64, device="npu", dtype=torch.bfloat16)
    sin = torch.randn(16, 64, device="npu", dtype=torch.bfloat16)

    q_e, k_e = _clone_qk(q, k)
    q_o, k_o = _clone_qk(q, k)
    out_e = eager(q_e, k_e, cos, sin)
    out_o = other(q_o, k_o, cos, sin)
    _assert_pair(
        (out_e[0].float(), out_e[1].float()),
        (out_o[0].float(), out_o[1].float()),
        atol=ROPE_NPU_ATOL,
        rtol=ROPE_NPU_RTOL,
    )

    go = (torch.randn_like(out_e[0]), torch.randn_like(out_e[1]))
    torch.autograd.backward(out_e, go)
    torch.autograd.backward(out_o, go)
    assert torch.allclose(q_e.grad.float(), q_o.grad.float(), atol=ROPE_NPU_ATOL, rtol=ROPE_NPU_RTOL)
    assert torch.allclose(k_e.grad.float(), k_o.grad.float(), atol=ROPE_NPU_ATOL, rtol=ROPE_NPU_RTOL)


def test_deepseek_v4_eager_matches_hf():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16, 128, dtype=torch.float32, requires_grad=True)
    angle = torch.randn(2, 16, 32, dtype=torch.float32)
    cos, sin = angle.cos(), angle.sin()

    x_h = x.detach().requires_grad_(True)
    x_e = x.detach().requires_grad_(True)
    out_h = hf_dsv4_rope(x_h, cos, sin, unsqueeze_dim=1)
    out_e = resolve_kernel("rope", "deepseek_v4", "eager").wrapper(x_e, cos, sin, unsqueeze_dim=1)
    assert torch.allclose(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = torch.randn_like(out_e)
    out_h.backward(go)
    out_e.backward(go)
    assert torch.allclose(x_e.grad, x_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="DeepSeek-V4 Triton RoPE needs CUDA")
def test_deepseek_v4_triton_matches_eager():
    pytest.importorskip("triton")
    eager = resolve_kernel("rope", "deepseek_v4", "eager").wrapper
    other = resolve_kernel("rope", "deepseek_v4", "triton").wrapper
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16, 128, device="cuda", dtype=torch.bfloat16)
    angle = torch.randn(2, 16, 32, device="cuda", dtype=torch.bfloat16)
    cos, sin = angle.cos(), angle.sin()

    x_e = x.detach().requires_grad_(True)
    x_o = x.detach().requires_grad_(True)
    out_e = eager(x_e, cos, sin, unsqueeze_dim=1)
    out_o = other(x_o, cos, sin, unsqueeze_dim=1)
    assert torch.allclose(out_e, out_o, atol=ROPE_FUSED_ATOL, rtol=ROPE_FUSED_RTOL)

    go = torch.randn_like(out_e)
    out_e.backward(go)
    out_o.backward(go)
    assert torch.allclose(x_e.grad, x_o.grad, atol=ROPE_FUSED_GRAD_ATOL, rtol=ROPE_FUSED_GRAD_RTOL)


def test_wan_eager_matches_official():
    torch.manual_seed(0)
    head_dim = 64
    x = torch.randn(2, 16, 4 * head_dim, dtype=torch.float32, requires_grad=True)
    angle = torch.randn(16, 1, head_dim // 2, dtype=torch.float64)
    freqs = torch.polar(torch.ones_like(angle), angle)

    x_h = x.detach().requires_grad_(True)
    out_h = _wan_official_rope_apply(x_h, freqs, head_dim)

    x_e = x.detach().requires_grad_(True)
    out_e = resolve_kernel("rope", "wan", "eager").wrapper(x_e, freqs, head_dim=head_dim)
    assert torch.allclose(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = torch.randn_like(out_e)
    out_h.backward(go)
    out_e.backward(go)
    assert torch.allclose(x_e.grad, x_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Wan Triton RoPE needs CUDA")
def test_wan_triton_matches_eager():
    pytest.importorskip("triton")
    eager = resolve_kernel("rope", "wan", "eager").wrapper
    other = resolve_kernel("rope", "wan", "triton").wrapper
    torch.manual_seed(0)
    head_dim = 64
    x = torch.randn(2, 16, 4 * head_dim, device="cuda", dtype=torch.bfloat16)
    angle = torch.randn(16, 1, head_dim // 2, device="cuda", dtype=torch.float64)
    freqs = torch.polar(torch.ones_like(angle), angle)

    x_e = x.detach().requires_grad_(True)
    x_o = x.detach().requires_grad_(True)
    out_e = eager(x_e, freqs, head_dim=head_dim)
    out_o = other(x_o, freqs, head_dim=head_dim)
    assert torch.allclose(out_e, out_o, atol=ROPE_FUSED_ATOL, rtol=ROPE_FUSED_RTOL)

    go = torch.randn_like(out_e)
    out_e.backward(go)
    out_o.backward(go)
    assert torch.allclose(x_e.grad, x_o.grad, atol=ROPE_FUSED_GRAD_ATOL, rtol=ROPE_FUSED_GRAD_RTOL)


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="NPU Wan RoPE needs NPU")
def test_wan_npu_matches_eager():
    eager = resolve_kernel("rope", "wan", "eager").wrapper
    other = resolve_kernel("rope", "wan", "npu").wrapper
    torch.manual_seed(0)
    head_dim = 64
    x = torch.randn(2, 16, 4 * head_dim, device="npu", dtype=torch.bfloat16)
    angle = torch.randn(16, 1, head_dim // 2, device="npu", dtype=torch.float64)
    freqs = torch.polar(torch.ones_like(angle), angle)

    x_e = x.detach().requires_grad_(True)
    x_o = x.detach().requires_grad_(True)
    out_e = eager(x_e, freqs, head_dim=head_dim)
    out_o = other(x_o, freqs, head_dim=head_dim)
    assert torch.allclose(out_e.float(), out_o.float(), atol=ROPE_NPU_ATOL, rtol=ROPE_NPU_RTOL)

    go = torch.randn_like(out_e)
    out_e.backward(go)
    out_o.backward(go)
    assert torch.allclose(x_e.grad.float(), x_o.grad.float(), atol=ROPE_NPU_ATOL, rtol=ROPE_NPU_RTOL)
