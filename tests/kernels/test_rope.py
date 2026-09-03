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
    ROPE_NPU_PROD_BF16_ATOL,
    ROPE_NPU_PROD_FP16_ATOL,
    ROPE_NPU_RTOL,
)
from veomni.kernels import resolve_kernel
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE


def _wan_reference_rope_apply(x: Tensor, freqs: Tensor, head_dim: int) -> Tensor:
    """Wan2.1 reference RoPE. transformers has no Wan.

    Copied from https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py ``rope_apply``:
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


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="liger RoPE needs CUDA")
def test_full_liger_matches_eager_unsqueeze_dim_2():
    pytest.importorskip("liger_kernel")
    eager = resolve_kernel("rope", "full", "eager").wrapper
    other = resolve_kernel("rope", "full", "liger_kernel").wrapper
    torch.manual_seed(4)
    # HF unsqueeze_dim=2: q/k are [B, S, H, D], tables are [B, S, D].
    q = torch.randn(2, 16, 8, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 16, 4, 64, device="cuda", dtype=torch.bfloat16)
    cos_half = torch.randn(2, 16, 32, device="cuda", dtype=torch.bfloat16)
    sin_half = torch.randn(2, 16, 32, device="cuda", dtype=torch.bfloat16)
    cos = torch.cat((cos_half, cos_half), dim=-1)
    sin = torch.cat((sin_half, sin_half), dim=-1)

    q_e, k_e = _clone_qk(q, k)
    q_o, k_o = _clone_qk(q, k)
    out_e = eager(q_e, k_e, cos, sin, unsqueeze_dim=2)
    out_o = other(q_o, k_o, cos, sin, unsqueeze_dim=2)
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


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="NPU RoPE needs NPU")
@pytest.mark.parametrize("batch, heads, seqlen, head_dim", [(1, 8, 256, 128), (2, 16, 64, 64)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_full_npu_production_shape(batch: int, heads: int, seqlen: int, head_dim: int, dtype: torch.dtype):
    eager = resolve_kernel("rope", "full", "eager").wrapper
    other = resolve_kernel("rope", "full", "npu").wrapper
    torch.manual_seed(3)
    q = torch.randn(batch, heads, seqlen, head_dim, device="npu", dtype=dtype)
    k = torch.randn(batch, heads, seqlen, head_dim, device="npu", dtype=dtype)
    half = torch.randn(batch, seqlen, head_dim // 2, device="npu", dtype=dtype)
    cos = torch.cat((half, half), dim=-1)
    half_s = torch.randn(batch, seqlen, head_dim // 2, device="npu", dtype=dtype)
    sin = torch.cat((half_s, half_s), dim=-1)
    atol = ROPE_NPU_PROD_BF16_ATOL if dtype == torch.bfloat16 else ROPE_NPU_PROD_FP16_ATOL
    out_e = eager(q, k, cos, sin, unsqueeze_dim=1)
    out_o = other(q, k, cos, sin, unsqueeze_dim=1)
    _assert_pair(
        (out_e[0].float(), out_e[1].float()),
        (out_o[0].float(), out_o[1].float()),
        atol=atol,
        rtol=atol,
    )


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="NPU RoPE needs NPU")
@pytest.mark.parametrize("head_dim, rotary_dim", [(128, 64), (256, 128)])
def test_partial_npu_production_shape(head_dim: int, rotary_dim: int):
    eager = resolve_kernel("rope", "partial", "eager").wrapper
    other = resolve_kernel("rope", "partial", "npu").wrapper
    torch.manual_seed(5)
    q = torch.randn(2, 4, 16, head_dim, device="npu", dtype=torch.bfloat16)
    k = torch.randn(2, 4, 16, head_dim, device="npu", dtype=torch.bfloat16)
    half = torch.randn(2, 16, rotary_dim // 2, device="npu", dtype=torch.bfloat16)
    cos = torch.cat((half, half), dim=-1)
    half_s = torch.randn(2, 16, rotary_dim // 2, device="npu", dtype=torch.bfloat16)
    sin = torch.cat((half_s, half_s), dim=-1)
    out_e = eager(q, k, cos, sin, unsqueeze_dim=1)
    out_o = other(q, k, cos, sin, unsqueeze_dim=1)
    _assert_pair(
        (out_e[0].float(), out_e[1].float()),
        (out_o[0].float(), out_o[1].float()),
        atol=ROPE_NPU_PROD_BF16_ATOL,
        rtol=ROPE_NPU_PROD_BF16_ATOL,
    )


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="NPU RoPE needs NPU")
def test_partial_npu_pass_through_preserved():
    other = resolve_kernel("rope", "partial", "npu").wrapper
    torch.manual_seed(6)
    rotary_dim = 32
    q = torch.randn(1, 2, 4, 64, device="npu", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 4, 64, device="npu", dtype=torch.bfloat16)
    half = torch.randn(1, 4, rotary_dim // 2, device="npu", dtype=torch.bfloat16)
    cos = torch.cat((half, half), dim=-1)
    half_s = torch.randn(1, 4, rotary_dim // 2, device="npu", dtype=torch.bfloat16)
    sin = torch.cat((half_s, half_s), dim=-1)
    out_q, out_k = other(q, k, cos, sin, unsqueeze_dim=1)
    assert torch.equal(out_q[..., rotary_dim:], q[..., rotary_dim:])
    assert torch.equal(out_k[..., rotary_dim:], k[..., rotary_dim:])


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


# Mirrors the real DeepSeek-V4 RoPE call sites. ``transposed`` marks the ones
# that reach the op as a ``[B, S, H, D].transpose(1, 2)`` view (Q, MQA KV, the
# attention output) rather than a contiguous tensor (compressor entries).
_DSV4_ROPE_CALL_SITES = [
    pytest.param(1, 8, 37, 512, 64, True, id="query"),
    pytest.param(2, 1, 64, 512, 64, True, id="mqa_kv"),
    pytest.param(1, 1, 13, 512, 64, False, id="compressed_entries"),
    pytest.param(2, 4, 33, 128, 64, True, id="indexer_query"),
    pytest.param(1, 2, 16, 64, 64, False, id="rope_spans_full_head"),
    pytest.param(2, 3, 33, 48, 24, True, id="rope_dim_not_power_of_two"),
]


def _dsv4_rope_inputs(
    batch: int,
    heads: int,
    seqlen: int,
    head_dim: int,
    rope_dim: int,
    transposed: bool,
    dtype: torch.dtype,
    device: str = "cuda",
) -> tuple[Tensor, Tensor, Tensor]:
    if transposed:
        x = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype).transpose(1, 2)
    else:
        x = torch.randn(batch, heads, seqlen, head_dim, device=device, dtype=dtype)
    angle = torch.randn(batch, seqlen, rope_dim // 2, device=device, dtype=dtype)
    return x, angle.cos(), angle.sin()


# The eager backward rounds each of its two branches to the activation dtype
# before summing them, so individual elements can cancel to exactly zero where
# the fused kernel's single rounding leaves a residue. That makes a relative
# comparison meaningless per element; bound the absolute error at ~2 ULP of the
# operand scale instead.
_DSV4_ROPE_GRAD_TOLERANCE = {
    torch.bfloat16: {"rtol": 1.6e-2, "atol": 1e-2},
    torch.float32: {"rtol": 1.3e-6, "atol": 1e-6},
}


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="DeepSeek-V4 Triton RoPE needs CUDA")
@pytest.mark.parametrize("batch, heads, seqlen, head_dim, rope_dim, transposed", _DSV4_ROPE_CALL_SITES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_deepseek_v4_triton_matches_eager(batch, heads, seqlen, head_dim, rope_dim, transposed, dtype):
    pytest.importorskip("triton")
    eager = resolve_kernel("rope", "deepseek_v4", "eager").wrapper
    other = resolve_kernel("rope", "deepseek_v4", "triton").wrapper
    torch.manual_seed(7)
    x, cos, sin = _dsv4_rope_inputs(batch, heads, seqlen, head_dim, rope_dim, transposed, dtype)
    grad = torch.randn(batch, heads, seqlen, head_dim, device="cuda", dtype=dtype)

    x_e = x.detach().clone().requires_grad_(True)
    x_o = x.detach().clone().requires_grad_(True)
    out_e = eager(x_e, cos, sin, unsqueeze_dim=1)
    out_o = other(x_o, cos, sin, unsqueeze_dim=1)
    assert out_o.shape == out_e.shape
    assert out_o.is_contiguous()
    torch.testing.assert_close(out_o, out_e)

    out_e.backward(grad)
    out_o.backward(grad)
    torch.testing.assert_close(x_o.grad, x_e.grad, **_DSV4_ROPE_GRAD_TOLERANCE[dtype])


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="DeepSeek-V4 Triton RoPE needs CUDA")
def test_deepseek_v4_triton_inverse_rotation_round_trips():
    pytest.importorskip("triton")
    rope = resolve_kernel("rope", "deepseek_v4", "triton").wrapper
    torch.manual_seed(7)
    x, cos, sin = _dsv4_rope_inputs(1, 4, 32, 512, 64, True, torch.float32)

    round_tripped = rope(rope(x, cos, sin, unsqueeze_dim=1), cos, -sin, unsqueeze_dim=1)

    torch.testing.assert_close(round_tripped, x.contiguous(), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="DeepSeek-V4 Triton RoPE needs CUDA")
def test_deepseek_v4_triton_saves_only_cos_sin():
    pytest.importorskip("triton")
    rope = resolve_kernel("rope", "deepseek_v4", "triton").wrapper
    torch.manual_seed(7)
    x, cos, sin = _dsv4_rope_inputs(1, 4, 32, 512, 64, True, torch.bfloat16)
    out = rope(x.detach().requires_grad_(True), cos, sin, unsqueeze_dim=1)
    assert [tensor.shape for tensor in out.grad_fn.saved_tensors] == [cos.shape, sin.shape]


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="DeepSeek-V4 Triton RoPE needs CUDA")
@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda x, cos, sin: (x, cos, sin, 2), id="unsqueeze_dim_not_one"),
        pytest.param(lambda x, cos, sin: (x[0], cos, sin, 1), id="x_not_4d"),
        pytest.param(lambda x, cos, sin: (x, cos.requires_grad_(True), sin, 1), id="cos_requires_grad"),
        pytest.param(lambda x, cos, sin: (x, cos[:, :-1], sin[:, :-1], 1), id="cos_seqlen_mismatch"),
        pytest.param(lambda x, cos, sin: (x[..., :-1], cos, sin, 1), id="odd_nope_dim"),
        pytest.param(lambda x, cos, sin: (x, cos[..., :0], sin[..., :0], 1), id="empty_rope_dim"),
        pytest.param(lambda x, cos, sin: (x, cos.cpu(), sin.cpu(), 1), id="cos_on_other_device"),
    ],
)
def test_deepseek_v4_triton_falls_back_when_unsupported(monkeypatch, mutate):
    pytest.importorskip("triton")
    from veomni.kernels._kernels.rope.deepseek_v4 import triton as dsv4_triton
    from veomni.kernels.registry import SavedState

    torch.manual_seed(7)
    x, cos, sin, unsqueeze_dim = mutate(*_dsv4_rope_inputs(1, 4, 32, 512, 64, True, torch.float32))

    monkeypatch.setattr(
        dsv4_triton,
        "_rotary_launch",
        lambda *a, **k: pytest.fail("unsupported input reached the Triton kernel"),
    )
    reached_eager = False

    def record_eager(tensor, *args, **kwargs):
        nonlocal reached_eager
        reached_eager = True
        return tensor, SavedState((cos, sin), dsv4_triton._Meta(False, unsqueeze_dim))

    monkeypatch.setattr(dsv4_triton._eager, "forward", record_eager)
    resolve_kernel("rope", "deepseek_v4", "triton").wrapper(x, cos, sin, unsqueeze_dim=unsqueeze_dim)
    assert reached_eager


def test_deepseek_v4_triton_fallback_matches_eager():
    from veomni.kernels._kernels.rope.deepseek_v4 import eager as dsv4_eager
    from veomni.kernels._kernels.rope.deepseek_v4 import triton as dsv4_triton

    torch.manual_seed(7)
    x, cos, sin = _dsv4_rope_inputs(1, 4, 32, 512, 64, True, torch.float32, device="cpu")
    out_e, _ = dsv4_eager.forward(x, cos, sin, unsqueeze_dim=1)
    out_o, _ = dsv4_triton.forward(x, cos, sin, unsqueeze_dim=1)
    torch.testing.assert_close(out_o, out_e)


def test_wan_eager_matches_reference():
    torch.manual_seed(0)
    head_dim = 64
    x = torch.randn(2, 16, 4 * head_dim, dtype=torch.float32, requires_grad=True)
    angle = torch.randn(16, 1, head_dim // 2, dtype=torch.float64)
    freqs = torch.polar(torch.ones_like(angle), angle)

    x_h = x.detach().requires_grad_(True)
    out_h = _wan_reference_rope_apply(x_h, freqs, head_dim)

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
