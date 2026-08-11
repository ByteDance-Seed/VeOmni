# Copyright (c) 2023, Tri Dao.
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

"""Fused Triton RoPE for DeepSeek-V4's partial + interleaved layout.

DeepSeek-V4 rotates only the trailing ``qk_rope_head_dim`` channels of each head
(``head_dim=512``, ``qk_rope_head_dim=64`` — 12.5%) using an interleaved
even/odd pairing, and receives half-width ``cos``/``sin`` tables shaped
``[B, S, rope_dim // 2]``. The eager reference in the generated modeling walks
that in ~7 separate passes: a strided ``rotate_half`` materialization, fp32
upcasts of both operands, the products and sum, the downcast, and a closing
``torch.cat`` that rewrites all 512 channels to change 64.

This kernel makes it a single pass — one read of ``x``, one write of the output —
by covering the full ``head_dim`` and passing the untouched ``nope`` channels
through in the same launch. Copying them separately with a sliced ``copy_``
measured ~3x slower per byte than the flat traffic, since a partial-channel copy
falls back to a strided elementwise kernel.

Numerics, relative to that reference: operands are upcast to fp32, ``cos``/``sin``
are consumed at their incoming precision (bf16 during training, fp32 under
``eval``), and the result is rounded back to ``x.dtype`` on store. Everything
below agrees with eager to within one ULP of the activation dtype, in both
directions. Two known sources, neither avoidable at reasonable cost:

- The compiler contracts ``x * cos - x_swap * sin`` into an FMA, dropping the
  intermediate rounding of each product. In bf16 the store's fp32 -> bf16
  rounding absorbs that (forward output comes out bit-identical in practice); in
  fp32 it shows up as ~1 ULP.
- RoPE is orthogonal, so the backward is the inverse rotation of the incoming
  gradient, computed here as one fp32 expression rounded once. The eager op chain
  instead rounds each of its two branches to bf16 (the ``.float()`` casts'
  backward) before autograd sums them; this kernel does not reproduce that double
  rounding, so its gradients are marginally *tighter*. Forcing the extra rounding
  mid-kernel does not survive the same FMA contraction.

Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/rotary.py
"""

import torch
import triton
import triton.language as tl

from ....utils.device import get_torch_device


# Guards ``BLOCK_D``, which covers the whole head, against unbounded register
# pressure. DeepSeek-V4 uses 512.
_MAX_HEAD_DIM = 1024


@triton.jit
def _dsv4_rotary_interleaved_kernel(
    OUT,
    X,
    COS,
    SIN,
    seqlen,
    headdim,
    nope_dim,
    stride_out_batch,
    stride_out_nheads,
    stride_out_seqlen,
    stride_out_headdim,
    stride_x_batch,
    stride_x_nheads,
    stride_x_seqlen,
    stride_x_headdim,
    stride_cos_batch,
    stride_cos_seqlen,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    CONJUGATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)

    X = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
    OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
    # cos/sin carry a batch dim but no head dim — they broadcast over heads.
    COS = COS + pid_batch * stride_cos_batch
    SIN = SIN + pid_batch * stride_cos_batch

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = tl.arange(0, BLOCK_D)
    mask_m = rm[:, None] < seqlen
    in_head = rd[None, :] < headdim
    is_rope = (rd[None, :] >= nope_dim) & in_head

    # ``nope_dim`` is even, so a rope channel's interleaved partner is ``rd ^ 1``.
    # Loading X[..., 0::2] and X[..., 1::2] separately would be slow; the swapped
    # load hits the same cache lines as the straight one.
    x = tl.load(X + rm[:, None] * stride_x_seqlen + rd[None, :] * stride_x_headdim, mask=mask_m & in_head, other=0.0)
    x_swap = tl.load(
        X + rm[:, None] * stride_x_seqlen + (rd[None, :] ^ 1) * stride_x_headdim, mask=mask_m & is_rope, other=0.0
    )

    # Half-width tables: each entry is shared by one even/odd channel pair.
    rd_half = (rd - nope_dim) // 2
    cos = tl.load(COS + rm[:, None] * stride_cos_seqlen + rd_half[None, :], mask=mask_m & is_rope, other=1.0).to(
        tl.float32
    )
    sin = tl.load(SIN + rm[:, None] * stride_cos_seqlen + rd_half[None, :], mask=mask_m & is_rope, other=0.0).to(
        tl.float32
    )
    if CONJUGATE:
        sin = -sin

    x_cos = x.to(tl.float32) * cos
    x_swap_sin = x_swap.to(tl.float32) * sin
    rotated = tl.where(rd[None, :] % 2 == 0, x_cos - x_swap_sin, x_cos + x_swap_sin)
    out = tl.where(is_rope, rotated.to(x.dtype), x)

    tl.store(OUT + rm[:, None] * stride_out_seqlen + rd[None, :] * stride_out_headdim, out, mask=mask_m & in_head)


def _rotary_launch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, conjugate: bool) -> torch.Tensor:
    """Rotate the trailing ``2 * cos.shape[-1]`` channels of ``x``.

    Args:
        x: ``[B, H, S, D]``. Arbitrary strides — the seq/head axes are passed
            through explicitly, so transposed views need no ``contiguous()``.
        cos, sin: ``[B, S, rope_dim // 2]``.
        conjugate: negate ``sin``, i.e. apply the inverse rotation. Used for the
            backward pass, since RoPE is orthogonal.

    Returns:
        A contiguous tensor with ``x``'s shape and dtype, matching what the
        eager reference's closing ``torch.cat`` produced.
    """
    batch, nheads, seqlen, headdim = x.shape
    nope_dim = headdim - 2 * cos.shape[-1]

    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    # cos/sin are indexed with a flat ``(rd - nope_dim) // 2`` offset, so the
    # trailing axis must be contiguous.
    cos, sin = cos.contiguous(), sin.contiguous()

    BLOCK_M = 4
    grid = (triton.cdiv(seqlen, BLOCK_M), batch, nheads)

    # Otherwise Triton tries to launch from cuda:0 and trips
    # "Pointer argument cannot be accessed from Triton".
    with get_torch_device().device(x.device.index):
        _dsv4_rotary_interleaved_kernel[grid](
            out,
            x,
            cos,
            sin,
            seqlen,
            headdim,
            nope_dim,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            cos.stride(0),
            cos.stride(1),
            triton.next_power_of_2(headdim),
            BLOCK_M,
            conjugate,
        )
    return out


class _ApplyRotaryPosEmbTriton(torch.autograd.Function):
    """RoPE is orthogonal, so the backward is the inverse rotation of the grad.

    Only ``cos``/``sin`` are stashed — unlike the eager op chain, which saves an
    fp32 copy of the rotated slice for each of its two multiplies.
    """

    @staticmethod
    def forward(ctx, x, cos, sin):
        ctx.save_for_backward(cos, sin)
        return _rotary_launch(x, cos, sin, conjugate=False)

    @staticmethod
    def backward(ctx, grad_out):
        cos, sin = ctx.saved_tensors
        return _rotary_launch(grad_out, cos, sin, conjugate=True), None, None


def _is_supported(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int) -> bool:
    if not (unsqueeze_dim == 1 and x.is_cuda and x.ndim == 4 and cos.ndim == 3):
        return False
    if not (cos.shape == sin.shape and cos.dtype == sin.dtype):
        return False
    if cos.device != x.device or sin.device != x.device:
        return False
    if cos.requires_grad or sin.requires_grad:
        return False
    batch, _, seqlen, headdim = x.shape
    rope_dim = 2 * cos.shape[-1]
    # ``nope_dim`` must be even for the ``rd ^ 1`` partner indexing to hold.
    # A zero-width ``rope_dim`` is rejected so the fallback keeps matching eager,
    # which raises on it rather than returning ``x`` unrotated.
    return (
        cos.shape[0] == batch
        and cos.shape[1] == seqlen
        and 0 < rope_dim <= headdim
        and headdim <= _MAX_HEAD_DIM
        and (headdim - rope_dim) % 2 == 0
    )


def apply_rotary_pos_emb_triton(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> torch.Tensor:
    """Drop-in replacement for DeepSeek-V4's ``apply_rotary_pos_emb``.

    Falls back to the upstream eager implementation for any shape/device/layout
    the kernel does not cover, so this stays safe to bind unconditionally.
    """
    if not _is_supported(x, cos, sin, unsqueeze_dim):
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import apply_rotary_pos_emb

        return apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=unsqueeze_dim)

    return _ApplyRotaryPosEmbTriton.apply(x, cos, sin)
