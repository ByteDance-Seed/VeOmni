# Copyright 2026 Advanced Micro Devices, Inc. and ByteDance Ltd. and/or its affiliates
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
#
# Portions of this file are adapted from AMD-AGI/Primus under the following license:
#
# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Adapted from AMD-AGI/Primus v26.8.0 (v4_attention_kernels/_triton_common/rmsnorm.py,
# hc_glue.py, sinkhorn.py, hc_collapse.py and hc_expand.py); modified for VeOmni.

# ruff: noqa
"""Triton forward/backward kernels for DeepSeek-V4 mHC.

``unweighted_rms_norm`` feeds the mix GEMM, which stays in ``F.linear``;
``mhc_mix_tail`` turns the GEMM output into ``(pre, post, comb)``;
``sinkhorn_normalize`` projects ``comb``; ``hc_collapse`` and ``hc_post_mix``
fold and refill the residual streams. ``hc_post_mix`` uses the DeepSeek-V4
orientation, where ``comb[..., k, h]`` weights residual stream ``k`` in output
stream ``h``. All arithmetic is FP32 and every row offset is 64-bit.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


SUPPORTED_HC_MULT = (1, 2, 4, 8, 16)

_TORCH_TO_TL_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

_STREAM_BLOCK_M = 16
_STREAM_BLOCK_D = 128
_STREAM_NUM_WARPS = 4


def _check_hc_mult(hc_mult: int) -> None:
    if hc_mult not in SUPPORTED_HC_MULT:
        raise ValueError(f"Triton mHC supports hc_mult in {SUPPORTED_HC_MULT}, got {hc_mult}")


def _leading_rows(shape: torch.Size, trailing: int) -> int:
    rows = 1
    for size in shape[: len(shape) - trailing]:
        rows *= size
    return rows


# ---------------------------------------------------------------------------
# Unweighted RMS norm with an FP32 result
# ---------------------------------------------------------------------------


@triton.jit
def _rms_fwd_kernel(X_PTR, OUT_PTR, RSTD_PTR, D, EPS: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_row = X_PTR + row * D
    acc = tl.zeros((), dtype=tl.float32)
    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        x = tl.load(x_row + cols, mask=cols < D, other=0.0).to(tl.float32)
        acc += tl.sum(x * x, axis=0)
    rstd = 1.0 / tl.sqrt(acc / D + EPS)
    tl.store(RSTD_PTR + row, rstd)
    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(OUT_PTR + row * D + cols, x * rstd, mask=mask)


@triton.jit
def _rms_bwd_kernel(X_PTR, RSTD_PTR, DY_PTR, DX_PTR, D, BLOCK_D: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_row = X_PTR + row * D
    dy_row = DY_PTR + row * D
    rstd = tl.load(RSTD_PTR + row)
    c = tl.zeros((), dtype=tl.float32)
    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_row + cols, mask=mask, other=0.0).to(tl.float32)
        c += tl.sum(dy * x, axis=0)
    coef = rstd * rstd * rstd * c / D
    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_row + cols, mask=mask, other=0.0).to(tl.float32)
        dx = rstd * dy - x * coef
        tl.store(DX_PTR + row * D + cols, dx.to(DX_PTR.dtype.element_ty), mask=mask)


def _rms_block_d(d: int) -> int:
    return min(triton.next_power_of_2(d), 2048)


class _UnweightedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float) -> torch.Tensor:
        d = x.shape[-1]
        x2 = x.reshape(-1, d).contiguous()
        rows = x2.shape[0]
        out = torch.empty((rows, d), dtype=torch.float32, device=x.device)
        rstd = torch.empty((rows,), dtype=torch.float32, device=x.device)
        _rms_fwd_kernel[(rows,)](x2, out, rstd, d, EPS=float(eps), BLOCK_D=_rms_block_d(d))
        ctx.save_for_backward(x2, rstd)
        ctx.x_shape = x.shape
        return out.view(x.shape)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        x2, rstd = ctx.saved_tensors
        d = x2.shape[-1]
        dy = grad.reshape(-1, d).contiguous()
        dx = torch.empty_like(x2)
        _rms_bwd_kernel[(x2.shape[0],)](x2, rstd, dy, dx, d, BLOCK_D=_rms_block_d(d))
        return dx.view(ctx.x_shape), None


def unweighted_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Return ``x * rsqrt(mean(x**2, -1) + eps)`` in FP32."""
    if not x.is_cuda:
        raise ValueError("Triton mHC RMS norm requires a GPU tensor")
    if x.dim() == 0 or x.shape[-1] == 0:
        raise ValueError(f"Triton mHC RMS norm requires a nonzero last dimension, got {tuple(x.shape)}")
    if x.dtype not in _TORCH_TO_TL_DTYPE:
        raise TypeError(f"Triton mHC RMS norm does not support {x.dtype}")
    return _UnweightedRMSNorm.apply(x, eps)


# ---------------------------------------------------------------------------
# Mix tail: (pre, post, comb) from the mix GEMM output
# ---------------------------------------------------------------------------


@triton.jit
def _mix_tail_fwd_kernel(
    MIXES_PTR,
    SCALE_PTR,
    BASE_PTR,
    PRE_PTR,
    POST_PTR,
    COMB_PTR,
    PRE_SIG_PTR,
    POST_SIG_PTR,
    COMB_SM_PTR,
    N,
    EPS: tl.constexpr,
    POST_MULT: tl.constexpr,
    K: tl.constexpr,
    BLOCK_LEADING: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = (pid * BLOCK_LEADING + tl.arange(0, BLOCK_LEADING)).to(tl.int64)
    mask = offs < N
    k_idx = tl.arange(0, K)
    KK: tl.constexpr = K * K
    ROW: tl.constexpr = (2 + K) * K

    scale0 = tl.load(SCALE_PTR + 0)
    scale1 = tl.load(SCALE_PTR + 1)
    scale2 = tl.load(SCALE_PTR + 2)
    base_pre = tl.load(BASE_PTR + k_idx)
    base_post = tl.load(BASE_PTR + K + k_idx)
    base_comb = tl.load(BASE_PTR + 2 * K + k_idx[:, None] * K + k_idx[None, :])

    pre_logit = tl.load(MIXES_PTR + offs[:, None] * ROW + k_idx[None, :], mask=mask[:, None], other=0.0)
    post_logit = tl.load(MIXES_PTR + offs[:, None] * ROW + K + k_idx[None, :], mask=mask[:, None], other=0.0)
    comb_logit = tl.load(
        MIXES_PTR + offs[:, None, None] * ROW + 2 * K + k_idx[None, :, None] * K + k_idx[None, None, :],
        mask=mask[:, None, None],
        other=0.0,
    )

    pre_sig = tl.sigmoid(pre_logit * scale0 + base_pre[None, :])
    post_sig = tl.sigmoid(post_logit * scale1 + base_post[None, :])
    comb_logit = comb_logit * scale2 + base_comb[None, :, :]
    comb_exp = tl.exp(comb_logit - tl.max(comb_logit, axis=2, keep_dims=True))
    comb_sm = comb_exp / tl.sum(comb_exp, axis=2, keep_dims=True)

    row_k = offs[:, None] * K + k_idx[None, :]
    row_kk = offs[:, None, None] * KK + k_idx[None, :, None] * K + k_idx[None, None, :]
    tl.store(PRE_SIG_PTR + row_k, pre_sig, mask=mask[:, None])
    tl.store(POST_SIG_PTR + row_k, post_sig, mask=mask[:, None])
    tl.store(COMB_SM_PTR + row_kk, comb_sm, mask=mask[:, None, None])
    tl.store(PRE_PTR + row_k, pre_sig + EPS, mask=mask[:, None])
    tl.store(POST_PTR + row_k, POST_MULT * post_sig, mask=mask[:, None])
    tl.store(COMB_PTR + row_kk, comb_sm + EPS, mask=mask[:, None, None])


@triton.jit
def _mix_tail_bwd_kernel(
    DPRE_PTR,
    DPOST_PTR,
    DCOMB_PTR,
    PRE_SIG_PTR,
    POST_SIG_PTR,
    COMB_SM_PTR,
    SCALE_PTR,
    DMIXES_PTR,
    DLOGIT_PTR,
    N,
    POST_MULT: tl.constexpr,
    K: tl.constexpr,
    BLOCK_LEADING: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = (pid * BLOCK_LEADING + tl.arange(0, BLOCK_LEADING)).to(tl.int64)
    mask = offs < N
    k_idx = tl.arange(0, K)
    KK: tl.constexpr = K * K
    ROW: tl.constexpr = (2 + K) * K

    scale0 = tl.load(SCALE_PTR + 0)
    scale1 = tl.load(SCALE_PTR + 1)
    scale2 = tl.load(SCALE_PTR + 2)
    row_k = offs[:, None] * K + k_idx[None, :]
    row_kk = offs[:, None, None] * KK + k_idx[None, :, None] * K + k_idx[None, None, :]
    pre_sig = tl.load(PRE_SIG_PTR + row_k, mask=mask[:, None], other=0.0)
    post_sig = tl.load(POST_SIG_PTR + row_k, mask=mask[:, None], other=0.0)
    comb_sm = tl.load(COMB_SM_PTR + row_kk, mask=mask[:, None, None], other=0.0)
    d_pre = tl.load(DPRE_PTR + row_k, mask=mask[:, None], other=0.0).to(tl.float32)
    d_post = tl.load(DPOST_PTR + row_k, mask=mask[:, None], other=0.0).to(tl.float32)
    d_comb = tl.load(DCOMB_PTR + row_kk, mask=mask[:, None, None], other=0.0).to(tl.float32)

    d_pre_logit = d_pre * pre_sig * (1.0 - pre_sig)
    d_post_logit = d_post * POST_MULT * post_sig * (1.0 - post_sig)
    d_comb_logit = comb_sm * (d_comb - tl.sum(d_comb * comb_sm, axis=2, keep_dims=True))

    mix_k = offs[:, None] * ROW + k_idx[None, :]
    mix_kk = offs[:, None, None] * ROW + 2 * K + k_idx[None, :, None] * K + k_idx[None, None, :]
    tl.store(DMIXES_PTR + mix_k, d_pre_logit * scale0, mask=mask[:, None])
    tl.store(DMIXES_PTR + mix_k + K, d_post_logit * scale1, mask=mask[:, None])
    tl.store(DMIXES_PTR + mix_kk, d_comb_logit * scale2, mask=mask[:, None, None])
    tl.store(DLOGIT_PTR + mix_k, d_pre_logit, mask=mask[:, None])
    tl.store(DLOGIT_PTR + mix_k + K, d_post_logit, mask=mask[:, None])
    tl.store(DLOGIT_PTR + mix_kk, d_comb_logit, mask=mask[:, None, None])


def _mix_tail_block(rows: int, k: int) -> int:
    cap = 64 if k <= 4 else (32 if k <= 8 else 8)
    return cap if rows >= cap else max(1, triton.next_power_of_2(rows))


class _MixTail(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mixes, scale, base, hc_mult: int, post_mult: float, eps: float):
        k = hc_mult
        mixes_c = mixes.contiguous().to(torch.float32)
        scale_c = scale.contiguous().to(torch.float32)
        base_c = base.contiguous().to(torch.float32)
        leading = mixes_c.shape[:-1]
        rows = _leading_rows(mixes_c.shape, 1)
        device = mixes_c.device
        pre = torch.empty((*leading, k), dtype=torch.float32, device=device)
        post = torch.empty((*leading, k), dtype=torch.float32, device=device)
        comb = torch.empty((*leading, k, k), dtype=torch.float32, device=device)
        pre_sig = torch.empty((rows, k), dtype=torch.float32, device=device)
        post_sig = torch.empty((rows, k), dtype=torch.float32, device=device)
        comb_sm = torch.empty((rows, k, k), dtype=torch.float32, device=device)
        block = _mix_tail_block(rows, k)
        _mix_tail_fwd_kernel[(triton.cdiv(rows, block),)](
            mixes_c,
            scale_c,
            base_c,
            pre,
            post,
            comb,
            pre_sig,
            post_sig,
            comb_sm,
            rows,
            EPS=float(eps),
            POST_MULT=float(post_mult),
            K=k,
            BLOCK_LEADING=block,
        )
        ctx.save_for_backward(mixes_c, scale_c, pre_sig, post_sig, comb_sm)
        ctx.hc_mult = k
        ctx.post_mult = float(post_mult)
        ctx.leading = tuple(leading)
        ctx.dtypes = (mixes.dtype, scale.dtype, base.dtype)
        return pre, post, comb

    @staticmethod
    def backward(ctx, d_pre, d_post, d_comb):
        mixes_c, scale_c, pre_sig, post_sig, comb_sm = ctx.saved_tensors
        k = ctx.hc_mult
        rows = pre_sig.shape[0]
        device = mixes_c.device
        d_pre = torch.zeros_like(pre_sig) if d_pre is None else d_pre.contiguous()
        d_post = torch.zeros_like(post_sig) if d_post is None else d_post.contiguous()
        d_comb = torch.zeros_like(comb_sm) if d_comb is None else d_comb.contiguous()
        d_mixes = torch.empty_like(mixes_c)
        d_logit = torch.empty((rows, (2 + k) * k), dtype=torch.float32, device=device)
        block = _mix_tail_block(rows, k)
        _mix_tail_bwd_kernel[(triton.cdiv(rows, block),)](
            d_pre,
            d_post,
            d_comb,
            pre_sig,
            post_sig,
            comb_sm,
            scale_c,
            d_mixes,
            d_logit,
            rows,
            POST_MULT=ctx.post_mult,
            K=k,
            BLOCK_LEADING=block,
        )
        mixes_flat = mixes_c.reshape(rows, (2 + k) * k)
        d_scale = torch.stack(
            [
                (mixes_flat[:, :k] * d_logit[:, :k]).sum(),
                (mixes_flat[:, k : 2 * k] * d_logit[:, k : 2 * k]).sum(),
                (mixes_flat[:, 2 * k :] * d_logit[:, 2 * k :]).sum(),
            ]
        )
        d_base = d_logit.sum(dim=0)
        mixes_dtype, scale_dtype, base_dtype = ctx.dtypes
        return (
            d_mixes.view(*ctx.leading, (2 + k) * k).to(mixes_dtype),
            d_scale.to(scale_dtype),
            d_base.to(base_dtype),
            None,
            None,
            None,
        )


def mhc_mix_tail(
    mixes: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_mult: int,
    post_mult: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return FP32 ``(sigmoid + eps, post_mult * sigmoid, softmax + eps)`` mixes.

    ``mixes`` is ``[..., (2 + hc) * hc]``; ``scale`` holds the pre / post /
    comb scales and ``base`` the matching biases.
    """
    _check_hc_mult(hc_mult)
    channels = (2 + hc_mult) * hc_mult
    if mixes.dim() == 0 or mixes.shape[-1] != channels:
        raise ValueError(f"mHC mixes must have {channels} channels")
    if scale.shape != (3,):
        raise ValueError(f"mHC pre expects scale shape (3,), got {tuple(scale.shape)}")
    if base.shape != (channels,):
        raise ValueError(f"mHC pre expects base shape ({channels},), got {tuple(base.shape)}")
    if not mixes.is_cuda or scale.device != mixes.device or base.device != mixes.device:
        raise ValueError("Triton mHC mixes, scale, and base must be GPU tensors on the same device")
    if any(tensor.dtype not in _TORCH_TO_TL_DTYPE for tensor in (mixes, scale, base)):
        raise TypeError("Triton mHC mixes, scale, and base must use supported floating dtypes")
    return _MixTail.apply(mixes, scale, base, hc_mult, post_mult, eps)


# ---------------------------------------------------------------------------
# Sinkhorn-Knopp normalization
# ---------------------------------------------------------------------------


@triton.jit
def _sinkhorn_fwd_kernel(
    X_PTR,
    Y_PTR,
    STATES_PTR,
    N,
    K: tl.constexpr,
    N_ITERS: tl.constexpr,
    EPS: tl.constexpr,
    SAVE_STATES: tl.constexpr,
    BLOCK_LEADING: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = (pid * BLOCK_LEADING + tl.arange(0, BLOCK_LEADING)).to(tl.int64)
    mask = offs[:, None, None] < N
    r_offs = tl.arange(0, K)
    c_offs = tl.arange(0, K)
    KK: tl.constexpr = K * K
    STATE_STRIDE: tl.constexpr = 2 * N_ITERS * KK
    base = offs[:, None, None] * KK + r_offs[None, :, None] * K + c_offs[None, None, :]
    state_base = offs[:, None, None] * STATE_STRIDE + r_offs[None, :, None] * K + c_offs[None, None, :]

    m = tl.load(X_PTR + base, mask=mask, other=0.0).to(tl.float32)
    if SAVE_STATES:
        tl.store(STATES_PTR + state_base, m, mask=mask)
    m = m / (tl.sum(m, axis=1, keep_dims=True) + EPS)
    if SAVE_STATES:
        tl.store(STATES_PTR + state_base + KK, m, mask=mask)
    for it in tl.static_range(N_ITERS - 1):
        m = m / (tl.sum(m, axis=2, keep_dims=True) + EPS)
        if SAVE_STATES:
            tl.store(STATES_PTR + state_base + (2 + 2 * it) * KK, m, mask=mask)
        m = m / (tl.sum(m, axis=1, keep_dims=True) + EPS)
        if SAVE_STATES:
            tl.store(STATES_PTR + state_base + (3 + 2 * it) * KK, m, mask=mask)
    tl.store(Y_PTR + base, m.to(Y_PTR.dtype.element_ty), mask=mask)


@triton.jit
def _sinkhorn_bwd_kernel(
    STATES_PTR,
    DY_PTR,
    DX_PTR,
    N,
    K: tl.constexpr,
    N_ITERS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_LEADING: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = (pid * BLOCK_LEADING + tl.arange(0, BLOCK_LEADING)).to(tl.int64)
    mask = offs[:, None, None] < N
    r_offs = tl.arange(0, K)
    c_offs = tl.arange(0, K)
    KK: tl.constexpr = K * K
    STATE_STRIDE: tl.constexpr = 2 * N_ITERS * KK
    base = offs[:, None, None] * KK + r_offs[None, :, None] * K + c_offs[None, None, :]
    state_base = offs[:, None, None] * STATE_STRIDE + r_offs[None, :, None] * K + c_offs[None, None, :]

    # Step s maps state s-1 to state s; odd steps normalize columns (axis 1),
    # even steps rows (axis 2). Walk them back with dx = (dy - sum(dy * y)) / s.
    dm = tl.load(DY_PTR + base, mask=mask, other=0.0).to(tl.float32)
    for i in range(N_ITERS - 1):
        col_step = 2 * N_ITERS - 1 - 2 * i
        before = tl.load(STATES_PTR + state_base + (col_step - 1) * KK, mask=mask, other=0.0)
        after = tl.load(STATES_PTR + state_base + col_step * KK, mask=mask, other=0.0)
        dm = (dm - tl.sum(dm * after, axis=1, keep_dims=True)) / (tl.sum(before, axis=1, keep_dims=True) + EPS)
        row_step = col_step - 1
        before = tl.load(STATES_PTR + state_base + (row_step - 1) * KK, mask=mask, other=0.0)
        after = tl.load(STATES_PTR + state_base + row_step * KK, mask=mask, other=0.0)
        dm = (dm - tl.sum(dm * after, axis=2, keep_dims=True)) / (tl.sum(before, axis=2, keep_dims=True) + EPS)
    before = tl.load(STATES_PTR + state_base, mask=mask, other=0.0)
    after = tl.load(STATES_PTR + state_base + KK, mask=mask, other=0.0)
    dm = (dm - tl.sum(dm * after, axis=1, keep_dims=True)) / (tl.sum(before, axis=1, keep_dims=True) + EPS)
    tl.store(DX_PTR + base, dm.to(DX_PTR.dtype.element_ty), mask=mask)


def _sinkhorn_block(rows: int, k: int) -> int:
    cap = 128 if k <= 4 else (32 if k <= 8 else 8)
    return cap if rows >= cap else max(1, triton.next_power_of_2(rows))


class _Sinkhorn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
        k = x.shape[-1]
        x_c = x.contiguous()
        rows = _leading_rows(x_c.shape, 2)
        save_states = bool(ctx.needs_input_grad[0])
        out = torch.empty_like(x_c)
        states = torch.empty(
            (rows, 2 * iters, k, k) if save_states else (1,),
            dtype=torch.float32,
            device=x_c.device,
        )
        block = _sinkhorn_block(rows, k)
        _sinkhorn_fwd_kernel[(triton.cdiv(rows, block),)](
            x_c,
            out,
            states,
            rows,
            K=k,
            N_ITERS=iters,
            EPS=float(eps),
            SAVE_STATES=save_states,
            BLOCK_LEADING=block,
        )
        ctx.save_for_backward(states)
        ctx.iters = iters
        ctx.eps = float(eps)
        ctx.rows = rows
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (states,) = ctx.saved_tensors
        k = grad.shape[-1]
        dy = grad.contiguous()
        dx = torch.empty_like(dy)
        block = _sinkhorn_block(ctx.rows, k)
        _sinkhorn_bwd_kernel[(triton.cdiv(ctx.rows, block),)](
            states,
            dy,
            dx,
            ctx.rows,
            K=k,
            N_ITERS=ctx.iters,
            EPS=ctx.eps,
            BLOCK_LEADING=block,
        )
        return dx, None, None


def sinkhorn_normalize(comb: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    """Column-normalize ``comb``, then alternate row / column normalization ``iters - 1`` times."""
    if comb.dim() < 2 or comb.shape[-2] != comb.shape[-1]:
        raise ValueError(f"Sinkhorn expects a [..., K, K] tensor, got {tuple(comb.shape)}")
    if not comb.is_cuda:
        raise ValueError("Triton mHC Sinkhorn requires a GPU tensor")
    if comb.dtype not in _TORCH_TO_TL_DTYPE:
        raise TypeError(f"Triton mHC Sinkhorn does not support {comb.dtype}")
    k = comb.shape[-1]
    _check_hc_mult(k)
    # The reference always applies the leading column normalization, so zero
    # iterations behaves like one.
    return _Sinkhorn.apply(comb, max(int(iters), 1), eps)


# ---------------------------------------------------------------------------
# Stream collapse: out[..., d] = sum_k pre[..., k] * x[..., k, d]
# ---------------------------------------------------------------------------


@triton.jit
def _collapse_fwd_kernel(
    X_PTR,
    PRE_PTR,
    OUT_PTR,
    M,
    D,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    rm = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rd = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    k_ax = tl.arange(0, K)
    mask_m = rm < M
    mask_d = rd < D
    x3 = tl.load(
        X_PTR + rm[:, None, None] * (K * D) + k_ax[None, :, None] * D + rd[None, None, :],
        mask=mask_m[:, None, None] & mask_d[None, None, :],
        other=0.0,
    ).to(tl.float32)
    pre = tl.load(PRE_PTR + rm[:, None] * K + k_ax[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    out = tl.sum(pre[:, :, None] * x3, axis=1)
    tl.store(
        OUT_PTR + rm[:, None] * D + rd[None, :],
        out.to(OUT_PTR.dtype.element_ty),
        mask=mask_m[:, None] & mask_d[None, :],
    )


@triton.jit
def _collapse_bwd_kernel(
    G_PTR,
    X_PTR,
    PRE_PTR,
    DX_PTR,
    DPRE_PARTIAL_PTR,
    M,
    D,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(1)
    rm = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rd = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    k_ax = tl.arange(0, K)
    mask_m = rm < M
    mask_d = rd < D
    mask3 = mask_m[:, None, None] & mask_d[None, None, :]
    stream = rm[:, None, None] * (K * D) + k_ax[None, :, None] * D + rd[None, None, :]
    x3 = tl.load(X_PTR + stream, mask=mask3, other=0.0).to(tl.float32)
    g = tl.load(G_PTR + rm[:, None] * D + rd[None, :], mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(
        tl.float32
    )
    pre = tl.load(PRE_PTR + rm[:, None] * K + k_ax[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    tl.store(DX_PTR + stream, (pre[:, :, None] * g[:, None, :]).to(DX_PTR.dtype.element_ty), mask=mask3)
    tl.store(
        DPRE_PARTIAL_PTR + pid_d.to(tl.int64) * M * K + rm[:, None] * K + k_ax[None, :],
        tl.sum(x3 * g[:, None, :], axis=2),
        mask=mask_m[:, None],
    )


class _Collapse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        k, d = x.shape[-2], x.shape[-1]
        x_c = x.contiguous()
        pre_c = pre.contiguous()
        rows = _leading_rows(x_c.shape, 2)
        out = torch.empty((*x_c.shape[:-2], d), dtype=x.dtype, device=x.device)
        grid = (triton.cdiv(rows, _STREAM_BLOCK_M), triton.cdiv(d, _STREAM_BLOCK_D))
        _collapse_fwd_kernel[grid](
            x_c,
            pre_c,
            out,
            rows,
            d,
            K=k,
            BLOCK_M=_STREAM_BLOCK_M,
            BLOCK_D=_STREAM_BLOCK_D,
            num_warps=_STREAM_NUM_WARPS,
        )
        ctx.save_for_backward(x_c, pre_c)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        x_c, pre_c = ctx.saved_tensors
        k, d = x_c.shape[-2], x_c.shape[-1]
        rows = _leading_rows(x_c.shape, 2)
        num_d = triton.cdiv(d, _STREAM_BLOCK_D)
        dx = torch.empty_like(x_c)
        dpre_partial = torch.empty((num_d, rows, k), dtype=torch.float32, device=x_c.device)
        _collapse_bwd_kernel[(triton.cdiv(rows, _STREAM_BLOCK_M), num_d)](
            grad.contiguous(),
            x_c,
            pre_c,
            dx,
            dpre_partial,
            rows,
            d,
            K=k,
            BLOCK_M=_STREAM_BLOCK_M,
            BLOCK_D=_STREAM_BLOCK_D,
            num_warps=_STREAM_NUM_WARPS,
        )
        return dx, dpre_partial.sum(dim=0).view(pre_c.shape).to(pre_c.dtype)


def hc_collapse(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """Return ``(pre.unsqueeze(-1) * x).sum(-2)`` in ``x.dtype``, accumulating in FP32.

    ``x`` is ``[..., K, D]`` and ``pre`` is ``[..., K]``; ``pre`` may be FP32
    while ``x`` is BF16.
    """
    if x.dim() < 2 or x.shape[-1] == 0:
        raise ValueError(f"mHC collapse expects [..., K, D] with nonzero D, got {tuple(x.shape)}")
    _check_hc_mult(x.shape[-2])
    if pre.shape != x.shape[:-1]:
        raise ValueError(f"mHC collapse weights {tuple(pre.shape)} do not match streams {tuple(x.shape)}")
    if not x.is_cuda or pre.device != x.device:
        raise ValueError("Triton mHC collapse streams and weights must be GPU tensors on the same device")
    if x.dtype not in _TORCH_TO_TL_DTYPE or pre.dtype not in _TORCH_TO_TL_DTYPE:
        raise TypeError(f"mHC collapse expects floating streams and weights, got {x.dtype} and {pre.dtype}")
    return _Collapse.apply(x, pre)


# ---------------------------------------------------------------------------
# Residual post-mix: new[..., h, d] = post[..., h] * out[..., d] + sum_k comb[..., k, h] * x[..., k, d]
# ---------------------------------------------------------------------------


@triton.jit
def _post_mix_fwd_kernel(
    X_PTR,
    OUT_PTR,
    POST_PTR,
    COMB_PTR,
    NEW_PTR,
    M,
    D,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    rm = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rd = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    k_ax = tl.arange(0, K)
    mask_m = rm < M
    mask_d = rd < D
    mask2 = mask_m[:, None] & mask_d[None, :]
    x3 = tl.load(
        X_PTR + rm[:, None, None] * (K * D) + k_ax[None, :, None] * D + rd[None, None, :],
        mask=mask_m[:, None, None] & mask_d[None, None, :],
        other=0.0,
    ).to(tl.float32)
    out_tile = tl.load(OUT_PTR + rm[:, None] * D + rd[None, :], mask=mask2, other=0.0).to(tl.float32)
    for h in range(K):
        comb_col = tl.load(
            COMB_PTR + rm[:, None] * (K * K) + k_ax[None, :] * K + h, mask=mask_m[:, None], other=0.0
        ).to(tl.float32)
        post_h = tl.load(POST_PTR + rm * K + h, mask=mask_m, other=0.0).to(tl.float32)
        new_h = post_h[:, None] * out_tile + tl.sum(comb_col[:, :, None] * x3, axis=1)
        tl.store(NEW_PTR + rm[:, None] * (K * D) + h * D + rd[None, :], new_h.to(NEW_PTR.dtype.element_ty), mask=mask2)


@triton.jit
def _post_mix_bwd_kernel(
    G_PTR,
    X_PTR,
    OUT_PTR,
    POST_PTR,
    COMB_PTR,
    DX_PTR,
    DOUT_PTR,
    DPOST_PARTIAL_PTR,
    DCOMB_PARTIAL_PTR,
    M,
    D,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(1)
    rm = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rd = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    k_ax = tl.arange(0, K)
    mask_m = rm < M
    mask_d = rd < D
    mask2 = mask_m[:, None] & mask_d[None, :]
    mask3 = mask_m[:, None, None] & mask_d[None, None, :]
    stream = rm[:, None, None] * (K * D) + k_ax[None, :, None] * D + rd[None, None, :]
    x3 = tl.load(X_PTR + stream, mask=mask3, other=0.0).to(tl.float32)
    g3 = tl.load(G_PTR + stream, mask=mask3, other=0.0).to(tl.float32)
    out_tile = tl.load(OUT_PTR + rm[:, None] * D + rd[None, :], mask=mask2, other=0.0).to(tl.float32)
    post = tl.load(POST_PTR + rm[:, None] * K + k_ax[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)

    # d_out[d] = sum_h post[h] * g[h, d]; d_post[h] = sum_d out[d] * g[h, d]
    tl.store(
        DOUT_PTR + rm[:, None] * D + rd[None, :],
        tl.sum(post[:, :, None] * g3, axis=1).to(DOUT_PTR.dtype.element_ty),
        mask=mask2,
    )
    partial_k = pid_d.to(tl.int64) * M * K + rm[:, None] * K + k_ax[None, :]
    tl.store(DPOST_PARTIAL_PTR + partial_k, tl.sum(out_tile[:, None, :] * g3, axis=2), mask=mask_m[:, None])

    # d_x[k, d] = sum_h comb[k, h] * g[h, d]
    for k in range(K):
        comb_row = tl.load(
            COMB_PTR + rm[:, None] * (K * K) + k * K + k_ax[None, :], mask=mask_m[:, None], other=0.0
        ).to(tl.float32)
        tl.store(
            DX_PTR + rm[:, None] * (K * D) + k * D + rd[None, :],
            tl.sum(comb_row[:, :, None] * g3, axis=1).to(DX_PTR.dtype.element_ty),
            mask=mask2,
        )

    # d_comb[k, h] = sum_d x[k, d] * g[h, d]
    partial_kk = pid_d.to(tl.int64) * M * (K * K) + rm[:, None] * (K * K)
    for h in range(K):
        g_h = tl.load(G_PTR + rm[:, None] * (K * D) + h * D + rd[None, :], mask=mask2, other=0.0).to(tl.float32)
        tl.store(
            DCOMB_PARTIAL_PTR + partial_kk + k_ax[None, :] * K + h,
            tl.sum(x3 * g_h[:, None, :], axis=2),
            mask=mask_m[:, None],
        )


class _PostMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, out, post, comb):
        k, d = x.shape[-2], x.shape[-1]
        x_c = x.contiguous()
        out_c = out.contiguous()
        post_c = post.contiguous()
        comb_c = comb.contiguous()
        rows = _leading_rows(x_c.shape, 2)
        new = torch.empty_like(x_c)
        grid = (triton.cdiv(rows, _STREAM_BLOCK_M), triton.cdiv(d, _STREAM_BLOCK_D))
        _post_mix_fwd_kernel[grid](
            x_c,
            out_c,
            post_c,
            comb_c,
            new,
            rows,
            d,
            K=k,
            BLOCK_M=_STREAM_BLOCK_M,
            BLOCK_D=_STREAM_BLOCK_D,
            num_warps=_STREAM_NUM_WARPS,
        )
        ctx.save_for_backward(x_c, out_c, post_c, comb_c)
        return new

    @staticmethod
    def backward(ctx, grad):
        x_c, out_c, post_c, comb_c = ctx.saved_tensors
        k, d = x_c.shape[-2], x_c.shape[-1]
        rows = _leading_rows(x_c.shape, 2)
        num_d = triton.cdiv(d, _STREAM_BLOCK_D)
        device = x_c.device
        dx = torch.empty_like(x_c)
        dout = torch.empty_like(out_c)
        dpost_partial = torch.empty((num_d, rows, k), dtype=torch.float32, device=device)
        dcomb_partial = torch.empty((num_d, rows, k, k), dtype=torch.float32, device=device)
        _post_mix_bwd_kernel[(triton.cdiv(rows, _STREAM_BLOCK_M), num_d)](
            grad.contiguous(),
            x_c,
            out_c,
            post_c,
            comb_c,
            dx,
            dout,
            dpost_partial,
            dcomb_partial,
            rows,
            d,
            K=k,
            BLOCK_M=_STREAM_BLOCK_M,
            BLOCK_D=_STREAM_BLOCK_D,
            num_warps=_STREAM_NUM_WARPS,
        )
        d_post = dpost_partial.sum(dim=0).view(post_c.shape).to(post_c.dtype)
        d_comb = dcomb_partial.sum(dim=0).view(comb_c.shape).to(comb_c.dtype)
        return dx, dout, d_post, d_comb


def hc_post_mix(residual: torch.Tensor, output: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """Return ``post.unsqueeze(-1) * output.unsqueeze(-2) + comb.transpose(-1, -2) @ residual``.

    ``residual`` is ``[..., K, D]``, ``output`` ``[..., D]``, ``post``
    ``[..., K]`` and ``comb`` ``[..., K, K]``. Accumulates in FP32 and returns
    ``residual.dtype``.
    """
    if residual.dim() < 2 or residual.shape[-1] == 0:
        raise ValueError(f"mHC post expects residual [..., K, D] with nonzero D, got {tuple(residual.shape)}")
    k = residual.shape[-2]
    _check_hc_mult(k)
    if output.shape != residual.shape[:-2] + residual.shape[-1:]:
        raise ValueError(f"mHC post output {tuple(output.shape)} does not match residual {tuple(residual.shape)}")
    if post.shape != residual.shape[:-1] or comb.shape != residual.shape[:-1] + (k,):
        raise ValueError(
            f"mHC post mixes {tuple(post.shape)} / {tuple(comb.shape)} do not match residual {tuple(residual.shape)}"
        )
    if any(tensor.device != residual.device for tensor in (output, post, comb)) or not residual.is_cuda:
        raise ValueError("Triton mHC post tensors must be GPU tensors on the same device")
    if output.dtype != residual.dtype:
        raise TypeError(f"mHC post output and residual must share a dtype, got {output.dtype} and {residual.dtype}")
    if any(tensor.dtype not in _TORCH_TO_TL_DTYPE for tensor in (residual, output, post, comb)):
        raise TypeError("Triton mHC post tensors must use supported floating dtypes")
    return _PostMix.apply(residual, output, post, comb)
