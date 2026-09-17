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

"""Nabla fused RMSNorm (Triton forward + backward).

Ported from the Nabla kernel-optimization project (Apache-2.0), where this
kernel was validated on H100 against liger-kernel v0.7.0 with an fp32 oracle:
output, grad_x and grad_weight all pass bf16 tolerances (atol=7e-2,
rtol=2e-2), and fwd+bwd latency is 1.41-1.48x liger at training shapes
(M in {2048, 4096, 8192}, hidden=2048, bf16) — the win comes from the
single-kernel backward (grad_weight accumulated in-place via ``tl.atomic_add``
into one fp32 buffer) instead of liger's partials + separate reduction pass.

Formulation is the standard one (matches ``LigerRMSNorm`` with offset=0.0,
casting_mode="llama"):

    out = x / sqrt(mean(x^2, dim=-1) + eps) * weight

The Triton path requires a 2-D contiguous CUDA input; N-D inputs are flattened
per call. Anything the Triton path cannot handle (CPU tensors, missing Triton,
non-contiguous weight) transparently falls back to an fp32-accumulation pure
torch implementation.
"""

from __future__ import annotations

import torch


try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def _torch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig_dtype = x.dtype
    x_f = x.float()
    variance = x_f.pow(2).mean(dim=-1, keepdim=True)
    out = x_f * torch.rsqrt(variance + eps) * weight.float()
    return out.to(orig_dtype)


def _num_warps_for(block: int) -> int:
    if block >= 8192:
        return 16
    if block >= 4096:
        return 8
    if block >= 2048:
        return 4
    return 2


if _HAS_TRITON:

    @triton.jit
    def _rmsnorm_fwd_kernel(
        X,
        W,
        Y,
        Rstd,
        stride_row,
        N,
        eps,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(Rstd + row, rstd)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(Y + row * stride_row + cols, y.to(Y.dtype.element_ty), mask=mask)

    @triton.jit
    def _rmsnorm_bwd_kernel(
        DY,
        X,
        W,
        Rstd,
        DX,
        DW,
        stride_row,
        N,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd + row)

        xhat = x * rstd
        wdy = w * dy
        c = tl.sum(xhat * wdy, axis=0) / N
        dx = (wdy - xhat * c) * rstd
        tl.store(DX + row * stride_row + cols, dx.to(DX.dtype.element_ty), mask=mask)
        # grad_weight_j = sum_over_rows(dy_j * xhat_j); fp32 accumulation.
        tl.atomic_add(DW + cols, dy * xhat, mask=mask)


def _rmsnorm_fwd_launch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    y = torch.empty_like(x)
    rstd = torch.empty((m,), dtype=torch.float32, device=x.device)
    block = triton.next_power_of_2(n)
    _rmsnorm_fwd_kernel[(m,)](x, weight, y, rstd, x.stride(0), n, eps, BLOCK=block, num_warps=_num_warps_for(block))
    return y, rstd


def _rmsnorm_bwd_launch(
    dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, rstd: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    dy = dy.contiguous()
    dx = torch.empty_like(x)
    dw_f32 = torch.zeros((n,), dtype=torch.float32, device=x.device)
    block = triton.next_power_of_2(n)
    _rmsnorm_bwd_kernel[(m,)](
        dy, x, weight, rstd, dx, dw_f32, x.stride(0), n, BLOCK=block, num_warps=_num_warps_for(block)
    )
    return dx, dw_f32.to(weight.dtype)


class NablaRMSNormFunction(torch.autograd.Function):
    """Fused RMSNorm forward + backward (Triton)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:  # type: ignore[override]
        y, rstd = _rmsnorm_fwd_launch(x, weight, eps)
        ctx.save_for_backward(x, weight, rstd)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        x, weight, rstd = ctx.saved_tensors
        dx, dw = _rmsnorm_bwd_launch(grad_output, x, weight, rstd)
        return dx, dw, None


def _can_use_triton(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        _HAS_TRITON
        and x.is_cuda
        and weight.is_cuda
        and x.dim() >= 2
        and x.is_contiguous()
        and weight.is_contiguous()
        and weight.numel() == x.shape[-1]
    )


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Functional fused RMSNorm, matching the OpSlot call shape ``(x, weight, eps)``.

    Accepts ``(B, S, H)`` (flattened internally) or ``(BxS, H)`` inputs. Triton
    fwd+bwd when possible, fp32-accumulation torch fallback otherwise.
    """
    if _can_use_triton(x, weight):
        shape = x.shape
        x2d = x.reshape(-1, shape[-1])  # contiguous guaranteed -> always a view
        return NablaRMSNormFunction.apply(x2d, weight, eps).view(*shape)
    return _torch_rms_norm(x, weight, eps)


class NablaRMSNorm(torch.nn.Module):
    """Drop-in RMSNorm module for the ``rms_norm_implementation="nabla"`` backend.

    ABI-compatible with ``LigerRMSNorm`` / HF ``{Model}RMSNorm``: constructed as
    ``NablaRMSNorm(hidden_size, eps=...)`` and called as ``module(hidden_states)``.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)
