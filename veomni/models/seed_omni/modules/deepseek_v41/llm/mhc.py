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

"""Manifold-constrained Hyper-Connections, wired the way V4.1 pipelines them.

The residual stream is ``hc_mult`` parallel copies. Each sublayer sits between a
*collapse* (the copies weighted down into one input) and an *expand* (the output
placed back across the copies, with the incoming stream mixed in through a
doubly-stochastic ``comb``). The three coefficient sets -- ``pre``, ``post``,
``comb`` -- all come out of one projection of the flattened stream, with ``comb``
Sinkhorn-projected onto the doubly-stochastic manifold.

The core arithmetic is identical to ``DeepseekV4HyperConnection``'s. The wiring
is not, and that is what this file exists for: **V4.1 uses each site's ``pre`` at
the *next* site**, not its own. Attention collapses with the previous block's
FFN ``pre``, the FFN collapses with this block's attention ``pre``, and the final
stream collapse uses the last block's FFN ``pre``.

That one-site pipeline is why neither V4 vehicle can be reused:

* ``DeepseekV4HyperConnection.forward`` returns ``(post, comb, collapsed)``. Its
  ``pre`` is consumed internally to build ``collapsed`` and never leaves, so
  there is nothing to hand to the next site.
* ``veomni.ops.kernels.mhc``'s ``mhc.pre`` variant makes the same assumption one
  level down -- ``mhc_pre_tile_kernels`` ends in
  ``mhc_pre_apply_mix(hidden_streams, pre)`` with the ``pre`` it just computed.

So the mixes and the collapse are split into two entry points here. The *expand*
has no such problem and goes through the existing ``mhc.post`` kernel, which is
also the heaviest of the three: it is the only one writing a full
``[B, S, hc_mult, D]`` tensor.

Splitting ``mhc.pre`` into registry variants of its own -- a ``mixes`` and a
``collapse`` over the same TileKernels sub-ops the fused path already calls --
would recover the rest of the kernel speedup. It is left out deliberately: it
touches shared ops, so it belongs to a change that can measure the difference
rather than to the one that first makes the model run.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ......ops import VeomniOp
from ......utils.device import IS_CUDA_AVAILABLE, get_torch_device
from ......utils.import_utils import is_package_available
from .configuration import DeepseekV41LlmConfig
from .reuse import DeepseekV4UnweightedRMSNorm, resolve_module_op_impl


_HAS_TRITON = is_package_available("triton")
if _HAS_TRITON:
    import triton
    import triton.language as tl


#: ``post`` is a placement weight in ``[0, 2]``, so the sigmoid is doubled. Named
#: because the value is shared with the kernel path, which takes it as an argument.
POST_MULTIPLIER = 2.0


if _HAS_TRITON:

    @triton.jit
    def _triton_mhc_mix_sinkhorn_kernel(
        Mixes_ptr,
        Base_ptr,
        Scale_ptr,
        Pre_ptr,
        Post_ptr,
        Comb_ptr,
        HC: tl.constexpr,
        SINKHORN_ITERS: tl.constexpr,
        HC_EPS: tl.constexpr,
        POST_MULT: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        MIX_DIM: tl.constexpr = (2 + HC) * HC
        row_base = Mixes_ptr + row * MIX_DIM

        pre_scale = tl.load(Scale_ptr + 0).to(tl.float32)
        post_scale = tl.load(Scale_ptr + 1).to(tl.float32)
        comb_scale = tl.load(Scale_ptr + 2).to(tl.float32)

        offs_hc = tl.arange(0, HC)
        pre_w = tl.load(row_base + offs_hc).to(tl.float32)
        pre_b = tl.load(Base_ptr + offs_hc).to(tl.float32)
        pre = tl.sigmoid(pre_w * pre_scale + pre_b) + HC_EPS
        tl.store(Pre_ptr + row * HC + offs_hc, pre)

        post_w = tl.load(row_base + HC + offs_hc).to(tl.float32)
        post_b = tl.load(Base_ptr + HC + offs_hc).to(tl.float32)
        post = POST_MULT * tl.sigmoid(post_w * post_scale + post_b)
        tl.store(Post_ptr + row * HC + offs_hc, post)

        r = tl.arange(0, HC)[:, None]
        c = tl.arange(0, HC)[None, :]
        comb_offs = 2 * HC + r * HC + c
        comb_w = tl.load(row_base + comb_offs).to(tl.float32)
        comb_b = tl.load(Base_ptr + comb_offs).to(tl.float32)
        logits = comb_w * comb_scale + comb_b

        row_max = tl.max(logits, axis=1, keep_dims=True)
        exp_val = tl.exp(logits - row_max)
        comb = exp_val / tl.sum(exp_val, axis=1, keep_dims=True) + HC_EPS
        comb = comb / (tl.sum(comb, axis=0, keep_dims=True) + HC_EPS)
        for _ in range(SINKHORN_ITERS - 1):
            comb = comb / (tl.sum(comb, axis=1, keep_dims=True) + HC_EPS)
            comb = comb / (tl.sum(comb, axis=0, keep_dims=True) + HC_EPS)

        tl.store(Comb_ptr + row * (HC * HC) + r * HC + c, comb)

    @triton.jit
    def _triton_mhc_collapse4_kernel(
        Streams_ptr,
        Pre_ptr,
        Out_ptr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        pid_d = tl.program_id(1).to(tl.int64)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        p0 = tl.load(Pre_ptr + row * 4 + 0).to(tl.float32)
        p1 = tl.load(Pre_ptr + row * 4 + 1).to(tl.float32)
        p2 = tl.load(Pre_ptr + row * 4 + 2).to(tl.float32)
        p3 = tl.load(Pre_ptr + row * 4 + 3).to(tl.float32)

        base = Streams_ptr + row * (4 * D) + offs_d
        s0 = tl.load(base + 0 * D, mask=mask_d, other=0.0).to(tl.float32)
        s1 = tl.load(base + 1 * D, mask=mask_d, other=0.0).to(tl.float32)
        s2 = tl.load(base + 2 * D, mask=mask_d, other=0.0).to(tl.float32)
        s3 = tl.load(base + 3 * D, mask=mask_d, other=0.0).to(tl.float32)

        acc = ((p0 * s0 + p1 * s1) + p2 * s2) + p3 * s3
        tl.store(Out_ptr + row * D + offs_d, acc.to(Out_ptr.dtype.element_ty), mask=mask_d)

    @triton.jit
    def _triton_mhc_collapse4_bwd_kernel(
        GradOut_ptr,
        Streams_ptr,
        Pre_ptr,
        GradStreams_ptr,
        GradPre_ptr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        pid_d = tl.program_id(1).to(tl.int64)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        g = tl.load(GradOut_ptr + row * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        p0 = tl.load(Pre_ptr + row * 4 + 0).to(tl.float32)
        p1 = tl.load(Pre_ptr + row * 4 + 1).to(tl.float32)
        p2 = tl.load(Pre_ptr + row * 4 + 2).to(tl.float32)
        p3 = tl.load(Pre_ptr + row * 4 + 3).to(tl.float32)

        base = Streams_ptr + row * (4 * D) + offs_d
        s0 = tl.load(base + 0 * D, mask=mask_d, other=0.0).to(tl.float32)
        s1 = tl.load(base + 1 * D, mask=mask_d, other=0.0).to(tl.float32)
        s2 = tl.load(base + 2 * D, mask=mask_d, other=0.0).to(tl.float32)
        s3 = tl.load(base + 3 * D, mask=mask_d, other=0.0).to(tl.float32)

        g_base = GradStreams_ptr + row * (4 * D) + offs_d
        tl.store(g_base + 0 * D, (g * p0).to(GradStreams_ptr.dtype.element_ty), mask=mask_d)
        tl.store(g_base + 1 * D, (g * p1).to(GradStreams_ptr.dtype.element_ty), mask=mask_d)
        tl.store(g_base + 2 * D, (g * p2).to(GradStreams_ptr.dtype.element_ty), mask=mask_d)
        tl.store(g_base + 3 * D, (g * p3).to(GradStreams_ptr.dtype.element_ty), mask=mask_d)

        dp0 = tl.sum(g * s0, axis=0)
        dp1 = tl.sum(g * s1, axis=0)
        dp2 = tl.sum(g * s2, axis=0)
        dp3 = tl.sum(g * s3, axis=0)
        tl.atomic_add(GradPre_ptr + row * 4 + 0, dp0)
        tl.atomic_add(GradPre_ptr + row * 4 + 1, dp1)
        tl.atomic_add(GradPre_ptr + row * 4 + 2, dp2)
        tl.atomic_add(GradPre_ptr + row * 4 + 3, dp3)

    @triton.jit
    def _triton_mhc_expand4_kernel(
        X_ptr,
        Res_ptr,
        Post_ptr,
        Comb_ptr,
        Out_ptr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        pid_d = tl.program_id(1).to(tl.int64)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        x = tl.load(X_ptr + row * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        base_res = Res_ptr + row * (4 * D) + offs_d
        s0 = tl.load(base_res + 0 * D, mask=mask_d, other=0.0).to(tl.float32)
        s1 = tl.load(base_res + 1 * D, mask=mask_d, other=0.0).to(tl.float32)
        s2 = tl.load(base_res + 2 * D, mask=mask_d, other=0.0).to(tl.float32)
        s3 = tl.load(base_res + 3 * D, mask=mask_d, other=0.0).to(tl.float32)

        p0 = tl.load(Post_ptr + row * 4 + 0).to(tl.float32)
        p1 = tl.load(Post_ptr + row * 4 + 1).to(tl.float32)
        p2 = tl.load(Post_ptr + row * 4 + 2).to(tl.float32)
        p3 = tl.load(Post_ptr + row * 4 + 3).to(tl.float32)

        c_base = Comb_ptr + row * 16
        c00 = tl.load(c_base + 0).to(tl.float32)
        c01 = tl.load(c_base + 1).to(tl.float32)
        c02 = tl.load(c_base + 2).to(tl.float32)
        c03 = tl.load(c_base + 3).to(tl.float32)
        c10 = tl.load(c_base + 4).to(tl.float32)
        c11 = tl.load(c_base + 5).to(tl.float32)
        c12 = tl.load(c_base + 6).to(tl.float32)
        c13 = tl.load(c_base + 7).to(tl.float32)
        c20 = tl.load(c_base + 8).to(tl.float32)
        c21 = tl.load(c_base + 9).to(tl.float32)
        c22 = tl.load(c_base + 10).to(tl.float32)
        c23 = tl.load(c_base + 11).to(tl.float32)
        c30 = tl.load(c_base + 12).to(tl.float32)
        c31 = tl.load(c_base + 13).to(tl.float32)
        c32 = tl.load(c_base + 14).to(tl.float32)
        c33 = tl.load(c_base + 15).to(tl.float32)

        o0 = (((c00 * s0 + c10 * s1) + c20 * s2) + c30 * s3) + p0 * x
        o1 = (((c01 * s0 + c11 * s1) + c21 * s2) + c31 * s3) + p1 * x
        o2 = (((c02 * s0 + c12 * s1) + c22 * s2) + c32 * s3) + p2 * x
        o3 = (((c03 * s0 + c13 * s1) + c23 * s2) + c33 * s3) + p3 * x

        base_out = Out_ptr + row * (4 * D) + offs_d
        tl.store(base_out + 0 * D, o0.to(Out_ptr.dtype.element_ty), mask=mask_d)
        tl.store(base_out + 1 * D, o1.to(Out_ptr.dtype.element_ty), mask=mask_d)
        tl.store(base_out + 2 * D, o2.to(Out_ptr.dtype.element_ty), mask=mask_d)
        tl.store(base_out + 3 * D, o3.to(Out_ptr.dtype.element_ty), mask=mask_d)

    @triton.jit
    def _triton_unweighted_rmsnorm_fp32_out_kernel(
        X_ptr,
        Y_ptr,
        EPS: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        offs_d = tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X_ptr + row * D + offs_d, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, axis=0) / D + EPS)
        tl.store(Y_ptr + row * D + offs_d, x * rstd, mask=mask)


class DeepseekV41HyperConnection(nn.Module):
    """One site's mHC coefficients: ``(pre, post, comb)`` from the stream itself.

    A site is an attention or an FFN within one block, and owns its own ``fn`` /
    ``base`` / ``scale``. Unlike V4's equivalent this returns ``pre`` rather than
    applying it, because in V4.1 the collapse it belongs to happens one site
    later -- see the module docstring.
    """

    def __init__(self, config: DeepseekV41LlmConfig) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps, config=config)
        self.veomni_mhc_post = VeomniOp("mhc", "post", resolve_module_op_impl(config, "mhc_implementation"))
        self._use_tile_kernels = resolve_module_op_impl(
            config, "mhc_implementation"
        ) == "tilelang" and is_package_available("tile_kernels")
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        # One learned scale per output: pre, post, comb.
        self.scale = nn.Parameter(torch.empty(3))
        self._initializer_range = config.initializer_range
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Match V4's HyperConnection init: ``fn ~ N(0, std)``, ``base = 0``, ``scale = 1``.

        These are raw ``Parameter``s, not ``nn.Linear``, so HF's default
        ``_init_weights`` would leave ``torch.empty`` garbage. Sinkhorn on nan
        ``comb`` is how a random-init toy forward sporadically explodes.
        """
        nn.init.normal_(self.fn, mean=0.0, std=self._initializer_range)
        nn.init.zeros_(self.base)
        nn.init.ones_(self.scale)

    def _get_fp32_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (
            self.fn.data_ptr(),
            self.fn._version,
            self.base.data_ptr(),
            self.base._version,
            self.scale.data_ptr(),
            self.scale._version,
        )
        cached = self.__dict__.get("_fp32_params_cache")
        if cached is None or cached[0] != key:
            cached = (
                key,
                self.fn.float().contiguous(),
                self.base.float().contiguous(),
                self.scale.float().contiguous(),
            )
            self.__dict__["_fp32_params_cache"] = cached
        return cached[1], cached[2], cached[3]

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``[B, S, hc, D]`` -> ``pre[B, S, hc]``, ``post[B, S, hc]``, ``comb[B, S, hc, hc]``."""
        hc = self.hc_mult
        # Normalised over the whole flattened hc*D vector, one statistic per
        # token -- not per copy. Normalising per copy would couple the copies mHC
        # deliberately keeps separate.
        if (
            not torch.is_grad_enabled()
            and IS_CUDA_AVAILABLE
            and _HAS_TRITON
            and hidden_streams.is_cuda
            and hidden_streams.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and hc in (2, 4, 8)
            and self.hc_sinkhorn_iters >= 1
        ):
            lead_shape = hidden_streams.shape[:-2]
            flat_in = hidden_streams.flatten(start_dim=2).contiguous()
            d_flat = flat_in.shape[-1]
            m = flat_in.numel() // d_flat
            fn_f, base_f, scale_f = self._get_fp32_params()
            if m > 0 and d_flat <= 32768:
                flat = torch.empty((m, d_flat), dtype=torch.float32, device=hidden_streams.device)
                block_d = triton.next_power_of_2(d_flat)
                num_warps = 8 if block_d <= 8192 else 16
                with get_torch_device().device(hidden_streams.device):
                    _triton_unweighted_rmsnorm_fp32_out_kernel[(m,)](
                        flat_in,
                        flat,
                        EPS=float(self.input_norm.eps),
                        D=d_flat,
                        BLOCK_D=block_d,
                        num_warps=num_warps,
                    )
            else:
                flat = self.input_norm(flat_in.float())
            mixes = F.linear(flat, fn_f).contiguous()
            pre = torch.empty((*lead_shape, hc), dtype=torch.float32, device=hidden_streams.device)
            post = torch.empty((*lead_shape, hc), dtype=torch.float32, device=hidden_streams.device)
            comb = torch.empty((*lead_shape, hc, hc), dtype=torch.float32, device=hidden_streams.device)
            if m > 0:
                with get_torch_device().device(hidden_streams.device):
                    _triton_mhc_mix_sinkhorn_kernel[(m,)](
                        mixes,
                        base_f,
                        scale_f,
                        pre,
                        post,
                        comb,
                        HC=hc,
                        SINKHORN_ITERS=self.hc_sinkhorn_iters,
                        HC_EPS=self.hc_eps,
                        POST_MULT=POST_MULTIPLIER,
                        num_warps=1,
                    )
            return pre, post, comb
        if (
            torch.is_grad_enabled()
            and IS_CUDA_AVAILABLE
            and getattr(self, "_use_tile_kernels", False)
            and hidden_streams.is_cuda
            and hidden_streams.dtype == torch.bfloat16
            and hidden_streams.ndim == 4
            and hc == 4
            and self.hc_sinkhorn_iters >= 1
        ):
            from tile_kernels.modeling.mhc.ops import (
                mhc_pre_norm_fn,
                mhc_pre_split_mixes,
                sinkhorn_normalize,
            )

            seq_len = hidden_streams.shape[1]
            pad_s = -seq_len % 128
            hs_in = F.pad(hidden_streams, (0, 0, 0, 0, 0, pad_s)) if pad_s else hidden_streams.contiguous()
            mixes = mhc_pre_norm_fn(
                hs_in,
                self.fn.float().contiguous(),
                None,
                float(self.input_norm.eps),
                fuse_grad_acc=False,
            )
            pre, post, comb = mhc_pre_split_mixes(
                mixes,
                self.scale.float().contiguous(),
                self.base.float().contiguous(),
                hc,
                POST_MULTIPLIER,
                self.hc_eps,
            )
            comb = sinkhorn_normalize(comb, repeat=self.hc_sinkhorn_iters, eps=self.hc_eps)
            if pad_s:
                pre = pre[:, :seq_len]
                post = post[:, :seq_len]
                comb = comb[:, :seq_len]
            return pre.squeeze(-1), post.squeeze(-1), comb

        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = POST_MULTIPLIER * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        return pre, post, comb


class _MhcCollapse4Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_streams: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        lead_shape = hidden_streams.shape[:-2]
        d = hidden_streams.shape[-1]
        m = hidden_streams.numel() // (4 * d)
        streams_c = hidden_streams.contiguous()
        pre_c = pre.contiguous()
        out = torch.empty((*lead_shape, d), dtype=hidden_streams.dtype, device=hidden_streams.device)
        if m > 0:
            with get_torch_device().device(hidden_streams.device):
                _triton_mhc_collapse4_kernel[(m, triton.cdiv(d, 1024))](
                    streams_c,
                    pre_c,
                    out,
                    D=d,
                    BLOCK_D=1024,
                    enable_fp_fusion=False,
                )
        ctx.save_for_backward(streams_c, pre_c)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        streams_c, pre_c = ctx.saved_tensors
        d = streams_c.shape[-1]
        m = streams_c.numel() // (4 * d)
        grad_out_c = grad_output.contiguous()
        grad_streams = torch.empty_like(streams_c)
        grad_pre = torch.zeros_like(pre_c, dtype=torch.float32)
        if m > 0:
            with get_torch_device().device(streams_c.device):
                _triton_mhc_collapse4_bwd_kernel[(m, triton.cdiv(d, 1024))](
                    grad_out_c,
                    streams_c,
                    pre_c,
                    grad_streams,
                    grad_pre,
                    D=d,
                    BLOCK_D=1024,
                    enable_fp_fusion=False,
                )
        return grad_streams, grad_pre.to(pre_c.dtype) if ctx.needs_input_grad[1] else None


def mhc_collapse(hidden_streams: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """Weight the ``hc_mult`` copies down into one sublayer input.

    ``[B, S, hc, D]`` and ``[B, S, hc]`` -> ``[B, S, D]``. ``pre`` comes from the
    *previous* site, which is the whole point of this being a free function
    rather than the tail of :meth:`DeepseekV41HyperConnection.forward`.
    """
    if (
        IS_CUDA_AVAILABLE
        and _HAS_TRITON
        and hidden_streams.is_cuda
        and hidden_streams.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and pre.dtype == torch.float32
        and hidden_streams.shape[-2] == 4
    ):
        if torch.is_grad_enabled() and (hidden_streams.requires_grad or pre.requires_grad):
            return _MhcCollapse4Function.apply(hidden_streams, pre)
        lead_shape = hidden_streams.shape[:-2]
        d = hidden_streams.shape[-1]
        m = hidden_streams.numel() // (4 * d)
        out = torch.empty((*lead_shape, d), dtype=hidden_streams.dtype, device=hidden_streams.device)
        if m > 0:
            with get_torch_device().device(hidden_streams.device):
                _triton_mhc_collapse4_kernel[(m, triton.cdiv(d, 1024))](
                    hidden_streams.contiguous(),
                    pre.contiguous(),
                    out,
                    D=d,
                    BLOCK_D=1024,
                    enable_fp_fusion=False,
                )
        return out
    return (pre.unsqueeze(-1) * hidden_streams.float()).sum(dim=2).to(hidden_streams.dtype)


def mhc_expand(
    output: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    op: VeomniOp | None = None,
) -> torch.Tensor:
    """Place a sublayer output back across the copies and mix the residual in.

    ``output[B, S, D]``, ``residual[B, S, hc, D]`` -> ``[B, S, hc, D]``.

    The eager form is written as a matmul against ``comb.transpose(-1, -2)``
    rather than as the reference's ``sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)``.
    They are the same contraction -- ``sum_i comb[i, j] * residual[i, d]`` -- but
    the reference's spelling materialises a ``[B, S, hc, hc, D]`` intermediate,
    which at hc_mult 4 is four times the stream it is mixing.
    """
    if (
        not torch.is_grad_enabled()
        and IS_CUDA_AVAILABLE
        and _HAS_TRITON
        and output.is_cuda
        and output.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and residual.dtype == output.dtype
        and post.dtype == torch.float32
        and comb.dtype == torch.float32
        and residual.shape[-2] == 4
    ):
        d = output.shape[-1]
        m = output.numel() // d
        out = torch.empty_like(residual)
        if m > 0:
            with get_torch_device().device(output.device):
                _triton_mhc_expand4_kernel[(m, triton.cdiv(d, 1024))](
                    output.contiguous(),
                    residual.contiguous(),
                    post.contiguous(),
                    comb.contiguous(),
                    out,
                    D=d,
                    BLOCK_D=1024,
                    enable_fp_fusion=False,
                )
        return out
    if op is None:
        op = VeomniOp("mhc", "post")
    return op(output, residual, post, comb)


def require_fp32_pre_mix(pre_mix: torch.Tensor) -> None:
    """Refuse a ``pre`` that something outside the model narrowed.

    Every ``pre`` the model builds is at least fp32 (fp64 in a float64 model),
    so a bf16 one at a block boundary is the FSDP2 mixed-precision input cast.
    That cast also hands each block a copy of the ``SharedAttentionState`` it
    publishes into, so the layers after a source never see its KV; and FSDP2
    skips it when gradient checkpointing replays the block, so the replay
    collapses with a different ``pre``, routes differently and fails the
    checkpoint's shape check.
    """
    if torch.finfo(pre_mix.dtype).bits < 32:
        raise RuntimeError(
            f"the hyper-connection pre mix reached a block as {pre_mix.dtype}, narrower than float32. Under FSDP2 mixed "
            "precision set `accelerator.fsdp_config.mixed_precision.cast_forward_inputs: false` for the "
            "DeepSeek-V4.1 llm."
        )


def identity_pre_mix(hidden_streams: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """The ``pre`` the first block collapses with: all weight on copy 0.

    Every copy holds the same embedding at that point, so this is one-hot rather
    than uniform only to keep the scale of the collapsed vector equal to the
    embedding's -- averaging would divide it by ``hc_mult``.
    """
    pre_mix = hidden_streams.new_zeros(hidden_streams.shape[0], hidden_streams.shape[1], hc_mult, dtype=torch.float32)
    pre_mix[:, :, 0] = 1.0
    return pre_mix


def expand_to_streams(hidden_states: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """``[B, S, D]`` -> ``[B, S, hc_mult, D]``, one copy per stream."""
    return hidden_states.unsqueeze(2).expand(-1, -1, hc_mult, -1).contiguous()


__all__ = [
    "POST_MULTIPLIER",
    "DeepseekV41HyperConnection",
    "expand_to_streams",
    "identity_pre_mix",
    "mhc_collapse",
    "mhc_expand",
    "require_fp32_pre_mix",
]
