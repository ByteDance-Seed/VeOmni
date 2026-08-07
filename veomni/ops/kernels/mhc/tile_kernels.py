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

"""DeepSeek V4 mHC adapters backed by deepseek-ai/TileKernels."""

import torch
import torch.nn.functional as F


_POST_MULTIPLIER = 2.0


def _require_supported_input(x: torch.Tensor, hc_mult: int) -> None:
    if not x.is_cuda:
        raise ValueError("TileKernels mHC requires a CUDA tensor")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"TileKernels mHC requires bfloat16 activations, got {x.dtype}")
    if x.ndim != 4:
        raise ValueError(f"TileKernels mHC expects [batch, sequence, hc, hidden], got {tuple(x.shape)}")
    if hc_mult != 4 or x.shape[-2] != 4:
        raise ValueError(f"TileKernels mHC backward currently requires hc_mult=4, got {hc_mult}")


def mhc_pre_tile_kernels(
    hidden_streams: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_eps: float,
    hc_mult: int,
    sinkhorn_iters: int,
    hc_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute mHC pre/post/comb mixes and collapse the residual streams."""
    from tile_kernels.modeling.mhc.ops import (
        mhc_pre_apply_mix,
        mhc_pre_big_fuse,
        mhc_pre_norm_fn,
        mhc_pre_split_mixes,
        sinkhorn_normalize,
    )

    _require_supported_input(hidden_streams, hc_mult)
    fn = fn.float().contiguous()
    scale = scale.float().contiguous()
    base = base.float().contiguous()

    if not torch.is_grad_enabled():
        post, comb, collapsed = mhc_pre_big_fuse(
            hidden_streams.contiguous(),
            fn,
            scale,
            base,
            rms_eps=norm_eps,
            mhc_pre_eps=hc_eps,
            mhc_sinkhorn_eps=hc_eps,
            mhc_post_mult_value=_POST_MULTIPLIER,
            sinkhorn_repeat=sinkhorn_iters,
            n_splits=16,
        )
    else:
        mixes = mhc_pre_norm_fn(
            hidden_streams.contiguous(),
            fn,
            None,
            norm_eps,
            fuse_grad_acc=False,
        )
        pre, post, comb = mhc_pre_split_mixes(
            mixes,
            scale,
            base,
            hc_mult,
            _POST_MULTIPLIER,
            hc_eps,
        )
        comb = sinkhorn_normalize(comb, repeat=sinkhorn_iters, eps=hc_eps)
        collapsed = mhc_pre_apply_mix(hidden_streams.contiguous(), pre)
    return post.squeeze(-1), comb, collapsed


class _MHCPostNoFusedGrad(torch.autograd.Function):
    """Use TileKernels post kernels without its storage-coupled grad shortcut."""

    @staticmethod
    def forward(ctx, x, residual, post, comb):
        from tile_kernels.modeling.mhc.ops import mhc_post_fwd

        output = mhc_post_fwd(x, residual, post, comb)
        ctx.save_for_backward(x, residual, post, comb)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        from tile_kernels.modeling.mhc.ops import mhc_post_bwd

        return mhc_post_bwd(*ctx.saved_tensors, grad_output, fuse_grad_acc=False)


def mhc_post_tile_kernels(
    output: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Mix a sublayer output back into the four mHC residual streams."""
    _require_supported_input(residual, residual.shape[-2])
    if output.dtype != torch.bfloat16:
        raise ValueError(f"TileKernels mHC post requires bfloat16 sublayer output, got {output.dtype}")
    return _MHCPostNoFusedGrad.apply(
        output.contiguous(),
        residual.contiguous(),
        post.float().unsqueeze(-1).contiguous(),
        comb.float().contiguous(),
    )


def mhc_head_tile_kernels(
    hidden_streams: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_eps: float,
    hc_mult: int,
    hc_eps: float,
) -> torch.Tensor:
    """Collapse the final mHC residual streams before the model norm/head."""
    from tile_kernels.modeling.mhc.ops import mhc_head_compute_mix, mhc_pre_apply_mix, mhc_pre_norm_fn

    _require_supported_input(hidden_streams, hc_mult)
    mhc_mix_dim = hc_mult * (2 + hc_mult)
    fn = fn.float().contiguous()
    if fn.shape[0] < mhc_mix_dim:
        fn = F.pad(fn, (0, 0, 0, mhc_mix_dim - fn.shape[0]))
    mixes = mhc_pre_norm_fn(
        hidden_streams.contiguous(),
        fn,
        None,
        norm_eps,
        fuse_grad_acc=False,
    )
    mix = mhc_head_compute_mix(
        mixes[..., :hc_mult].contiguous(),
        scale.float().reshape(1).contiguous(),
        base.float().contiguous(),
        hc_eps,
    )
    return mhc_pre_apply_mix(hidden_streams.contiguous(), mix.unsqueeze(-1))


__all__ = ["mhc_head_tile_kernels", "mhc_post_tile_kernels", "mhc_pre_tile_kernels"]
