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

"""mHC pre TileKernels adapter (SM90+)."""

from __future__ import annotations

import torch
from torch import Tensor

from ..helper import POST_MULTIPLIER, require_tilelang_input


def wrapper(
    hidden_streams: Tensor,
    fn: Tensor,
    scale: Tensor,
    base: Tensor,
    norm_eps: float,
    hc_mult: int,
    sinkhorn_iters: int,
    hc_eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """TileKernels pre / Sinkhorn / collapse. Lazy-imports ``tile_kernels``."""
    from tile_kernels.modeling.mhc.ops import (
        mhc_pre_apply_mix,
        mhc_pre_big_fuse,
        mhc_pre_norm_fn,
        mhc_pre_split_mixes,
        sinkhorn_normalize,
    )

    require_tilelang_input(hidden_streams, hc_mult)
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
            mhc_post_mult_value=POST_MULTIPLIER,
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
            POST_MULTIPLIER,
            hc_eps,
        )
        comb = sinkhorn_normalize(comb, repeat=sinkhorn_iters, eps=hc_eps)
        collapsed = mhc_pre_apply_mix(hidden_streams.contiguous(), pre)
    return post.squeeze(-1), comb, collapsed
