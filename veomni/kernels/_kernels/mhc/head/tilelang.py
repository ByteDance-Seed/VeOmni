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

"""mHC head TileKernels adapter (SM90+)."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from ..helper import require_tilelang_input


def wrapper(
    hidden_streams: Tensor,
    fn: Tensor,
    scale: Tensor,
    base: Tensor,
    norm_eps: float,
    hc_mult: int,
    hc_eps: float,
) -> Tensor:
    """TileKernels final collapse. Lazy-imports ``tile_kernels``."""
    from tile_kernels.modeling.mhc.ops import mhc_head_compute_mix, mhc_pre_apply_mix, mhc_pre_norm_fn

    require_tilelang_input(hidden_streams, hc_mult)
    mix_dim = hc_mult * (2 + hc_mult)
    fn = fn.float().contiguous()
    if fn.shape[0] < mix_dim:
        fn = F.pad(fn, (0, 0, 0, mix_dim - fn.shape[0]))
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
