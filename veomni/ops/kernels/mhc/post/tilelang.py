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

"""mHC post TileKernels raw pair (SM90+)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ....registry import SavedState
from ..helper import require_tilelang_input


@dataclass(frozen=True)
class _Meta:
    """Public dtypes so backward can undo the float / unsqueeze layout."""

    output_dtype: torch.dtype
    residual_dtype: torch.dtype
    post_dtype: torch.dtype
    comb_dtype: torch.dtype


def forward(
    output: Tensor,
    residual: Tensor,
    post: Tensor,
    comb: Tensor,
) -> tuple[Tensor, SavedState]:
    """TileKernels post mix. *post* is modeling ``[B, S, H]``."""
    from tile_kernels.modeling.mhc.ops import mhc_post_fwd

    require_tilelang_input(residual, residual.shape[-2])
    if output.dtype != torch.bfloat16:
        raise ValueError(f"TileKernels mHC post requires bfloat16 sublayer output, got {output.dtype}")

    x = output.contiguous()
    residual_c = residual.contiguous()
    post_c = post.float().unsqueeze(-1).contiguous()
    comb_c = comb.float().contiguous()
    y = mhc_post_fwd(x, residual_c, post_c, comb_c)
    return y, SavedState(
        (x, residual_c, post_c, comb_c),
        _Meta(output.dtype, residual.dtype, post.dtype, comb.dtype),
    )


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor | None, ...]:
    """Return grads for ``(output, residual, post, comb)`` on the modeling face."""
    from tile_kernels.modeling.mhc.ops import mhc_post_bwd

    meta = saved.metadata
    assert isinstance(meta, _Meta)
    grad_output_c, grad_residual, grad_post, grad_comb = mhc_post_bwd(*saved.tensors, grad_output, fuse_grad_acc=False)
    if grad_post is not None:
        grad_post = grad_post.squeeze(-1)
    return (
        None if grad_output_c is None else grad_output_c.to(dtype=meta.output_dtype),
        None if grad_residual is None else grad_residual.to(dtype=meta.residual_dtype),
        None if grad_post is None else grad_post.to(dtype=meta.post_dtype),
        None if grad_comb is None else grad_comb.to(dtype=meta.comb_dtype),
    )
