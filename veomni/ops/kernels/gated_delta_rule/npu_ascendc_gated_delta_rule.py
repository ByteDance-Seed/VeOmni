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
"""NPU AscendC chunk gated delta rule: layout adapter over ``flash_gated_delta_rule``.

The generated ``Qwen3_5GatedDeltaNet.forward`` feeds ``q/k/v`` in the FLA layout
``[B, T, H, D]`` (split on the channel dim, then reshaped to heads) — this is what
the ``fla`` GPU kernel and the vendored ``npu`` Triton kernel consume. The AscendC
entry ``flash_gated_delta_rule`` instead expects ``[B, H, T, D]`` (and validates the
shape, raising otherwise), because MindSpeed-MM's AscendC path produces that layout
directly from its ``triton_with_transpose`` conv + head-dim split.

Rather than replicate MM's per-implementation ``forward`` branching (which would
require patchgen modeling changes), this thin adapter absorbs the layout difference
at the op boundary: it transposes ``q/k/v`` from ``[B, T, H, D]`` to ``[B, H, T, D]``
on the way in. ``g``/``beta`` are already ``[B, T, H]`` (what the entry wants) and the
returned ``o`` is already ``[B, T, H, V]``, so no output transpose is needed. Numerically
this equals MM's AscendC path; the only cost is the input transpose (a copy).

``flash_gated_delta_rule`` is imported at module top so that binding the ``npu_ascendc``
backend surfaces a missing ``fla_npu``/``torch_npu`` as an actionable error at
``OpSlot.bind()`` time (the package ``__init__`` factory only imports this module when
the backend is selected).
"""

from __future__ import annotations

from typing import Optional

import torch

from ._ascend.flash_gated_delta_rule import flash_gated_delta_rule


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    **kwargs,
):
    # q/k/v arrive as [B, T, H, D]; the AscendC entry wants [B, H, T, D].
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    return flash_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        **kwargs,
    )
