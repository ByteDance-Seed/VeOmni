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
"""NPU causal conv1d: adapter over the vendored Triton kernel.

The generated Qwen3.5 modeling calls ``causal_conv1d_fn`` with the *FLA*
signature::

    causal_conv1d_fn(x=, weight=, bias=, activation=, seq_idx=, backend=,
                     cu_seqlens=)[0]

The vendored NPU kernel (`._ascend.causal_conv1d`) has a different signature
(``x, weight, bias, residual, initial_state, activation, cu_seqlens,
output_final_state``) and expects the depthwise conv weight transposed. This
thin adapter bridges the two so the modeling call site stays untouched (no
hand-edits under ``generated/``):

- swallows ``seq_idx`` / ``backend`` (FLA-only kwargs the NPU kernel ignores);
- transposes ``weight`` from ``[D, W]`` (``conv1d.weight.squeeze(1)``) to the
  ``[W, D]`` layout the kernel wants;
- returns ``(y, final_state)`` so the call site's ``[0]`` selects ``y``.

The transpose reproduces exactly what the validated NPU run does, so numerics
are unchanged relative to the pre-registry hard-coded path.
"""

from __future__ import annotations

import torch


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    seq_idx: torch.Tensor | None = None,
    backend: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    **_ignored,
):
    from ._ascend.causal_conv1d import causal_conv1d as _ascend_causal_conv1d

    # weight arrives as [D, W] (conv1d.weight.squeeze(1)); the kernel wants [W, D].
    return _ascend_causal_conv1d(
        x=x,
        weight=weight.transpose(0, 1),
        bias=bias,
        activation=activation,
        cu_seqlens=cu_seqlens,
        output_final_state=False,
    )
