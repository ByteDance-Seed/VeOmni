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

"""mHC head Triton adapter (AMD ROCm)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..helper import require_triton_input, require_triton_tensors


def wrapper(
    hidden_streams: Tensor,
    fn: Tensor,
    scale: Tensor,
    base: Tensor,
    norm_eps: float,
    hc_mult: int,
    hc_eps: float,
) -> Tensor:
    """Collapse the residual streams with the Triton norm and collapse kernels."""
    require_triton_input(hidden_streams, hc_mult)
    expected_fn = (hc_mult, hc_mult * hidden_streams.shape[-1])
    if fn.shape != expected_fn:
        raise ValueError(f"Triton mHC head expects fn shape {expected_fn}, got {tuple(fn.shape)}")
    if scale.shape != (1,):
        raise ValueError(f"Triton mHC head expects scale shape (1,), got {tuple(scale.shape)}")
    if base.shape != (hc_mult,):
        raise ValueError(f"Triton mHC head expects base shape ({hc_mult},), got {tuple(base.shape)}")
    require_triton_tensors(hidden_streams, fn=fn, scale=scale, base=base)

    from ..vendor.triton_mhc import hc_collapse, unweighted_rms_norm

    mixes = F.linear(unweighted_rms_norm(hidden_streams.flatten(start_dim=2), norm_eps), fn.float())
    pre = torch.sigmoid(mixes * scale.float() + base.float()) + hc_eps
    return hc_collapse(hidden_streams, pre)
