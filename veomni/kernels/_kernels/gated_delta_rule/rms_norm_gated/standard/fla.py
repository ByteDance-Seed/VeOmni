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

"""standard rms_norm_gated FLA adapter."""

from __future__ import annotations

from torch import Tensor


def wrapper(x: Tensor, gate: Tensor, weight: Tensor, *, eps: float = 1e-6) -> Tensor:
    """FLA fused ``rms_norm(x) * silu(gate)``. Lazy-imports ``fla``."""
    from fla.modules.fused_norm_gate import rms_norm_gated

    return rms_norm_gated(x, gate, weight, None, activation="silu", eps=eps)
