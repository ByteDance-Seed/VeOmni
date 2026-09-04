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

"""standard causal_conv1d FLA adapter."""

from __future__ import annotations

from torch import Tensor

from ...optional import optional_tensor


def wrapper(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    *,
    activation: str | None = "silu",
    seq_idx: Tensor | None = None,
    backend: str | None = None,
) -> Tensor:
    """FLA Triton causal conv1d. Returns ``y`` only. Lazy-imports ``fla``."""
    del seq_idx
    from fla.modules.convolution import causal_conv1d

    output, _final_state = causal_conv1d(
        x=x,
        weight=weight,
        bias=optional_tensor(bias),
        activation=activation,
        cu_seqlens=optional_tensor(cu_seqlens),
        backend=backend or "triton",
    )
    return output
