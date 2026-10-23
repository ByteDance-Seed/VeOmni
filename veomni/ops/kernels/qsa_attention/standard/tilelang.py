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

"""standard QSA gather-based TileLang adapter (NVIDIA CUDA)."""

from __future__ import annotations

from torch import Tensor

from .....utils import logging
from . import eager


logger = logging.get_logger(__name__)


def wrapper(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    selected_indices: Tensor,
    *,
    scaling: float,
    dropout: float = 0.0,
    attention_mask: Tensor | None = None,
) -> tuple[Tensor, None]:
    """Gather the selected K/V rows without materializing a dense score tensor.

    The contract is the one ``eager`` implements. The kernel reads the compact
    ``[B, S, K]`` selection directly, which is what makes 16K-token training
    feasible, and returns no attention weights.

    The selection already encodes causality and packed-sample boundaries, so an
    additive or boolean *attention_mask* has no kernel equivalent and raises.
    A nonzero *dropout* falls back to the dense eager reference with a one-time
    warning, because the kernel has no dropout path.
    """
    if attention_mask is not None:
        raise ValueError(
            f"tilelang QSA attention requires no attention_mask; got attention_mask={type(attention_mask).__name__}."
        )
    if dropout:
        logger.warning_once(
            f"tilelang QSA attention has no dropout path; attention_dropout={dropout} falls back to the "
            "dense eager QSA reference, which materializes the [B, H, S, S] score tensor."
        )
        return eager.wrapper(query, key, value, selected_indices, scaling=scaling, dropout=dropout)

    from ..vendor.tilelang_qsa import qsa_attn_tilelang

    return qsa_attn_tilelang(query, key, value, selected_indices, scaling), None
