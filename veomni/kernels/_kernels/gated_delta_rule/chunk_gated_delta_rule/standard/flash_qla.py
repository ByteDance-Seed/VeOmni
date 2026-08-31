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

"""standard chunk_gated_delta_rule FlashQLA adapter (Hopper SM90)."""

from __future__ import annotations

from torch import Tensor

from ...optional import optional_tensor


def wrapper(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    *,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_size: int = 64,
    cu_seqlens_list: list[int] | None = None,
    chunk_indices: object = None,
    chunk_indices_list: object = None,
    scale: float | None = None,
) -> tuple[Tensor, Tensor | None]:
    """FlashQLA chunk gated delta rule. Lazy-imports ``flash_qla``."""
    from flash_qla.ops.gated_delta_rule import chunk_gated_delta_rule

    del chunk_size, cu_seqlens_list, chunk_indices, chunk_indices_list, scale
    return chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=optional_tensor(initial_state),
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=optional_tensor(cu_seqlens),
    )
