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

"""SavedState layout shared by standard and dit async Ulysses QKV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..registry import KernelEntry, SavedState


# Positional tensors of raw QKV ``forward`` (hidden + Q/K/V weights/biases + QK norms).
QKV_TENSOR_COUNT = 11


@dataclass
class QKVMeta:
    """Non-tensor QKV save payload.

    ``rms_q`` / ``rms_k`` are ``append_inner`` specs for the rmsnorm path.
    LayerNorm tensors stay in the trailing save list instead.
    """

    seq_dimension: int
    head_dimension: int
    unpadded_dim_size: int
    head_dim: int | None
    group: Any
    norm_type: str | None
    normalized_shape: Any
    eps: float | None
    need_repeat_kv: bool = False
    n_repeat: int = 1
    original_num_kv_heads: int = 0
    rms: KernelEntry | None = None
    rms_q: tuple[Any, int] | None = None
    rms_k: tuple[Any, int] | None = None


def qkv_grads(
    grad_hidden: Any,
    grad_q_weight: Any,
    grad_q_bias: Any,
    grad_k_weight: Any,
    grad_k_bias: Any,
    grad_v_weight: Any,
    grad_v_bias: Any,
    grad_norm_q_weight: Any = None,
    grad_norm_q_bias: Any = None,
    grad_norm_k_weight: Any = None,
    grad_norm_k_bias: Any = None,
) -> tuple[Any, ...]:
    """Pack QKV grads in the same order as the raw ``forward`` tensors."""
    return (
        grad_hidden,
        grad_q_weight,
        grad_q_bias,
        grad_k_weight,
        grad_k_bias,
        grad_v_weight,
        grad_v_bias,
        grad_norm_q_weight,
        grad_norm_q_bias,
        grad_norm_k_weight,
        grad_norm_k_bias,
    )


def unpack_qkv(saved: SavedState) -> tuple[Any, ...]:
    """Split the shared prefix tensors from the norm-specific trailing list."""
    meta = saved.metadata
    assert isinstance(meta, QKVMeta)
    (
        hidden_states,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        *rest,
    ) = saved.tensors
    return meta, hidden_states, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, rest
