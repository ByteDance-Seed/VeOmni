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

"""HF-style modeling helper around a tensor-native load-balancing kernel."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


GateLogits = Tensor | tuple[Tensor, ...] | None


def load_balancing_loss(
    gate_logits: GateLogits,
    num_experts: int | None = None,
    top_k: int = 2,
    attention_mask: Tensor | None = None,
    *,
    kernel: Callable,
) -> Tensor | int:
    """Adapt HF per-layer router logits to the unified ``[N, E]`` kernel.

    The helper owns modeling policy only: the legacy ``None``/non-tuple
    behavior, concatenating per-layer logits, optional expert-count
    validation, and the empty-mask sentinel. Backend selection stays on the
    instance-local ``kernel`` handle.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    concatenated = torch.cat(
        [layer.reshape(-1, layer.shape[-1]) for layer in gate_logits],
        dim=0,
    ).contiguous()
    if num_experts is not None and concatenated.shape[-1] != num_experts:
        raise ValueError(f"gate_logits last dim ({concatenated.shape[-1]}) != num_experts ({num_experts})")

    mask = attention_mask if attention_mask is not None else concatenated.new_empty(0)
    return kernel(concatenated, mask, top_k=top_k)
