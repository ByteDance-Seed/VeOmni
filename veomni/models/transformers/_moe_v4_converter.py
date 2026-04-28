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

"""
Shared on-the-fly checkpoint tensor converter for transformers v4 MoE models.

The transformers-v4 VeOmni patches for `qwen3_moe`, `deepseek_v3`, and
`qwen3_omni_moe` (in fused mode) store expert weights as three separate stacked
nn.Parameters (`gate_proj`, `up_proj`, `down_proj`) shaped `[E, ...]`. Vanilla
HuggingFace checkpoints store the same data per-expert under nn.ModuleList.
This module bridges that gap at load time so users can pass HF checkpoints
directly without running the offline `scripts/moe_ckpt_merge/moe_merge.py`.

Output keys match what the offline merge script produces (see
`_merge_experts_for_group` in moe_merge.py):

    {prefix}.experts.gate_proj  [E, I, H]
    {prefix}.experts.up_proj    [E, I, H]
    {prefix}.experts.down_proj  [E, H, I]

The v5 converters (e.g. `qwen3_moe/checkpoint_tensor_converter.py`) cat gate+up
into a single `gate_up_proj` because v5 modeling uses one fused parameter; v4
modeling keeps them separate, so we stack each projection independently.
"""

from typing import Callable, Dict, List, Optional, Pattern, Tuple

import torch

from ..checkpoint_tensor_loading import ConvertedCheckpointTensor


class MoEV4StackingConverter:
    """Buffer per-expert tensors and emit one stacked tensor per (prefix, projection).

    Args:
        pattern: Compiled regex with three capture groups:
            (1) layer prefix up to and including ``.mlp`` (e.g.
                ``"model.layers.3.mlp"``), used as the emit-key prefix and as
                the bucket key.
            (2) expert id (decimal integer).
            (3) projection name — one of ``gate_proj``, ``up_proj``, ``down_proj``.
        num_experts_for: Callable mapping a prefix to the expected number of
            experts at that prefix. For flat single-tower models pass
            ``lambda _: N``; for multi-tower omni models, dispatch on the
            tower component of the prefix.
    """

    def __init__(self, pattern: Pattern[str], num_experts_for: Callable[[str], int]):
        self._pattern = pattern
        self._num_experts_for = num_experts_for
        # {(prefix, proj_name): {expert_id: tensor}}
        self._buffer: Dict[Tuple[str, str], Dict[int, torch.Tensor]] = {}

    def can_handle(self, name: str) -> bool:
        return bool(self._pattern.match(name))

    def convert(self, name: str, tensor: "torch.Tensor") -> Optional[ConvertedCheckpointTensor]:
        match = self._pattern.match(name)
        if not match:
            return None

        prefix, expert_id_str, proj_name = match.group(1), match.group(2), match.group(3)
        expert_id = int(expert_id_str)
        bucket = self._buffer.setdefault((prefix, proj_name), {})
        bucket[expert_id] = tensor

        num_experts = self._num_experts_for(prefix)
        if len(bucket) < num_experts:
            return None

        stacked = torch.stack([bucket[i] for i in range(num_experts)])
        del self._buffer[(prefix, proj_name)]
        return ConvertedCheckpointTensor(name=f"{prefix}.experts.{proj_name}", tensor=stacked)

    def finalize(self) -> List[ConvertedCheckpointTensor]:
        if not self._buffer:
            return []
        unflushed = {f"{p}.experts.{proj}": sorted(b.keys()) for (p, proj), b in self._buffer.items()}
        raise RuntimeError(
            "MoE v4 checkpoint converter: incomplete checkpoint detected. "
            f"Unflushed per-expert buffers (collected expert ids per key): {unflushed}"
        )
