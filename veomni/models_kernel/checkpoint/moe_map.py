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

"""Shared per-expert HF index -> fused expert FQN mapping (no tensor ops)."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Pattern


# Default pattern for flat MoE towers (qwen3_moe, deepseek_v3, qwen3_omni thinker, etc.).
PER_EXPERT_SPLIT_TO_FUSED_PATTERN = re.compile(r"^(.+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def convert_per_expert_fqn_mapping_to_fused(
    fqn_to_index_mapping: dict[str, int],
    pattern: Pattern[str] = PER_EXPERT_SPLIT_TO_FUSED_PATTERN,
) -> dict[str, int]:
    """Map per-expert HF index keys to fused expert FQNs for a given regex *pattern*.

    Output keys match ``CheckpointTensorConverter`` emit names (no ``.weight`` suffix).
    Non-matching keys are copied unchanged.
    """
    gate_up_shard_indices: dict[str, list[int]] = defaultdict(list)
    down_shard_indices: dict[str, list[int]] = defaultdict(list)
    converted: dict[str, int] = {}

    for fqn, shard_idx in fqn_to_index_mapping.items():
        match = pattern.match(fqn)
        if not match:
            converted[fqn] = shard_idx
            continue

        prefix, _expert_id, proj_name = match.groups()
        if proj_name == "down_proj":
            down_shard_indices[prefix].append(shard_idx)
        else:
            gate_up_shard_indices[prefix].append(shard_idx)

    for prefix, indices in down_shard_indices.items():
        converted[f"{prefix}.experts.down_proj"] = min(indices)

    for prefix, indices in gate_up_shard_indices.items():
        converted[f"{prefix}.experts.gate_up_proj"] = min(indices)

    return converted
