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
Runtime checkpoint tensor converter for Qwen3-MoE models.

Converts HuggingFace per-expert checkpoint format to v5 fused format
at load time, eliminating the need for offline checkpoint merging.

    HF checkpoint format (per-expert):
        model.layers.{i}.mlp.experts.{j}.gate_proj.weight  [I, H]
        model.layers.{i}.mlp.experts.{j}.up_proj.weight    [I, H]
        model.layers.{i}.mlp.experts.{j}.down_proj.weight  [H, I]

    Target v5 format:
        model.layers.{i}.mlp.experts.gate_up_proj  [E, 2*I, H]
        model.layers.{i}.mlp.experts.down_proj     [E, H, I]
"""

from typing import Dict

from ..._moe_fused_weight_map import (
    PER_EXPERT_SPLIT_TO_FUSED_PATTERN,
    convert_per_expert_fqn_mapping_to_fused,
)
from ..._moe_per_expert_converter import PerExpertFusedCheckpointTensorConverter


class Qwen3MoeCheckpointTensorConverter(PerExpertFusedCheckpointTensorConverter):
    """Per-expert -> fused converter for Qwen3-MoE (also used by Qwen3.5-MoE)."""

    model_name = "Qwen3MoE"


def create_qwen3_moe_checkpoint_tensor_converter(model):
    """Factory function registered on model classes via _create_checkpoint_tensor_converter."""
    return Qwen3MoeCheckpointTensorConverter(
        num_experts=model.config.num_experts,
    )


def convert_qwen3_moe_fqn_to_index_mapping(fqn_to_index_mapping: Dict[str, int]) -> Dict[str, int]:
    """Align HF safetensors index keys with fused expert parameter names."""
    return convert_per_expert_fqn_mapping_to_fused(fqn_to_index_mapping, PER_EXPERT_SPLIT_TO_FUSED_PATTERN)
