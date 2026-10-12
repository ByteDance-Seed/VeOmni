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
Runtime checkpoint tensor converter for DeepseekV3 MoE models.

Converts HuggingFace per-expert checkpoint format to v5 fused format
at load time, eliminating the need for offline checkpoint merging.

    HF checkpoint format (per-expert):
        model.layers.{i}.mlp.experts.{j}.gate_proj.weight  [I, H]
        model.layers.{i}.mlp.experts.{j}.up_proj.weight    [I, H]
        model.layers.{i}.mlp.experts.{j}.down_proj.weight  [H, I]

    Target v5 format (direct layout, no transpose):
        model.layers.{i}.mlp.experts.gate_up_proj  [E, 2*I, H]
        model.layers.{i}.mlp.experts.down_proj     [E, H, I]
"""

from typing import Dict

from ..._moe_fused_weight_map import (
    PER_EXPERT_SPLIT_TO_FUSED_PATTERN,
    convert_per_expert_fqn_mapping_to_fused,
)
from ..._moe_per_expert_converter import PerExpertFusedCheckpointTensorConverter


class DeepseekV3CheckpointTensorConverter(PerExpertFusedCheckpointTensorConverter):
    """Per-expert -> fused converter for DeepseekV3.

    ``num_experts`` is ``config.n_routed_experts`` (equal to
    ``config.num_local_experts`` in v5 DeepseekV3).
    """

    model_name = "DeepseekV3"


def create_deepseek_v3_checkpoint_tensor_converter(model):
    """Factory function registered on model classes via _create_checkpoint_tensor_converter."""
    return DeepseekV3CheckpointTensorConverter(
        num_experts=model.config.n_routed_experts,
    )


def convert_deepseek_v3_fqn_to_index_mapping(fqn_to_index_mapping: Dict[str, int]) -> Dict[str, int]:
    """Align HF safetensors index keys with fused expert parameter names."""
    return convert_per_expert_fqn_mapping_to_fused(fqn_to_index_mapping, PER_EXPERT_SPLIT_TO_FUSED_PATTERN)
