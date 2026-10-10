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
Runtime checkpoint tensor converter for GLM-MoE-DSA (GLM-5) models.

The released GLM-5 checkpoint stores routed experts per expert; transformers
fuses them through the ``qwen2_moe`` conversion recipe, which VeOmni's loader
does not run. ``mlp.shared_experts.*`` keys do not match the per-expert regex
and load by name.

    HF checkpoint format (per-expert):
        model.layers.{i}.mlp.experts.{j}.gate_proj.weight  [I, H]
        model.layers.{i}.mlp.experts.{j}.up_proj.weight    [I, H]
        model.layers.{i}.mlp.experts.{j}.down_proj.weight  [H, I]

    Target v5 format:
        model.layers.{i}.mlp.experts.gate_up_proj  [E, 2*I, H]
        model.layers.{i}.mlp.experts.down_proj     [E, H, I]

The checkpoint also ships its MTP block as trunk layer ``num_hidden_layers``
(``model.layers.78`` for GLM-5), which the model does not build; its fused
outputs are dropped as unexpected keys.
"""

from typing import Dict

from ..._moe_fused_weight_map import (
    PER_EXPERT_SPLIT_TO_FUSED_PATTERN,
    convert_per_expert_fqn_mapping_to_fused,
)
from ..._moe_per_expert_converter import PerExpertFusedCheckpointTensorConverter


class GlmMoeDsaCheckpointTensorConverter(PerExpertFusedCheckpointTensorConverter):
    """Per-expert -> fused converter for GLM-MoE-DSA routed experts."""

    model_name = "GlmMoeDsa"


def create_glm_moe_dsa_checkpoint_tensor_converter(model):
    """Factory function registered on model classes via _create_checkpoint_tensor_converter."""
    return GlmMoeDsaCheckpointTensorConverter(
        num_experts=model.config.n_routed_experts,
    )


def convert_glm_moe_dsa_fqn_to_index_mapping(fqn_to_index_mapping: Dict[str, int]) -> Dict[str, int]:
    """Align HF safetensors index keys with fused expert parameter names."""
    return convert_per_expert_fqn_mapping_to_fused(fqn_to_index_mapping, PER_EXPERT_SPLIT_TO_FUSED_PATTERN)
