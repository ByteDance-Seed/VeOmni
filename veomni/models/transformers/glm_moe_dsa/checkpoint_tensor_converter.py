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
# See the License for the specific language governing limitations
# under the License.

"""Runtime checkpoint tensor conversion for GLM-MoE-DSA experts."""

from typing import Dict

from veomni.models.checkpoint.expert_fusion import PerExpertSplitToFusedConverter
from veomni.models.checkpoint.moe_map import (
    PER_EXPERT_SPLIT_TO_FUSED_PATTERN,
    convert_per_expert_fqn_mapping_to_fused,
)


_EXPERT_PATTERN = PER_EXPERT_SPLIT_TO_FUSED_PATTERN


class GlmMoeDsaCheckpointTensorConverter(PerExpertSplitToFusedConverter):
    """Stack split HF expert tensors into the fused v5 expert layout."""

    family_name = "GLM-MoE-DSA"
    expert_pattern = _EXPERT_PATTERN


def create_glm_moe_dsa_checkpoint_tensor_converter(model):
    """Create a converter using the model's routed-expert count."""
    return GlmMoeDsaCheckpointTensorConverter(num_experts=model.config.n_routed_experts)


def convert_glm_moe_dsa_fqn_to_index_mapping(fqn_to_index_mapping: Dict[str, int]) -> Dict[str, int]:
    """Align split HF checkpoint index keys with fused expert parameters."""
    return convert_per_expert_fqn_mapping_to_fused(fqn_to_index_mapping, _EXPERT_PATTERN)
