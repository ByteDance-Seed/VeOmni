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

"""
Runtime checkpoint tensor converter for GLM-5.3-Flash (``glm5_next``).

HF conversion_mapping uses the qwen2_moe-style per-expert recipe for
``glm5_next``; VeOmni's loader does not run that recipe, so convert at load:

    HF checkpoint format (per-expert):
        ...mlp.experts.{j}.gate_proj.weight  [I, H]
        ...mlp.experts.{j}.up_proj.weight    [I, H]
        ...mlp.experts.{j}.down_proj.weight  [H, I]

    Target v5 format:
        ...mlp.experts.gate_up_proj  [E, 2*I, H]
        ...mlp.experts.down_proj     [E, H, I]

Released Flash checkpoints may already be fused; the per-expert regex only
matches split keys, so fused tensors pass through unchanged.
"""

from typing import Dict

from ..._moe_per_expert_converter import PerExpertFusedCheckpointTensorConverter
from ...checkpoint.moe_map import (
    PER_EXPERT_SPLIT_TO_FUSED_PATTERN,
    convert_per_expert_fqn_mapping_to_fused,
)


class Glm5NextCheckpointTensorConverter(PerExpertFusedCheckpointTensorConverter):
    """Per-expert -> fused converter for GLM-5.3-Flash routed experts."""

    model_name = "Glm5Next"


def create_glm5_next_checkpoint_tensor_converter(model):
    """Factory registered on model classes via ``_create_checkpoint_tensor_converter``."""
    text_config = getattr(model.config, "text_config", model.config)
    return Glm5NextCheckpointTensorConverter(
        num_experts=text_config.n_routed_experts,
    )


def convert_glm5_next_fqn_to_index_mapping(fqn_to_index_mapping: Dict[str, int]) -> Dict[str, int]:
    """Align HF safetensors index keys with fused expert parameter names."""
    return convert_per_expert_fqn_mapping_to_fused(fqn_to_index_mapping, PER_EXPERT_SPLIT_TO_FUSED_PATTERN)
