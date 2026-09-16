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
# See the License for the specific language governing limitations
# under the License.

"""
Runtime checkpoint tensor converter for Qwen3-Omni-MoE (thinker) models.

HuggingFace's own loader would convert the on-disk per-expert keys into the
fused modeling layout via `conversion_mapping.py` (the `qwen2_moe` recipe:
`MergeModulelist(dim=0) + Concatenate(dim=1)`). VeOmni's loader reads
safetensors directly and never invokes that recipe, so we reproduce it here.

    HF on-disk layout (per-expert split):
        thinker.model.layers.{i}.mlp.experts.{j}.gate_proj.weight  [I, H]
        thinker.model.layers.{i}.mlp.experts.{j}.up_proj.weight    [I, H]
        thinker.model.layers.{i}.mlp.experts.{j}.down_proj.weight  [H, I]

    Target v5 modeling layout (fused):
        thinker.model.layers.{i}.mlp.experts.gate_up_proj  [E, 2*I, H]
        thinker.model.layers.{i}.mlp.experts.down_proj     [E, H, I]

VeOmni's training save path can emit the v5 fused layout directly (e.g.
`save_pretrained(save_original_format=False)`). Those keys do not match the
per-expert regex here, so `maybe_convert_checkpoint_tensor` passes them
through untouched — the fused shape already equals the modeling layout, so
dispatch copies them into place.

The talker tower (`talker.model.layers.{i}.mlp.experts.*`) uses the same
per-expert convention; the regex matches both tower prefixes so standalone
talker tensors (if ever loaded through this converter) are handled uniformly.
"""

from typing import Dict

from veomni.models.checkpoint.expert_fusion import PerExpertSplitToFusedConverter
from veomni.models.checkpoint.moe_map import (
    PER_EXPERT_SPLIT_TO_FUSED_PATTERN,
    convert_per_expert_fqn_mapping_to_fused,
)


_EXPERT_PATTERN = PER_EXPERT_SPLIT_TO_FUSED_PATTERN


class Qwen3OmniMoeCheckpointTensorConverter(PerExpertSplitToFusedConverter):
    """Stack per-expert gate/up/down tensors into v5 fused layout at load time."""

    family_name = "Qwen3OmniMoe"
    expert_pattern = _EXPERT_PATTERN


def create_qwen3_omni_moe_checkpoint_tensor_converter(model):
    """Factory registered on model classes via `_create_checkpoint_tensor_converter`.

    Resolves the text config from whichever top-level config is attached to the
    model:
    - ``Qwen3OmniMoeConfig`` (top) → ``config.thinker_config.text_config``
    - ``Qwen3OmniMoeThinkerConfig`` → ``config.text_config``
    - ``Qwen3OmniMoeThinkerTextConfig`` (the inner text submodel) → ``config``
    """
    config = model.config
    thinker_config = getattr(config, "thinker_config", config)
    text_config = getattr(thinker_config, "text_config", thinker_config)
    return Qwen3OmniMoeCheckpointTensorConverter(num_experts=text_config.num_experts)


def convert_qwen3_omni_moe_fqn_to_index_mapping(fqn_to_index_mapping: Dict[str, int]) -> Dict[str, int]:
    """Align HF safetensors index keys with fused thinker expert parameter names."""
    return convert_per_expert_fqn_mapping_to_fused(fqn_to_index_mapping, _EXPERT_PATTERN)
