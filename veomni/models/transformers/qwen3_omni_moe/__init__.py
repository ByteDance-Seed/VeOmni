# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
from ...loader import MODEL_CONFIG_REGISTRY, MODEL_PROCESSOR_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("qwen3_omni_moe")
def register_qwen3_omni_moe_config():
    from .configuration_qwen3_omni_moe import Qwen3OmniMoeConfig

    return Qwen3OmniMoeConfig


@MODELING_REGISTRY.register("qwen3_omni_moe")
def register_qwen3_omni_moe_modeling(architecture: str):
    from .modeling_qwen3_omni_moe import (
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeTalkerForConditionalGeneration,
        Qwen3OmniMoeTalkerModel,
        Qwen3OmniMoeThinkerForConditionalGeneration,
        Qwen3OmniMoeThinkerTextModel,
    )

    if "ForConditionalGeneration" in architecture:
        return Qwen3OmniMoeForConditionalGeneration
    if "ThinkerTextModel" in architecture:
        return Qwen3OmniMoeThinkerTextModel
    if "ThinkerForConditionalGeneration" in architecture:
        return Qwen3OmniMoeThinkerForConditionalGeneration
    if "TalkerModel" in architecture:
        return Qwen3OmniMoeTalkerModel
    if "TalkerForConditionalGeneration" in architecture:
        return Qwen3OmniMoeTalkerForConditionalGeneration
    return Qwen3OmniMoeForConditionalGeneration


@MODEL_PROCESSOR_REGISTRY.register("Qwen3OmniMoeProcessor")
def register_qwen3_omni_moe_processor():
    from .processing_qwen3_omni_moe import Qwen3OmniMoeProcessor

    return Qwen3OmniMoeProcessor
