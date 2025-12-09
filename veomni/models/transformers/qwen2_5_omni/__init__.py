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
from ...loader import MODEL_CONFIG_REGISTRY, MODEL_PROCESSOR_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("qwen2_5_omni")
def register_qwen2_5_omni_config():
    from .configuration_qwen2_5_omni import Qwen2_5OmniConfig

    return Qwen2_5OmniConfig


@MODELING_REGISTRY.register("qwen2_5_omni")
def register_qwen2_5_omni_modeling(architecture: str):
    from .modeling_qwen2_5_omni import (
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniTalkerForConditionalGeneration,
        Qwen2_5OmniTalkerModel,
        Qwen2_5OmniThinkerForConditionalGeneration,
        Qwen2_5OmniThinkerTextModel,
        Qwen2_5OmniToken2WavBigVGANModel,
        Qwen2_5OmniToken2WavDiTModel,
        Qwen2_5OmniToken2WavModel,
    )

    if "ForConditionalGeneration" in architecture:
        return Qwen2_5OmniForConditionalGeneration
    if "ThinkerTextModel" in architecture:
        return Qwen2_5OmniThinkerTextModel
    if "ThinkerForConditionalGeneration" in architecture:
        return Qwen2_5OmniThinkerForConditionalGeneration
    if "TalkerModel" in architecture:
        return Qwen2_5OmniTalkerModel
    if "TalkerForConditionalGeneration" in architecture:
        return Qwen2_5OmniTalkerForConditionalGeneration
    if "Token2WavDiTModel" in architecture:
        return Qwen2_5OmniToken2WavDiTModel
    if "Token2WavBigVGANModel" in architecture:
        return Qwen2_5OmniToken2WavBigVGANModel
    if "Token2WavModel" in architecture:
        return Qwen2_5OmniToken2WavModel
    return Qwen2_5OmniForConditionalGeneration


@MODEL_PROCESSOR_REGISTRY.register("Qwen2_5OmniProcessor")
def register_qwen2_5_omni_processor():
    from .processing_qwen2_5_omni import Qwen2_5OmniProcessor

    return Qwen2_5OmniProcessor
