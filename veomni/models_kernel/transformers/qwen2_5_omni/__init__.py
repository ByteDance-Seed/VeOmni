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

"""Qwen2.5-Omni modeling that calls local ``VeomniOp`` handles."""

from veomni.models_kernel.registry import MODEL_CONFIG_REGISTRY, MODEL_PROCESSOR_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("qwen2_5_omni")
def register_qwen2_5_omni_config():
    from .configuration_qwen2_5_omni import Qwen2_5OmniConfig

    return Qwen2_5OmniConfig


def _get_qwen2_5_omni_modeling_classes():
    from .generated.patched_modeling_qwen2_5_omni_gpu import (
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniThinkerForConditionalGeneration,
        Qwen2_5OmniThinkerTextModel,
    )

    return (
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniThinkerForConditionalGeneration,
        Qwen2_5OmniThinkerTextModel,
    )


@MODELING_REGISTRY.register("qwen2_5_omni")
def register_qwen2_5_omni_modeling(architecture: str | None):
    top_cls, thinker_cls, text_cls = _get_qwen2_5_omni_modeling_classes()
    architecture = architecture or ""

    if "ThinkerTextModel" in architecture:
        return text_cls
    if "TalkerForConditionalGeneration" in architecture:
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniTalkerForConditionalGeneration,
        )

        return Qwen2_5OmniTalkerForConditionalGeneration
    if "TalkerModel" in architecture:
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniTalkerModel

        return Qwen2_5OmniTalkerModel
    if "ThinkerForConditionalGeneration" in architecture:
        return thinker_cls
    return top_cls


@MODELING_REGISTRY.register("qwen2_5_omni_thinker")
def register_qwen2_5_omni_thinker_modeling(_architecture: str | None):
    _, thinker_cls, _ = _get_qwen2_5_omni_modeling_classes()
    return thinker_cls


@MODELING_REGISTRY.register("qwen2_5_omni_text")
def register_qwen2_5_omni_text_modeling(_architecture: str | None):
    _, _, text_cls = _get_qwen2_5_omni_modeling_classes()
    return text_cls


@MODEL_PROCESSOR_REGISTRY.register("Qwen2_5OmniProcessor")
def register_qwen2_5_omni_processor():
    from .processing_qwen2_5_omni import Qwen2_5OmniProcessor

    return Qwen2_5OmniProcessor
