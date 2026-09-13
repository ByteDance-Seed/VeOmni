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

"""Qwen3-Omni-MoE modeling that calls local ``VeomniOp`` handles."""

from veomni.lora.target_mapping import convert_fused_moe_lora_targets
from veomni.models_kernel.registry import MODEL_CONFIG_REGISTRY, MODEL_PROCESSOR_REGISTRY, MODELING_REGISTRY
from veomni.utils.device import IS_NPU_AVAILABLE


def _convert_qwen3_omni_moe_wrapped_lora_targets_to_parameters(_model, lora_modules, target_parameter_patterns):
    return convert_fused_moe_lora_targets(
        lora_modules,
        target_parameter_patterns,
        "thinker.model.layers.*.mlp.experts.gate_up_proj",
        "thinker.model.layers.*.mlp.experts.down_proj",
    )


def _convert_qwen3_omni_moe_thinker_lora_targets_to_parameters(_model, lora_modules, target_parameter_patterns):
    return convert_fused_moe_lora_targets(
        lora_modules,
        target_parameter_patterns,
        "model.layers.*.mlp.experts.gate_up_proj",
        "model.layers.*.mlp.experts.down_proj",
    )


def _convert_qwen3_omni_moe_text_lora_targets_to_parameters(_model, lora_modules, target_parameter_patterns):
    return convert_fused_moe_lora_targets(
        lora_modules,
        target_parameter_patterns,
        "layers.*.mlp.experts.gate_up_proj",
        "layers.*.mlp.experts.down_proj",
    )


@MODEL_CONFIG_REGISTRY.register("qwen3_omni_moe")
def register_qwen3_omni_moe_config():
    from .configuration_qwen3_omni_moe import Qwen3OmniMoeConfig

    return Qwen3OmniMoeConfig


def _get_qwen3_omni_moe_modeling_classes():
    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_qwen3_omni_moe_npu import (
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeThinkerForConditionalGeneration,
            Qwen3OmniMoeThinkerTextModel,
        )
    else:
        from .generated.patched_modeling_qwen3_omni_moe_gpu import (
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeThinkerForConditionalGeneration,
            Qwen3OmniMoeThinkerTextModel,
        )

    from .checkpoint_tensor_converter import (
        convert_qwen3_omni_moe_fqn_to_index_mapping,
        create_qwen3_omni_moe_checkpoint_tensor_converter,
    )

    for model_cls in (
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeThinkerForConditionalGeneration,
        Qwen3OmniMoeThinkerTextModel,
    ):
        model_cls._create_checkpoint_tensor_converter = staticmethod(create_qwen3_omni_moe_checkpoint_tensor_converter)
        model_cls._convert_fqn_to_index_mapping = staticmethod(convert_qwen3_omni_moe_fqn_to_index_mapping)

    Qwen3OmniMoeForConditionalGeneration._convert_lora_targets_to_parameters = staticmethod(
        _convert_qwen3_omni_moe_wrapped_lora_targets_to_parameters
    )
    Qwen3OmniMoeThinkerForConditionalGeneration._convert_lora_targets_to_parameters = staticmethod(
        _convert_qwen3_omni_moe_thinker_lora_targets_to_parameters
    )
    Qwen3OmniMoeThinkerTextModel._convert_lora_targets_to_parameters = staticmethod(
        _convert_qwen3_omni_moe_text_lora_targets_to_parameters
    )

    return (
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeThinkerForConditionalGeneration,
        Qwen3OmniMoeThinkerTextModel,
    )


@MODELING_REGISTRY.register("qwen3_omni_moe")
def register_qwen3_omni_moe_modeling(architecture: str | None):
    top_cls, thinker_cls, text_cls = _get_qwen3_omni_moe_modeling_classes()
    architecture = architecture or ""

    if "ThinkerTextModel" in architecture:
        return text_cls
    if "ThinkerForConditionalGeneration" in architecture:
        return thinker_cls
    if "TalkerForConditionalGeneration" in architecture:
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeTalkerForConditionalGeneration,
        )

        return Qwen3OmniMoeTalkerForConditionalGeneration
    if "TalkerModel" in architecture:
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeTalkerModel

        return Qwen3OmniMoeTalkerModel
    return top_cls


@MODELING_REGISTRY.register("qwen3_omni_moe_thinker")
def register_qwen3_omni_moe_thinker_modeling(_architecture: str | None):
    _, thinker_cls, _ = _get_qwen3_omni_moe_modeling_classes()
    return thinker_cls


@MODELING_REGISTRY.register("qwen3_omni_moe_text")
def register_qwen3_omni_moe_text_modeling(_architecture: str | None):
    _, _, text_cls = _get_qwen3_omni_moe_modeling_classes()
    return text_cls


@MODEL_PROCESSOR_REGISTRY.register("Qwen3OmniMoeProcessor")
def register_qwen3_omni_moe_processor():
    from .processing_qwen3_omni_moe import Qwen3OmniMoeProcessor

    return Qwen3OmniMoeProcessor
