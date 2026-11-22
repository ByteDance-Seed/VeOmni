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

"""GLM-5.3-Flash (``glm5_next``) modeling that calls local ``VeomniOp`` handles."""

from veomni.lora.target_mapping import convert_fused_moe_lora_targets
from veomni.models.registry import MODELING_REGISTRY


def _convert_wrapped_lora_targets_to_parameters(_model, lora_modules, target_parameter_patterns):
    return convert_fused_moe_lora_targets(
        lora_modules,
        target_parameter_patterns,
        "model.language_model.layers.*.mlp.experts.gate_up_proj",
        "model.language_model.layers.*.mlp.experts.down_proj",
    )


def _convert_model_lora_targets_to_parameters(_model, lora_modules, target_parameter_patterns):
    return convert_fused_moe_lora_targets(
        lora_modules,
        target_parameter_patterns,
        "language_model.layers.*.mlp.experts.gate_up_proj",
        "language_model.layers.*.mlp.experts.down_proj",
    )


def _convert_text_lora_targets_to_parameters(_model, lora_modules, target_parameter_patterns):
    return convert_fused_moe_lora_targets(
        lora_modules,
        target_parameter_patterns,
        "layers.*.mlp.experts.gate_up_proj",
        "layers.*.mlp.experts.down_proj",
    )


@MODELING_REGISTRY.register("glm5_next")
def register_glm5_next_modeling(architecture: str):
    from .checkpoint_tensor_converter import (
        convert_glm5_next_fqn_to_index_mapping,
        create_glm5_next_checkpoint_tensor_converter,
    )
    from .modeling_glm5_next import (
        Glm5NextForConditionalGeneration,
        Glm5NextModel,
        Glm5NextTextModel,
    )

    classes = (
        Glm5NextForConditionalGeneration,
        Glm5NextModel,
        Glm5NextTextModel,
    )
    for model_cls in classes:
        model_cls._create_checkpoint_tensor_converter = staticmethod(create_glm5_next_checkpoint_tensor_converter)
        model_cls._convert_fqn_to_index_mapping = staticmethod(convert_glm5_next_fqn_to_index_mapping)

    Glm5NextForConditionalGeneration._convert_lora_targets_to_parameters = staticmethod(
        _convert_wrapped_lora_targets_to_parameters
    )
    Glm5NextModel._convert_lora_targets_to_parameters = staticmethod(_convert_model_lora_targets_to_parameters)
    Glm5NextTextModel._convert_lora_targets_to_parameters = staticmethod(_convert_text_lora_targets_to_parameters)

    if "ForConditionalGeneration" in architecture:
        return Glm5NextForConditionalGeneration
    if "TextModel" in architecture:
        return Glm5NextTextModel
    if architecture.endswith("Model") or "Glm5NextModel" == architecture:
        return Glm5NextModel
    return Glm5NextForConditionalGeneration


@MODELING_REGISTRY.register("glm5_next_text")
def register_glm5_next_text_modeling(architecture: str):
    from .checkpoint_tensor_converter import (
        convert_glm5_next_fqn_to_index_mapping,
        create_glm5_next_checkpoint_tensor_converter,
    )
    from .modeling_glm5_next import Glm5NextTextModel

    Glm5NextTextModel._create_checkpoint_tensor_converter = staticmethod(create_glm5_next_checkpoint_tensor_converter)
    Glm5NextTextModel._convert_fqn_to_index_mapping = staticmethod(convert_glm5_next_fqn_to_index_mapping)
    Glm5NextTextModel._convert_lora_targets_to_parameters = staticmethod(_convert_text_lora_targets_to_parameters)
    return Glm5NextTextModel
