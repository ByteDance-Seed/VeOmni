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
from ....lora.target_mapping import convert_fused_moe_lora_targets
from ....utils.device import IS_NPU_AVAILABLE
from ...loader import MODELING_REGISTRY


def _convert_qwen3_moe_wrapped_lora_targets_to_parameters(_model, lora_modules, target_parameter_patterns):
    return convert_fused_moe_lora_targets(
        lora_modules,
        target_parameter_patterns,
        "model.layers.*.mlp.experts.gate_up_proj",
        "model.layers.*.mlp.experts.down_proj",
    )


def _convert_qwen3_moe_model_lora_targets_to_parameters(_model, lora_modules, target_parameter_patterns):
    return convert_fused_moe_lora_targets(
        lora_modules,
        target_parameter_patterns,
        "layers.*.mlp.experts.gate_up_proj",
        "layers.*.mlp.experts.down_proj",
    )


@MODELING_REGISTRY.register("qwen3_moe")
def register_qwen3_moe_modeling(architecture: str):
    from .checkpoint_tensor_converter import (
        convert_qwen3_moe_fqn_to_index_mapping,
        create_qwen3_moe_checkpoint_tensor_converter,
    )

    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_qwen3_moe_npu import (
            Qwen3MoeForCausalLM,
            Qwen3MoeForQuestionAnswering,
            Qwen3MoeForTokenClassification,
            Qwen3MoeModel,
        )
    else:
        from .generated.patched_modeling_qwen3_moe_gpu import (
            Qwen3MoeForCausalLM,
            Qwen3MoeForQuestionAnswering,
            Qwen3MoeForTokenClassification,
            Qwen3MoeModel,
        )

    for model_cls in (
        Qwen3MoeForCausalLM,
        Qwen3MoeForQuestionAnswering,
        Qwen3MoeForTokenClassification,
        Qwen3MoeModel,
    ):
        model_cls._create_checkpoint_tensor_converter = staticmethod(create_qwen3_moe_checkpoint_tensor_converter)
        model_cls._convert_fqn_to_index_mapping = staticmethod(convert_qwen3_moe_fqn_to_index_mapping)
        model_cls._convert_lora_targets_to_parameters = staticmethod(
            _convert_qwen3_moe_wrapped_lora_targets_to_parameters
        )
    Qwen3MoeModel._convert_lora_targets_to_parameters = staticmethod(
        _convert_qwen3_moe_model_lora_targets_to_parameters
    )

    if "ForCausalLM" in architecture:
        return Qwen3MoeForCausalLM
    elif "ForTokenClassification" in architecture:
        return Qwen3MoeForTokenClassification
    elif "ForQuestionAnswering" in architecture:
        return Qwen3MoeForQuestionAnswering
    elif "Model" in architecture:
        return Qwen3MoeModel
    else:
        return Qwen3MoeForCausalLM
