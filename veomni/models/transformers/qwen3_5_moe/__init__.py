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


def _convert_qwen3_5_moe_conditional_lora_targets_to_parameters(_model, lora_modules, target_parameter_patterns):
    return convert_fused_moe_lora_targets(
        lora_modules,
        target_parameter_patterns,
        "model.language_model.layers.*.mlp.experts.gate_up_proj",
        "model.language_model.layers.*.mlp.experts.down_proj",
    )


def _convert_qwen3_5_moe_causal_lora_targets_to_parameters(_model, lora_modules, target_parameter_patterns):
    return convert_fused_moe_lora_targets(
        lora_modules,
        target_parameter_patterns,
        "model.layers.*.mlp.experts.gate_up_proj",
        "model.layers.*.mlp.experts.down_proj",
    )


#
# NPU branch is opt-in; everything else (CUDA, CPU-only) falls back to the GPU
# generated file. The GPU generated module imports cleanly without an active
# CUDA device, so a CPU-only environment (e.g. CI lint, doc build) can still
# register the class.


@MODELING_REGISTRY.register("qwen3_5_moe")
def register_qwen3_5_moe_modeling(architecture: str):
    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_qwen3_5_moe_npu import (
            Qwen3_5MoeForCausalLM,
            Qwen3_5MoeForConditionalGeneration,
        )
    else:
        from .generated.patched_modeling_qwen3_5_moe_gpu import (
            Qwen3_5MoeForCausalLM,
            Qwen3_5MoeForConditionalGeneration,
        )

    Qwen3_5MoeForConditionalGeneration._convert_lora_targets_to_parameters = staticmethod(
        _convert_qwen3_5_moe_conditional_lora_targets_to_parameters
    )
    Qwen3_5MoeForCausalLM._convert_lora_targets_to_parameters = staticmethod(
        _convert_qwen3_5_moe_causal_lora_targets_to_parameters
    )
    if "ForCausalLM" in architecture:
        return Qwen3_5MoeForCausalLM
    elif "ForConditionalGeneration" in architecture:
        return Qwen3_5MoeForConditionalGeneration
    else:
        return Qwen3_5MoeForCausalLM


@MODELING_REGISTRY.register("qwen3_5_moe_text")
def register_qwen3_5_moe_text_modeling(architecture: str):
    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_qwen3_5_moe_npu import Qwen3_5MoeForCausalLM
    else:
        from .generated.patched_modeling_qwen3_5_moe_gpu import Qwen3_5MoeForCausalLM

    Qwen3_5MoeForCausalLM._convert_lora_targets_to_parameters = staticmethod(
        _convert_qwen3_5_moe_causal_lora_targets_to_parameters
    )
    return Qwen3_5MoeForCausalLM
