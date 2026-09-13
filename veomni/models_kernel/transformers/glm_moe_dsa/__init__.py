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

"""GLM-MoE-DSA modeling that calls local ``VeomniOp`` handles."""

from veomni.models_kernel.registry import MODELING_REGISTRY
from veomni.utils.device import IS_NPU_AVAILABLE


@MODELING_REGISTRY.register("glm_moe_dsa")
def register_glm_moe_dsa_modeling(architecture: str):
    from .checkpoint_tensor_converter import (
        convert_glm_moe_dsa_fqn_to_index_mapping,
        create_glm_moe_dsa_checkpoint_tensor_converter,
    )

    if IS_NPU_AVAILABLE:
        from .generated import patched_modeling_glm_moe_dsa_npu as gen
    else:
        from .generated import patched_modeling_glm_moe_dsa_gpu as gen

    GlmMoeDsaForCausalLM = gen.GlmMoeDsaForCausalLM
    GlmMoeDsaModel = gen.GlmMoeDsaModel

    for model_cls in (GlmMoeDsaForCausalLM, GlmMoeDsaModel):
        model_cls._create_checkpoint_tensor_converter = staticmethod(create_glm_moe_dsa_checkpoint_tensor_converter)
        model_cls._convert_fqn_to_index_mapping = staticmethod(convert_glm_moe_dsa_fqn_to_index_mapping)

    if "ForCausalLM" in architecture:
        return GlmMoeDsaForCausalLM
    elif "Model" in architecture:
        return GlmMoeDsaModel
    else:
        return GlmMoeDsaForCausalLM
