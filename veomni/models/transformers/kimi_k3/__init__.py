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
from veomni.models.registry import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY
from veomni.utils.device import IS_NPU_AVAILABLE


@MODEL_CONFIG_REGISTRY.register("kimi_k3")
def register_kimi_k3_config():
    from .source.configuration_kimi_k3 import KimiK3Config

    return KimiK3Config


@MODEL_CONFIG_REGISTRY.register("kimi_linear")
def register_kimi_linear_config():
    from .source.configuration_kimi_k3 import KimiLinearConfig

    return KimiLinearConfig


def _load_kimi_k3_modeling():
    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_kimi_k3_npu import (
            KimiK3ForConditionalGeneration,
            KimiLinearForCausalLM,
            KimiLinearModel,
        )
    else:
        from .generated.patched_modeling_kimi_k3_gpu import (
            KimiK3ForConditionalGeneration,
            KimiLinearForCausalLM,
            KimiLinearModel,
        )
    return KimiK3ForConditionalGeneration, KimiLinearForCausalLM, KimiLinearModel


@MODELING_REGISTRY.register("kimi_k3")
def register_kimi_k3_modeling(architecture: str):
    from .checkpoint_tensor_converter import (
        convert_kimi_k3_fqn_to_index_mapping,
        create_kimi_k3_checkpoint_tensor_converter,
    )

    KimiK3ForConditionalGeneration, KimiLinearForCausalLM, KimiLinearModel = _load_kimi_k3_modeling()

    for model_cls in (
        KimiK3ForConditionalGeneration,
        KimiLinearForCausalLM,
        KimiLinearModel,
    ):
        model_cls._create_checkpoint_tensor_converter = staticmethod(create_kimi_k3_checkpoint_tensor_converter)
        model_cls._convert_fqn_to_index_mapping = staticmethod(convert_kimi_k3_fqn_to_index_mapping)

    if "ForConditionalGeneration" in architecture:
        return KimiK3ForConditionalGeneration
    elif "ForCausalLM" in architecture:
        return KimiLinearForCausalLM
    elif "Model" in architecture:
        return KimiLinearModel
    else:
        return KimiK3ForConditionalGeneration


@MODELING_REGISTRY.register("kimi_linear")
def register_kimi_linear_modeling(architecture: str):
    return register_kimi_k3_modeling(architecture)
