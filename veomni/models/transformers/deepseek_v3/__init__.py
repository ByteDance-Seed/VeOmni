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
from ....utils.device import IS_NPU_AVAILABLE
from ...loader import MODELING_REGISTRY


@MODELING_REGISTRY.register("deepseek_v3")
def register_deepseek_v3_modeling(architecture: str):
    from .checkpoint_tensor_converter import create_deepseek_v3_checkpoint_tensor_converter
    from .device_patch import apply_veomni_deepseek_v3_device_patch

    if IS_NPU_AVAILABLE:
        from .generated import patched_modeling_deepseek_v3_npu as gen
    else:
        from .generated import patched_modeling_deepseek_v3_gpu as gen

    apply_veomni_deepseek_v3_device_patch(gen)

    DeepseekV3ForCausalLM = gen.DeepseekV3ForCausalLM
    DeepseekV3ForSequenceClassification = gen.DeepseekV3ForSequenceClassification
    DeepseekV3ForTokenClassification = gen.DeepseekV3ForTokenClassification
    DeepseekV3Model = gen.DeepseekV3Model

    for model_cls in (
        DeepseekV3ForCausalLM,
        DeepseekV3ForSequenceClassification,
        DeepseekV3ForTokenClassification,
        DeepseekV3Model,
    ):
        model_cls._create_checkpoint_tensor_converter = staticmethod(create_deepseek_v3_checkpoint_tensor_converter)

    if "ForCausalLM" in architecture:
        return DeepseekV3ForCausalLM
    elif "ForTokenClassification" in architecture:
        return DeepseekV3ForTokenClassification
    elif "ForSequenceClassification" in architecture:
        return DeepseekV3ForSequenceClassification
    elif "Model" in architecture:
        return DeepseekV3Model
    else:
        return DeepseekV3ForCausalLM
