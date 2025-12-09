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
from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("deepseek_v3")
def register_deepseek_v3_config():
    from .configuration_deepseek import DeepseekV3Config

    return DeepseekV3Config


@MODELING_REGISTRY.register("deepseek_v3")
def register_deepseek_v3_modeling(architecture: str):
    from .modeling_deepseek import DeepseekV3ForCausalLM, DeepseekV3ForSequenceClassification, DeepseekV3Model

    if "ForCausalLM" in architecture:
        return DeepseekV3ForCausalLM
    elif "ForSequenceClassification" in architecture:
        return DeepseekV3ForSequenceClassification
    elif "Model" in architecture:
        return DeepseekV3Model
    else:  # None
        return DeepseekV3ForCausalLM
