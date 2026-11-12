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


@MODEL_CONFIG_REGISTRY.register("llada_moe")
def register_llada_moe_config():
    from .configuration_llada_moe import LLaDAMoEConfig

    return LLaDAMoEConfig


@MODELING_REGISTRY.register("llada_moe")
def register_llada_moe_modeling(architecture: str):
    from .modeling_llada_moe import LLaDAMoEForCausalLM, LLaDAMoEModel

    if architecture.endswith("Model"):
        return LLaDAMoEModel
    return LLaDAMoEForCausalLM
