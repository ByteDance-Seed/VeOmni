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
from veomni.models.loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("qwen3_next")
def register_qwen3_next_config():
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig

    return Qwen3NextConfig


@MODELING_REGISTRY.register("qwen3_next")
def register_qwen3_next_modeling(architecture: str):
    from .modeling_qwen3_next import (
        Qwen3NextForCausalLM,
        Qwen3NextForQuestionAnswering,
        Qwen3NextForSequenceClassification,
        Qwen3NextForTokenClassification,
        Qwen3NextModel,
    )

    if "ForCausalLM" in architecture:
        return Qwen3NextForCausalLM
    elif "ForSequenceClassification" in architecture:
        return Qwen3NextForSequenceClassification
    elif "ForTokenClassification" in architecture:
        return Qwen3NextForTokenClassification
    elif "ForQuestionAnswering" in architecture:
        return Qwen3NextForQuestionAnswering
    elif "Model" in architecture:
        return Qwen3NextModel
    else:
        return Qwen3NextForCausalLM
