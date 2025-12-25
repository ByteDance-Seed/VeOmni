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
from ...loader import MODELING_REGISTRY


@MODELING_REGISTRY.register("qwen2")
def register_qwen2_modeling(architecture: str):
    from .modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model

    if "ForCausalLM" in architecture:
        return Qwen2ForCausalLM
    elif "Model" in architecture:
        return Qwen2Model
    else:
        return Qwen2ForCausalLM
