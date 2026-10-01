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

from ...loader import MODELING_REGISTRY


@MODELING_REGISTRY.register("gemma4")
def register_gemma4_modeling(architecture: str | None):
    from .generated.patched_modeling_gemma4_gpu import Gemma4ForConditionalGeneration, Gemma4Model

    architecture = architecture or "Gemma4ForConditionalGeneration"
    if "ForConditionalGeneration" in architecture:
        return Gemma4ForConditionalGeneration
    if "Model" in architecture:
        return Gemma4Model
    return Gemma4ForConditionalGeneration


@MODELING_REGISTRY.register("gemma4_text")
def register_gemma4_text_modeling(architecture: str | None):
    from .generated.patched_modeling_gemma4_gpu import Gemma4ForCausalLM, Gemma4TextModel

    architecture = architecture or "Gemma4ForCausalLM"
    if "ForCausalLM" in architecture:
        return Gemma4ForCausalLM
    if "Model" in architecture:
        return Gemma4TextModel
    return Gemma4ForCausalLM
