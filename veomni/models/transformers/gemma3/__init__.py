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

from ....utils.device import IS_NPU_AVAILABLE
from ...loader import MODELING_REGISTRY


@MODELING_REGISTRY.register("gemma3_text")
def register_gemma3_text_modeling(architecture: str | None):
    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_gemma3_npu import Gemma3ForCausalLM, Gemma3TextModel
    else:
        from .generated.patched_modeling_gemma3_gpu import Gemma3ForCausalLM, Gemma3TextModel

    architecture = architecture or "Gemma3ForCausalLM"
    if "ForCausalLM" in architecture:
        return Gemma3ForCausalLM
    if "Model" in architecture:
        return Gemma3TextModel
    return Gemma3ForCausalLM


@MODELING_REGISTRY.register("gemma3")
def register_gemma3_modeling(architecture: str | None):
    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_gemma3_npu import Gemma3ForConditionalGeneration, Gemma3Model
    else:
        from .generated.patched_modeling_gemma3_gpu import Gemma3ForConditionalGeneration, Gemma3Model

    architecture = architecture or "Gemma3ForConditionalGeneration"
    if "ForConditionalGeneration" in architecture:
        return Gemma3ForConditionalGeneration
    if "Model" in architecture:
        return Gemma3Model
    return Gemma3ForConditionalGeneration
