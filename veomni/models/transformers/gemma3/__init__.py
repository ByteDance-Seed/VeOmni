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


@MODELING_REGISTRY.register("gemma3_text")
def register_gemma3_text_modeling(architecture: str | None):
    from .generated.patched_modeling_gemma3_gpu import Gemma3ForCausalLM, Gemma3TextModel

    architecture = architecture or "Gemma3ForCausalLM"
    if "ForCausalLM" in architecture:
        return Gemma3ForCausalLM
    if "Model" in architecture:
        return Gemma3TextModel
    return Gemma3ForCausalLM
