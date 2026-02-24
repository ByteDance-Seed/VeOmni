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
from ....utils.import_utils import is_transformers_version_greater_or_equal_to
from ...loader import MODELING_REGISTRY


# qwen3_5 is added in transformers 5.2.0.
if is_transformers_version_greater_or_equal_to("5.2.0"):

    @MODELING_REGISTRY.register("qwen3_5")
    def register_qwen3_5_modeling(architecture: str):
        from .generated.patched_modeling_qwen3_5_gpu import Qwen3_5ForCausalLM, Qwen3_5Model

        if "ForCausalLM" in architecture:
            return Qwen3_5ForCausalLM
        elif "Model" in architecture:
            return Qwen3_5Model
        else:
            return Qwen3_5ForCausalLM
