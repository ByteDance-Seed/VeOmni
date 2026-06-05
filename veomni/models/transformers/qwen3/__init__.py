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


@MODELING_REGISTRY.register("qwen3")
def register_qwen3_modeling(architecture: str):
    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_qwen3_npu import (
            Qwen3ForCausalLM,
            Qwen3ForSequenceClassification,
            Qwen3ForTokenClassification,
            Qwen3Model,
        )
    else:
        from .generated.patched_modeling_qwen3_gpu import (
            Qwen3ForCausalLM,
            Qwen3ForSequenceClassification,
            Qwen3ForTokenClassification,
            Qwen3Model,
        )

    if "ForCausalLM" in architecture:
        return Qwen3ForCausalLM
    elif "ForTokenClassification" in architecture:
        return Qwen3ForTokenClassification
    elif "ForSequenceClassification" in architecture:
        return Qwen3ForSequenceClassification
    elif "Model" in architecture:
        return Qwen3Model
    else:
        return Qwen3ForCausalLM
