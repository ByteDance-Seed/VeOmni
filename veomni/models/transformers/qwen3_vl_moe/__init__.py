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
from ....utils.import_utils import is_transformers_version_greater_or_equal_to
from ...loader import MODELING_REGISTRY


@MODELING_REGISTRY.register("qwen3_vl_moe")
def register_qwen3_vl_moe_modeling(architecture: str):
    if is_transformers_version_greater_or_equal_to("5.2.0"):
        from .checkpoint_tensor_converter import create_qwen3_vl_moe_checkpoint_tensor_converter

        if IS_NPU_AVAILABLE:
            from .generated.patched_modeling_qwen3_vl_moe_npu import (
                Qwen3VLMoeForConditionalGeneration,
                Qwen3VLMoeModel,
                Qwen3VLMoeTextModel,
            )
        else:
            from .generated.patched_modeling_qwen3_vl_moe_gpu import (
                Qwen3VLMoeForConditionalGeneration,
                Qwen3VLMoeModel,
                Qwen3VLMoeTextModel,
            )

        for model_cls in (Qwen3VLMoeForConditionalGeneration, Qwen3VLMoeModel, Qwen3VLMoeTextModel):
            model_cls._create_checkpoint_tensor_converter = staticmethod(
                create_qwen3_vl_moe_checkpoint_tensor_converter
            )
    else:
        from .modeling_qwen3_vl_moe import (
            Qwen3VLMoeForConditionalGeneration,
            Qwen3VLMoeModel,
            apply_veomni_qwen3vlmoe_patch,
        )

        apply_veomni_qwen3vlmoe_patch()

    if "ForConditionalGeneration" in architecture:
        return Qwen3VLMoeForConditionalGeneration
    elif "Model" in architecture:
        return Qwen3VLMoeModel
    else:
        return Qwen3VLMoeForConditionalGeneration
