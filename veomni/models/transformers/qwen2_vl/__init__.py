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
from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("qwen2_vl")
def register_qwen2_vl_model_config():
    if is_transformers_version_greater_or_equal_to("5.0.0"):
        from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig

        return Qwen2VLConfig
    else:
        from .configuration_qwen2_vl import Qwen2VLConfig, apply_veomni_qwen2vl_patch

        apply_veomni_qwen2vl_patch()

        return Qwen2VLConfig


@MODELING_REGISTRY.register("qwen2_vl")
def register_qwen2_vl_modeling(architecture: str):
    if is_transformers_version_greater_or_equal_to("5.0.0"):
        from .generated.patched_modeling_qwen2_vl_gpu import (
            Qwen2VLForConditionalGeneration,
            Qwen2VLModel,
        )
    else:
        from .modeling_qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLModel, apply_veomni_qwen2vl_patch

        apply_veomni_qwen2vl_patch()

    if "ForConditionalGeneration" in architecture:
        return Qwen2VLForConditionalGeneration
    elif "Model" in architecture:
        return Qwen2VLModel
    else:
        return Qwen2VLForConditionalGeneration
