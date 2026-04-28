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
from ...loader import MODELING_REGISTRY, raise_if_not_migrated_to_v5


@MODELING_REGISTRY.register("deepseek_v3")
def register_deepseek_v3_modeling(architecture: str):
    raise_if_not_migrated_to_v5("deepseek_v3")

    from transformers import (
        DeepseekV3ForCausalLM,
        DeepseekV3ForSequenceClassification,
        DeepseekV3Model,
    )

    from .checkpoint_tensor_converter_v4 import create_deepseek_v3_v4_checkpoint_tensor_converter
    from .modeling_deepseek_v3 import apply_veomni_deepseek_v3_patch

    apply_veomni_deepseek_v3_patch()

    # Stack per-expert HF weights into v4's three 3-D nn.Parameters at load time
    # (PatchDeepseekV3NaiveMoe.gate_proj/up_proj/down_proj). Skips the first
    # `first_k_dense_replace` layers automatically (regex won't match dense layers).
    for model_cls in (DeepseekV3ForCausalLM, DeepseekV3ForSequenceClassification, DeepseekV3Model):
        model_cls._create_checkpoint_tensor_converter = staticmethod(create_deepseek_v3_v4_checkpoint_tensor_converter)

    if "ForCausalLM" in architecture:
        return DeepseekV3ForCausalLM
    elif "ForSequenceClassification" in architecture:
        return DeepseekV3ForSequenceClassification
    elif "Model" in architecture:
        return DeepseekV3Model
    else:
        return DeepseekV3ForCausalLM
