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


@MODELING_REGISTRY.register("deepseek_v4")
def register_deepseek_v4_modeling(architecture: str):
    from .checkpoint_tensor_converter import (
        convert_deepseek_v4_fqn_to_index_mapping,
        create_deepseek_v4_checkpoint_tensor_converter,
    )
    from .generated import patched_modeling_deepseek_v4_gpu as gen

    DeepseekV4ForCausalLM = gen.DeepseekV4ForCausalLM
    DeepseekV4Model = gen.DeepseekV4Model

    for model_cls in (DeepseekV4ForCausalLM, DeepseekV4Model):
        model_cls._create_checkpoint_tensor_converter = staticmethod(create_deepseek_v4_checkpoint_tensor_converter)
        model_cls._convert_fqn_to_index_mapping = staticmethod(convert_deepseek_v4_fqn_to_index_mapping)

    if "ForCausalLM" in architecture:
        return DeepseekV4ForCausalLM
    elif "Model" in architecture:
        return DeepseekV4Model
    else:
        return DeepseekV4ForCausalLM
