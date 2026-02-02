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

import transformers.models.qwen2_5_vl.configuration_qwen2_5_vl as hf_qwen25vl
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import PretrainedConfig, Qwen2_5_VLConfig


# https://github.com/huggingface/transformers/pull/41758
def Qwen2_5_VLConfig___getattribute__(self: Qwen2_5_VLConfig, key: str):
    if "text_config" in PretrainedConfig.__getattribute__(self, "__dict__") and key not in [
        "dtype",
        "_attn_implementation_internal",
        "_name_or_path",
        "model_type",
    ]:
        text_config = PretrainedConfig.__getattribute__(self, "text_config")
        if key in text_config.__dict__:
            return getattr(text_config, key)

    return PretrainedConfig.__getattribute__(self, key)


def apply_veomni_qwen25_vl_patch():
    hf_qwen25vl.Qwen2_5_VLConfig.__getattribute__ = Qwen2_5_VLConfig___getattribute__
