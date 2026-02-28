# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
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


import transformers.models.qwen2_5_omni.configuration_qwen2_5_omni as hf_qwen25omni
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniConfig as _Qwen2_5OmniConfig


class Qwen2_5OmniConfig(_Qwen2_5OmniConfig):
    def __init__(
        self,
        **kwargs,
    ):
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(tie_word_embeddings=False, **kwargs)


def apply_veomni_qwen25_omni_patch():
    hf_qwen25omni.Qwen2_5OmniConfig = Qwen2_5OmniConfig


__all__ = ["Qwen2_5OmniConfig"]
