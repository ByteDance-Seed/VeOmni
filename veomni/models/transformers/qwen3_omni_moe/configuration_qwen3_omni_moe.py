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


import transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe as hf_qwen3_omni_moe
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig as _Qwen3OmniMoeConfig,
)


class Qwen3OmniMoeConfig(_Qwen3OmniMoeConfig):
    def __init__(
        self,
        **kwargs,
    ):
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(tie_word_embeddings=False, **kwargs)


def apply_veomni_qwen3_omni_moe_patch():
    hf_qwen3_omni_moe.Qwen3OmniMoeConfig = Qwen3OmniMoeConfig


__all__ = ["Qwen3OmniMoeConfig"]
