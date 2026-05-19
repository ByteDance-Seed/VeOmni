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
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniThinkerConfig

from ..base import BaseFoundationConfigMixin


class Qwen25OmniFoundationModelConfig(BaseFoundationConfigMixin, Qwen2_5OmniThinkerConfig):
    model_type = "qwen2_5_omni_foundation"

    def __init__(self, hidden_size=None, vocab_size=None, **kwargs):
        if hidden_size is None and "text_config" in kwargs:
            hidden_size = kwargs["text_config"]["hidden_size"]
        if vocab_size is None and "text_config" in kwargs:
            vocab_size = kwargs["text_config"]["vocab_size"]

        """
            Qwen2.5Omni didn't set tie_word_embeddings, so it is set to True by default.
            However, Qwen2.5Omnimodel didn't set get_output_embeddings, so the `embed_tokens` can't tie with `lm_head`
            Logically, `tie_word_embeddings=False`
        """
        kwargs.pop("tie_word_embeddings", None)

        super().__init__(hidden_size=hidden_size, vocab_size=vocab_size, tie_word_embeddings=False, **kwargs)
