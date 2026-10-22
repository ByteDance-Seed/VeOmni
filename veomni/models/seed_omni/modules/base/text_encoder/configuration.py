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

"""Configuration for :class:`TextEncoder`."""

from typing import Optional

from ...module_configuration_base import OmniModuleConfig


class TextEncoderConfig(OmniModuleConfig):
    """Config for the generic word-token embedding + LM head module."""

    model_type = "text_encoder"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        tie_word_embeddings: bool = True,
        lm_head_bias: bool = False,
        pad_token_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.tie_word_embeddings = tie_word_embeddings
        self.lm_head_bias = lm_head_bias
        self.pad_token_id = pad_token_id


__all__ = ["TextEncoderConfig"]
