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

"""Config for :class:`LladaUniTextEncoder`.

LLaDA2.0-Uni shares one **unified vocabulary** (text + 16384 image VQ codes at
``[157184, 173567]``) and an *untied* ``lm_head`` (separate from
``embed_tokens``). The dLLM mask token is ``<|mask|>`` = 156895.
"""

from typing import Optional

from ...text_encoder.configuration import TextEncoderConfig


class LladaUniTextEncoderConfig(TextEncoderConfig):
    """TextEncoder config for LLaDA2.0-Uni (unified vocab + GlmImage chat template)."""

    model_type = "llada2_uni_text_encoder"

    def __init__(
        self,
        vocab_size: int = 173568,
        hidden_size: int = 2048,
        tie_word_embeddings: bool = False,
        lm_head_bias: bool = False,
        pad_token_id: Optional[int] = 156892,
        mask_token_id: int = 156895,
        mask_eps: float = 1e-3,
        text_vocab_size: int = 156891,
        image_token_offset: int = 157184,
        image_start_token_id: int = 156901,
        image_end_token_id: int = 156902,
        boi_token_id: int = 156904,
        reserved_token_base: int = 156895,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            tie_word_embeddings=tie_word_embeddings,
            lm_head_bias=lm_head_bias,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.mask_token_id = mask_token_id
        self.mask_eps = mask_eps
        self.text_vocab_size = text_vocab_size
        self.image_token_offset = image_token_offset
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.boi_token_id = boi_token_id
        self.reserved_token_base = reserved_token_base


__all__ = ["LladaUniTextEncoderConfig"]
