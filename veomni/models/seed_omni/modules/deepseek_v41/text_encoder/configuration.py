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

"""Config for :class:`DeepseekV41TextEncoder`."""

from typing import Optional, Union

from ...base.text_encoder.configuration import TextEncoderConfig


class DeepseekV41TextEncoderConfig(TextEncoderConfig):
    """``embed_tokens`` + ``lm_head`` + the V4.1 chat template's defaults.

    ``tie_word_embeddings`` is ``False``: the released checkpoint carries ``embed``
    and ``head`` as separate tensors (and the head is stored in a different dtype),
    so tying them would quietly train one matrix for two jobs.

    ``thinking_mode`` / ``reasoning_effort`` are the run-wide defaults the chat
    template starts from; a sample overrides them from its own ``meta``.
    ``image_token_id`` is not used to *build* anything here -- the vision module
    owns the span -- but it is what the n-gram module's mask and any id-level
    debugging key off, so it belongs with the tokenizer-side config.
    """

    model_type = "deepseek_v41_text_encoder"

    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 5120,
        tie_word_embeddings: bool = False,
        lm_head_bias: bool = False,
        thinking_mode: str = "thinking",
        reasoning_effort: Optional[Union[str, int]] = None,
        image_token_id: int = 129264,
        pad_token_id: int = 2,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            tie_word_embeddings=tie_word_embeddings,
            lm_head_bias=lm_head_bias,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.thinking_mode = thinking_mode
        self.reasoning_effort = reasoning_effort
        self.image_token_id = image_token_id


__all__ = ["DeepseekV41TextEncoderConfig"]
