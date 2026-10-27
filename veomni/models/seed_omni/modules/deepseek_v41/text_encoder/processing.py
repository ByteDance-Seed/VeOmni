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

"""Worker-side preprocessor for :class:`DeepseekV41TextEncoder`."""

from __future__ import annotations

from typing import Any

from .....auto import build_tokenizer
from ...text_encoder.chat_template import TextEncoderChatTemplate
from ...text_encoder.processing import TextEncoderPreprocessor
from .chat_template import DeepseekV41ChatTemplate


class DeepseekV41TextEncoderPreprocessor(TextEncoderPreprocessor):
    """DeepSeek-V4.1 chat template + tokenize worker.

    ``thinking_mode`` / ``reasoning_effort`` come from the module's own
    ``config.json`` via ``config_overrides``, so a preprocessor built in a
    DataLoader worker renders the same wire format the live model was configured
    with -- the two are built independently of each other and would otherwise only
    disagree at loss time.
    """

    @classmethod
    def build_chat_template(
        cls,
        module_path: str,
        *,
        config_overrides: dict[str, Any] | None = None,
    ) -> TextEncoderChatTemplate:
        from .configuration import DeepseekV41TextEncoderConfig

        config = DeepseekV41TextEncoderConfig.from_pretrained(module_path, **(config_overrides or {}))
        return DeepseekV41ChatTemplate(
            build_tokenizer(module_path),
            thinking_mode=config.thinking_mode,
            reasoning_effort=config.reasoning_effort,
        )


__all__ = ["DeepseekV41TextEncoderPreprocessor"]
