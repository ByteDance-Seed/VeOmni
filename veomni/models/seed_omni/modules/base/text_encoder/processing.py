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

"""Shared CPU worker for every SeedOmni text encoder module."""

from __future__ import annotations

from typing import Any

from ....utils.conversation import ConversationItem
from ...module_processing_base import ModulePreprocessorBase
from .chat_template import TextEncoderChatTemplate


class TextEncoderPreprocessor(ModulePreprocessorBase):
    """Worker-side ``apply_chat_template`` → tokenize → merge for text encoders."""

    def __init__(self, chat_template: TextEncoderChatTemplate) -> None:
        self._chat_template = chat_template
        self._tokenizer = chat_template.tokenizer

    @classmethod
    def build_chat_template(
        cls,
        module_path: str,
        *,
        config_overrides: dict[str, Any] | None = None,
    ) -> TextEncoderChatTemplate:
        """Build this module's chat template from its checkpoint subfolder."""
        del module_path, config_overrides
        raise NotImplementedError(f"{cls.__name__} must implement build_chat_template()")

    @classmethod
    def from_pretrained(
        cls,
        module_path: str,
        *,
        config_overrides: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Build straight from the checkpoint dir — no model instance needed."""
        return cls(cls.build_chat_template(module_path, config_overrides=config_overrides), **kwargs)

    def _tokenize_conversation_kwargs(self, inference: bool, **kwargs: Any) -> dict[str, Any]:
        """Extra kwargs forwarded to :meth:`TextEncoderChatTemplate.tokenize_conversation`."""
        del kwargs
        return {"add_generation_prompt": inference}

    def preprocess_conversations(
        self,
        conversation_list: list[list[ConversationItem]],
        inference: bool = False,
        **kwargs: Any,
    ) -> None:
        tc_kwargs = self._tokenize_conversation_kwargs(inference, **kwargs)
        for sample in conversation_list or []:
            parts = self._chat_template.tokenize_conversation(sample, **tc_kwargs)
            sample.clear()
            sample.extend(parts)

    def __call__(self, batch: dict[str, Any], inference: bool = False, **kwargs: Any) -> None:
        self.preprocess_conversations(batch["conversation_list"], inference=inference, **kwargs)


__all__ = ["TextEncoderPreprocessor"]
