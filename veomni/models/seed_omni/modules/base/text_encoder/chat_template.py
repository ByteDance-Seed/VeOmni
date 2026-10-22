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

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from ....utils.conversation import ConversationItem, ItemRole, ItemType, ItemValue


@dataclass(frozen=True)
class ChatMarkers:
    """Default wire-format markers shared by every text encoder (bos / eos only)."""

    bos_token: str | None
    eos_token: str | None


class TextEncoderChatTemplate:
    chat_markers: ChatMarkers

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.chat_markers = ChatMarkers(
            bos_token=getattr(tokenizer, "bos_token", None),
            eos_token=getattr(tokenizer, "eos_token", None),
        )
        self.bos_token_id = self._resolve_token_id(tokenizer, "bos_token_id")
        self.eos_token_id = self._resolve_token_id(tokenizer, "eos_token_id")

    @staticmethod
    def _resolve_token_id(tokenizer: Any, attr: str) -> int | None:
        token_id = getattr(tokenizer, attr, None)
        return int(token_id) if token_id is not None else None

    @abstractmethod
    def apply_chat_template(self, sample: list[ConversationItem]) -> list[ConversationItem]:
        """Apply the chat template only — produce ``str`` rows with ``meta['loss_mask']``."""
        raise NotImplementedError

    @abstractmethod
    def apply_generation_prompt(self, sample: list[ConversationItem]) -> list[ConversationItem]:
        """Append the assistant generation prefix to a templated prompt (inference)."""
        raise NotImplementedError

    def tokenize_conversation(
        self,
        sample: list[ConversationItem],
        *,
        add_generation_prompt: bool = False,
    ) -> list[ConversationItem]:
        """Full text pipeline: apply chat template → tokenize → merge tokenized text."""
        parts = self.apply_chat_template(sample)
        if add_generation_prompt:
            parts = self.apply_generation_prompt(parts)
        self.tokenize(parts)
        return self.merge_text_embeds(parts)

    def _build_conversation_item(
        self,
        type: ItemType,
        value: ItemValue,
        role: ItemRole,
        loss_mask: int | None = None,
        meta: dict | None = None,
    ) -> ConversationItem:
        """Build one conversation row; ``text`` rows always get ``meta["loss_mask"]``."""
        part_meta = dict(meta or {})
        if type == "text":
            part_meta["loss_mask"] = int(role == "assistant") if loss_mask is None else int(loss_mask)
        return ConversationItem(type=type, value=value, role=role, meta=part_meta)

    @staticmethod
    def pack_input_ids(parts: list[ConversationItem]) -> list[torch.Tensor]:
        """Collect ``type='text'`` token-id tensors (``value``); one tensor per text row."""
        return [part.value for part in parts if part.type == "text"]

    def tokenize(self, parts: list[ConversationItem]) -> None:
        """Tokenize each ``text`` part in place: ``str`` value → token-id tensor."""
        for part in parts:
            if part.type == "text":
                text = part.value
                loss_mask = int(part.meta.pop("loss_mask"))
                input_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
                labels = input_ids if loss_mask else [-100] * len(input_ids)
                part.value = torch.tensor(input_ids, dtype=torch.long)
                part.meta["labels"] = torch.tensor(labels, dtype=torch.long)
                part.meta["attention_mask"] = torch.ones(len(input_ids), dtype=torch.long)

    @staticmethod
    def merge_text_embeds(parts: list[ConversationItem]) -> list[ConversationItem]:
        """Merge adjacent same-role ``text`` parts (concat ids / labels / mask)."""
        merged: list[ConversationItem] = []
        for part in parts:
            if merged and merged[-1].type == "text" and part.type == "text" and merged[-1].role == part.role:
                prev = merged[-1]
                prev.value = torch.cat([prev.value, part.value])
                prev.meta["labels"] = torch.cat([prev.meta["labels"], part.meta["labels"]])
                prev.meta["attention_mask"] = torch.cat([prev.meta["attention_mask"], part.meta["attention_mask"]])
                continue
            merged.append(part)
        return merged


__all__ = ["ChatMarkers", "TextEncoderChatTemplate"]
