"""The Cosmos3 prompt format: one user turn, the checkpoint's own chat template.

Cosmos3 trains no text target, so a sample's only text is the prompt. It is
tokenized the way the reference packs it: the tokenizer's chat template with the
generation prompt, truncated to ``max_sequence_length``, then
``[eos, <|vision_start|>]`` so the latent span follows a generation cue. Every
id is a prompt id, hence ``-100`` labels throughout.

The prompt is priced *before* the two markers. The reference's token counter
charges a sample's eos and vision markers as three tokens on top of the prompt
and the latent patches, and the VAE preprocessor adds exactly those
(``EXTRA_TOKENS_PER_SAMPLE``), so counting the markers here too would charge
them twice.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from ....utils.conversation import _OMNI_TOKEN_LEN, ConversationItem
from ...text_encoder.chat_template import TextEncoderChatTemplate


class Cosmos3ChatTemplate(TextEncoderChatTemplate):
    def __init__(
        self,
        tokenizer: Any,
        *,
        generation_marker_ids: Sequence[int] = (),
        max_sequence_length: int = 2048,
    ):
        super().__init__(tokenizer)
        self.generation_marker_ids = torch.tensor(list(generation_marker_ids), dtype=torch.long)
        self.max_sequence_length = int(max_sequence_length)

    def apply_chat_template(self, sample: list[ConversationItem]) -> list[ConversationItem]:
        """Keep the prompt as one unsupervised row; the clip passes through untouched."""
        out: list[ConversationItem] = []
        for item in sample:
            if item.type != "text":
                out.append(item)
                continue
            if item.role != "user":
                raise ValueError(f"Cosmos3 trains no text target; got a {item.role!r} text item.")
            out.append(self._build_conversation_item("text", str(item.value), "user", loss_mask=0, meta=item.meta))
        return out

    def apply_generation_prompt(self, sample: list[ConversationItem]) -> list[ConversationItem]:
        # Every prompt row already ends in the generation prompt (see :meth:`tokenize`).
        return list(sample)

    def tokenize(self, parts: list[ConversationItem]) -> None:
        for part in parts:
            if part.type != "text":
                continue
            part.meta.pop("loss_mask", None)
            token_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": part.value}],
                tokenize=True,
                add_generation_prompt=True,
                add_vision_id=False,
                return_dict=False,
            )
            input_ids = torch.tensor(list(token_ids), dtype=torch.long)[: self.max_sequence_length]
            price = int(input_ids.numel())
            input_ids = torch.cat([input_ids, self.generation_marker_ids])
            part.value = input_ids
            part.meta["labels"] = torch.full_like(input_ids, -100)
            part.meta["attention_mask"] = torch.ones_like(input_ids)
            part.meta[_OMNI_TOKEN_LEN] = price
