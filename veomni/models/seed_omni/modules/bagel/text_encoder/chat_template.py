"""BAGEL chat template for text/materialized image prompt rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from ....utils.conversation import ConversationItem, is_dummy
from ...base.text_encoder.chat_template import ChatMarkers, TextEncoderChatTemplate
from ..sources import BAGEL_SIGLIP_CONTEXT, BAGEL_VAE_CONTEXT
from .processing import (
    copy_image_item,
    is_bagel_vision_marker,
)


_OMNI_TOKENIZED = "_omni_tokenized"


@dataclass(frozen=True)
class BagelChatMarkers(ChatMarkers):
    vision_start_token: str
    vision_end_token: str


class BagelChatTemplate(TextEncoderChatTemplate):
    """BAGEL-specific template over V2 conversation rows.

    The template is intentionally row-preserving: text rows are tokenized, and
    image rows are bracketed by marker text rows, but marker-image-marker packing
    remains a MoT responsibility.
    """

    chat_markers: BagelChatMarkers

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__(tokenizer)

        bos_token = "<|im_start|>"
        eos_token = "<|im_end|>"
        vision_start_token = "<|vision_start|>"
        vision_end_token = "<|vision_end|>"

        self.bos_token_id = self._resolve_token_id_from_token(tokenizer, bos_token)
        self.eos_token_id = self._resolve_token_id_from_token(tokenizer, eos_token)
        self.vision_start_token_id = self._resolve_token_id_from_token(tokenizer, vision_start_token)
        self.vision_end_token_id = self._resolve_token_id_from_token(tokenizer, vision_end_token)

        self.chat_markers = BagelChatMarkers(
            bos_token=bos_token,
            eos_token=eos_token,
            vision_start_token=vision_start_token,
            vision_end_token=vision_end_token,
        )

    @staticmethod
    def _resolve_token_id_from_token(tokenizer: PreTrainedTokenizerBase, token: str) -> int:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
            raise ValueError(f"BAGEL tokenizer is missing required token: {token!r}.")
        return int(token_id)

    def tokenize_conversation(
        self,
        sample: list[ConversationItem],
        *,
        add_generation_prompt: bool = False,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> list[ConversationItem]:
        parts = self.apply_chat_template(
            sample,
            inference=add_generation_prompt,
            generation_kwargs=generation_kwargs,
        )
        if add_generation_prompt:
            parts = self.apply_generation_prompt(parts)
        self.tokenize(parts)
        return self.merge_text_embeds(parts)

    def apply_chat_template(
        self,
        sample: list[ConversationItem],
        *,
        inference: bool = False,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> list[ConversationItem]:
        infer_type = None if generation_kwargs is None else generation_kwargs.get("infer_type")
        routed = self._route_prompt_context_images(sample, inference=inference, infer_type=infer_type)
        return self._insert_prompt_context_image_markers(routed)

    def apply_generation_prompt(self, sample: list[ConversationItem]) -> list[ConversationItem]:
        return sample

    def tokenize(self, parts: list[ConversationItem]) -> None:
        for item in parts:
            if item.type != "text" or is_dummy(item) or item.meta.get(_OMNI_TOKENIZED):
                continue

            token_ids = self._token_ids_for_text_item(item)
            loss_mask = int(item.meta.pop("loss_mask", int(item.role == "assistant")))
            item.value = token_ids
            item.meta["input_ids"] = token_ids.detach()
            item.meta["attention_mask"] = torch.ones_like(token_ids, dtype=torch.long)
            item.meta["labels"] = (
                token_ids.detach().clone() if loss_mask else torch.full_like(token_ids, -100, dtype=torch.long)
            )
            item.meta[_OMNI_TOKENIZED] = True

            if (
                item.value in {self.chat_markers.bos_token, self.chat_markers.eos_token}
                and int(token_ids.numel()) != 1
            ):
                raise ValueError("BAGEL start/end tokens must tokenize to exactly one token.")

    @staticmethod
    def merge_text_embeds(parts: list[ConversationItem]) -> list[ConversationItem]:
        return parts

    def _token_ids_for_text_item(self, item: ConversationItem) -> torch.Tensor:
        value = item.value
        if torch.is_tensor(value):
            token_ids = value.detach().reshape(-1).to(dtype=torch.long)
        else:
            token_ids = torch.tensor(
                self.tokenizer(str(value), add_special_tokens=False)["input_ids"],
                dtype=torch.long,
            )
        return token_ids

    def _route_prompt_context_images(
        self,
        sample: list[ConversationItem],
        *,
        inference: bool,
        infer_type: object,
    ) -> list[ConversationItem]:
        routed: list[ConversationItem] = []
        for item in sample:
            if item.type == "text" and not is_bagel_vision_marker(item):
                if not item.meta.get(_OMNI_TOKENIZED) and not torch.is_tensor(item.value):
                    item.value = f"{self.chat_markers.bos_token}{item.value}{self.chat_markers.eos_token}"
                    item.meta["loss_mask"] = int(item.role == "assistant")
                routed.append(item)
                continue

            if item.type != "image" or is_dummy(item):
                routed.append(item)
                continue

            if item.source in {BAGEL_SIGLIP_CONTEXT, BAGEL_VAE_CONTEXT}:
                routed.append(item)
                continue

            if item.role == "assistant":
                item.source = BAGEL_VAE_CONTEXT
                routed.append(item)
                continue

            if item.role == "user" and inference and infer_type == "infer_edit":
                vae_item = copy_image_item(item)
                vae_item.source = BAGEL_VAE_CONTEXT
                item.source = BAGEL_SIGLIP_CONTEXT
                routed.extend([vae_item, item])
                continue

            # Training edit detection is intentionally not implemented yet. Raw
            # user images outside infer_edit are the SigLIP prompt/context branch.
            if item.role == "user":
                item.source = BAGEL_SIGLIP_CONTEXT
            routed.append(item)
        return routed

    def _insert_prompt_context_image_markers(self, sample: list[ConversationItem]) -> list[ConversationItem]:
        out: list[ConversationItem] = []

        index = 0
        while index < len(sample):
            item = sample[index]

            if item.type != "image" or is_dummy(item):
                out.append(item)
                index += 1
                continue

            # Add vision marker for image items: <vision_start> + image + <vision_end>.
            if not (out and out[-1].source == item.source and is_bagel_vision_marker(out[-1], source=item.source)):
                out.append(
                    ConversationItem(
                        type="text",
                        value=self.chat_markers.vision_start_token,
                        role=item.role,
                        source=item.source,
                        meta={"loss_mask": 0},
                    )
                )

            out.append(item)

            if (
                index + 1 < len(sample)
                and sample[index + 1].source == item.source
                and is_bagel_vision_marker(sample[index + 1], source=item.source)
            ):
                out.append(sample[index + 1])
                index += 2
            else:
                out.append(
                    ConversationItem(
                        type="text",
                        value=self.chat_markers.vision_end_token,
                        role=item.role,
                        source=item.source,
                        meta={"loss_mask": 0},
                    )
                )
                index += 1

        return out


__all__ = [
    "BagelChatMarkers",
    "BagelChatTemplate",
]
