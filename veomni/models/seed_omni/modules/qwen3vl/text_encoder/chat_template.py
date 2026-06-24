"""Qwen3-VL ChatML template (text + image + video) as readable Python.

Mirrors the upstream ``chat_template.json``:

* each turn is wrapped in ``<|im_start|>{role}\\n … <|im_end|>\\n``;
* image / video become ``<|vision_start|><|image_pad|><|vision_end|>`` /
  ``<|vision_start|><|video_pad|><|vision_end|>`` — in the V2 segment model the
  ``<|*_pad|>`` run is *not* tokenized; the sibling ``image`` / ``video`` item
  already carries the merged vision tokens, so the template emits
  ``<|vision_start|>`` text · the media item · ``<|vision_end|>`` text.

Qwen3-VL has no audio modality (audio-in-video is an Omni feature — see
``design.md`` § av-video, design-only).

Reuses :class:`TextEncoderChatTemplate` for tokenize / merge / pack; only the
ChatML templating (:meth:`Qwen3VLChatTemplate.apply_chat_template`) and the
generation prompt are model-specific.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ....utils.conversation import ConversationItem
from ...base.text_encoder.chat_template import TextEncoderChatTemplate


@dataclass(frozen=True)
class Qwen3VLChatMarkers:
    """Wire-format ChatML markers (fixed literals — not tokenizer-dependent)."""

    im_start_token: str = "<|im_start|>"
    im_end_token: str = "<|im_end|>"
    assistant_prefix: str = "<|im_start|>assistant\n"
    vision_start_token: str = "<|vision_start|>"
    vision_end_token: str = "<|vision_end|>"


class Qwen3VLChatTemplate(TextEncoderChatTemplate):
    chat_markers: Qwen3VLChatMarkers

    def __init__(self, tokenizer: Any):
        super().__init__(tokenizer)  # resolves bos / eos markers + ids
        self.chat_markers = Qwen3VLChatMarkers()
        # Extra ChatML stop token (eos_token_id comes from the base).
        self.im_end_token_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))

    def apply_chat_template(self, sample: list[ConversationItem]) -> list[ConversationItem]:
        """Apply Qwen3-VL ChatML to a raw conversation (text + image + video)."""
        markers = self.chat_markers
        out: list[ConversationItem] = []
        dummy_parts: list[ConversationItem] = []
        prev_role: str | None = None

        def close_turn(role: str) -> None:
            out.append(
                self._build_conversation_item(
                    "text", markers.im_end_token + "\n", role, loss_mask=int(role == "assistant")
                )
            )

        for item in sample:
            role = item.role
            if role == "dummy":
                dummy_parts.append(item)
                continue

            if role != prev_role:
                if prev_role is not None:
                    close_turn(prev_role)
                out.append(
                    self._build_conversation_item("text", markers.im_start_token + role + "\n", role, loss_mask=0)
                )
                prev_role = role

            if item.type == "text":
                out.append(self._build_conversation_item("text", str(item.value), role))
            elif item.type in ("image", "video"):
                # Image and video both wrap in <|vision_start|> … <|vision_end|>
                # (the model uses <|image_pad|> / <|video_pad|> inside). Qwen3-VL has
                # no audio modality — audio-in-video is an Omni feature (design-only,
                # see design.md § av-video).
                out.append(self._build_conversation_item("text", markers.vision_start_token, role, loss_mask=0))
                out.append(item)  # media row passed through verbatim (keeps value/source/meta)
                out.append(self._build_conversation_item("text", markers.vision_end_token, role, loss_mask=0))
            else:
                raise ValueError(f"Qwen3-VL text encoder only supports text/image/video items, got {item.type!r}")

        if prev_role is not None:
            close_turn(prev_role)
        out.extend(dummy_parts)
        return out

    def apply_generation_prompt(self, sample: list[ConversationItem]) -> list[ConversationItem]:
        """Append the assistant generation prefix after a templated (turn-closed) prompt."""
        out = list(sample)
        out.append(self._build_conversation_item("text", self.chat_markers.assistant_prefix, "assistant", loss_mask=0))
        return out
