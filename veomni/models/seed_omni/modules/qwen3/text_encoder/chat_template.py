"""Qwen3 ChatML template as readable Python.

Wraps each turn in ``<|im_start|>{role}\\n{content}<|im_end|>\\n`` (standard Qwen
ChatML — every turn is closed, matching :class:`Qwen3VLChatTemplate`); the closing
``<|im_end|>`` is supervised only on assistant turns. Reuses
:class:`TextEncoderChatTemplate` for tokenize / merge / pack; only the ChatML
templating and generation prompt are model-specific.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ....utils.conversation import ConversationItem
from ...text_encoder.chat_template import TextEncoderChatTemplate


@dataclass(frozen=True)
class Qwen3ChatMarkers:
    """Wire-format ChatML markers (fixed literals — not tokenizer-dependent)."""

    im_start_token: str = "<|im_start|>"
    im_end_token: str = "<|im_end|>"
    assistant_prefix: str = "<|im_start|>assistant\n"


@dataclass(frozen=True)
class Qwen3VLChatMarkers(Qwen3ChatMarkers):
    """Wire-format ChatML markers with vision start/end delimiters."""

    vision_start_token: str = "<|vision_start|>"
    vision_end_token: str = "<|vision_end|>"


class Qwen3ChatTemplate(TextEncoderChatTemplate):
    chat_markers: Qwen3ChatMarkers

    def __init__(self, tokenizer: Any):
        super().__init__(tokenizer)  # resolves bos / eos markers + ids
        self.chat_markers = Qwen3ChatMarkers()
        # Extra ChatML stop token (eos_token_id comes from the base).
        self.im_end_token_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))

    def apply_chat_template(self, sample: list[ConversationItem]) -> list[ConversationItem]:
        """Apply ChatML template to a raw text-only conversation.

        Every turn is closed with ``<|im_end|>\\n`` (standard Qwen ChatML, matching
        :class:`Qwen3VLChatTemplate`); the closing marker is supervised only for
        assistant turns.
        """
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
            if item.is_dummy:
                dummy_parts.append(item)
                continue
            if item.type != "text":
                raise ValueError(f"Qwen3 text encoder only supports text items, got {item.type!r}")

            if role != prev_role:
                if prev_role is not None:
                    close_turn(prev_role)
                out.append(
                    self._build_conversation_item("text", markers.im_start_token + role + "\n", role, loss_mask=0)
                )
                prev_role = role

            out.append(self._build_conversation_item("text", str(item.value), role))

        if prev_role is not None:
            close_turn(prev_role)
        out.extend(dummy_parts)
        return out

    def apply_generation_prompt(self, sample: list[ConversationItem]) -> list[ConversationItem]:
        """Append the assistant generation prefix after a templated (turn-closed) prompt."""
        out = list(sample)
        out.append(self._build_conversation_item("text", self.chat_markers.assistant_prefix, "assistant", loss_mask=0))
        return out


class Qwen3VLChatTemplate(Qwen3ChatTemplate):
    """Image/video-aware Qwen3 ChatML template (wraps media items in ``<|vision_start|> … <|vision_end|>``)."""

    chat_markers: Qwen3VLChatMarkers

    def __init__(self, tokenizer: Any):
        super().__init__(tokenizer)
        self.chat_markers = Qwen3VLChatMarkers()

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
            if item.is_dummy:
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
                out.append(self._build_conversation_item("text", markers.vision_start_token, role, loss_mask=0))
                out.append(item)
                out.append(self._build_conversation_item("text", markers.vision_end_token, role, loss_mask=0))
            else:
                raise ValueError(f"Qwen3-VL text encoder only supports text/image/video items, got {item.type!r}")

        if prev_role is not None:
            close_turn(prev_role)
        out.extend(dummy_parts)
        return out
