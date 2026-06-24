"""Janus chat template as readable Python (mirrors ``chat_template.jinja``).

:class:`JanusChatTemplate` subclasses
:class:`~veomni.models.seed_omni.modules.base.text_encoder.chat_template.TextEncoderChatTemplate`
and overrides only the Janus-specific templating:

1. :meth:`JanusChatTemplate.apply_chat_template` — insert bos / (optional) system
   / role markers / boi–image–eoi spans; set ``meta["loss_mask"]`` on template
   rows that differ from the default ``int(role == "assistant")``.
2. :meth:`JanusChatTemplate.render_template_string` — concatenate the
   human-readable wire string (debug / demo only).

Tokenize / merge / pack are inherited from the base template; the text encoder
runs ``apply_chat_template`` → ``tokenize`` → ``merge_text_embeds`` → ``pack_input_ids``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ....utils.conversation import ConversationItem
from ...base.text_encoder.chat_template import TextEncoderChatTemplate


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

# Upstream Jinja emits this literal for ``content['type'] == 'image'``.
IMAGE_PLACEHOLDER = "<image_placeholder>"

# Janus chat-template defaults (mirror the upstream Jinja system preamble + role markers).
_JANUS_SYSTEM_PROMPT = (
    "You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language."
    "\n\n"
)
_JANUS_USER_PREFIX = "<|User|>: "
_JANUS_ASSISTANT_PREFIX = "\n\n<|Assistant|>:"


@dataclass(frozen=True)
class JanusChatMarkers:
    """Wire-format strings — use ``tokenizer.bos_token``, ``tokenizer.boi_token``, etc."""

    bos_token: str
    eos_token: str
    boi_token: str
    eoi_token: str
    system_prompt: str = _JANUS_SYSTEM_PROMPT
    user_prefix: str = _JANUS_USER_PREFIX
    assistant_prefix: str = _JANUS_ASSISTANT_PREFIX


class JanusChatTemplate(TextEncoderChatTemplate):
    chat_markers: JanusChatMarkers

    def __init__(self, tokenizer: PreTrainedTokenizer):
        chat_markers = JanusChatMarkers(
            bos_token=tokenizer.bos_token,
            eos_token=tokenizer.eos_token,
            boi_token=tokenizer.boi_token,
            eoi_token=tokenizer.eoi_token,
            system_prompt=_JANUS_SYSTEM_PROMPT,
            user_prefix=_JANUS_USER_PREFIX,
            assistant_prefix=_JANUS_ASSISTANT_PREFIX,
        )
        self.tokenizer = tokenizer
        self.chat_markers = chat_markers
        self.bos_token_id = int(tokenizer.bos_token_id)
        self.eos_token_id = int(tokenizer.eos_token_id)
        self.boi_token_id = int(tokenizer.boi_token_id)
        self.eoi_token_id = int(tokenizer.eoi_token_id)
        self.pad_token_id = int(tokenizer.pad_token_id)

    def apply_chat_template(self, sample: list[ConversationItem]) -> list[ConversationItem]:
        """Apply Janus chat template to a raw conversation (text + image parts).

        User and assistant images both use explicit ``<boi>`` … ``<eoi>`` text
        rows bracketing a sibling ``image`` row (SigLIP patch embeds).  No
        ``<image_placeholder>`` text row — nothing in the inference graph
        tokenises or embeds that literal.
        """
        out: list[ConversationItem] = []
        dummy_parts: list[ConversationItem] = []  # dummy from other modules
        out.append(self._build_conversation_item("text", self.chat_markers.bos_token, "user"))
        # HF Janus ``task != "gen"`` prepends the default system prompt for I2T.
        # TODO: shared by training + inference.  Official Janus I2T prepends the VL
        # system preamble; T2I does not.  Current micro-batches are either pure I2T
        # (user image present) or pure T2I (text-only user turn) — use that as a
        # proxy.  Ideally every sample would carry an explicit system row upstream.
        if self._sample_has_user_image(sample):
            out.append(self._build_conversation_item("text", self.chat_markers.system_prompt, "user"))
        prev_role: str | None = None
        prev_was_user_image = (
            False  # True after a user image; prepend \n to the next user text (HF Jinja same-turn layout).
        )
        for item in sample:
            role = item.role
            if role != prev_role:
                if role == "user":
                    out.append(self._build_conversation_item("text", self.chat_markers.user_prefix, "user"))
                elif role == "assistant":
                    out.append(
                        self._build_conversation_item(
                            type="text",
                            value=self.chat_markers.assistant_prefix,
                            role="assistant",
                            loss_mask=0,
                        )
                    )
                prev_role = role
                prev_was_user_image = False

            if item.type == "text":
                text = str(item.value)
                if prev_was_user_image and role == "user" and not text.startswith("\n"):
                    text = "\n" + text
                out.append(self._build_conversation_item("text", text, role))
                prev_was_user_image = False
            elif item.type == "image" and role != "dummy":
                out.append(self._build_conversation_item("text", self.chat_markers.boi_token, role))
                out.append(item)  # media row passed through verbatim (keeps value/source/meta)
                out.append(self._build_conversation_item("text", self.chat_markers.eoi_token, role))
                prev_was_user_image = role == "user"
            elif role == "dummy":
                dummy_parts.append(item)
            else:
                raise ValueError(f"Unsupported part type: {item.type}")
        if prev_role == "assistant":
            out.append(self._build_conversation_item("text", self.chat_markers.eos_token, "assistant"))
        out.extend(dummy_parts)
        return out

    def apply_generation_prompt(self, sample: list[ConversationItem]) -> list[ConversationItem]:
        """Append the assistant generation prefix to a templated prompt (inference)."""
        out = list(sample)
        out.append(self._build_conversation_item("text", self.chat_markers.assistant_prefix, "assistant", loss_mask=0))
        return out

    def render_template_string(self, parts: list[ConversationItem]) -> str:
        """
        For debug or demo.
            Build the on-the-wire prompt string (Jinja-visible layout).
        """
        chunks: list[str] = []
        for part in parts:
            if part.type == "text":
                chunks.append(str(part.value or ""))
            elif part.type == "image":
                chunks.append(IMAGE_PLACEHOLDER)
            else:
                raise ValueError(f"Unsupported part type: {part.type}")
        return "".join(chunks)

    @staticmethod
    def _sample_has_user_image(sample: list[ConversationItem]) -> bool:
        """Return True when the raw conversation includes a user ``image`` row."""
        return any(item.type == "image" and item.role == "user" for item in sample)
