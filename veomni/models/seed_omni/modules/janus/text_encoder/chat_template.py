"""Janus chat template as readable Python (mirrors ``chat_template.jinja``).

:class:`JanusChatTemplate` subclasses
:class:`~veomni.models.seed_omni.modules.text_encoder.chat_template.TextEncoderChatTemplate`
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
from typing import TYPE_CHECKING, Any

from ....utils.conversation import ConversationItem
from ...text_encoder.chat_template import TextEncoderChatTemplate


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
# Upstream's Jinja emits a space after ``<|Assistant|>:`` only when a response
# follows: ``...{%if not (loop.last and not add_generation_prompt and
# message['content'][0]['text']=='')%} {%endif%}``. A bare generation prompt
# ends at the colon. The tokenizer is byte-level and adds no spacing of its own,
# so the two spellings are different tokens and the distinction has to be kept.
_JANUS_ASSISTANT_TEXT_SEPARATOR = " "


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
    assistant_text_separator: str = _JANUS_ASSISTANT_TEXT_SEPARATOR


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
        # Whether the assistant turn just closed ended on a generated image, which
        # decides if the EOS terminating it is supervised (see the two EOS sites).
        prev_turn_ended_on_image = False
        # Whether the current assistant turn already emitted a text span, so the
        # ``<|Assistant|>: `` separator is added to the first one only.
        prev_turn_has_text = False
        for item in sample:
            if item.is_dummy:
                dummy_parts.append(item)
                continue
            role = item.role
            if role != prev_role:
                if role == "user":
                    if prev_role == "assistant":
                        # Close the preceding assistant turn before the next user
                        # one: Janus' wire format is ``...<|Assistant|>:answer</s>
                        # <|User|>: next``. Without this the two answers run
                        # together and only the sample-final EOS is supervised, so
                        # the model never learns the intermediate stop token. A turn
                        # that ended on a generated image leaves its EOS unscored,
                        # exactly as the sample-final one does below.
                        eos_loss_mask = 0 if prev_turn_ended_on_image else None
                        out.append(
                            self._build_conversation_item(
                                "text", self.chat_markers.eos_token, "assistant", loss_mask=eos_loss_mask
                            )
                        )
                    out.append(self._build_conversation_item("text", self.chat_markers.user_prefix, "user"))
                elif role == "assistant":
                    # The bare marker. Upstream emits ``<|Assistant|>: `` — with a
                    # trailing space — only when a response follows, and that space
                    # belongs to the *first supervised token*: a byte-level
                    # tokenizer has one id for ``Hello`` and another for `` Hello``,
                    # so putting the separator in this unscored part tokenized it as
                    # a standalone space and left the answer's first token different
                    # from the native template. It is prepended to the first text
                    # span below instead, where it merges with the response.
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
                prev_turn_has_text = False
                if role == "assistant":
                    prev_turn_ended_on_image = False

            if item.type == "text":
                text = str(item.value)
                if prev_was_user_image and role == "user" and not text.startswith("\n"):
                    text = "\n" + text
                if role == "assistant" and not prev_turn_has_text:
                    # First text span of an assistant turn: upstream renders
                    # ``<|Assistant|>: Hello``, so the separator merges into this
                    # span's first token rather than sitting in the unscored
                    # marker part (where a byte-level tokenizer would emit it as a
                    # standalone space and leave ``Hello`` on a different id).
                    text = self.chat_markers.assistant_text_separator + text
                    prev_turn_has_text = True
                out.append(self._build_conversation_item("text", text, role))
                prev_was_user_image = False
                if role == "assistant":
                    prev_turn_ended_on_image = False
            elif item.type == "image":
                # ``<boi>`` stays out of the text CE: Janus was never trained to emit
                # it (upstream VeOmni masks ``image_start_id`` for the same reason), so
                # supervising it costs tens of nats and swamps the real text loss.
                # ``<eoi>`` is supervised — after a fixed-length image span the model
                # predicts it near-perfectly, and the reference stack scores it too.
                boi = self.chat_markers.boi_token
                if role == "assistant" and not prev_turn_has_text:
                    # An image opening a response takes the separator too: upstream
                    # renders ``<|Assistant|>: <image_placeholder>``, and the boundary
                    # before ``<boi>`` is the same position a text answer starts at.
                    boi = self.chat_markers.assistant_text_separator + boi
                    prev_turn_has_text = True
                out.append(self._build_conversation_item("text", boi, role, loss_mask=0))
                out.append(item)  # media row passed through verbatim (keeps value/role/meta)
                out.append(self._build_conversation_item("text", self.chat_markers.eoi_token, role))
                prev_was_user_image = role == "user"
                if role == "assistant":
                    prev_turn_ended_on_image = True
            else:
                raise ValueError(f"Unsupported part type: {item.type}")
        if prev_role == "assistant":
            # Janus leaves the eos that follows a generated image untrained (upstream
            # VeOmni masks it for the same reason), so scoring it on a T2I turn only
            # adds noise. A text answer -- I2T caption or text-only SFT -- does
            # supervise its eos; that is how the model learns to stop.
            eos_loss_mask = 0 if self._ends_on_generated_image(sample) else None
            out.append(
                self._build_conversation_item(
                    "text", self.chat_markers.eos_token, "assistant", loss_mask=eos_loss_mask
                )
            )
        out.extend(dummy_parts)
        return out

    def apply_generation_prompt(
        self,
        sample: list[ConversationItem],
        *,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> list[ConversationItem]:
        """Append the assistant generation prefix to a templated prompt (inference).

        The two response modalities need different tails, and that is what the
        checkpoint was trained on (byte-level BPE, so the difference is real):

        * **text** (``infer_und``) — training renders ``<|Assistant|>: Hello`` and
          the space folds into ``Hello``, so the model predicts the merged
          ``" Hello"`` token straight off the marker. The prompt therefore stops at
          the bare marker, matching upstream's
          ``{%if add_generation_prompt%}<|Assistant|>:{%endif%}``.
        * **image** (anything else) — training renders ``<|Assistant|>: <boi>``,
          where ``<boi>`` is a special token so nothing folds and the space stays
          its own token. The prompt carries the separator, and
          ``emit_image_start`` then feeds ``<boi>`` — the same two positions a
          training sample has.
        """
        infer_type = None if generation_kwargs is None else generation_kwargs.get("infer_type")
        value = self.chat_markers.assistant_prefix
        if infer_type != "infer_und":
            value += self.chat_markers.assistant_text_separator
        out = list(sample)
        out.append(self._build_conversation_item("text", value, "assistant", loss_mask=0))
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
        return any(item.type == "image" and item.role == "user" and not item.is_dummy for item in sample)

    @staticmethod
    def _ends_on_generated_image(sample: list[ConversationItem]) -> bool:
        """Return True when the assistant's last row is an ``image`` (a T2I turn)."""
        for item in reversed(sample):
            if item.is_dummy:
                continue
            return item.type == "image" and item.role == "assistant"
        return False
