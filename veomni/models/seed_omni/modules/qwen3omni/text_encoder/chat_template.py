"""Qwen3-Omni ChatML template (text + image + video + audio) as readable Python.

Mirrors the upstream ``chat_template.json``:

* each turn is wrapped in ``<|im_start|>{role}\\n … <|im_end|>\\n``;
* image / video become ``<|vision_start|><|image_pad|><|vision_end|>`` /
  ``<|vision_start|><|video_pad|><|vision_end|>``;
* standalone audio becomes ``<|audio_start|><|audio_pad|><|audio_end|>``.

As in Qwen3-VL, the ``<|*_pad|>`` run is *not* tokenized here. The sibling
encoder module already put its embeds on the media item, so the template emits
the opening marker as text, the media item verbatim, then the closing marker.
That keeps this template independent of how many tokens a clip or image
actually expands to -- a count only the encoder can know.

Reuses :class:`TextEncoderChatTemplate` for tokenize / merge / pack; only the
ChatML templating and the generation prompt are model-specific.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ....utils.conversation import ConversationItem
from ...base.llm_packing import item_token_len
from ...text_encoder.chat_template import TextEncoderChatTemplate
from ..carrier import is_thinker_input


# The placeholder token each modality expands to at the token level. V2 never
# materialises these runs — the media item carries the embeds directly — but the
# talker's 3D RoPE is computed by upstream code that scans a token sequence, so
# :meth:`Qwen3OmniChatTemplate.pack_talker_input_ids` rebuilds them on demand.
MEDIA_PAD_TOKENS = {"image": "<|image_pad|>", "video": "<|video_pad|>", "audio": "<|audio_pad|>"}

# The checkpoint README's system turn for speech output, verbatim.
#
# The text lives here rather than in a config or a launch flag because it is a
# property of the checkpoint, like the ChatML markers in this module: the
# thinker was trained to answer in speakable prose under *this* wording, and a
# paraphrase is a different conditioning. Without it the thinker answers in
# numbered prose that never reaches ``<|im_end|>`` inside the token budget, so
# the FSM never hands over to the talker and no audio is produced at all.
#
# Which scenarios get it is still the graph's call — see
# ``Qwen3OmniTextEncoderConfig.add_speech_system_prompt``. Text-only rounds
# leave it off: it caps replies at 50 words and imposes a voice persona, which
# would skew a text evaluation.
SPEECH_SYSTEM_PROMPT = (
    "You are Qwen-Omni, a smart voice assistant created by Alibaba Qwen. You are a virtual voice assistant "
    "with no gender or age.\n"
    "You are communicating with the user.\n"
    "In user messages, \u201cI/me/my/we/our\u201d refer to the user and \u201cyou/your\u201d refer to the "
    "assistant. In your replies, address the user as \u201cyou/your\u201d and yourself as "
    "\u201cI/me/my\u201d; never mirror the user\u2019s pronouns\u2014always shift perspective. Keep original "
    "pronouns only in direct quotes; if a reference is unclear, ask a brief clarifying question.\n"
    "Interact with users using short(no more than 50 words), brief, straightforward language, maintaining a "
    "natural tone.\n"
    "Never use formal phrasing, mechanical expressions, bullet points, overly structured language. \n"
    "Your output must consist only of the spoken content you want the user to hear. \n"
    "Do not include any descriptions of actions, emotions, sounds, or voice changes. \n"
    "Do not use asterisks, brackets, parentheses, or any other symbols to indicate tone or actions. \n"
    "You must answer users' audio or text questions, do not directly describe the video content. \n"
    "You should communicate in the same language strictly as the user unless they request otherwise.\n"
    "When you are uncertain (e.g., you can't see/hear clearly, don't understand, or the user makes a comment "
    "rather than asking a question), use appropriate questions to guide the user to continue the "
    "conversation.\n"
    "Keep replies concise and conversational, as if talking face-to-face."
)


@dataclass(frozen=True)
class Qwen3OmniChatMarkers:
    """Wire-format ChatML markers (fixed literals — not tokenizer-dependent)."""

    im_start_token: str = "<|im_start|>"
    im_end_token: str = "<|im_end|>"
    assistant_prefix: str = "<|im_start|>assistant\n"
    vision_start_token: str = "<|vision_start|>"
    vision_end_token: str = "<|vision_end|>"
    audio_start_token: str = "<|audio_start|>"
    audio_end_token: str = "<|audio_end|>"


class Qwen3OmniChatTemplate(TextEncoderChatTemplate):
    chat_markers: Qwen3OmniChatMarkers

    def __init__(self, tokenizer: Any, add_speech_system_prompt: bool = False):
        super().__init__(tokenizer)  # resolves bos / eos markers + ids
        self.add_speech_system_prompt = bool(add_speech_system_prompt)
        self.chat_markers = Qwen3OmniChatMarkers()
        # Extra ChatML stop token (eos_token_id comes from the base).
        self.im_end_token_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
        self.media_pad_token_ids = {
            item_type: int(tokenizer.convert_tokens_to_ids(token)) for item_type, token in MEDIA_PAD_TOKENS.items()
        }
        # Speech-span framing tokens. They live in the *thinker* vocabulary — the
        # talker consumes their thinker embeddings, not codec ids — so the
        # tokenizer is their single source of truth and no config repeats them.
        self.tts_bos_token_id = int(tokenizer.convert_tokens_to_ids("<tts_text_bos>"))
        self.tts_eos_token_id = int(tokenizer.convert_tokens_to_ids("<tts_text_eod>"))
        self.tts_pad_token_id = int(tokenizer.convert_tokens_to_ids("<tts_pad>"))
        # ChatML turn boundaries, for the talker. It cannot split turns by
        # ``item.role``: a history assistant turn and the trailing generation
        # prompt share the role ``assistant``, and merging them would corrupt
        # the speech-span prefix length. Upstream splits on ``<|im_start|>``, so
        # the talker does too, and reads the role from the token after it.
        self.im_start_token_id = int(tokenizer.convert_tokens_to_ids("<|im_start|>"))
        self.role_token_ids = {
            role: int(tokenizer.convert_tokens_to_ids(role)) for role in ("system", "user", "assistant")
        }

    def pack_talker_input_ids(self, parts: list[ConversationItem]) -> torch.Tensor:
        """Rebuild the token-level id sequence for the prompt, placeholders included.

        The talker's ``forward`` derives its 3D RoPE positions by scanning a
        token sequence for ``<|vision_start|>`` / ``<|audio_start|>`` and
        counting the placeholder run after each. V2 deliberately does not
        materialise those runs (that is the point of the segment carrier), so
        this reconstructs them: a media item contributes its placeholder token
        repeated once per embedding row. The two counts are equal by
        construction — the sibling encoder produced one row per placeholder.

        Must be called *before* the text encoder overwrites text rows with
        embeddings, while ``item.value`` is still a token-id tensor.
        """
        ids: list[torch.Tensor] = []
        for item in parts:
            if not is_thinker_input(item):
                continue
            if item.type == "text":
                ids.append(item.value.reshape(-1).cpu())
                continue
            pad_token_id = self.media_pad_token_ids.get(item.type)
            if pad_token_id is None:
                raise ValueError(
                    f"qwen3omni: cannot rebuild talker input ids for item type {item.type!r} — "
                    f"known media types are {sorted(self.media_pad_token_ids)}."
                )
            ids.append(torch.full((item_token_len(item),), pad_token_id, dtype=torch.long))
        if not ids:
            return torch.zeros(0, dtype=torch.long)
        return torch.cat(ids, dim=0)

    def _maybe_prepend_speech_system(self, parts: list[ConversationItem]) -> list[ConversationItem]:
        """Open a speech round with :data:`SPEECH_SYSTEM_PROMPT` when the graph asks for it.

        Prepended here rather than carried in the data so that generation and
        talker training cannot drift apart: both render through this template,
        so both see the same opening turn without a corpus column to keep in
        sync.

        Takes *rendered* parts and writes its own ChatML markers, under the role
        of the turn it precedes, instead of riding a ``role="system"`` row
        through the loop above. The tokens are the same either way, but a system
        role on the carrier would be a new member of ``ItemRole`` that every
        other text encoder then has to refuse; folding system text into a
        neighbouring turn is already how this repo handles it elsewhere (see
        ``tulu_3_sft_mixture_preprocess``). ``merge_text_embeds`` joins the rows
        afterwards, and ``_chatml_turns`` still finds the turn because it splits
        on ``<|im_start|>`` tokens rather than on roles.

        Three rows rather than one pre-rendered string, matching how every other
        turn here is laid out: :meth:`tokenize` runs per row, so joining the
        marker to the prompt would let the tokenizer merge across that seam and
        shift the ids away from what the marker tokenizes to on its own.

        No parts means no turn to open — a micro-batch of nothing but dummies or
        talker labels is not a round, and a system block alone would be one.
        """
        if not self.add_speech_system_prompt or not parts:
            return parts
        markers = self.chat_markers
        role = parts[0].role
        return [
            self._build_conversation_item("text", markers.im_start_token + "system\n", role, loss_mask=0),
            self._build_conversation_item("text", SPEECH_SYSTEM_PROMPT, role, loss_mask=0),
            self._build_conversation_item("text", markers.im_end_token + "\n", role, loss_mask=0),
            *parts,
        ]

    def apply_chat_template(self, sample: list[ConversationItem]) -> list[ConversationItem]:
        """Apply Qwen3-Omni ChatML to a raw conversation (text + image + video + audio)."""
        markers = self.chat_markers
        out: list[ConversationItem] = []
        dummy_parts: list[ConversationItem] = []
        deferred_parts: list[ConversationItem] = []
        prev_role: str | None = None

        def close_turn(role: str) -> None:
            out.append(
                self._build_conversation_item(
                    "text", markers.im_end_token + "\n", role, loss_mask=int(role == "assistant")
                )
            )

        def wrap_media(item: ConversationItem, role: str, start: str, end: str) -> None:
            out.append(self._build_conversation_item("text", start, role, loss_mask=0))
            out.append(item)  # media row passed through verbatim (keeps value/source/meta)
            out.append(self._build_conversation_item("text", end, role, loss_mask=0))

        for item in sample:
            role = item.role
            if item.is_dummy:
                dummy_parts.append(item)
                continue
            if role == "assistant" and item.type == "audio":
                # The talker's label, not a turn. Held back like the dummies so
                # it stays on the carrier for the codec without taking a position
                # in the ChatML the thinker reads. Downstream walks recognise it
                # by role/type through ``carrier.is_thinker_input``, so it needs
                # no marker of its own.
                deferred_parts.append(item)
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
                # A clip's sound track gets no span of its own: without the time
                # interleave (av_video_design.md) the vision span carries the
                # frames and the track is skipped.
                wrap_media(item, role, markers.vision_start_token, markers.vision_end_token)
            elif item.type == "audio":
                wrap_media(item, role, markers.audio_start_token, markers.audio_end_token)
            else:
                raise ValueError(
                    f"Qwen3-Omni text encoder only supports text/image/video/audio items, got {item.type!r}"
                )

        if prev_role is not None:
            close_turn(prev_role)
        # Before the held-back rows are re-attached: those trail the thinker
        # sequence, so ``out[0]`` must still be the first turn's marker for the
        # system block to land in front of it under that turn's role.
        out = self._maybe_prepend_speech_system(out)
        out.extend(deferred_parts)
        out.extend(dummy_parts)
        return out

    def apply_generation_prompt(
        self,
        sample: list[ConversationItem],
        *,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> list[ConversationItem]:
        """Append the assistant generation prefix after a templated (turn-closed) prompt.

        ``generation_kwargs`` is accepted and ignored: this family's reply is
        always text first, and whether it is then spoken is the generation
        graph's decision (``infer_type``), not a property of the prompt.
        """
        del generation_kwargs
        out = list(sample)
        out.append(self._build_conversation_item("text", self.chat_markers.assistant_prefix, "assistant", loss_mask=0))
        return out


__all__ = ["Qwen3OmniChatMarkers", "Qwen3OmniChatTemplate", "SPEECH_SYSTEM_PROMPT"]
