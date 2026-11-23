"""Unit tests for the Qwen3-Omni ChatML template (text + image + video + audio)."""

import numpy as np
import pytest
import torch

from veomni.data.seed_omni.utils.media_metadata import AUDIO_METADATA_KEY, AudioMetadata
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.chat_template import (
    SPEECH_SYSTEM_PROMPT,
    Qwen3OmniChatTemplate,
)
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.modeling import (
    talker_prompt_segments,
    talker_supervised_mask,
)
from veomni.models.seed_omni.utils.conversation import ConversationItem


class FakeTokenizer:
    """Minimal tokenizer for chat-template construction (markers are fixed literals)."""

    eos_token_id = 0

    def convert_tokens_to_ids(self, token):
        return 1

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def _template() -> Qwen3OmniChatTemplate:
    return Qwen3OmniChatTemplate(FakeTokenizer())


def test_chat_template_closes_each_turn():
    sample = [
        ConversationItem(type="text", value="hi", role="user"),
        ConversationItem(type="text", value="hello", role="assistant"),
    ]
    parts = _template().apply_chat_template(sample)
    assert [p.value for p in parts] == [
        "<|im_start|>user\n",
        "hi",
        "<|im_end|>\n",
        "<|im_start|>assistant\n",
        "hello",
        "<|im_end|>\n",
    ]
    # assistant content + its closing <|im_end|> are supervised; user turn is not.
    assert parts[4].meta["loss_mask"] == 1
    assert parts[5].meta["loss_mask"] == 1
    assert parts[1].meta["loss_mask"] == 0
    assert parts[2].meta["loss_mask"] == 0


def test_chat_template_wraps_audio_with_audio_markers():
    sample = [
        ConversationItem(type="audio", value=np.zeros(16000, dtype=np.float32), role="user"),
        ConversationItem(type="text", value="what is this", role="user"),
    ]
    parts = _template().apply_chat_template(sample)
    assert [(p.type, p.value if p.type == "text" else "<wav>") for p in parts] == [
        ("text", "<|im_start|>user\n"),
        ("text", "<|audio_start|>"),
        ("audio", "<wav>"),
        ("text", "<|audio_end|>"),
        ("text", "what is this"),
        ("text", "<|im_end|>\n"),
    ]
    # markers must not be supervised
    assert parts[1].meta["loss_mask"] == 0
    assert parts[3].meta["loss_mask"] == 0


@pytest.mark.parametrize("media_type", ["image", "video"])
def test_chat_template_wraps_vision_with_vision_markers(media_type):
    sample = [
        ConversationItem(type=media_type, value="<pixels>", role="user"),
        ConversationItem(type="text", value="describe", role="user"),
    ]
    parts = _template().apply_chat_template(sample)
    assert [(p.type, p.value if p.type == "text" else "<media>") for p in parts] == [
        ("text", "<|im_start|>user\n"),
        ("text", "<|vision_start|>"),
        (media_type, "<media>"),
        ("text", "<|vision_end|>"),
        ("text", "describe"),
        ("text", "<|im_end|>\n"),
    ]


def test_chat_template_interleaves_audio_and_vision():
    """Both modalities in one turn keep their own marker pair and stay in order."""
    sample = [
        ConversationItem(type="image", value="<pixels>", role="user"),
        ConversationItem(type="audio", value=np.zeros(8, dtype=np.float32), role="user"),
    ]
    parts = _template().apply_chat_template(sample)
    assert [p.value if p.type == "text" else p.type for p in parts] == [
        "<|im_start|>user\n",
        "<|vision_start|>",
        "image",
        "<|vision_end|>",
        "<|audio_start|>",
        "audio",
        "<|audio_end|>",
        "<|im_end|>\n",
    ]


def test_a_clip_with_sound_gets_the_same_vision_span_as_a_silent_one():
    """Without the time interleave the sound track gets no span of its own.

    The clip is wrapped in the vision markers like any video and the track is
    skipped, so the template output matches the silent clip's item for item.
    """

    def _clip(meta: dict) -> list[ConversationItem]:
        return [ConversationItem(type="video", value="<frames>", role="user", meta=meta)]

    sounding = _clip({AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=16_000, num_samples=8)})

    def _shape(parts):
        return [(p.type, p.value if p.type == "text" else "<media>") for p in parts]

    assert _shape(_template().apply_chat_template(sounding)) == _shape(_template().apply_chat_template(_clip({})))


def test_unsupported_item_type_is_rejected():
    sample = [ConversationItem(type="output", value="x", role="assistant")]
    with pytest.raises(ValueError, match="text/image/video/audio"):
        _template().apply_chat_template(sample)


def test_generation_prompt_appends_assistant_prefix():
    tmpl = _template()
    sample = tmpl.apply_chat_template([ConversationItem(type="text", value="hi", role="user")])
    parts = tmpl.apply_generation_prompt(sample)
    assert parts[-1].value == "<|im_start|>assistant\n"
    assert parts[-1].meta["loss_mask"] == 0
    # the user turn was already closed by the chat template
    assert parts[-2].value == "<|im_end|>\n"


def test_chat_template_passes_media_rows_through_untouched():
    """Rebuilt media rows must keep their tags (regression).

    ``tokenize_conversation`` re-materialises every sample, so the media rows are
    emitted verbatim rather than rebuilt: their ``value``, ``role``, ``is_dummy``
    and ``meta`` are how the sibling encoder selects them. A rebuild that dropped
    ``meta`` would leave ``_OMNI_GRID`` / ``_OMNI_MEL_LEN`` behind and every
    tower's pre-hook would find nothing to encode.
    """
    sample = [
        ConversationItem(type="image", value="<pixels>", role="user", meta={"_omni_grid": [1, 2, 2]}),
        ConversationItem(type="audio", value=np.zeros(8, dtype=np.float32), role="user"),
    ]
    parts = _template().apply_chat_template(sample)
    media_rows = [p for p in parts if p.type in ("image", "audio")]
    assert media_rows[0].meta["_omni_grid"] == [1, 2, 2]
    assert [p.role for p in media_rows] == ["user", "user"]


def test_dummy_rows_are_moved_to_tail():
    sample = [
        ConversationItem(type="text", value="hi", role="user"),
        ConversationItem(type="audio", value=np.zeros(8, dtype=np.float32), role="user", is_dummy=True),
    ]
    parts = _template().apply_chat_template(sample)
    assert parts[-1].is_dummy
    assert parts[-1].type == "audio"


def test_assistant_audio_is_held_out_of_the_chatml():
    """It is the talker's label, so it takes no position in what the thinker reads."""
    sample = [
        ConversationItem(type="text", value="say hi", role="user"),
        ConversationItem(type="text", value="hi there", role="assistant"),
        ConversationItem(type="audio", value=np.zeros(24000, dtype=np.float32), role="assistant"),
    ]
    parts = _template().apply_chat_template(sample)

    assert [p.value for p in parts if p.type == "text"] == [
        "<|im_start|>user\n",
        "say hi",
        "<|im_end|>\n",
        "<|im_start|>assistant\n",
        "hi there",
        "<|im_end|>\n",
    ]
    # Still on the carrier — that is how the codec finds it — but at the tail,
    # unwrapped, with no audio markers around it.
    assert parts[-1].type == "audio" and parts[-1].role == "assistant"
    assert "<|audio_start|>" not in [p.value for p in parts if p.type == "text"]


def test_the_user_side_of_the_same_modality_is_still_a_turn():
    """Role is the whole difference: one is heard, the other is spoken."""
    heard = ConversationItem(type="audio", value=np.zeros(16000, dtype=np.float32), role="user")
    spoken = ConversationItem(type="audio", value=np.zeros(24000, dtype=np.float32), role="assistant")

    parts = _template().apply_chat_template([heard, spoken])

    # The heard clip is wrapped in audio markers inside the user turn; the
    # spoken one is at the tail with none.
    assert [p.value for p in parts if p.type == "text"] == [
        "<|im_start|>user\n",
        "<|audio_start|>",
        "<|audio_end|>",
        "<|im_end|>\n",
    ]
    assert parts[2] is heard
    assert parts[-1] is spoken


def test_the_talker_token_view_skips_the_spoken_audio():
    conversation = _template().tokenize_conversation(
        [
            ConversationItem(type="text", value="say hi", role="user"),
            ConversationItem(type="text", value="hi", role="assistant"),
            ConversationItem(type="audio", value=np.zeros(24000, dtype=np.float32), role="assistant"),
        ]
    )

    segments = talker_prompt_segments(conversation)

    assert all(segment.type != "audio" for segment in segments)
    assert sum(segment.length for segment in segments) == sum(
        int(item.value.size(0)) for item in conversation if item.type == "text"
    )


def test_the_boundary_survives_the_real_tokenize_pipeline():
    """`loss_mask` is popped by `tokenize` and the prefix is merged into the
    response, so the boundary has to be read from the per-token labels."""
    conversation = _template().tokenize_conversation(
        [
            ConversationItem(type="text", value="say hi", role="user"),
            ConversationItem(type="text", value="hi", role="assistant"),
        ]
    )

    supervised = talker_supervised_mask(conversation)

    # The rule is inoperable at item granularity: `tokenize` removed the key it
    # used to read, and the merge fused the masked assistant prefix into the
    # supervised response.
    assert all("loss_mask" not in item.meta for item in conversation)
    assert len([item for item in conversation if item.role == "assistant"]) == 1
    # It still resolves, inside that merged item.
    assert supervised.any() and not supervised.all()
    first = int(supervised.nonzero()[0])
    assert 0 < first < supervised.numel()


def test_an_inference_prompt_supervises_nothing():
    """The response has not been generated yet, so the prompt is everything given."""
    conversation = _template().tokenize_conversation(
        [ConversationItem(type="text", value="say hi", role="user")],
        add_generation_prompt=True,
    )

    assert not talker_supervised_mask(conversation).any()


def test_the_supervised_mask_lines_up_with_the_segments():
    """The talker reads them positionally against one another."""
    conversation = _template().tokenize_conversation(
        [
            ConversationItem(type="audio", value=np.zeros(16000, dtype=np.float32), role="user"),
            ConversationItem(type="text", value="what did I say", role="user"),
            ConversationItem(type="text", value="you said hi", role="assistant"),
            ConversationItem(type="audio", value=np.zeros(24000, dtype=np.float32), role="assistant"),
        ]
    )
    # The user's clip is a media row the encoder would have filled; give it rows.
    for item in conversation:
        if item.type == "audio" and item.role == "user":
            item.value = torch.zeros(7, 4)

    segments = talker_prompt_segments(conversation)
    supervised = talker_supervised_mask(conversation)

    assert supervised.numel() == sum(segment.length for segment in segments)
    # A media row is never supervised.
    assert not supervised[:7].any()


# ── the speech system turn (add_speech_system_prompt) ─────────────────────────

_SPEECH_BLOCK = f"<|im_start|>system\n{SPEECH_SYSTEM_PROMPT}<|im_end|>\n"


def _speech_round() -> list[ConversationItem]:
    """A talker round: spoken reply plus the waveform its RVQ targets come from.

    Rebuilt per call — ``tokenize`` overwrites ``item.value`` in place.
    """
    return [
        ConversationItem(type="text", value="say hi", role="user"),
        ConversationItem(type="text", value="hi", role="assistant"),
        ConversationItem(type="audio", value=np.zeros(24000, dtype=np.float32), role="assistant"),
    ]


def _speech_template() -> Qwen3OmniChatTemplate:
    return Qwen3OmniChatTemplate(FakeTokenizer(), add_speech_system_prompt=True)


def test_speech_system_turn_is_off_by_default():
    """A text round must not inherit the voice persona: it caps replies at 50 words."""
    parts = _template().apply_chat_template([ConversationItem(type="text", value="hi", role="user")])
    assert [p.value for p in parts] == ["<|im_start|>user\n", "hi", "<|im_end|>\n"]


def test_speech_system_turn_opens_the_round_when_enabled():
    parts = _speech_template().apply_chat_template([ConversationItem(type="text", value="hi", role="user")])
    # Laid out per row like every other turn, so each row tokenizes on its own.
    assert [p.value for p in parts] == [
        "<|im_start|>system\n",
        SPEECH_SYSTEM_PROMPT,
        "<|im_end|>\n",
        "<|im_start|>user\n",
        "hi",
        "<|im_end|>\n",
    ]
    # The conditioning is the prompt, never a target — supervising it would train
    # the thinker to emit its own system turn.
    assert all(p.meta["loss_mask"] == 0 for p in parts)


def test_the_speech_system_turn_introduces_no_system_role():
    """It writes its own markers under the role of the turn it opens.

    A ``role="system"`` row would add a member to ``ItemRole`` that every other
    text encoder then has to refuse, for no gain: the tokens are identical
    either way, and ``merge_text_embeds`` folds the block into the turn that
    follows it. Folding system text into a neighbouring turn is already how
    this repo handles it (see ``tulu_3_sft_mixture_preprocess``).
    """
    template = _speech_template()
    assert all(p.role == "user" for p in template.apply_chat_template(_speech_round()[:1]))
    assert [p.role for p in template.tokenize_conversation(_speech_round()[:1])] == ["user"]


def test_an_empty_round_gets_no_speech_system_turn():
    """No turns is not a round, and a system block alone would make it one."""
    assert _speech_template().apply_chat_template([]) == []


def test_the_speech_system_turn_is_never_supervised():
    """It is prompt, not target — supervising it would train the talker to
    speak the instructions."""
    conversation = _speech_template().tokenize_conversation(_speech_round())

    rendered = "".join(chr(token) for item in conversation if item.type == "text" for token in item.value.tolist())
    assert rendered.startswith(f"{_SPEECH_BLOCK}<|im_start|>user\n")

    supervised = talker_supervised_mask(conversation)
    assert not supervised[: len(_SPEECH_BLOCK)].any()  # FakeTokenizer emits one id per character
    assert supervised.any(), "the assistant span must still be supervised"


def test_the_talker_token_view_carries_the_speech_system_turn():
    """The talker's 3D RoPE scans this view, so it must match the thinker's.

    The system turn takes real positions in the packed sequence; dropping it
    from the token view would shift every later position the talker computes.
    """
    conversation = _speech_template().tokenize_conversation(_speech_round())

    segments = talker_prompt_segments(conversation)

    assert sum(segment.length for segment in segments) == sum(
        int(item.value.size(0)) for item in conversation if item.type == "text"
    )
    assert all(segment.type != "audio" for segment in segments)
