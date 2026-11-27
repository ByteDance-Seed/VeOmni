"""Unit tests for the Qwen3-Omni ChatML template (text + image + video + audio)."""

import numpy as np
import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
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


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class FakeTokenizer:
    """Minimal character-level tokenizer for chat-template tests."""

    eos_token_id = 0

    def convert_tokens_to_ids(self, token):
        return 1

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def _template(*, add_speech_system_prompt: bool = False) -> Qwen3OmniChatTemplate:
    return Qwen3OmniChatTemplate(FakeTokenizer(), add_speech_system_prompt=add_speech_system_prompt)


def test_chat_template_turns_generation_prompt_and_validation():
    tmpl = _template()
    parts = tmpl.apply_chat_template(
        [
            ConversationItem(type="text", value="hi", role="user"),
            ConversationItem(type="text", value="hello", role="assistant"),
        ]
    )
    assert [p.value for p in parts] == [
        "<|im_start|>user\n",
        "hi",
        "<|im_end|>\n",
        "<|im_start|>assistant\n",
        "hello",
        "<|im_end|>\n",
    ]
    assert [parts[1].meta["loss_mask"], parts[2].meta["loss_mask"]] == [0, 0]
    assert [parts[4].meta["loss_mask"], parts[5].meta["loss_mask"]] == [1, 1]

    gen = tmpl.apply_generation_prompt(
        tmpl.apply_chat_template([ConversationItem(type="text", value="hi", role="user")])
    )
    assert [gen[-2].value, gen[-1].value] == ["<|im_end|>\n", "<|im_start|>assistant\n"]
    assert gen[-1].meta["loss_mask"] == 0

    with pytest.raises(ValueError, match="text/image/video/audio"):
        tmpl.apply_chat_template([ConversationItem(type="output", value="x", role="assistant")])


def test_chat_template_wraps_multimodal_items_preserves_meta_and_orders_dummies():
    tmpl = _template()
    sample = [
        ConversationItem(type="image", value="<pixels>", role="user", meta={"_omni_grid": [1, 2, 2]}),
        ConversationItem(
            type="video",
            value="<frames>",
            role="user",
            meta={AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=16_000, num_samples=8)},
        ),
        ConversationItem(type="audio", value=np.zeros(8, dtype=np.float32), role="user"),
        ConversationItem(type="text", value="describe", role="user"),
        ConversationItem(type="audio", value=np.zeros(8, dtype=np.float32), role="user", is_dummy=True),
    ]
    parts = tmpl.apply_chat_template(sample)
    assert [(p.type, p.value if p.type == "text" else ("<dummy>" if p.is_dummy else p.type)) for p in parts] == [
        ("text", "<|im_start|>user\n"),
        ("text", "<|vision_start|>"),
        ("image", "image"),
        ("text", "<|vision_end|>"),
        ("text", "<|vision_start|>"),
        ("video", "video"),
        ("text", "<|vision_end|>"),
        ("text", "<|audio_start|>"),
        ("audio", "audio"),
        ("text", "<|audio_end|>"),
        ("text", "describe"),
        ("text", "<|im_end|>\n"),
        ("audio", "<dummy>"),
    ]
    # Media markers are not supervised, media metadata survives verbatim, and dummies move to the tail.
    assert parts[7].meta["loss_mask"] == 0 and parts[9].meta["loss_mask"] == 0
    assert parts[2].meta["_omni_grid"] == [1, 2, 2]
    assert parts[-1].is_dummy and parts[-1].type == "audio"


def test_assistant_audio_is_held_out_while_user_audio_stays_in_turn():
    heard = ConversationItem(type="audio", value=np.zeros(16000, dtype=np.float32), role="user")
    spoken = ConversationItem(type="audio", value=np.zeros(24000, dtype=np.float32), role="assistant")
    parts = _template().apply_chat_template(
        [
            heard,
            ConversationItem(type="text", value="say hi", role="user"),
            ConversationItem(type="text", value="hi there", role="assistant"),
            spoken,
        ]
    )
    assert [p.value for p in parts if p.type == "text"] == [
        "<|im_start|>user\n",
        "<|audio_start|>",
        "<|audio_end|>",
        "say hi",
        "<|im_end|>\n",
        "<|im_start|>assistant\n",
        "hi there",
        "<|im_end|>\n",
    ]
    assert parts[2] is heard
    assert parts[-1] is spoken


def test_talker_prompt_segments_and_supervised_mask():
    tmpl = _template()
    conversation = tmpl.tokenize_conversation(
        [
            ConversationItem(type="audio", value=np.zeros(16000, dtype=np.float32), role="user"),
            ConversationItem(type="text", value="what did I say", role="user"),
            ConversationItem(type="text", value="you said hi", role="assistant"),
            ConversationItem(type="audio", value=np.zeros(24000, dtype=np.float32), role="assistant"),
        ]
    )
    for item in conversation:
        if item.type == "audio" and item.role == "user":
            item.value = torch.zeros(7, 4)

    segments = talker_prompt_segments(conversation)
    supervised = talker_supervised_mask(conversation)

    # Spoken assistant audio is excluded; user audio stays and is unsupervised.
    assert [s.type for s in segments].count("audio") == 1
    assert supervised.numel() == sum(s.length for s in segments)
    assert not supervised[:7].any()
    # Boundary survives tokenize_conversation popping loss_mask and merging assistant prefix.
    assert all("loss_mask" not in item.meta for item in conversation)
    assert supervised.any() and not supervised.all()
    assert 0 < int(supervised.nonzero()[0]) < supervised.numel()

    infer_conv = tmpl.tokenize_conversation(
        [ConversationItem(type="text", value="say hi", role="user")], add_generation_prompt=True
    )
    assert not talker_supervised_mask(infer_conv).any()


def test_speech_system_prompt_option():
    speech_block = f"<|im_start|>system\n{SPEECH_SYSTEM_PROMPT}<|im_end|>\n"
    speech_tmpl = _template(add_speech_system_prompt=True)

    # Off by default; empty input stays empty even when enabled.
    assert [
        p.value for p in _template().apply_chat_template([ConversationItem(type="text", value="hi", role="user")])
    ] == [
        "<|im_start|>user\n",
        "hi",
        "<|im_end|>\n",
    ]
    assert speech_tmpl.apply_chat_template([]) == []

    parts = speech_tmpl.apply_chat_template([ConversationItem(type="text", value="hi", role="user")])
    assert [p.value for p in parts] == [
        "<|im_start|>system\n",
        SPEECH_SYSTEM_PROMPT,
        "<|im_end|>\n",
        "<|im_start|>user\n",
        "hi",
        "<|im_end|>\n",
    ]
    assert all(p.role == "user" and p.meta["loss_mask"] == 0 for p in parts)

    conversation = speech_tmpl.tokenize_conversation(
        [
            ConversationItem(type="text", value="say hi", role="user"),
            ConversationItem(type="text", value="hi", role="assistant"),
            ConversationItem(type="audio", value=np.zeros(24000, dtype=np.float32), role="assistant"),
        ]
    )
    rendered = "".join(chr(t) for item in conversation if item.type == "text" for t in item.value.tolist())
    assert rendered.startswith(f"{speech_block}<|im_start|>user\n")

    supervised = talker_supervised_mask(conversation)
    assert not supervised[: len(speech_block)].any()
    assert supervised.any()

    segments = talker_prompt_segments(conversation)
    assert all(s.type != "audio" for s in segments)
    assert sum(s.length for s in segments) == sum(int(i.value.size(0)) for i in conversation if i.type == "text")
