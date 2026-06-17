"""Unit tests for Qwen3-VL ChatML template helpers (text + image)."""

from veomni.models.seed_omni.conversation import ConversationItem
from veomni.models.seed_omni.modules.qwen3vl.text_encoder.chat_template import (
    Qwen3VLChatMarkers,
    apply_qwen3vl_chat_template,
    apply_qwen3vl_generation_prompt,
)


def _markers() -> Qwen3VLChatMarkers:
    return Qwen3VLChatMarkers(
        im_start_token="<|im_start|>",
        im_end_token="<|im_end|>",
        eos_token="<|im_end|>",
        assistant_prefix="<|im_start|>assistant\n",
        vision_start_token="<|vision_start|>",
        vision_end_token="<|vision_end|>",
    )


def test_chat_template_closes_each_turn():
    sample = [
        ConversationItem(type="text", value="hi", role="user"),
        ConversationItem(type="text", value="hello", role="assistant"),
    ]
    parts = apply_qwen3vl_chat_template(sample, _markers())
    values = [p.value for p in parts]
    assert values == [
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


def test_chat_template_wraps_image_with_vision_markers():
    sample = [
        ConversationItem(type="image", value="<pixels>", role="user"),
        ConversationItem(type="text", value="describe", role="user"),
    ]
    parts = apply_qwen3vl_chat_template(sample, _markers())
    types = [(p.type, p.value if p.type == "text" else "<img>") for p in parts]
    assert types == [
        ("text", "<|im_start|>user\n"),
        ("text", "<|vision_start|>"),
        ("image", "<img>"),
        ("text", "<|vision_end|>"),
        ("text", "describe"),
        ("text", "<|im_end|>\n"),
    ]


def test_chat_template_wraps_video_with_vision_markers():
    sample = [
        ConversationItem(type="video", value="<frames>", role="user"),
        ConversationItem(type="text", value="summarize", role="user"),
    ]
    parts = apply_qwen3vl_chat_template(sample, _markers())
    types = [(p.type, p.value if p.type == "text" else "<media>") for p in parts]
    assert types == [
        ("text", "<|im_start|>user\n"),
        ("text", "<|vision_start|>"),
        ("video", "<media>"),
        ("text", "<|vision_end|>"),
        ("text", "summarize"),
        ("text", "<|im_end|>\n"),
    ]


def test_generation_prompt_appends_assistant_prefix():
    sample = apply_qwen3vl_chat_template([ConversationItem(type="text", value="hi", role="user")], _markers())
    parts = apply_qwen3vl_generation_prompt(sample, _markers())
    assert parts[-1].value == "<|im_start|>assistant\n"
    assert parts[-1].meta["loss_mask"] == 0
    # the user turn was already closed by the chat template
    assert parts[-2].value == "<|im_end|>\n"


def test_dummy_rows_are_moved_to_tail():
    sample = [
        ConversationItem(type="text", value="hi", role="user"),
        ConversationItem(type="image", value="<dummy>", role="dummy"),
    ]
    parts = apply_qwen3vl_chat_template(sample, _markers())
    assert parts[-1].role == "dummy"
    assert parts[-1].type == "image"
