"""Unit tests for Qwen3 ChatML template helpers."""

from veomni.models.seed_omni.conversation import ConversationItem
from veomni.models.seed_omni.modules.qwen3.text_encoder.chat_template import (
    Qwen3ChatMarkers,
    apply_qwen3_chat_template,
    apply_qwen3_generation_prompt,
)


def _markers() -> Qwen3ChatMarkers:
    return Qwen3ChatMarkers(
        im_start_token="<|im_start|>",
        im_end_token="</turn>",
        eos_token="<|endoftext|>",
        assistant_prefix="<|im_start|>assistant\n",
    )


def test_apply_qwen3_chat_template_wraps_roles():
    sample = [
        ConversationItem(type="text", value="hi", role="user"),
        ConversationItem(type="text", value="hello", role="assistant"),
    ]
    parts = apply_qwen3_chat_template(sample, _markers())
    assert parts[0].value == "<|im_start|>user\n"
    assert parts[1].value == "hi"
    assert parts[2].value == "<|im_start|>assistant\n"
    assert parts[3].value == "hello"
    assert parts[4].value == "</turn>\n"
    assert parts[3].meta["loss_mask"] == 1


def test_apply_qwen3_generation_prompt_appends_assistant_prefix():
    sample = [
        ConversationItem(type="text", value="<|im_start|>user\n", role="user", meta={"loss_mask": 0}),
        ConversationItem(type="text", value="hi", role="user", meta={"loss_mask": 0}),
    ]
    parts = apply_qwen3_generation_prompt(sample, _markers())
    assert parts[-1].value == "<|im_start|>assistant\n"
    assert parts[-2].value == "</turn>\n"
