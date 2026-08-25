"""Unit tests for the Qwen3-VL ChatML template (text + image + video, class-based)."""

from veomni.models.seed_omni.modules.qwen3vl.text_encoder.chat_template import Qwen3VLChatTemplate
from veomni.models.seed_omni.utils.conversation import ConversationItem


class FakeTokenizer:
    """Minimal tokenizer for chat-template construction (markers are fixed literals)."""

    eos_token_id = 0

    def convert_tokens_to_ids(self, token):
        return 1

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def _template() -> Qwen3VLChatTemplate:
    return Qwen3VLChatTemplate(FakeTokenizer())


def test_chat_template_closes_each_turn():
    sample = [
        ConversationItem(type="text", value="hi", role="user"),
        ConversationItem(type="text", value="hello", role="assistant"),
    ]
    parts = _template().apply_chat_template(sample)
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
    parts = _template().apply_chat_template(sample)
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
    parts = _template().apply_chat_template(sample)
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
    tmpl = _template()
    sample = tmpl.apply_chat_template([ConversationItem(type="text", value="hi", role="user")])
    parts = tmpl.apply_generation_prompt(sample)
    assert parts[-1].value == "<|im_start|>assistant\n"
    assert parts[-1].meta["loss_mask"] == 0
    # the user turn was already closed by the chat template
    assert parts[-2].value == "<|im_end|>\n"


def test_chat_template_preserves_media_source():
    """Rebuilt image/video rows must keep ``item.source`` (regression).

    ``tokenize_conversation`` re-materialises every sample; dropping ``source``
    here would make the vision encoder's ``sources=[...]`` filter return an empty
    batch and crash ``torch.stack`` on the next forward.
    """
    sample = [
        ConversationItem(type="image", value="<pixels>", role="user", source="qwen3vl_vision"),
        ConversationItem(type="video", value="<frames>", role="user", source="qwen3vl_vision"),
    ]
    parts = _template().apply_chat_template(sample)
    media_rows = [p for p in parts if p.type in ("image", "video")]
    assert len(media_rows) == 2
    assert all(p.source == "qwen3vl_vision" for p in media_rows)


def test_dummy_rows_are_moved_to_tail():
    sample = [
        ConversationItem(type="text", value="hi", role="user"),
        ConversationItem(type="image", value="<dummy>", role="dummy"),
    ]
    parts = _template().apply_chat_template(sample)
    assert parts[-1].role == "dummy"
    assert parts[-1].type == "image"
