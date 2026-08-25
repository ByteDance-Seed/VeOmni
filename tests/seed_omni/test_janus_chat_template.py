"""Janus chat-template expansion (class-based mirror of Jinja)."""

from veomni.models.seed_omni.modules.janus.text_encoder.chat_template import (
    IMAGE_PLACEHOLDER,
    JanusChatTemplate,
)
from veomni.models.seed_omni.utils.conversation import ConversationItem


# A distinctive substring of the fixed Janus default system prompt.
_SYSTEM_PROMPT_MARKER = "helpful language and vision assistant"


class FakeTokenizer:
    """Minimal tokenizer exposing the marker tokens / ids Janus resolves at init."""

    bos_token = "<s>"
    eos_token = "</s>"
    boi_token = "<boi>"
    eoi_token = "<eoi>"
    bos_token_id = 1
    eos_token_id = 2
    boi_token_id = 3
    eoi_token_id = 4
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def _template() -> JanusChatTemplate:
    return JanusChatTemplate(FakeTokenizer())


def test_expand_user_text_image_text_uses_boi_eoi_spans():
    sample = [
        ConversationItem(type="text", value="describe", role="user"),
        ConversationItem(type="image", value=None, role="user"),
        ConversationItem(type="text", value="more", role="user"),
        ConversationItem(type="text", value="hi", role="assistant"),
    ]
    tmpl = _template()
    parts = tmpl.apply_chat_template(sample)
    rendered = tmpl.render_template_string(parts)
    assert rendered.startswith("<s>")
    assert _SYSTEM_PROMPT_MARKER in rendered  # user image → system prompt prepended
    assert "<|User|>: " in rendered
    assert IMAGE_PLACEHOLDER in rendered
    assert "<boi>" in rendered
    assert "<eoi>" in rendered
    assert "describe" in rendered
    assert "\nmore" in rendered
    assert "\n\n<|Assistant|>:" in rendered
    assert "hi" in rendered
    assert rendered.endswith("</s>")
    assert all(p.type in ("text", "image") for p in parts)
    assert not any(p.meta.get("image_slot") for p in parts)


def test_apply_chat_template_preserves_media_source():
    """Rebuilt image rows must keep ``item.source`` (regression).

    The text encoder's ``tokenize_conversation`` re-materialises every sample, so
    if ``apply_chat_template`` drops ``source`` from the rebuilt image row, the
    VQVAE encoder's ``sources=[janus_vqvae]`` filter sees an empty batch and
    ``torch.stack`` crashes on the next forward.
    """
    sample = [
        ConversationItem(type="text", value="draw a cat", role="user"),
        ConversationItem(type="image", value=None, role="assistant", source="janus_vqvae"),
    ]
    parts = _template().apply_chat_template(sample)
    image_rows = [p for p in parts if p.type == "image"]
    assert len(image_rows) == 1
    assert image_rows[0].source == "janus_vqvae"


def test_apply_chat_template_preserves_dummy_source():
    """Worker-appended dummy rows are passed through verbatim (source kept)."""
    sample = [
        ConversationItem(type="text", value="hello", role="user"),
        ConversationItem(type="text", value="hi", role="assistant"),
        ConversationItem(type="image", value=None, role="dummy", source="janus_vqvae"),
    ]
    parts = _template().apply_chat_template(sample)
    dummy_rows = [p for p in parts if p.role == "dummy"]
    assert len(dummy_rows) == 1
    assert dummy_rows[0].source == "janus_vqvae"


def test_expand_text_only_t2i_skips_system_prompt():
    """T2I-style user text (no user image) matches official empty system_prompt."""
    sample = [
        ConversationItem(type="text", value="a cat on the moon", role="user"),
        ConversationItem(type="text", value="", role="assistant"),
    ]
    tmpl = _template()
    parts = tmpl.apply_chat_template(sample)
    rendered = tmpl.render_template_string(parts)
    assert rendered.startswith("<s><|User|>: ")
    assert _SYSTEM_PROMPT_MARKER not in rendered
