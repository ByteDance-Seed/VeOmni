"""Janus chat-template expansion (readable Python mirror of Jinja)."""

from veomni.models.seed_omni.conversation import ConversationItem
from veomni.models.seed_omni.modules.janus.text_encoder.chat_template import (
    IMAGE_PLACEHOLDER,
    JanusChatMarkers,
    apply_janus_chat_template,
    render_template_string,
)


def test_expand_user_text_image_text_uses_boi_eoi_spans():
    sample = [
        ConversationItem(type="text", value="describe", role="user"),
        ConversationItem(type="image", value=None, role="user"),
        ConversationItem(type="text", value="more", role="user"),
        ConversationItem(type="text", value="hi", role="assistant"),
    ]
    markers = JanusChatMarkers(
        bos_token="<s>",
        eos_token="</s>",
        boi_token="<boi>",
        eoi_token="<eoi>",
        system_prompt="SYS\n\n",
        user_prefix="<|User|>: ",
        assistant_prefix="\n\n<|Assistant|>:",
    )
    parts = apply_janus_chat_template(sample, markers)
    rendered = render_template_string(parts)
    assert rendered.startswith("<s>")
    assert "SYS" in rendered
    assert "<|User|>: " in rendered
    assert IMAGE_PLACEHOLDER not in rendered
    assert "<boi>" in rendered
    assert "<eoi>" in rendered
    assert "describe" in rendered
    assert "\nmore" in rendered
    assert "\n\n<|Assistant|>:" in rendered
    assert "hi" in rendered
    assert rendered.endswith("</s>")
    assert all(p.type in ("text", "image") for p in parts)
    assert not any(p.meta.get("image_slot") for p in parts)


def test_expand_text_only_t2i_skips_system_prompt():
    """T2I-style user text (no user image) matches official empty system_prompt."""
    sample = [
        ConversationItem(type="text", value="a cat on the moon", role="user"),
        ConversationItem(type="text", value="", role="assistant"),
    ]
    markers = JanusChatMarkers(
        bos_token="<s>",
        eos_token="</s>",
        boi_token="<boi>",
        eoi_token="<eoi>",
        system_prompt="SYS\n\n",
        user_prefix="<|User|>: ",
        assistant_prefix="\n\n<|Assistant|>:",
    )
    parts = apply_janus_chat_template(sample, markers)
    rendered = render_template_string(parts)
    assert rendered.startswith("<s><|User|>: ")
    assert "SYS" not in rendered
