"""Unit tests for the Qwen3 ChatML template (class-based)."""

from veomni.models.seed_omni.modules.qwen3.text_encoder.chat_template import Qwen3ChatTemplate
from veomni.models.seed_omni.utils.conversation import ConversationItem


class FakeTokenizer:
    """Minimal tokenizer for chat-template construction (markers are fixed literals)."""

    eos_token_id = 0

    def convert_tokens_to_ids(self, token):
        return 1

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def _template() -> Qwen3ChatTemplate:
    return Qwen3ChatTemplate(FakeTokenizer())


def test_apply_qwen3_chat_template_wraps_roles():
    sample = [
        ConversationItem(type="text", value="hi", role="user"),
        ConversationItem(type="text", value="hello", role="assistant"),
    ]
    parts = _template().apply_chat_template(sample)
    # Standard Qwen ChatML: every turn closed with <|im_end|>\n.
    assert parts[0].value == "<|im_start|>user\n"
    assert parts[1].value == "hi"
    assert parts[2].value == "<|im_end|>\n"  # user turn closed (unsupervised)
    assert parts[2].meta["loss_mask"] == 0
    assert parts[3].value == "<|im_start|>assistant\n"
    assert parts[4].value == "hello"
    assert parts[4].meta["loss_mask"] == 1
    assert parts[5].value == "<|im_end|>\n"  # assistant turn closed (supervised)
    assert parts[5].meta["loss_mask"] == 1


def test_apply_qwen3_generation_prompt_appends_assistant_prefix():
    # Input is an already-templated (turn-closed) prompt; generation prompt only
    # appends the assistant prefix.
    sample = _template().apply_chat_template([ConversationItem(type="text", value="hi", role="user")])
    parts = _template().apply_generation_prompt(sample)
    assert parts[-1].value == "<|im_start|>assistant\n"
    assert parts[-2].value == "<|im_end|>\n"  # closing of the user turn from apply_chat_template
