import pytest

from veomni.data.chat_template import (
    CHAT_TEMPLATE_REGISTRY,
    GptOssTokenizerTemplate,
    MultimodalChatTemplate,
    Qwen2VLChatTemplate,
    TokenizerTemplate,
    build_chat_template,
)
from veomni.utils.constants import IGNORE_INDEX


class _PrefixStableTokenizer:
    chat_template = "{{ messages }}"
    unk_token_id = -1

    def convert_tokens_to_ids(self, token):
        return {"<|return|>": 51, "<|end|>": 50}.get(token, self.unk_token_id)

    def apply_chat_template(self, messages, **kwargs):
        role_ids = {"system": 1, "user": 2, "assistant": 3, "tool": 4}
        input_ids = [99]
        for message in messages:
            input_ids.extend([role_ids[message["role"]], *message["content"]])
        return {"input_ids": input_ids}


def test_tokenizer_template_masks_non_assistant_turns_and_truncates():
    template = TokenizerTemplate(_PrefixStableTokenizer())
    messages = [
        {"role": "user", "content": [10, 11]},
        {"role": "assistant", "content": [20, 21]},
    ]

    encoded = template.encode_messages(messages, max_seq_len=4)

    assert encoded == {
        "input_ids": [11, 3, 20, 21],
        "attention_mask": [1, 1, 1, 1],
        "labels": [IGNORE_INDEX, 3, 20, 21],
    }


def test_gpt_oss_tokenizer_template_supports_terminal_token_rewrite():
    class TerminalRewritingTokenizer(_PrefixStableTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            encoded = super().apply_chat_template(messages, **kwargs)
            # GPT-OSS renders a terminal assistant turn with <|return|>, then
            # changes it to <|end|> when another turn follows.
            for index, message in enumerate(messages[:-1]):
                if message["role"] == "assistant":
                    encoded["input_ids"][index * 2 + 2] = 50
            if messages[-1]["role"] == "assistant":
                encoded["input_ids"][-1] = 51
            return encoded

    template = GptOssTokenizerTemplate(TerminalRewritingTokenizer())
    encoded = template.encode_messages(
        [
            {"role": "user", "content": [10]},
            {"role": "assistant", "content": [20]},
            {"role": "user", "content": [30]},
        ]
    )

    assert encoded == {
        "input_ids": [99, 2, 10, 3, 50, 2, 30],
        "attention_mask": [1, 1, 1, 1, 1, 1, 1],
        "labels": [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 3, 50, IGNORE_INDEX, IGNORE_INDEX],
    }


def test_tokenizer_template_rejects_structural_prefix_rewrite():
    class StructurallyRewritingTokenizer(_PrefixStableTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            encoded = super().apply_chat_template(messages, **kwargs)
            if len(messages) > 1:
                encoded["input_ids"].insert(1, 77)
            return encoded

    template = TokenizerTemplate(StructurallyRewritingTokenizer())

    with pytest.raises(ValueError, match="structurally rewrote"):
        template.encode_messages(
            [
                {"role": "user", "content": [10]},
                {"role": "assistant", "content": [20]},
            ]
        )


def test_tokenizer_template_rejects_terminal_rewrite():
    class TerminalRewritingTokenizer(_PrefixStableTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            if len(messages) == 1:
                return {"input_ids": [99, 3, 51]}
            return {"input_ids": [99, 3, 50, 2, 30]}

    template = TokenizerTemplate(TerminalRewritingTokenizer())

    with pytest.raises(ValueError, match="prefix-stable"):
        template.encode_messages(
            [
                {"role": "assistant", "content": [20]},
                {"role": "user", "content": [30]},
            ]
        )


@pytest.mark.parametrize("inserted_tokens", [[], [77]])
def test_gpt_oss_tokenizer_template_rejects_insertion_at_terminal_boundary(inserted_tokens):
    class BoundaryInsertionTokenizer(_PrefixStableTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            if len(messages) == 1:
                return {"input_ids": [99, 3, 51]}
            # An inserted <|end|> can look like a terminal replacement when
            # only absolute positions are compared, but the old terminal is
            # displaced instead of replaced.
            return {"input_ids": [99, 3, 50, *inserted_tokens, 51, 2, 30]}

    template = GptOssTokenizerTemplate(BoundaryInsertionTokenizer())

    with pytest.raises(ValueError, match="structurally rewrote"):
        template.encode_messages(
            [
                {"role": "assistant", "content": [20]},
                {"role": "user", "content": [30]},
            ]
        )


class _VisionTokenizer:
    """Minimal stand-in for a Qwen-VL tokenizer, enough to construct a template."""

    pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return {"<|image_pad|>": 1, "<|video_pad|>": 2, "<|vision_start|>": 3, "<|vision_end|>": 4}[token]

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 97 for c in text]


def test_qwen2_5vl_is_an_alias_of_qwen2vl():
    # The two names must share one class; a copy would let the two drift apart.
    assert CHAT_TEMPLATE_REGISTRY["qwen2_5vl"] is CHAT_TEMPLATE_REGISTRY["qwen2vl"]
    assert CHAT_TEMPLATE_REGISTRY["qwen2vl"] is Qwen2VLChatTemplate


@pytest.mark.parametrize(
    "template_name, is_multimodal",
    [
        ("chatml", False),
        ("default", False),
        ("gpt_oss", False),
        ("llama2", False),
        ("tokenizer", False),
        ("qwen2vl", True),
        ("qwen2_5vl", True),
        ("qwen3vl", True),
    ],
)
def test_one_registry_holds_both_template_kinds(template_name, is_multimodal):
    template_cls = CHAT_TEMPLATE_REGISTRY[template_name]
    assert issubclass(template_cls, MultimodalChatTemplate) is is_multimodal


def test_build_chat_template_accepts_the_expected_kind():
    assert isinstance(
        build_chat_template("qwen3vl", _VisionTokenizer(), expect_multimodal=True), MultimodalChatTemplate
    )
    assert not isinstance(
        build_chat_template("chatml", _VisionTokenizer(), expect_multimodal=False), MultimodalChatTemplate
    )


def test_build_chat_template_without_expectation_skips_the_kind_check():
    assert isinstance(build_chat_template("qwen3vl", _VisionTokenizer()), MultimodalChatTemplate)


@pytest.mark.parametrize(
    "template_name, expect_multimodal, message",
    [
        ("chatml", True, "is text-only"),
        ("qwen3vl", False, "is multimodal"),
    ],
)
def test_build_chat_template_rejects_the_wrong_kind(template_name, expect_multimodal, message):
    # Both kinds now live in one registry, so a config naming the wrong one
    # resolves. Without this guard it would only fail later inside a dataloader
    # worker, on the differing encode_messages signature.
    with pytest.raises(ValueError, match=message):
        build_chat_template(template_name, _VisionTokenizer(), expect_multimodal=expect_multimodal)


def test_build_chat_template_still_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown ChatTemplate name"):
        build_chat_template("no_such_template", _VisionTokenizer())
