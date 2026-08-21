from types import SimpleNamespace

import pytest

from veomni.data.chat_template import (
    CHAT_TEMPLATE_REGISTRY,
    GptOssTokenizerTemplate,
    MultimodalChatTemplate,
    Qwen2VLChatTemplate,
    Qwen3VLChatTemplate,
    TokenizerTemplate,
    build_chat_template,
)
from veomni.utils.constants import IGNORE_INDEX, TYPE2INDEX


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


def test_build_chat_template_builds_both_kinds():
    # One build function covers both, taking whatever the caller's modality has:
    # the VLM trainer holds a processor, the text trainers only a tokenizer.
    multimodal = build_chat_template("qwen3vl", _Processor(_VisionTokenizer()))
    assert isinstance(multimodal, MultimodalChatTemplate)
    assert multimodal.tokenizer is multimodal.processor.tokenizer

    assert not isinstance(build_chat_template("chatml", _VisionTokenizer()), MultimodalChatTemplate)


def test_build_chat_template_still_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown ChatTemplate name"):
        build_chat_template("no_such_template", _VisionTokenizer())


@pytest.mark.parametrize("template_name", ["qwen2vl", "qwen3vl"])
def test_qwen_vl_rejects_pads_that_collide_with_unk(template_name):
    # A tokenizer missing the pads answers with its unk id, so the remap would
    # rewrite every unk in the batch into a modality sentinel -- marking
    # positions that hold no pixels, whether or not the sample carries any.
    class _PadlessTokenizer(_VisionTokenizer):
        unk_token_id = 7

        def convert_tokens_to_ids(self, token):
            return {"<|vision_start|>": 3, "<|vision_end|>": 4}.get(token, self.unk_token_id)

    with pytest.raises(ValueError, match=r"requires the <\|image_pad\|> and <\|video_pad\|> tokenizer tokens"):
        build_chat_template(template_name, _Processor(_PadlessTokenizer()))


@pytest.mark.parametrize("template_name", ["qwen2vl", "qwen3vl"])
def test_qwen_vl_accepts_a_tokenizer_that_only_knows_the_image_pad(template_name):
    # An image-only processor is a supported shape, and a tokenizer with no unk
    # id answers None for the video pad, which the remap compares against and
    # leaves alone. Rejecting it here would break a setup that trains today.
    class _ImageOnlyTokenizer(_VisionTokenizer):
        unk_token_id = None

        def convert_tokens_to_ids(self, token):
            return {"<|image_pad|>": 1, "<|vision_start|>": 3, "<|vision_end|>": 4}.get(token)

    template = build_chat_template(template_name, _Processor(_ImageOnlyTokenizer()))
    assert template.image_token_id == 1
    assert template.video_token_id is None


class _SpecialTokenTokenizer:
    """Emits one id per special token, so placeholders can be counted.

    ``_VisionTokenizer`` maps characters, which cannot represent ``<|video_pad|>``
    as a single id and so cannot express the placeholder-count contract.
    """

    pad_token_id = 0
    _SPECIALS = {
        "<|image_pad|>": 1,
        "<|video_pad|>": 2,
        "<|vision_start|>": 3,
        "<|vision_end|>": 4,
        "<|im_start|>": 5,
        "<|im_end|>": 6,
    }

    def convert_tokens_to_ids(self, token):
        return self._SPECIALS[token]

    def encode(self, text, add_special_tokens=False):
        ids, position = [], 0
        while position < len(text):
            for token, token_id in self._SPECIALS.items():
                if text.startswith(token, position):
                    ids.append(token_id)
                    position += len(token)
                    break
            else:
                # Offset well past the special ids and the TYPE2INDEX sentinels.
                ids.append(ord(text[position]) + 1000)
                position += 1
        return ids


class _Processor:
    """The minimum a multimodal template reads off a processor.

    Real processors carry the tokenizer and the video grid parameters together,
    which is why templates take one instead of a bare tokenizer.
    """

    def __init__(self, tokenizer, temporal_patch_size=2):
        self.tokenizer = tokenizer
        self.video_processor = SimpleNamespace(temporal_patch_size=temporal_patch_size)


def _video_metadata(total_num_frames, fps=2.0, frames_indices=None):
    from transformers.video_utils import VideoMetadata

    return VideoMetadata(
        total_num_frames=total_num_frames,
        fps=fps,
        frames_indices=list(range(total_num_frames)) if frames_indices is None else frames_indices,
    )


# Frame/token pairs read off the real Qwen3VLVideoProcessor (temporal_patch_size=2,
# merge_size=2) at 128x128: the processor pads an odd frame count up, so 15 and 16
# frames both yield grid_t=8 and 128 tokens.
@pytest.mark.parametrize("num_frames, num_video_tokens", [(4, 72), (8, 64), (15, 128), (16, 128), (17, 144)])
def test_qwen3vl_emits_one_video_placeholder_per_processor_token(num_frames, num_video_tokens):
    # The vision tower produces exactly num_video_tokens embeddings, and
    # process_sample_qwen_vl builds video_mask from these placeholder positions.
    # Any shortfall silently misaligns visual features against text positions.
    template = build_chat_template("qwen3vl", _Processor(_SpecialTokenTokenizer()))

    encoded = template.encode_messages(
        [("user", ("video", None))],
        {"video": [num_video_tokens]},
        video_metadata=[_video_metadata(num_frames)],
    )

    emitted = int((encoded["input_ids"] == TYPE2INDEX["input"]["video"]).sum())
    assert emitted == num_video_tokens


@pytest.mark.parametrize("num_frames, num_video_tokens", [(8, 65), (17, 100), (16, 1)])
def test_qwen3vl_rejects_a_video_token_count_it_cannot_lay_out_evenly(num_frames, num_video_tokens):
    # The chunk count is re-derived from frames_indices instead of being taken
    # from video_grid_thw, so it only matches the processor while the two agree.
    # Flooring the split would emit fewer placeholders than there are embeddings
    # and misalign every visual feature after the shortfall, silently.
    template = build_chat_template("qwen3vl", _Processor(_SpecialTokenTokenizer()))

    with pytest.raises(ValueError, match="divide evenly"):
        template.encode_messages(
            [("user", ("video", None))],
            {"video": [num_video_tokens]},
            video_metadata=[_video_metadata(num_frames)],
        )


@pytest.mark.parametrize("template_name", ["qwen2vl", "qwen3vl"])
@pytest.mark.parametrize("modality", ["image", "video"])
def test_encode_messages_names_the_modality_whose_counts_ran_out(template_name, modality):
    # Bare StopIteration from inside a dataloader worker gives no clue which
    # modality was short.
    template = build_chat_template(template_name, _Processor(_SpecialTokenTokenizer()))

    with pytest.raises(ValueError, match=f"{modality.capitalize()} token number is missing"):
        template.encode_messages(
            [("user", (modality, None))],
            {},
            video_metadata=[_video_metadata(16)],
        )


def test_qwen3vl_does_not_pad_the_caller_video_metadata():
    # VideoMetadata.frames_indices is declared list[int]. The odd frame count
    # forces the merge-size padding, which must not land on the caller's list.
    metadata = _video_metadata(15)
    frames_indices_before = list(metadata.frames_indices)

    template = build_chat_template("qwen3vl", _Processor(_SpecialTokenTokenizer()))
    template.encode_messages([("user", ("video", None))], {"video": [128]}, video_metadata=[metadata])

    assert list(metadata.frames_indices) == frames_indices_before


@pytest.mark.parametrize("num_image_tokens", [1, 7, 64])
def test_qwen2vl_emits_one_image_placeholder_per_processor_token(num_image_tokens):
    template = build_chat_template("qwen2vl", _Processor(_SpecialTokenTokenizer()))

    encoded = template.encode_messages([("user", ("image", None))], {"image": [num_image_tokens]})

    emitted = int((encoded["input_ids"] == TYPE2INDEX["input"]["image"]).sum())
    assert emitted == num_image_tokens


@pytest.mark.parametrize("template_name", ["qwen2vl", "qwen3vl"])
def test_input_ids_carry_no_negative_id_other_than_the_input_sentinels(template_name):
    # process_sample_qwen_vl only zeroes the input sentinels, so any other
    # negative id survives into the embedding lookup. An image in an assistant
    # turn used to produce one.
    template = build_chat_template(template_name, _Processor(_SpecialTokenTokenizer()))

    encoded = template.encode_messages(
        [("user", ("text", "draw a cat")), ("assistant", ("image", None))],
        {"image": [4]},
    )

    negatives = {int(i) for i in encoded["input_ids"] if i < 0}
    assert negatives <= {TYPE2INDEX["input"]["image"], TYPE2INDEX["input"]["video"]}


def test_qwen3vl_leaves_the_video_processor_alone_without_video():
    # The grid parameter is only meaningful for video, and a processor built for
    # an image-only model has no video side to read it from.
    class _ImageOnlyProcessor:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    template = build_chat_template("qwen3vl", _ImageOnlyProcessor(_SpecialTokenTokenizer()))

    encoded = template.encode_messages(
        [("user", ("image", None), ("text", "what is this"))],
        {"image": [4]},
    )

    assert int((encoded["input_ids"] == TYPE2INDEX["input"]["image"]).sum()) == 4


@pytest.mark.parametrize("temporal_patch_size, num_chunks", [(2, 4), (4, 2)])
def test_qwen3vl_chunks_time_the_way_its_processor_patched_it(temporal_patch_size, num_chunks):
    # 8 frames give 4 chunks at 2 and 2 chunks at 4. The template reads this off
    # its own processor so the layout cannot drift from the grid that processor
    # derived; models reusing this template need not all patch time by 2.
    template = build_chat_template(
        "qwen3vl", _Processor(_SpecialTokenTokenizer(), temporal_patch_size=temporal_patch_size)
    )

    encoded = template.encode_messages(
        [("user", ("video", None))],
        {"video": [64]},
        video_metadata=[_video_metadata(8)],
    )

    # One vision_start per time chunk.
    assert int((encoded["input_ids"] == 3).sum()) == num_chunks
    assert int((encoded["input_ids"] == TYPE2INDEX["input"]["video"]).sum()) == 64


def test_qwen2vl_tolerates_the_video_metadata_qwen3vl_needs():
    # One call site feeds every multimodal template, and configs/multimodal/qwen2_vl
    # ships this combination.
    template = build_chat_template("qwen2vl", _Processor(_SpecialTokenTokenizer()))

    encoded = template.encode_messages(
        [("user", ("text", "hello")), ("assistant", ("text", "hi"))],
        {},
        video_metadata=[],
    )

    assert len(encoded["input_ids"]) > 0


def test_qwen_vl_variants_share_one_tokenize_and_remap():
    # The variants differ only in how they render messages. If a subclass grows
    # its own copy of the tail, a change to the modality contract can land in one
    # and miss the other.
    assert Qwen3VLChatTemplate._tokenize_and_remap is Qwen2VLChatTemplate._tokenize_and_remap


def test_qwen_vl_variants_render_the_same_ids_for_a_text_only_turn():
    # Same input, same tail: the only intended difference is Qwen2-VL's system
    # prompt, so dropping it must make the two outputs identical.
    class NoSystemPrompt(Qwen2VLChatTemplate):
        def _get_system_message(self):
            return None

    messages = [("user", ("text", "hi")), ("assistant", ("text", "hello"))]
    qwen2 = NoSystemPrompt(_Processor(_SpecialTokenTokenizer())).encode_messages(messages, {})
    qwen3 = Qwen3VLChatTemplate(_Processor(_SpecialTokenTokenizer())).encode_messages(messages, {})

    for key in ("input_ids", "attention_mask", "labels"):
        assert qwen2[key].tolist() == qwen3[key].tolist(), key


@pytest.mark.parametrize("template_name", ["qwen2vl", "qwen3vl"])
def test_encode_messages_does_not_consume_the_caller_token_counts(template_name):
    # The caller derives these counts from grid_thw and owns the dict; a template
    # must read them, not consume them.
    num_tokens = {"image": [4]}
    template = build_chat_template(template_name, _Processor(_SpecialTokenTokenizer()))

    template.encode_messages([("user", ("image", None))], num_tokens)

    assert num_tokens == {"image": [4]}
