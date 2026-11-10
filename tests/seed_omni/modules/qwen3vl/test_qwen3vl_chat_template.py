"""Unit tests for the Qwen3-VL ChatML template (text + image + video, class-based)."""

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.models.seed_omni.modules.qwen3vl.text_encoder.chat_template import Qwen3VLChatTemplate
from veomni.models.seed_omni.utils.conversation import _IMG_TAG_KEY, ConversationItem


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


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


def test_chat_template_passes_media_rows_through():
    """Rebuilt image/video rows must be the original items (regression).

    ``tokenize_conversation`` re-materialises every sample; rebuilding a media row
    without its ``role`` / ``meta`` here would make the vision encoder's
    user-image filter return an empty batch and crash on the next forward.
    """
    media = [
        ConversationItem(type="image", value="<pixels>", role="user", meta={_IMG_TAG_KEY: "und"}),
        ConversationItem(type="video", value="<frames>", role="user", meta={_IMG_TAG_KEY: "und"}),
    ]
    parts = _template().apply_chat_template(media)
    media_rows = [p for p in parts if p.type in ("image", "video")]
    assert len(media_rows) == 2
    assert all(row is item for row, item in zip(media_rows, media))


def test_dummy_rows_are_moved_to_tail():
    dummy = ConversationItem(type="image", value="<dummy>", role="user", is_dummy=True, meta={_IMG_TAG_KEY: "und"})
    sample = [ConversationItem(type="text", value="hi", role="user"), dummy]
    parts = _template().apply_chat_template(sample)
    assert parts[-1] is dummy
    assert [p for p in parts if p.is_dummy] == [dummy]


def test_chat_template_splits_a_video_into_one_timestamped_block_per_frame():
    """HF emits ``<t seconds><|vision_start|>…<|vision_end|>`` per temporal patch, and
    the backbone's M-RoPE restarts per frame to match. A single block for the whole
    clip would hand the model a different token sequence than the checkpoint saw."""
    import torch

    from veomni.models.seed_omni.modules.qwen3vl.vision.processing import _OMNI_GRID

    tmpl = _template()
    # A video item holds *raw patch rows*: 3 frames x (4x4) = 48 rows.
    video = ConversationItem(type="video", value=torch.zeros(48, 8), role="user", meta={_OMNI_GRID: [3, 4, 4]})
    parts = tmpl.apply_chat_template([video])
    rows = [(p.type, p.value if p.type == "text" else tuple(p.value.shape)) for p in parts]

    timestamps = [value for kind, value in rows if kind == "text" and value.endswith("seconds>")]
    # No decode metadata, so the stamps are synthetic (one patch apart) -- the
    # layout is what this asserts; real timestamps need `VideoMetadata`.
    assert timestamps == ["<0.0 seconds>", "<2.0 seconds>", "<4.0 seconds>"]
    # One block per frame, each carrying that frame's own raw patch rows.
    video_rows = [value for kind, value in rows if kind == "video"]
    assert video_rows == [(16, 8)] * 3, "each frame keeps its own raw patch rows"
    assert sum(kind == "text" and value == "<|vision_start|>" for kind, value in rows) == 3


def test_chat_template_uses_real_timestamps_when_the_item_has_decode_metadata():
    """The `<t seconds>` stamps have to be the clip's real ones — frame index over
    source fps, then the temporal patch's mid-time, as
    `Qwen3VLProcessor._calculate_timestamps` computes them."""
    import torch

    from veomni.data.seed_omni.utils.media_metadata import VIDEO_METADATA_KEY, VideoMetadata
    from veomni.models.seed_omni.modules.qwen3vl.vision.processing import _OMNI_GRID

    # 8 source frames at 2 fps -> 4 temporal patches of 2 frames, mid-times
    # 0.25 / 1.25 / 2.25 / 3.25. 4 frames per 4x4 grid = 64 raw patch rows.
    metadata = VideoMetadata(fps=2.0, total_num_frames=8, frames_indices=list(range(8)), requested_fps=2.0)
    video = ConversationItem(
        type="video",
        value=torch.zeros(4 * 4 * 4, 8),
        role="user",
        meta={_OMNI_GRID: [4, 4, 4], VIDEO_METADATA_KEY: metadata},
    )
    parts = _template().apply_chat_template([video])
    stamps = [p.value for p in parts if p.type == "text" and p.value.endswith("seconds>")]
    assert stamps == ["<0.2 seconds>", "<1.2 seconds>", "<2.2 seconds>", "<3.2 seconds>"]


def test_video_split_rewrites_each_segments_grid_and_token_len():
    """Each segment is a complete visual item, so its metadata must describe that
    segment alone: grid `[1, h, w]`, and its own share of the merged-token count.

    Copying the clip's `[t, h, w]` makes every downstream consumer build `t`
    frames' worth of M-RoPE/positions for one frame's rows — a shape error in
    training and in inference alike. The row counts a split produces are asserted
    against the grid the split *writes*, not the one it was given, which is what
    the earlier test missed."""
    import torch

    from veomni.models.seed_omni.modules.qwen3vl.llm.modeling import qwen3vl_vision_position_ids
    from veomni.models.seed_omni.modules.qwen3vl.vision.processing import _OMNI_GRID
    from veomni.models.seed_omni.utils.conversation import _OMNI_TOKEN_LEN

    # 3 frames x (4x4 patches) = 48 raw rows; the clip contributes 12 merged tokens.
    video = ConversationItem(
        type="video",
        value=torch.zeros(48, 8),
        role="user",
        meta={_OMNI_GRID: [3, 4, 4], _OMNI_TOKEN_LEN: 12},
    )
    segments = [p for p in _template().apply_chat_template([video]) if p.type == "video"]
    assert len(segments) == 3

    for segment in segments:
        grid = segment.meta[_OMNI_GRID]
        assert grid == [1, 4, 4], "a segment declares one frame, not the whole clip"
        # The row count this segment carries must be what its grid implies.
        assert segment.value.shape[0] == grid[0] * grid[1] * grid[2]
        # And the merged-token count must match the positions the backbone rebuilds.
        positions = qwen3vl_vision_position_ids(0, torch.tensor(grid), merge=2)
        assert positions.shape[1] == segment.meta[_OMNI_TOKEN_LEN] == 4

    # Splitting neither duplicates nor loses tokens.
    assert sum(s.meta[_OMNI_TOKEN_LEN] for s in segments) == 12


def test_video_split_feeds_the_packer_one_frame_per_visual():
    """The packed preprocessor must receive per-frame grids that match each
    segment's row count, so the packer's `t*h*w` split sees one frame per item."""
    import torch

    from veomni.models.seed_omni.modules.qwen3vl.text_encoder.processing import Qwen3VLTextEncoderPreprocessor
    from veomni.models.seed_omni.modules.qwen3vl.vision.processing import _OMNI_GRID
    from veomni.models.seed_omni.processing_omni import OmniProcessor

    preprocessor = Qwen3VLTextEncoderPreprocessor(_template(), packed_preprocess=True, spatial_merge_size=2)
    processor = OmniProcessor({"qwen3vl_text_encoder": preprocessor})
    sample = [
        ConversationItem(type="text", value="describe", role="user"),
        ConversationItem(type="video", value=torch.zeros(48, 8), role="user", meta={_OMNI_GRID: [3, 4, 4]}),
        ConversationItem(type="text", value="ok", role="assistant"),
    ]
    packed = processor.preprocess(sample)
    assert packed["image_grid_thw"].tolist() == [[1, 4, 4]] * 3
    # One merged visual token per frame lands in the packed sequence.
    assert int(packed["visual_pos_mask"].sum()) == 12
