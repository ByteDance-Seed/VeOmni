"""Qwen3-Omni vision tower: ViT round-trip and the ``second_per_grid`` contract.

The tower's weights are upstream's (patchgen-generated), so the tests focus on
what VeOmni owns: the carrier round-trip (patchify to merged tokens scattered
back onto items) and the TMRoPE timeline metadata the backbone *requires* for
video items.
"""

import json
import os

import pytest
import torch

from veomni.data.seed_omni.utils.media_metadata import VIDEO_METADATA_KEY, VideoMetadata
from veomni.data.seed_omni.utils.video import VideoInputs
from veomni.models.seed_omni.modules.qwen3omni.vision.configuration import Qwen3OmniVisionEncoderConfig
from veomni.models.seed_omni.modules.qwen3omni.vision.modeling import (
    Qwen3OmniVisionEncoder,
    build_qwen3omni_vit_metadata,
)
from veomni.models.seed_omni.modules.qwen3omni.vision.processing import (
    _OMNI_GRID,
    Qwen3OmniVisionPreprocessor,
    _second_per_grid,
)
from veomni.models.seed_omni.utils.conversation import ConversationItem


CHECKPOINT = "/opt/tiger/models/Qwen3-Omni-30B-A3B-Instruct"
requires_checkpoint = pytest.mark.skipif(
    not os.path.isdir(CHECKPOINT), reason=f"needs the Qwen3-Omni checkpoint at {CHECKPOINT}"
)


def _tiny_vision():
    """Structurally faithful tower from the real config, shrunk, random weights."""
    with open(os.path.join(CHECKPOINT, "config.json")) as handle:
        raw = json.load(handle)["thinker_config"]["vision_config"]
    tiny = dict(raw)
    tiny.update(
        depth=4,
        hidden_size=32,
        out_hidden_size=64,
        num_heads=4,
        intermediate_size=32,
        deepstack_visual_indexes=[1, 2],
        dtype="float32",
    )
    torch.manual_seed(0)
    return Qwen3OmniVisionEncoder(Qwen3OmniVisionEncoderConfig(vision_config=tiny)).eval()


def _patch_item(item_type, grid, cfg, role="user", meta=None):
    """An item in the post-CPU-preprocessor form: patches on value, grid on meta."""
    t, h, w = grid
    pixel_row = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    torch.manual_seed(sum(grid))
    return ConversationItem(
        type=item_type,
        value=torch.randn(t * h * w, pixel_row),
        role=role,
        meta={_OMNI_GRID: list(grid), **(meta or {})},
    )


@requires_checkpoint
def test_generate_scatters_merged_tokens_and_deepstack_per_item():
    """Each item must receive exactly its own slice of the flat ViT output.

    The tower returns every item's merged tokens concatenated, so a wrong split
    would silently give one item another's patches — shapes alone would still
    look plausible, hence the per-item token-count assertions.
    """
    module = _tiny_vision()
    cfg = module.config.vision_config
    merge_area = cfg.spatial_merge_size**2
    grids = [(1, 4, 4), (1, 2, 6), (2, 2, 2)]
    items = [
        _patch_item("image", grids[0], cfg),
        _patch_item("image", grids[1], cfg),
        _patch_item("video", grids[2], cfg, meta={"second_per_grid": 1.0}),
    ]
    conversation_list = [ConversationItem(type="text", value=torch.zeros(2), role="user"), *items]

    with torch.no_grad():
        module.generate(conversation_list=conversation_list)

    for item, grid in zip(items, grids, strict=True):
        expected_merged = (grid[0] * grid[1] * grid[2]) // merge_area
        assert item.value.shape == (expected_merged, cfg.out_hidden_size)
        # grid_thw is republished for the backbone's TMRoPE; the private stash is gone.
        assert item.meta["grid_thw"].tolist() == list(grid)
        assert _OMNI_GRID not in item.meta
        assert len(item.meta["deepstack"]) == len(cfg.deepstack_visual_indexes)
        for layer in item.meta["deepstack"]:
            assert layer.shape == (expected_merged, cfg.out_hidden_size)

    # The video item's timeline metadata must survive the scatter — the backbone
    # reads it after this and raises if it is missing.
    assert items[2].meta["second_per_grid"] == 1.0


@requires_checkpoint
def test_generate_is_a_noop_without_visual_items():
    module = _tiny_vision()
    conversation_list = [ConversationItem(type="text", value=torch.zeros(3), role="user")]
    outputs = module.generate(conversation_list=conversation_list)
    assert outputs["conversation_list"] is conversation_list


@requires_checkpoint
def test_images_and_videos_share_one_vit_forward():
    """Mixing modalities must not change any item's result.

    One forward per step is an FSDP requirement, so image and video patches are
    concatenated into a single batch. That is only safe because ViT attention is
    segmented per frame by cu_seqlens: this pins that no cross-item leakage
    occurs by comparing against encoding each item alone.
    """
    module = _tiny_vision()
    cfg = module.config.vision_config

    image_grid, video_grid = (1, 4, 4), (2, 2, 2)
    together = [
        _patch_item("image", image_grid, cfg),
        _patch_item("video", video_grid, cfg, meta={"second_per_grid": 1.0}),
    ]
    alone = [
        [_patch_item("image", image_grid, cfg)],
        [_patch_item("video", video_grid, cfg, meta={"second_per_grid": 1.0})],
    ]

    with torch.no_grad():
        module.generate(conversation_list=list(together))
        for single in alone:
            module.generate(conversation_list=list(single))

    torch.testing.assert_close(together[0].value, alone[0][0].value)
    torch.testing.assert_close(together[1].value, alone[1][0].value)


def _video_item(*, fps, frames_indices, requested_fps=None) -> ConversationItem:
    return ConversationItem(
        type="video",
        value=None,
        role="user",
        meta={
            VIDEO_METADATA_KEY: VideoMetadata(
                fps=fps,
                total_num_frames=(frames_indices[-1] + 1) if frames_indices else 0,
                frames_indices=list(frames_indices),
                requested_fps=requested_fps,
            )
        },
    )


def test_second_per_grid_spans_the_frames_that_were_actually_kept():
    """The value the backbone scales TMRoPE by, measured off the kept frames.

    Frames 0, 15, 30… of a 30 fps source are 0.5 s apart, so two of them per
    temporal patch span 1 s.
    """
    item = _video_item(fps=30.0, frames_indices=[0, 15, 30, 45])
    assert _second_per_grid(item, 2) == pytest.approx(1.0)
    assert _second_per_grid(item, 1) == pytest.approx(0.5)


def test_second_per_grid_does_not_divide_by_the_rate_that_was_asked_for():
    """The #1165 regression guard, on the scalar rather than on timestamps.

    A 25 fps source asked for 2 fps cannot be sampled at 2 fps: the sampler
    keeps whole source frames, so it lands on every 12th or 13th one and the
    kept frames sit ~0.48 s apart. ``temporal_patch_size / requested_fps`` says
    0.5 s, and the error compounds across the clip — with in-video audio, the
    two streams slide apart on a clock they are meant to share.
    """
    item = _video_item(fps=25.0, frames_indices=[0, 12, 24, 36, 48], requested_fps=2.0)

    measured = _second_per_grid(item, 1)
    assert measured == pytest.approx(12.0 / 25.0)
    assert measured != pytest.approx(1.0 / 2.0)


def test_a_video_item_with_no_metadata_at_all_is_refused():
    """The preprocessor cannot invent a timeline, and guessing one is the bug."""
    item = ConversationItem(type="video", value=None, role="user")
    with pytest.raises(ValueError, match="carries no meta"):
        _second_per_grid(item, 2)


@requires_checkpoint
def test_preprocessor_sets_second_per_grid_for_videos_only():
    """Video items get a timeline; still images keep the backbone's 1.0 default.

    This is the hand-off the backbone's packer depends on: it raises for a video
    item without ``second_per_grid``, so this preprocessor is the only supplier.
    """
    preprocessor = Qwen3OmniVisionPreprocessor.from_pretrained(CHECKPOINT)
    temporal_patch_size = preprocessor._video_processor.temporal_patch_size

    # Real payload and real metadata rather than stand-ins: the payload goes
    # through the HF processor, and the timeline comes off the same dataclass the
    # data layer writes — a stub would keep passing if either were re-derived,
    # and a wrong TMRoPE timeline fails silently.
    frames = torch.randint(0, 255, (4, 3, 64, 64), dtype=torch.uint8)
    video_item = _video_item(fps=8.0, frames_indices=[0, 2, 4, 6], requested_fps=4.0)
    video_item.value = VideoInputs(video=frames)
    image_item = ConversationItem(
        type="image", value=torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8), role="user"
    )

    preprocessor.preprocess_conversations([[image_item, video_item]], inference=True)

    # Frames 2 apart on an 8 fps source are 0.25 s apart.
    assert video_item.meta["second_per_grid"] == pytest.approx(temporal_patch_size * 0.25)
    # Still images have no timeline of their own; the backbone defaults them to 1.0.
    assert "second_per_grid" not in image_item.meta
    # Both were patchified, with grids stashed for the main process.
    for item in (image_item, video_item):
        assert isinstance(item.value, torch.Tensor)
        assert len(item.meta[_OMNI_GRID]) == 3


@requires_checkpoint
def test_processors_come_from_the_checkpoint_not_class_defaults():
    """Qwen3-Omni ships no video_preprocessor_config.json, so this could regress silently.

    ``Qwen2VLVideoProcessor``'s class default ``patch_size`` is 14 while
    Qwen3-Omni uses 16; loading from the checkpoint dir is what makes it fall
    back to the shared ``preprocessor_config.json``.
    """
    preprocessor = Qwen3OmniVisionPreprocessor.from_pretrained(CHECKPOINT)
    with open(os.path.join(CHECKPOINT, "preprocessor_config.json")) as handle:
        expected = json.load(handle)
    for processor in (preprocessor._image_processor, preprocessor._video_processor):
        assert processor.patch_size == expected["patch_size"]
        assert processor.temporal_patch_size == expected["temporal_patch_size"]
        assert processor.merge_size == expected["merge_size"]
    # No resampling: this is why the clip's own fps is the right divisor.
    assert preprocessor._video_processor.do_sample_frames is False


@requires_checkpoint
def test_encode_is_bit_identical_to_upstream():
    """The patchgen ViT must match upstream exactly when SP is off.

    Guards the wiring rather than the weights: the patched tower takes
    ``hidden_states`` (not ``pixel_values``) and casts its interpolated
    positional embedding back to the input dtype, so a mis-wired ``_encode``
    would still run and still return plausible shapes.
    """
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeVisionEncoder

    module = _tiny_vision()
    cfg = module.config.vision_config
    reference = Qwen3OmniMoeVisionEncoder._from_config(cfg).eval()
    missing, unexpected = reference.load_state_dict(module.visual.state_dict(), strict=False)
    assert not missing and not unexpected

    grids = [(1, 4, 4), (2, 2, 2)]
    items = [_patch_item("image", grids[0], cfg), _patch_item("video", grids[1], cfg)]
    pixel_values = torch.cat([item.value for item in items], dim=0)
    grid_thw = torch.tensor([list(grid) for grid in grids], dtype=torch.long)

    with torch.no_grad():
        embeds, deepstack = module._encode(pixel_values, grid_thw, None)
        expected = reference(hidden_states=pixel_values.type(reference.dtype), grid_thw=grid_thw)

    torch.testing.assert_close(embeds, expected.pooler_output, rtol=0, atol=0)
    for actual, want in zip(deepstack, expected.deepstack_features, strict=True):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)


def test_vit_metadata_segments_attention_per_video_frame():
    """cu_seqlens must break a clip into one segment per frame, not one per clip."""
    metadata = build_qwen3omni_vit_metadata([[2, 2, 2], [1, 4, 4]], spatial_merge_size=2)
    # Frames of 4 patches each (2x2), then one 16-patch image (4x4).
    assert metadata["cu_seqlens"].tolist() == [0, 4, 8, 24]
    assert metadata["max_seqlen"] == 16
    assert metadata["grid_thw_list"] == [[2, 2, 2], [1, 4, 4]]


def _accelerated_tower():
    """Structure-only tower for the packed hooks — no checkpoint needed."""
    from veomni.models.seed_omni.modules.qwen3omni.vision.accelerated.accelerated import (
        Qwen3OmniVisionEncoderAccelerated,
    )

    config = Qwen3OmniVisionEncoderConfig(
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "out_hidden_size": 64,
            "num_heads": 2,
            "intermediate_size": 32,
            "deepstack_visual_indexes": [0],
        }
    )
    return Qwen3OmniVisionEncoderAccelerated(config).eval()
