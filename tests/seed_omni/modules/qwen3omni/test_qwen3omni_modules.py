"""Tests for the Qwen3-Omni audio tower OmniModule.

The risky part of this module is length bookkeeping: the tower eats one flat
concatenation of every clip in the batch and returns one flat tensor, so the
split on the way back out is *derived* from a length formula rather than
observed. If that formula disagrees with the tower by even one frame, embeds
scatter onto the wrong items and nothing raises.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from tests.tools.training_utils import make_eager_ops_config
from veomni.data.seed_omni.utils.media_metadata import (
    AUDIO_METADATA_KEY,
    VIDEO_METADATA_KEY,
    AudioMetadata,
    VideoMetadata,
)
from veomni.data.seed_omni.utils.video import VideoInputs
from veomni.models.seed_omni.modules.qwen3omni.audio.configuration import Qwen3OmniAudioEncoderConfig
from veomni.models.seed_omni.modules.qwen3omni.audio.modeling import (
    Qwen3OmniAudioEncoder,
    audio_output_lengths,
    encode_qwen3omni_audio_items,
)
from veomni.models.seed_omni.modules.qwen3omni.audio.processing import (
    _OMNI_MEL_LEN,
    Qwen3OmniAudioPreprocessor,
)
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.accelerated.accelerated import (
    Qwen3OmniTextEncoderAccelerated,
)
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.configuration import Qwen3OmniTextEncoderConfig
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.modeling import Qwen3OmniTextEncoder
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.processing import Qwen3OmniTextEncoderPreprocessor
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

from .test_qwen3omni_talker import CONFIG_DIR


NUM_MEL_BINS = 128
SAMPLING_RATE = 16000
# thinker_config.position_id_per_seconds as shipped by Qwen3-Omni.
POSITION_ID_PER_SECONDS = 13


def _tiny_config() -> Qwen3OmniAudioEncoderConfig:
    """Smallest tower that still runs the real conv stack and one attention layer."""
    return Qwen3OmniAudioEncoderConfig(
        ops_implementation=make_eager_ops_config(),
        audio_config={
            "num_mel_bins": NUM_MEL_BINS,
            "encoder_layers": 1,
            "encoder_attention_heads": 2,
            "d_model": 32,
            "encoder_ffn_dim": 64,
            "downsample_hidden_size": 16,
            "output_dim": 48,
            "n_window": 50,
            "n_window_infer": 100,
        },
    )


def _preprocessor() -> Qwen3OmniAudioPreprocessor:
    from transformers import WhisperFeatureExtractor

    return Qwen3OmniAudioPreprocessor(WhisperFeatureExtractor(feature_size=NUM_MEL_BINS, sampling_rate=SAMPLING_RATE))


def test_preprocessor_trims_whisper_padding_and_records_mel_length():
    # WhisperFeatureExtractor pads everything to 30s / 3000 frames. A 1s clip
    # must not arrive at the GPU as 3000 frames.
    prep = _preprocessor()
    conversation_list = [[ConversationItem(type="audio", value=np.zeros(SAMPLING_RATE), role="user")]]
    prep.preprocess_conversations(conversation_list, inference=True)

    item = conversation_list[0][0]
    mel_len, mel_bins = item.value.shape
    assert mel_bins == NUM_MEL_BINS
    assert item.meta[_OMNI_MEL_LEN] == mel_len
    assert mel_len < 3000, "Whisper's 30s padding was not trimmed"


def test_preprocessor_handles_torch_and_numpy_waveforms_alike():
    prep = _preprocessor()
    conversation_list = [
        [
            ConversationItem(type="audio", value=np.zeros(SAMPLING_RATE), role="user"),
            ConversationItem(type="audio", value=torch.zeros(SAMPLING_RATE), role="user"),
        ]
    ]
    prep.preprocess_conversations(conversation_list, inference=True)
    shapes = [tuple(item.value.shape) for item in conversation_list[0]]
    assert shapes[0] == shapes[1]


def test_preprocessor_leaves_non_audio_items_untouched():
    prep = _preprocessor()
    text = ConversationItem(type="text", value="hello", role="user")
    image = ConversationItem(type="image", value="not-a-waveform", role="user")
    prep.preprocess_conversations([[text, image]], inference=True)
    assert text.value == "hello"
    assert image.value == "not-a-waveform"


def test_a_second_of_audio_costs_position_id_per_seconds_tokens():
    """The invariant that lets audio and video share TMRoPE's clock.

    The backbone spends one position id per audio token and advances a video's
    temporal row by ``second_per_grid * position_id_per_seconds``. Those agree
    only if a second of audio really is ``position_id_per_seconds`` tokens, so
    that identity is the contract -- not an incidental property of the formula.
    """
    prep = _preprocessor()
    seconds = 4
    item = ConversationItem(type="audio", value=np.zeros(SAMPLING_RATE * seconds, dtype=np.float32), role="user")
    prep.preprocess_conversations([[item]], inference=True)

    tokens = int(audio_output_lengths(torch.tensor([item.meta[_OMNI_MEL_LEN]]), n_window=50)[0])
    assert tokens == seconds * POSITION_ID_PER_SECONDS


def test_a_clip_declaring_another_rate_is_resampled_and_keeps_its_duration():
    # The regression this guards: WhisperFeatureExtractor never resamples, so
    # 22.05kHz read as 16kHz used to stretch a 4s clip to 5.54s of TMRoPE clock
    # (+38%) and drift against any video track by the same factor.
    pytest.importorskip("librosa")
    prep = _preprocessor()
    seconds = 4
    lengths = {}
    for rate in (SAMPLING_RATE, 22050, 24000):
        item = ConversationItem(
            type="audio",
            value=np.zeros(int(rate * seconds), dtype=np.float32),
            role="user",
            meta={AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=rate, num_samples=int(rate * seconds))},
        )
        prep.preprocess_conversations([[item]], inference=True)
        lengths[rate] = int(audio_output_lengths(torch.tensor([item.meta[_OMNI_MEL_LEN]]), n_window=50)[0])

    assert lengths[SAMPLING_RATE] == seconds * POSITION_ID_PER_SECONDS
    # Resampling is not frame-exact at the boundary, so allow a token of slack;
    # the point is that the duration is preserved rather than scaled by rate/16k.
    for rate, tokens in lengths.items():
        assert abs(tokens - lengths[SAMPLING_RATE]) <= 1, f"{rate}Hz clip landed on a different span"


def test_a_clip_without_a_declared_rate_is_taken_as_the_encoder_rate():
    # Back-compat with every producer that hands over already-16kHz samples,
    # which is what VeOmni's own audio loader resamples to.
    prep = _preprocessor()
    declared = ConversationItem(
        type="audio",
        value=np.zeros(SAMPLING_RATE, dtype=np.float32),
        role="user",
        meta={AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=SAMPLING_RATE, num_samples=SAMPLING_RATE)},
    )
    bare = ConversationItem(type="audio", value=np.zeros(SAMPLING_RATE, dtype=np.float32), role="user")
    prep.preprocess_conversations([[declared, bare]], inference=True)
    assert declared.meta[_OMNI_MEL_LEN] == bare.meta[_OMNI_MEL_LEN]


def test_multichannel_audio_is_rejected_rather_than_flattened():
    # Interleaved stereo flattened to mono would double the clip's apparent
    # duration -- the same class of silent misplacement as a wrong rate.
    prep = _preprocessor()
    stereo = ConversationItem(type="audio", value=np.zeros((SAMPLING_RATE, 2), dtype=np.float32), role="user")
    with pytest.raises(ValueError, match="mono"):
        prep.preprocess_conversations([[stereo]], inference=True)


def test_a_channel_dimension_of_one_is_accepted():
    prep = _preprocessor()
    item = ConversationItem(type="audio", value=torch.zeros(1, SAMPLING_RATE), role="user")
    prep.preprocess_conversations([[item]], inference=True)
    assert item.meta[_OMNI_MEL_LEN] > 0


@pytest.mark.parametrize("mel_lengths", [[100], [37], [100, 250], [13, 100, 201]])
def test_split_lengths_match_what_the_tower_actually_returns(mel_lengths):
    torch.manual_seed(0)
    model = Qwen3OmniAudioEncoder(_tiny_config()).eval()

    items = [
        ConversationItem(
            type="audio",
            value=torch.randn(length, NUM_MEL_BINS),
            role="user",
            meta={_OMNI_MEL_LEN: length},
        )
        for length in mel_lengths
    ]
    with torch.no_grad():
        encode_qwen3omni_audio_items(model, items)

    n_window = model.config.audio_config.n_window
    expected = audio_output_lengths(torch.tensor(mel_lengths), n_window).tolist()
    assert [item.value.size(0) for item in items] == expected
    # A silent zip mismatch would be worse than a crash, so also assert the
    # embeds really were replaced (not still mel features).
    assert all(item.value.size(-1) == model.config.audio_config.output_dim for item in items)


def test_rejects_a_window_size_the_upstream_tower_cannot_run():
    # A bad n_window otherwise surfaces as a split_with_sizes mismatch deep
    # inside attention, which says nothing about the real cause.
    config = _tiny_config()
    config.audio_config.n_window = 25
    with pytest.raises(ValueError, match="hardcodes 100 frames"):
        Qwen3OmniAudioEncoder(config)


def test_length_formula_agrees_with_upstream_at_the_shipped_window_size():
    # Upstream's helper hardcodes a 100-frame chunk, which is exactly
    # n_window=50 as shipped. Our config-driven version must agree there --
    # and the parametrised test above covers the case where it must not.
    from veomni.models.transformers.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        _get_feat_extract_output_lengths,
    )

    lengths = torch.tensor([1, 13, 37, 99, 100, 101, 150, 250, 1000])
    assert audio_output_lengths(lengths, n_window=50).tolist() == _get_feat_extract_output_lengths(lengths).tolist()


def test_encode_is_a_noop_on_an_empty_item_list():
    model = Qwen3OmniAudioEncoder(_tiny_config()).eval()
    with torch.no_grad():
        encode_qwen3omni_audio_items(model, [])


def test_generate_only_consumes_user_audio_items():
    torch.manual_seed(0)
    model = Qwen3OmniAudioEncoder(_tiny_config()).eval()
    assistant = ConversationItem(
        type="audio", value=torch.randn(100, NUM_MEL_BINS), role="assistant", meta={_OMNI_MEL_LEN: 100}
    )
    user = ConversationItem(type="audio", value=torch.randn(100, NUM_MEL_BINS), role="user", meta={_OMNI_MEL_LEN: 100})
    conversation_list = [assistant, user]
    with torch.no_grad():
        out = model.generate(conversation_list=conversation_list)

    assert out["conversation_list"] is conversation_list
    assert assistant.value.shape == (100, NUM_MEL_BINS), "assistant audio should not be encoded"
    assert user.value.size(-1) == model.config.audio_config.output_dim


def _accelerated_tower():
    from veomni.models.seed_omni.modules.qwen3omni.audio.accelerated.accelerated import (
        Qwen3OmniAudioEncoderAccelerated,
    )

    return Qwen3OmniAudioEncoderAccelerated(_tiny_config()).eval()


def test_a_clip_from_the_data_layer_lands_on_the_towers_wall_clock():
    """22.05 kHz is what a real corpus holds, and neither module runs at it.

    The tower spends one position id per audio token and the backbone emits 13
    per second of *real* time, while a video's temporal row advances by wall-clock
    seconds — so a dropped rate would slide the two apart on a clock they share.
    """
    from transformers import WhisperFeatureExtractor

    from veomni.models.seed_omni.modules.qwen3omni.audio.modeling import audio_output_lengths

    seconds = 1.5
    position_id_per_seconds = 13  # thinker_config.position_id_per_seconds
    item = ConversationItem(
        type="audio",
        value=np.asarray(np.sin(2 * np.pi * 220 * np.arange(int(seconds * 22050)) / 22050), dtype=np.float32),
        role="user",
        meta={AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=22050, num_samples=int(seconds * 22050))},
    )

    tower = Qwen3OmniAudioPreprocessor(WhisperFeatureExtractor(feature_size=128, sampling_rate=16000))
    tower.preprocess_conversations([[item]], inference=True)

    tokens = int(audio_output_lengths(torch.tensor([item.meta[_OMNI_MEL_LEN]]), n_window=50)[0])
    assert tokens == pytest.approx(seconds * position_id_per_seconds, abs=1)


def test_the_tower_refuses_a_clip_its_window_would_silently_truncate():
    """``WhisperFeatureExtractor`` cuts at 30 s and reports nothing.

    In training that pairs a clipped encoder input with a text target covering
    the whole recording — no error, just a ceiling on quality.
    """
    from transformers import WhisperFeatureExtractor

    tower = Qwen3OmniAudioPreprocessor(WhisperFeatureExtractor(feature_size=128, sampling_rate=16000))
    item = ConversationItem(
        type="audio",
        value=np.zeros(40 * 16000, dtype=np.float32),
        role="user",
        meta={AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=16000, num_samples=40 * 16000)},
    )

    with pytest.raises(ValueError, match="silently truncate"):
        tower.preprocess_conversations([[item]], inference=True)


def test_the_audio_preprocessor_stamps_the_towers_output_row_count():
    """`dyn_bsz` counts a sample's backbone tokens before any model exists, so the
    worker has to stamp what the row *will* contribute.

    Stamping the mel-frame count would over-report by ~8x (13 rows per 100 mel
    frames); leaving the key off drops the row from the count entirely, which is
    what made an audio-heavy sample look short and let the batcher pack too many
    of them.
    """
    from transformers import WhisperFeatureExtractor

    from veomni.models.seed_omni.modules.qwen3omni.audio.processing import audio_output_lengths

    tower = Qwen3OmniAudioPreprocessor(WhisperFeatureExtractor(feature_size=128, sampling_rate=16000), n_window=50)
    item = ConversationItem(
        type="audio",
        value=np.zeros(16000 * 2, dtype=np.float32),  # 2 s at the tower's rate
        role="user",
        meta={AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=16000, num_samples=16000 * 2)},
    )

    tower.preprocess_conversations([[item]], inference=True)

    mel_len = item.meta[_OMNI_MEL_LEN]
    expected = int(audio_output_lengths(torch.tensor([mel_len]), n_window=50)[0])
    assert item.meta["_omni_token_len"] == expected
    # The prediction is the encoder's row count, not the mel-frame count.
    assert expected < mel_len


def test_assistant_speech_is_not_thinker_input():
    """The one rule every carrier walk agrees on, replacing three marker tests.

    A user clip is backbone input; the assistant's speech is a talker label and
    never enters the thinker. Same ``type``, different ``role`` — which is why the
    rule lives in this family rather than in the shared packer helpers.
    """
    from veomni.models.seed_omni.modules.qwen3omni.carrier import is_thinker_input

    user = ConversationItem(type="audio", value=torch.zeros(8), role="user")
    label = ConversationItem(type="audio", value=torch.zeros(8), role="assistant")
    dummy = ConversationItem(type="audio", value=torch.zeros(8), role="user", is_dummy=True)

    assert is_thinker_input(user)
    assert not is_thinker_input(label)
    assert not is_thinker_input(dummy)


# ── text encoder ─────────────────────────────────────────────────────────────


def test_config_inherits_base_defaults():
    """transformers 5.x auto-generates ``__init__`` for annotation-less config subclasses.

    Without the explicit forward in ``Qwen3OmniTextEncoderConfig`` none of
    ``TextEncoderConfig``'s defaults land, and ``lm_head_bias`` in particular goes
    missing — which breaks the untied path below.
    """
    config = Qwen3OmniTextEncoderConfig()
    assert config.model_type == "qwen3omni_text_encoder"
    assert config.vocab_size == 32000
    assert config.lm_head_bias is False


def test_untied_module_builds_and_decodes():
    """Qwen3-Omni's thinker ships ``tie_word_embeddings=False``, so this is the only path."""
    config = Qwen3OmniTextEncoderConfig(vocab_size=64, hidden_size=32, tie_word_embeddings=False)
    module = Qwen3OmniTextEncoder(config)
    assert isinstance(module.lm_head, torch.nn.Linear)
    assert module.lm_head.bias is None

    # ``encode`` returns per-segment ``(L, D)`` rows for the carrier, not a batch.
    embeds = module.encode(torch.tensor([[1, 2, 3]]))["inputs_embeds"]
    assert embeds.shape == (3, 32)


def test_preprocessor_class_is_reachable_from_the_model():
    """The processing binding resolves the preprocessor off the native class."""
    assert Qwen3OmniTextEncoder.preprocessor_class is Qwen3OmniTextEncoderPreprocessor


def test_tts_embeds_reads_the_sharded_table_it_actually_has():
    """`tts_embeds` must go through the embedding module, not a raw table read.

    The table is a `ShardedEmbedding`, whose `forward` takes *global* ids and
    performs the `emb` all-to-all itself (falling through to `nn.Embedding` when
    the table is not split). Reading a `_embed_tokens` attribute raises
    `AttributeError` — on both the speech-inference and talker-training paths,
    i.e. before the talker ever runs.
    """
    from unittest import mock

    module = Qwen3OmniTextEncoder(Qwen3OmniTextEncoderConfig(vocab_size=64, hidden_size=32, tie_word_embeddings=False))
    # ``tts_embeds`` reads its ids off the chat template, which a real checkpoint
    # binds; stub just the three ids it needs.
    module._chat_template = mock.MagicMock(tts_bos_token_id=1, tts_eos_token_id=2, tts_pad_token_id=3)

    # No ``_embed_tokens`` exists anywhere in the MRO — the bug was assuming one.
    assert not any("_embed_tokens" in cls.__dict__ for cls in type(module).__mro__)

    with torch.no_grad():
        embeds = module.tts_embeds()

    assert embeds.shape == (1, 3, 32)
    # ...and the ids really reached the table (bos / eos / pad rows, in order).
    with torch.no_grad():
        expected = module.embed_tokens(torch.tensor([[1, 2, 3]]))
    assert torch.equal(embeds, expected)


# ── vision tower ─────────────────────────────────────────────────────────────


def _tiny_vision():
    """Structurally faithful tower from the real config, shrunk, random weights."""
    with open(os.path.join(CONFIG_DIR, "config.json")) as handle:
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
    return Qwen3OmniVisionEncoder(
        Qwen3OmniVisionEncoderConfig(vision_config=tiny, ops_implementation=make_eager_ops_config())
    ).eval()


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


def test_generate_is_a_noop_without_visual_items():
    module = _tiny_vision()
    conversation_list = [ConversationItem(type="text", value=torch.zeros(3), role="user")]
    outputs = module.generate(conversation_list=conversation_list)
    assert outputs["conversation_list"] is conversation_list


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


def test_preprocessor_sets_second_per_grid_for_videos_only():
    """Video items get a timeline; still images keep the backbone's 1.0 default.

    This is the hand-off the backbone's packer depends on: it raises for a video
    item without ``second_per_grid``, so this preprocessor is the only supplier.
    """
    preprocessor = Qwen3OmniVisionPreprocessor.from_pretrained(CONFIG_DIR)
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


def test_processors_come_from_the_dir_not_class_defaults():
    """Qwen3-Omni ships no video_preprocessor_config.json, so this could regress silently.

    ``Qwen2VLVideoProcessor``'s class default ``patch_size`` is 14 while
    Qwen3-Omni uses 16; loading from a directory is what makes it fall back to
    the shared ``preprocessor_config.json``.
    """
    preprocessor = Qwen3OmniVisionPreprocessor.from_pretrained(CONFIG_DIR)
    with open(os.path.join(CONFIG_DIR, "preprocessor_config.json")) as handle:
        expected = json.load(handle)
    for processor in (preprocessor._image_processor, preprocessor._video_processor):
        assert processor.patch_size == expected["patch_size"]
        assert processor.temporal_patch_size == expected["temporal_patch_size"]
        assert processor.merge_size == expected["merge_size"]
    # No resampling: this is why the clip's own fps is the right divisor.
    assert preprocessor._video_processor.do_sample_frames is False


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


def test_the_decode_walk_skips_the_assistant_speech_target():
    """The one override that makes the joint graph work.

    The base decode walk skips only dummy rows. This carrier also holds the
    assistant's speech, which never entered the thinker's forward and still holds a
    raw waveform — concatenating it with the thinker's hidden states fails. The
    override must therefore be *this* class's method, not the inherited one: a
    nested or misplaced definition silently keeps the base behaviour and the joint
    graph breaks only when a real audio target reaches it.
    """
    from veomni.models.seed_omni.modules.qwen3omni.text_encoder.accelerated.accelerated import (
        TrainingMixin as Qwen3OmniTrainingMixin,
    )
    from veomni.models.seed_omni.modules.text_encoder.accelerated.accelerated import (
        TrainingMixin as BaseTrainingMixin,
    )

    assert "_prepare_decode_inputs" in Qwen3OmniTrainingMixin.__dict__, (
        "the override must be a member of this mixin; nested inside a function it overrides nothing"
    )
    assert Qwen3OmniTrainingMixin._prepare_decode_inputs is not BaseTrainingMixin._prepare_decode_inputs, (
        "the override must differ from the base walk"
    )

    # And it must actually drop the waveform row: two text rows with a spoken
    # assistant turn between them walk as two text segments, not three.
    module = Qwen3OmniTextEncoderAccelerated(
        Qwen3OmniTextEncoderConfig(vocab_size=64, hidden_size=8, tie_word_embeddings=False)
    )
    import torch as _torch

    sample = [
        ConversationItem(
            type="text",
            value=_torch.zeros(2, 8),
            role="user",
            meta={"labels": _torch.zeros(2, dtype=_torch.long)},
        ),
        ConversationItem(type="audio", value=_torch.zeros(2400), role="assistant"),
        ConversationItem(
            type="text",
            value=_torch.zeros(3, 8),
            role="assistant",
            meta={"labels": _torch.zeros(3, dtype=_torch.long)},
        ),
    ]
    hidden, shift_labels = module._prepare_decode_inputs([sample])

    assert hidden.shape[0] == 5, f"expected the two text rows only, got {tuple(hidden.shape)}"
    assert shift_labels.shape[0] == 5


# ── shard handoff ────────────────────────────────────────────────────────────
#
# The alignment test drives a text-only batch, so neither media producer's shard
# branch is reached there. These two cover the branches themselves — they were
# added after an import in the audio one shipped broken: the module resolved only
# when the branch ran, and nothing ran it.


def test_the_audio_window_splits_into_per_clip_slices():
    """Each clip receives its intersection with this rank's window.

    Lengths come from ``audio_output_lengths`` rather than from the value, because
    the tower returns one unsegmented tensor and a sharded value only holds a
    window. Two clips of unequal length are used so a wrong derivation cannot pass
    by symmetry.
    """
    from veomni.models.seed_omni.modules.qwen3omni.audio.accelerated.accelerated import _scatter_audio_window
    from veomni.models.seed_omni.modules.qwen3omni.audio.modeling import audio_output_lengths
    from veomni.models.seed_omni.modules.qwen3omni.audio.processing import _OMNI_MEL_LEN

    n_window = 100
    mel_lens = [200, 300]
    full = [int(v) for v in audio_output_lengths(torch.tensor(mel_lens), n_window).tolist()]
    sp_size = 2
    window = (sum(full) + (-sum(full) % sp_size)) // sp_size

    def _items():
        return [
            ConversationItem(type="audio", value=torch.zeros(n, 4), role="user", meta={_OMNI_MEL_LEN: mel})
            for n, mel in zip(full, mel_lens)
        ]

    covered = 0
    for rank in range(sp_size):
        items = _items()
        _scatter_audio_window(items, torch.randn(window, 4), n_window=n_window, sp_rank=rank, sp_size=sp_size)
        rows = [item.value.size(0) for item in items]
        # No item ever receives padding: the window's pad tail is dropped, so the
        # slices add up to at most the window width.
        assert sum(rows) <= window, (rank, rows, window)
        covered += sum(rows)

    # Every real row is emitted exactly once across the group.
    assert covered == sum(full), (covered, full)


def test_the_vision_window_splits_merged_tokens_and_deepstack():
    """Merged tokens and their DeepStack rows are sliced by the same schedule.

    They are indexed by visual position, so a mismatch would pair one item's
    activation with another's feature — silently, since the shapes still line up.
    """
    from veomni.models.seed_omni.modules.qwen3omni.vision.modeling import scatter_qwen3omni_visual_embeds

    class _Slot:
        def __init__(self, num_merged, item, grid):
            self.num_merged = num_merged
            self.item = item
            self.grid = grid

    sizes = [4, 2]
    sp_size = 2
    window = (sum(sizes) + (-sum(sizes) % sp_size)) // sp_size
    num_layers = 2

    for rank in range(sp_size):
        items = [ConversationItem(type="image", value=None, role="user", meta={}) for _ in sizes]
        slots = [_Slot(n, item, torch.tensor((1, 2, 2))) for n, item in zip(sizes, items)]
        embeds = torch.randn(window, 8)
        deepstack = [torch.randn(window, 8) for _ in range(num_layers)]

        scatter_qwen3omni_visual_embeds(slots, embeds, deepstack, sp_rank=rank, sp_size=sp_size)

        for item in items:
            assert item.value.size(0) == len(item.meta["deepstack"][0])
            assert len(item.meta["deepstack"]) == num_layers
        # Nothing from the pad tail may reach an item.
        assert sum(item.value.size(0) for item in items) <= window
