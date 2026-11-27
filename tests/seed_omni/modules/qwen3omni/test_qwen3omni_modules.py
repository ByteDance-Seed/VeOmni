"""Tests for Qwen3-Omni audio, text_encoder, and vision OmniModules."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from transformers import WhisperFeatureExtractor
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeVisionEncoder

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.training_utils import make_eager_ops_config
from veomni.data.seed_omni.utils.media_metadata import (
    AUDIO_METADATA_KEY,
    VIDEO_METADATA_KEY,
    AudioMetadata,
    VideoMetadata,
)
from veomni.data.seed_omni.utils.video import VideoInputs
from veomni.models.seed_omni.modules.qwen3omni.audio.accelerated.accelerated import _scatter_audio_window
from veomni.models.seed_omni.modules.qwen3omni.audio.configuration import Qwen3OmniAudioEncoderConfig
from veomni.models.seed_omni.modules.qwen3omni.audio.modeling import (
    Qwen3OmniAudioEncoder,
    audio_output_lengths,
    encode_qwen3omni_audio_items,
)
from veomni.models.seed_omni.modules.qwen3omni.audio.processing import _OMNI_MEL_LEN, Qwen3OmniAudioPreprocessor
from veomni.models.seed_omni.modules.qwen3omni.carrier import is_thinker_input
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.accelerated.accelerated import (
    Qwen3OmniTextEncoderAccelerated,
)
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.accelerated.accelerated import (
    TrainingMixin as Qwen3OmniTrainingMixin,
)
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.configuration import Qwen3OmniTextEncoderConfig
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.modeling import Qwen3OmniTextEncoder
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.processing import Qwen3OmniTextEncoderPreprocessor
from veomni.models.seed_omni.modules.qwen3omni.vision.configuration import Qwen3OmniVisionEncoderConfig
from veomni.models.seed_omni.modules.qwen3omni.vision.modeling import (
    Qwen3OmniVisionEncoder,
    build_qwen3omni_vit_metadata,
    scatter_qwen3omni_visual_embeds,
)
from veomni.models.seed_omni.modules.qwen3omni.vision.processing import (
    _OMNI_GRID,
    Qwen3OmniVisionPreprocessor,
    _second_per_grid,
)
from veomni.models.seed_omni.modules.text_encoder.accelerated.accelerated import TrainingMixin as BaseTrainingMixin
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.models.transformers.qwen3_omni_moe.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
CONFIG_DIR = Path(__file__).parents[4] / "tests" / "toy_config" / "qwen3omni_toy"


NUM_MEL_BINS = 128
SAMPLING_RATE = 16000
POSITION_ID_PER_SECONDS = 13


# ── audio tower ──────────────────────────────────────────────────────────────


def _tiny_audio_config() -> Qwen3OmniAudioEncoderConfig:
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


def _audio_preprocessor() -> Qwen3OmniAudioPreprocessor:
    return Qwen3OmniAudioPreprocessor(
        WhisperFeatureExtractor(feature_size=NUM_MEL_BINS, sampling_rate=SAMPLING_RATE), n_window=50
    )


def test_audio_preprocessor_trims_resamples_and_stamps_token_len():
    pytest.importorskip("librosa")
    prep = _audio_preprocessor()

    # Whisper 30s padding is trimmed, torch/numpy/1xN mono inputs match, non-audio stays untouched.
    np_item = ConversationItem(type="audio", value=np.zeros(SAMPLING_RATE * 2, dtype=np.float32), role="user")
    pt_item = ConversationItem(type="audio", value=torch.zeros(1, SAMPLING_RATE * 2), role="user")
    declared_16k = ConversationItem(
        type="audio",
        value=np.zeros(SAMPLING_RATE * 2, dtype=np.float32),
        role="user",
        meta={AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=SAMPLING_RATE, num_samples=SAMPLING_RATE * 2)},
    )
    text = ConversationItem(type="text", value="hello", role="user")
    prep.preprocess_conversations([[np_item, pt_item, declared_16k, text]], inference=True)

    mel_len, mel_bins = np_item.value.shape
    assert mel_bins == NUM_MEL_BINS and mel_len < 3000
    assert np_item.meta[_OMNI_MEL_LEN] == mel_len
    assert tuple(pt_item.value.shape) == (mel_len, mel_bins)
    assert declared_16k.meta[_OMNI_MEL_LEN] == mel_len
    assert text.value == "hello"

    # `_omni_token_len` records output token count (not mel frames), matching 13 tok/s wall clock.
    expected_tokens = int(audio_output_lengths(torch.tensor([mel_len]), n_window=50)[0])
    assert np_item.meta["_omni_token_len"] == expected_tokens == 2 * POSITION_ID_PER_SECONDS
    assert expected_tokens < mel_len

    # Other sample rates resample while preserving wall-clock duration.
    seconds = 4
    for rate in (22050, 24000):
        item = ConversationItem(
            type="audio",
            value=np.zeros(int(rate * seconds), dtype=np.float32),
            role="user",
            meta={AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=rate, num_samples=int(rate * seconds))},
        )
        prep.preprocess_conversations([[item]], inference=True)
        tokens = int(audio_output_lengths(torch.tensor([item.meta[_OMNI_MEL_LEN]]), n_window=50)[0])
        assert abs(tokens - seconds * POSITION_ID_PER_SECONDS) <= 1

    # Multichannel waveforms and clips exceeding Whisper's 30s window are rejected.
    with pytest.raises(ValueError, match="mono"):
        prep.preprocess_conversations(
            [[ConversationItem(type="audio", value=np.zeros((SAMPLING_RATE, 2), dtype=np.float32), role="user")]],
            inference=True,
        )
    with pytest.raises(ValueError, match="silently truncate"):
        prep.preprocess_conversations(
            [
                [
                    ConversationItem(
                        type="audio",
                        value=np.zeros(40 * SAMPLING_RATE, dtype=np.float32),
                        role="user",
                        meta={
                            AUDIO_METADATA_KEY: AudioMetadata(
                                sampling_rate=SAMPLING_RATE, num_samples=40 * SAMPLING_RATE
                            )
                        },
                    )
                ]
            ],
            inference=True,
        )


@pytest.mark.parametrize("mel_lengths", [[100], [37], [13, 100, 201]])
def test_audio_encoder_output_lengths_and_split(mel_lengths):
    torch.manual_seed(0)
    model = Qwen3OmniAudioEncoder(_tiny_audio_config()).eval()

    lengths = torch.tensor([1, 13, 37, 99, 100, 101, 150, 250, 1000])
    assert audio_output_lengths(lengths, n_window=50).tolist() == _get_feat_extract_output_lengths(lengths).tolist()

    items = [
        ConversationItem(
            type="audio", value=torch.randn(length, NUM_MEL_BINS), role="user", meta={_OMNI_MEL_LEN: length}
        )
        for length in mel_lengths
    ]
    with torch.no_grad():
        encode_qwen3omni_audio_items(model, items)
        encode_qwen3omni_audio_items(model, [])

    expected = audio_output_lengths(torch.tensor(mel_lengths), model.config.audio_config.n_window).tolist()
    assert [item.value.size(0) for item in items] == expected
    assert all(item.value.size(-1) == model.config.audio_config.output_dim for item in items)

    # Audio container preserves requested attn_implementation and short-circuits empty forward.
    assert model._check_and_adjust_attn_implementation("sdpa") == "sdpa"
    assert model.forward(input_features=None, feature_lens=None) == {"audio_embeds": None}

    bad_cfg = _tiny_audio_config()
    bad_cfg.audio_config.n_window = 25
    with pytest.raises(ValueError, match="hardcodes 100 frames"):
        Qwen3OmniAudioEncoder(bad_cfg)


def test_audio_generate_and_is_thinker_input_skip_assistant_speech():
    torch.manual_seed(0)
    model = Qwen3OmniAudioEncoder(_tiny_audio_config()).eval()
    assistant = ConversationItem(
        type="audio", value=torch.randn(100, NUM_MEL_BINS), role="assistant", meta={_OMNI_MEL_LEN: 100}
    )
    user = ConversationItem(type="audio", value=torch.randn(100, NUM_MEL_BINS), role="user", meta={_OMNI_MEL_LEN: 100})
    dummy = ConversationItem(type="audio", value=torch.zeros(8), role="user", is_dummy=True)

    assert is_thinker_input(user) and not is_thinker_input(assistant) and not is_thinker_input(dummy)

    conversation_list = [assistant, user]
    with torch.no_grad():
        out = model.generate(conversation_list=conversation_list)
    assert out["conversation_list"] is conversation_list
    assert assistant.value.shape == (100, NUM_MEL_BINS)
    assert user.value.size(-1) == model.config.audio_config.output_dim


def test_audio_sp_window_scatter_covers_all_real_rows():
    n_window = 100
    mel_lens = [200, 300]
    full = [int(v) for v in audio_output_lengths(torch.tensor(mel_lens), n_window).tolist()]
    sp_size = 2
    window = (sum(full) + (-sum(full) % sp_size)) // sp_size

    covered = 0
    for rank in range(sp_size):
        items = [
            ConversationItem(type="audio", value=torch.zeros(n, 4), role="user", meta={_OMNI_MEL_LEN: mel})
            for n, mel in zip(full, mel_lens)
        ]
        _scatter_audio_window(items, torch.randn(window, 4), n_window=n_window, sp_rank=rank, sp_size=sp_size)
        rows = [item.value.size(0) for item in items]
        assert sum(rows) <= window
        covered += sum(rows)
    assert covered == sum(full)


# ── text encoder ─────────────────────────────────────────────────────────────


def test_text_encoder_untied_build_and_tts_embeds():
    default_cfg = Qwen3OmniTextEncoderConfig()
    assert (default_cfg.model_type, default_cfg.vocab_size, default_cfg.lm_head_bias) == (
        "qwen3omni_text_encoder",
        32000,
        False,
    )
    assert Qwen3OmniTextEncoder.preprocessor_class is Qwen3OmniTextEncoderPreprocessor

    module = Qwen3OmniTextEncoder(Qwen3OmniTextEncoderConfig(vocab_size=64, hidden_size=32, tie_word_embeddings=False))
    assert isinstance(module.lm_head, torch.nn.Linear) and module.lm_head.bias is None
    assert module.encode(torch.tensor([[1, 2, 3]]))["inputs_embeds"].shape == (3, 32)

    # `tts_embeds` must route through `embed_tokens` (a ShardedEmbedding), not a private `_embed_tokens`.
    assert not any("_embed_tokens" in cls.__dict__ for cls in type(module).__mro__)
    module._chat_template = mock.MagicMock(tts_bos_token_id=1, tts_eos_token_id=2, tts_pad_token_id=3)
    with torch.no_grad():
        assert torch.equal(module.tts_embeds(), module.embed_tokens(torch.tensor([[1, 2, 3]])))


def test_text_encoder_decode_skips_assistant_speech():
    assert "_prepare_decode_inputs" in Qwen3OmniTrainingMixin.__dict__
    assert Qwen3OmniTrainingMixin._prepare_decode_inputs is not BaseTrainingMixin._prepare_decode_inputs

    module = Qwen3OmniTextEncoderAccelerated(
        Qwen3OmniTextEncoderConfig(vocab_size=64, hidden_size=8, tie_word_embeddings=False)
    )
    sample = [
        ConversationItem(
            type="text", value=torch.zeros(2, 8), role="user", meta={"labels": torch.zeros(2, dtype=torch.long)}
        ),
        ConversationItem(type="audio", value=torch.zeros(2400), role="assistant"),
        ConversationItem(
            type="text", value=torch.zeros(3, 8), role="assistant", meta={"labels": torch.zeros(3, dtype=torch.long)}
        ),
    ]
    hidden, shift_labels = module._prepare_decode_inputs([sample])
    assert hidden.shape[0] == 5 and shift_labels.shape[0] == 5


# ── vision tower ─────────────────────────────────────────────────────────────


def _tiny_vision() -> Qwen3OmniVisionEncoder:
    with open(os.path.join(CONFIG_DIR, "config.json")) as handle:
        tiny = dict(json.load(handle)["thinker_config"]["vision_config"])
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


def _patch_item(item_type, grid, cfg, role="user", meta=None) -> ConversationItem:
    t, h, w = grid
    pixel_row = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    torch.manual_seed(sum(grid))
    return ConversationItem(
        type=item_type,
        value=torch.randn(t * h * w, pixel_row),
        role=role,
        meta={_OMNI_GRID: list(grid), **(meta or {})},
    )


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


def test_vision_encode_matches_upstream_and_shared_forward_has_no_leakage():
    module = _tiny_vision()
    cfg = module.config.vision_config
    reference = Qwen3OmniMoeVisionEncoder._from_config(cfg).eval()
    missing, unexpected = reference.load_state_dict(module.visual.state_dict(), strict=False)
    assert not missing and not unexpected

    grids = [(1, 4, 4), (2, 2, 2)]
    together = [
        _patch_item("image", grids[0], cfg),
        _patch_item("video", grids[1], cfg, meta={"second_per_grid": 1.0}),
    ]
    alone = [
        [_patch_item("image", grids[0], cfg)],
        [_patch_item("video", grids[1], cfg, meta={"second_per_grid": 1.0})],
    ]
    pixel_values = torch.cat([item.value for item in together], dim=0)
    grid_thw = torch.tensor([list(g) for g in grids], dtype=torch.long)

    with torch.no_grad():
        embeds, deepstack = module._encode(pixel_values, grid_thw, None)
        expected = reference(hidden_states=pixel_values.type(reference.dtype), grid_thw=grid_thw)
        module.generate(conversation_list=list(together))
        for single in alone:
            module.generate(conversation_list=list(single))

    torch.testing.assert_close(embeds, expected.pooler_output, rtol=0, atol=0)
    for actual, want in zip(deepstack, expected.deepstack_features, strict=True):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)
    torch.testing.assert_close(together[0].value, alone[0][0].value)
    torch.testing.assert_close(together[1].value, alone[1][0].value)

    # Per-frame cu_seqlens segmentation.
    metadata = build_qwen3omni_vit_metadata([[2, 2, 2], [1, 4, 4]], spatial_merge_size=2)
    assert metadata["cu_seqlens"].tolist() == [0, 4, 8, 24] and metadata["max_seqlen"] == 16


def test_vision_generate_scatters_tokens_and_deepstack():
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
        assert item.meta["grid_thw"].tolist() == list(grid) and _OMNI_GRID not in item.meta
        assert len(item.meta["deepstack"]) == len(cfg.deepstack_visual_indexes)
        assert all(layer.shape == (expected_merged, cfg.out_hidden_size) for layer in item.meta["deepstack"])
    assert items[2].meta["second_per_grid"] == 1.0

    text_only = [ConversationItem(type="text", value=torch.zeros(3), role="user")]
    assert module.generate(conversation_list=text_only)["conversation_list"] is text_only
    assert module._check_and_adjust_attn_implementation("sdpa") == "sdpa"

    import inspect

    from veomni.models.seed_omni.modules.qwen3omni.text_encoder.modeling import Qwen3OmniTextEncoder

    assert inspect.signature(Qwen3OmniTextEncoder.generate).parameters["generation_kwargs"].default is None


def test_vision_preprocessor_loads_dir_config_and_computes_second_per_grid():
    preprocessor = Qwen3OmniVisionPreprocessor.from_pretrained(CONFIG_DIR)
    with open(os.path.join(CONFIG_DIR, "preprocessor_config.json")) as handle:
        expected = json.load(handle)
    for proc in (preprocessor._image_processor, preprocessor._video_processor):
        assert (proc.patch_size, proc.temporal_patch_size, proc.merge_size) == (
            expected["patch_size"],
            expected["temporal_patch_size"],
            expected["merge_size"],
        )
    assert preprocessor._video_processor.do_sample_frames is False

    # Measured off actually kept frames, not requested_fps.
    assert _second_per_grid(_video_item(fps=30.0, frames_indices=[0, 15, 30, 45]), 2) == pytest.approx(1.0)
    assert _second_per_grid(
        _video_item(fps=25.0, frames_indices=[0, 12, 24, 36, 48], requested_fps=2.0), 1
    ) == pytest.approx(12.0 / 25.0)
    with pytest.raises(ValueError, match="carries no meta"):
        _second_per_grid(ConversationItem(type="video", value=None, role="user"), 2)

    # Full preprocess sets second_per_grid for videos only and stashes grids.
    video_item = _video_item(fps=8.0, frames_indices=[0, 2, 4, 6], requested_fps=4.0)
    video_item.value = VideoInputs(video=torch.randint(0, 255, (4, 3, 64, 64), dtype=torch.uint8))
    image_item = ConversationItem(
        type="image", value=torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8), role="user"
    )
    preprocessor.preprocess_conversations([[image_item, video_item]], inference=True)

    assert video_item.meta["second_per_grid"] == pytest.approx(
        preprocessor._video_processor.temporal_patch_size * 0.25
    )
    assert "second_per_grid" not in image_item.meta
    assert all(isinstance(i.value, torch.Tensor) and len(i.meta[_OMNI_GRID]) == 3 for i in (image_item, video_item))


def test_vision_sp_window_scatter_slices_tokens_and_deepstack_together():
    class _Slot:
        def __init__(self, num_merged, item, grid):
            self.num_merged, self.item, self.grid = num_merged, item, grid

    sizes, sp_size, num_layers = [4, 2], 2, 2
    window = (sum(sizes) + (-sum(sizes) % sp_size)) // sp_size
    covered = 0
    for rank in range(sp_size):
        items = [ConversationItem(type="image", value=None, role="user", meta={}) for _ in sizes]
        slots = [_Slot(n, item, torch.tensor((1, 2, 2))) for n, item in zip(sizes, items)]
        scatter_qwen3omni_visual_embeds(
            slots,
            torch.randn(window, 8),
            [torch.randn(window, 8) for _ in range(num_layers)],
            sp_rank=rank,
            sp_size=sp_size,
        )
        for item in items:
            assert item.value.size(0) == len(item.meta["deepstack"][0])
            assert len(item.meta["deepstack"]) == num_layers
        rows = sum(item.value.size(0) for item in items)
        assert rows <= window
        covered += rows
    assert covered == sum(sizes)
