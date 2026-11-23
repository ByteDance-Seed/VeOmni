"""Tests for the Qwen3-Omni audio tower OmniModule.

The risky part of this module is length bookkeeping: the tower eats one flat
concatenation of every clip in the batch and returns one flat tensor, so the
split on the way back out is *derived* from a length formula rather than
observed. If that formula disagrees with the tower by even one frame, embeds
scatter onto the wrong items and nothing raises.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.tools.training_utils import make_eager_ops_config
from veomni.data.seed_omni.utils.media_metadata import AUDIO_METADATA_KEY, AudioMetadata
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
from veomni.models.seed_omni.utils.conversation import ConversationItem


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
