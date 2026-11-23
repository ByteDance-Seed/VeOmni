"""Audio and video on the carrier, as the Qwen3-Omni graph consumes them.

Two things this family needs from the shared data layer are worth pinning here
rather than in the generic ingestion suite, because they are its own:

* **The rate survives the trip.** A Qwen3-Omni thinker's tower runs at 16 kHz and
  the codec that turns assistant speech into talker targets runs at 24 kHz, so
  the data layer must not normalise. The rate is also what places a clip on
  TMRoPE's shared clock, so a dropped or wrong one is a silent alignment bug
  rather than an audio-quality one.
* **Assistant speech is a label, not input.** The same ``type="audio"`` row is an
  encoder input on the user's side and a training target on the assistant's, and
  ``role`` is the only thing separating them.

The generic loader's own behaviour (stereo downmix, container fallbacks, the
empty-clip guard) is covered by ``tests/seed_omni/data/test_audio_ingestion.py``
and is not repeated here.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest
import torch

from veomni.data.seed_omni.preprocess import conv_preprocess
from veomni.data.seed_omni.utils.media_metadata import AUDIO_METADATA_KEY, AudioMetadata
from veomni.models.seed_omni.modules.qwen3omni.audio.processing import (
    _OMNI_MEL_LEN,
    Qwen3OmniAudioPreprocessor,
)
from veomni.models.seed_omni.modules.qwen3omni.codec.processing import Qwen3OmniCodecPreprocessor
from veomni.models.seed_omni.utils.conversation import ConversationItem, is_backbone_input


def _wav_bytes(seconds: float, rate: int) -> bytes:
    """A PCM wav declaring ``rate``, so the loader has a real header to read."""
    frames = int(seconds * rate)
    tone = np.sin(2 * np.pi * 220 * np.arange(frames) / rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((tone * 16384).astype("<i2").tobytes())
    return buffer.getvalue()


def _assistant_audio(seconds: float, rate: int) -> ConversationItem:
    frames = int(seconds * rate)
    return ConversationItem(
        type="audio",
        value=np.zeros(frames, dtype=np.float32),
        role="assistant",
        meta={AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=rate, num_samples=frames)},
    )


# ── the talker corpus layout ───────────────────────────────────────────────────


def test_the_talker_layout_pairs_each_answer_with_its_own_speech():
    """Text and waveform are both required, and they are not interchangeable.

    The talker is taught to say this text, so the text is the speech span's
    content while the waveform is the label the RVQ codes come from.
    """
    constructed, refs = conv_preprocess(
        "voice_assistant_talker",
        [{"from": "human", "value": "what is your name"}, {"from": "gpt", "value": "I am Omni"}],
        {"audios": [_wav_bytes(0.5, 24000)]},
    )

    assert [[part[0] for part in turn[1:]] for turn in constructed] == [["text"], ["text", "audio"]]
    assert [turn[0] for turn in constructed] == ["user", "assistant"]
    assert len(refs["audio"]) == 1


def test_a_talker_row_without_speech_is_refused():
    """Every sample needs one: the talker all-gathers once per utterance, so a
    partly-speech mix would give the ranks different collective counts."""
    with pytest.raises(ValueError, match="spoken assistant turn"):
        conv_preprocess(
            "voice_assistant_talker",
            [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}],
            {"audios": []},
        )
    # No assistant turn at all is the other shape the same guard rejects.
    with pytest.raises(ValueError, match="no assistant turn to speak"):
        conv_preprocess(
            "voice_assistant_talker",
            [{"from": "human", "value": "hi"}],
            {"audios": []},
        )


def test_a_talker_row_with_unpaired_clip_counts_is_refused_both_ways():
    """Pairing is positional, so a mismatch attaches the wrong speech to the
    wrong text rather than failing."""
    with pytest.raises(ValueError, match="unequal count"):
        conv_preprocess(
            "voice_assistant_talker",
            [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}],
            {"audios": [_wav_bytes(0.2, 24000), _wav_bytes(0.2, 24000)]},
        )


def test_the_omni_layout_orders_the_user_clip_before_the_assistant_one():
    """Both audio turns are ``type="audio"``; only ``role`` and order separate
    them, and swapping them would train the talker on the question."""
    constructed, refs = conv_preprocess(
        "voice_assistant_talker_omni",
        [{"from": "human", "value": "describe this"}, {"from": "gpt", "value": "a cat"}],
        {
            "audios": [_wav_bytes(0.2, 16000), _wav_bytes(0.2, 24000)],
            "images": [object()],
            "user_audio": True,
        },
    )

    user_turn = constructed[0]
    assert [kind[0] for kind in user_turn[1:]] == ["image", "audio", "text"]
    assert len(refs["audio"]) == 2 and len(refs["image"]) == 1
    # The assistant's own turn carries text then its speech, so the waveform
    # that becomes the label is last.
    assert [kind[0] for kind in constructed[1][1:]] == ["text", "audio"]


# ── the codec brings assistant speech to its own rate ─────────────────────────


def test_the_codec_retunes_assistant_speech_and_moves_the_timeline_with_it():
    """``encode_wav`` refuses any other rate, and this is what satisfies it.

    Payload and metadata move together: a row left claiming 22.05 kHz x 22050
    samples would read as 0.92 s once the payload became 24 kHz.
    """
    preprocessor = Qwen3OmniCodecPreprocessor(sampling_rate=24000)
    item = _assistant_audio(1.0, 22050)

    preprocessor.preprocess_conversations([[item]])

    metadata = item.meta[AUDIO_METADATA_KEY]
    assert metadata.sampling_rate == 24000
    assert metadata.num_samples == item.value.numel()
    assert metadata.duration_seconds == pytest.approx(1.0, rel=1e-3)


def test_what_the_codec_leaves_on_the_row_is_what_encode_wav_can_read():
    """A tensor, not the numpy array the data layer loaded.

    ``encode_pre`` hands ``item.value`` to ``encode_wav`` verbatim and its shape
    normaliser calls ``.dim()``.
    """
    from veomni.models.seed_omni.modules.qwen3omni.codec.modeling import _as_mono_batch

    preprocessor = Qwen3OmniCodecPreprocessor(sampling_rate=24000)
    items = [_assistant_audio(1.0, 22050), _assistant_audio(1.0, 24000)]

    preprocessor.preprocess_conversations([items])

    for item in items:
        assert isinstance(item.value, torch.Tensor), f"{item.value.__class__.__name__} would break encode_wav"
        assert _as_mono_batch(item.value).shape[:2] == (1, 1)


def test_a_clip_already_at_the_codec_rate_is_not_resampled():
    """Not merely "same length" — resampling 24k to 24k would still filter it."""
    preprocessor = Qwen3OmniCodecPreprocessor(sampling_rate=24000)
    item = _assistant_audio(1.0, 24000)
    item.value = np.linspace(-1, 1, 24000, dtype=np.float32)
    original = item.value.copy()

    preprocessor.preprocess_conversations([[item]])

    assert torch.equal(item.value, torch.from_numpy(original))


def test_the_codec_never_touches_the_user_side():
    """Same row type, opposite meaning — the audio tower owns the user's side."""
    preprocessor = Qwen3OmniCodecPreprocessor(sampling_rate=24000)
    item = ConversationItem(
        type="audio",
        value=np.zeros(22050, dtype=np.float32),
        role="user",
        meta={AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=22050, num_samples=22050)},
    )

    preprocessor.preprocess_conversations([[item]])

    assert item.meta[AUDIO_METADATA_KEY].sampling_rate == 22050
    assert item.value.size == 22050


def test_the_codec_refuses_assistant_audio_that_declares_no_rate():
    preprocessor = Qwen3OmniCodecPreprocessor(sampling_rate=24000)
    item = ConversationItem(type="audio", value=np.zeros(1000, dtype=np.float32), role="assistant", meta={})

    with pytest.raises(ValueError, match="states no meta"):
        preprocessor.preprocess_conversations([[item]])


def test_the_codec_does_nothing_at_inference():
    """The assistant's speech is generation's output there, not its input."""
    preprocessor = Qwen3OmniCodecPreprocessor(sampling_rate=24000)
    item = _assistant_audio(1.0, 22050)

    preprocessor.preprocess_conversations([[item]], inference=True)

    assert item.meta[AUDIO_METADATA_KEY].sampling_rate == 22050


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


# ── the carrier predicate the packers and scatters share ──────────────────────


def test_assistant_speech_is_not_backbone_input_but_dummies_and_user_audio_are():
    """The rule every packer and scatter pairs rows by.

    It cannot be derived from ``type``: the user's clip and the assistant's are
    both ``type="audio"``. It is a statement the row itself makes, set by the
    chat template when it defers the speech target out of the sequence.
    """
    from veomni.models.seed_omni.utils.conversation import _NOT_BACKBONE_INPUT

    real = ConversationItem(type="text", value="hi", role="user")
    dummy = ConversationItem(type="image", value=torch.zeros(4, 8), role="user", is_dummy=True)
    user_audio = ConversationItem(type="audio", value=torch.zeros(8), role="user")
    label = ConversationItem(type="audio", value=torch.zeros(8), role="assistant", meta={_NOT_BACKBONE_INPUT: True})

    assert is_backbone_input(real)
    assert is_backbone_input(user_audio)
    assert not is_backbone_input(dummy)
    assert not is_backbone_input(label)


def test_the_chat_template_marks_the_speech_target_it_defers():
    """Held out of the chat template *and* marked, because the packers skip by
    the marker and cannot re-derive it from ``role`` alone."""
    from veomni.models.seed_omni.modules.qwen3omni.text_encoder.chat_template import Qwen3OmniChatTemplate
    from veomni.models.seed_omni.utils.conversation import _NOT_BACKBONE_INPUT

    class _Tokenizer:
        bos_token = None
        eos_token = "<|im_end|>"
        bos_token_id = None
        eos_token_id = 1

        def convert_tokens_to_ids(self, token):
            return abs(hash(token)) % 1000

    template = Qwen3OmniChatTemplate(_Tokenizer())
    speech = ConversationItem(type="audio", value=np.zeros(2400, dtype=np.float32), role="assistant")
    parts = template.apply_chat_template(
        [
            ConversationItem(type="text", value="say hi", role="user"),
            ConversationItem(type="text", value="hi", role="assistant"),
            speech,
        ]
    )

    assert any(p is speech for p in parts), "the row must stay on the carrier for the codec"
    assert speech.meta[_NOT_BACKBONE_INPUT] is True


def test_the_shared_packer_skips_the_marked_row():
    """A packer that kept it would read a raw waveform's *sample* count as a row
    count, and every projection after the row would shift without raising."""
    from veomni.models.seed_omni.modules.base.llm_packing import pack_llm_conversations_for_forward
    from veomni.models.seed_omni.utils.conversation import _NOT_BACKBONE_INPUT

    sample = [
        ConversationItem(
            type="text", value=torch.zeros(3, 8), role="user", meta={"labels": torch.zeros(3, dtype=torch.long)}
        ),
        ConversationItem(
            type="audio",
            value=torch.zeros(2400),
            role="assistant",
            meta={_NOT_BACKBONE_INPUT: True},
        ),
    ]
    inputs_embeds, *_ = pack_llm_conversations_for_forward([sample], torch.device("cpu"))

    assert tuple(inputs_embeds.shape) == (1, 3, 8)
