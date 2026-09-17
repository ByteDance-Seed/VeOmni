"""Audio into the SeedOmni carrier, and the rate that has to survive the trip.

The data layer loads a clip at whatever rate the file holds and declares it;
each module then converts to its own. That split exists because the two modules
reading a conversation disagree — a Qwen3-Omni thinker's tower runs at 16 kHz,
the codec that turns assistant speech into talker targets at 24 kHz — so any
rate the data layer picked would be wrong for one of them, and the second
conversion cannot restore what the first removed.

The rate is also what places a clip on TMRoPE's shared clock, so a dropped or
wrong rate is a silent alignment bug rather than an audio-quality one.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import wave
from unittest import mock

import numpy as np
import pytest
import torch

from veomni.data.seed_omni.preprocess import SEED_OMNI_PREPROCESSOR_REGISTRY, conv_preprocess
from veomni.data.seed_omni.seedomni_transform import process_seedomni_example
from veomni.data.seed_omni.utils import audio as audio_utils
from veomni.data.seed_omni.utils.audio import SAMPLING_RATE_KEY, load_audio


def _wav_bytes(seconds: float, rate: int, channels: int = 1) -> bytes:
    """A PCM wav declaring ``rate``, so the loader has a real header to read."""
    frames = int(seconds * rate)
    tone = np.sin(2 * np.pi * 220 * np.arange(frames) / rate)
    samples = (tone * 16384).astype("<i2")
    if channels > 1:
        samples = np.repeat(samples[:, None], channels, axis=1).reshape(-1)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


def _sample(audios: list[bytes], turns: list[tuple[str, str]]) -> dict:
    return {
        "source_name": "voice_assistant",
        "conversations": [{"from": speaker, "value": value} for speaker, value in turns],
        "audios": audios,
    }


def test_a_clip_arrives_at_its_own_rate_not_a_normalised_one():
    """22.05 kHz is neither module's rate, and it must reach them as 22.05 kHz."""
    items = process_seedomni_example(_sample([_wav_bytes(1.5, 22050)], [("human", "hi"), ("gpt", "hello")]))[0][
        "conversation_list"
    ]
    audio = next(item for item in items if item.type == "audio")

    assert audio.meta[SAMPLING_RATE_KEY] == 22050
    assert audio.value.dtype == np.float32
    assert audio.value.shape == (33075,)


def test_the_declared_rate_and_the_sample_count_agree_on_the_duration():
    """The one invariant every downstream clock reads: samples / rate == seconds."""
    items = process_seedomni_example(_sample([_wav_bytes(2.0, 44100)], [("human", "hi"), ("gpt", "yes")]))[0][
        "conversation_list"
    ]
    audio = next(item for item in items if item.type == "audio")

    assert audio.value.size / audio.meta[SAMPLING_RATE_KEY] == pytest.approx(2.0, abs=1e-3)


def test_a_stereo_file_is_downmixed_where_its_layout_is_known():
    """The modules take mono and refuse to guess a mix, so this layer owes them one."""
    samples, rate = load_audio(_wav_bytes(1.0, 16000, channels=2))

    assert rate == 16000
    assert samples.ndim == 1
    assert samples.size == 16000


def test_a_one_frame_stereo_file_is_not_read_as_two_samples():
    """The decoded layout is known, so it is not guessed.

    ``sf.read`` gives ``(frames, channels)``. A guess that the channel axis is
    the shorter one is right for any real clip but inverts here, turning two
    channels into two samples.
    """
    samples, rate = load_audio(_wav_bytes(1 / 16000, 16000, channels=2))

    assert samples.shape == (1,)
    assert rate == 16000


@pytest.mark.parametrize("shape", [(1000, 2), (2, 1000)], ids=["frames-first", "channels-first"])
def test_a_bare_stereo_array_is_downmixed_whichever_way_round_it_is(shape):
    """The channel axis is guessed here, and a wrong guess is silent.

    A bare array carries no layout, so the shorter axis is taken to be the
    channels. Inverting that collapses the clip to one value per channel — in
    *both* orientations, so a single-orientation test cannot catch it, and the
    empty-clip guard does not either since two samples is not zero.
    """
    samples, rate = load_audio(np.zeros(shape, dtype=np.float32), audio_sampling_rate=16000)

    assert samples.shape == (1000,)
    assert rate == 16000


def test_an_empty_clip_is_refused_rather_than_averaged_into_nan():
    """Averaging an empty axis yields NaN with only a warning.

    Those NaN samples surface much later as a NaN loss, with nothing pointing
    back to the file that produced them.
    """
    with pytest.raises(ValueError, match="carries no samples"):
        load_audio(_wav_bytes(0.0, 16000, channels=2))


@pytest.mark.parametrize("as_bytes", [False, True], ids=["path", "bytes"])
def test_a_container_libsndfile_cannot_read_falls_back_to_audioread(tmp_path, as_bytes):
    """m4a / aac / wma need a second backend, and bytes is the likely form.

    A parquet corpus stores clips inline, so an m4a source arrives as bytes —
    the backend takes a filename, hence the spill. The fallback must keep the
    native rate: librosa's own default would resample to 22.05 kHz.

    ffmpeg is needed here to *author* the file; decoding may go through it or
    gstreamer, so a box with only the latter still exercises the path.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not available to author a non-libsndfile container")
    path = tmp_path / "clip.m4a"
    subprocess.run(
        [
            ffmpeg,
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1.5:sample_rate=32000",
            "-ac",
            "1",
            "-y",
            str(path),
            "-loglevel",
            "error",
        ],
        check=True,
    )

    samples, rate = load_audio(path.read_bytes() if as_bytes else str(path))

    assert rate == 32000
    assert samples.size / rate == pytest.approx(1.5, abs=0.05)


def test_a_bare_array_must_be_told_its_rate():
    """An array carries no header, and guessing one would misplace it on the clock."""
    with pytest.raises(ValueError, match="declares no rate"):
        load_audio(np.zeros(1000, dtype=np.float32))

    samples, rate = load_audio(np.zeros(1000, dtype=np.float32), audio_sampling_rate=8000)
    assert (samples.size, rate) == (1000, 8000)


def test_a_repeated_ref_is_decoded_once_but_not_shared():
    """Decoded once for speed, copied for safety.

    The two rows holding one clip can be the user's and the assistant's, which
    different modules read, so sharing one array would let the first in-place
    write anyone adds corrupt the other module's input.
    """
    clip = _wav_bytes(0.5, 16000)
    decodes = 0
    real_load = audio_utils.load_audio

    def counting_load(*args, **kwargs):
        nonlocal decodes
        decodes += 1
        return real_load(*args, **kwargs)

    with mock.patch.object(audio_utils, "load_audio", counting_load):
        loaded = audio_utils.fetch_audios([clip, clip])

    assert decodes == 1
    assert loaded[0][0] is not loaded[1][0]
    assert np.array_equal(loaded[0][0], loaded[1][0])


def test_a_generated_waveform_is_written_however_the_vocoder_shaped_it(tmp_path):
    """The shape and dtype the codec actually emits: batched, and bf16 under AMP.

    Both conversions the writer makes are load-bearing: soundfile takes only
    1-D or 2-D, so an unsqueezed clip raises rather than writing, and
    ``Tensor.numpy()`` refuses bfloat16 outright.
    """
    rate = 24000
    frames = rate // 10
    tone = np.sin(2 * np.pi * 220 * np.arange(frames) / rate).astype(np.float32)
    path = tmp_path / "generated_audio_0.wav"

    audio_utils.save_audio(str(path), torch.from_numpy(tone).to(torch.bfloat16)[None, None, :], rate)

    written, written_rate = load_audio(str(path))
    assert written_rate == rate
    assert written.shape == tone.shape
    # bfloat16 keeps 8 mantissa bits, so the tolerance is set by the dtype the
    # vocoder ran in, not by the wav.
    np.testing.assert_allclose(written, tone, atol=1e-2)


def test_a_generated_clip_with_no_rate_is_refused_rather_than_written_at_a_guess(tmp_path):
    """A default rate writes a file that plays at the wrong speed, and nothing
    downstream can tell that apart from a model that generated it wrong."""
    with pytest.raises(ValueError, match=SAMPLING_RATE_KEY):
        audio_utils.save_audio(str(tmp_path / "generated_audio_0.wav"), torch.zeros(2400), None)


def test_a_multi_channel_waveform_is_refused_by_name(tmp_path):
    """Anything still 2-D after the squeeze is a real batch or true stereo, and
    neither names a single mono utterance.

    Left to soundfile this is not silent — it reads the 2400 axis as channels
    and libsndfile rejects the format — but the error names neither the item nor
    its shape, which is the whole diagnostic.
    """
    with pytest.raises(ValueError, match="not a single"):
        audio_utils.save_audio(str(tmp_path / "generated_audio_0.wav"), torch.zeros(2, 2400), 24000)


def test_an_empty_generated_clip_is_refused_rather_than_written_as_a_bare_header(tmp_path):
    """Symmetry with the load side, which refuses an empty clip by name.

    ``squeeze`` leaves a zero-length clip 1-D, so it passes the mono check;
    without this guard soundfile writes a valid 44-byte header and the caller
    logs a successful write, leaving a playable-looking file with nothing in it.
    """
    path = tmp_path / "generated_audio_0.wav"
    with pytest.raises(ValueError, match="no samples"):
        audio_utils.save_audio(str(path), torch.zeros(1, 1, 0), 24000)
    assert not path.exists()


def test_a_one_sample_clip_is_mono_not_a_batching_mistake(tmp_path):
    """``squeeze`` takes a single sample down to a scalar, which is still mono.

    Degenerate, but reporting it as "not a single mono waveform. Emit one item
    per utterance" would send the reader looking for a batching bug.
    """
    path = tmp_path / "generated_audio_0.wav"
    audio_utils.save_audio(str(path), torch.ones(1, 1, 1), 24000)

    samples, rate = load_audio(str(path))
    assert (samples.size, rate) == (1, 24000)


def test_more_clips_than_spoken_turns_is_refused_rather_than_silently_dropped():
    with pytest.raises(ValueError, match="spoken turn"):
        conv_preprocess(
            "voice_assistant",
            [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}],
            {"audios": [_wav_bytes(0.5, 16000), _wav_bytes(0.5, 16000)]},
        )


def test_fewer_clips_than_spoken_turns_is_refused_too():
    """Both directions, and both by a named error.

    A bare ``next()`` running out would raise ``StopIteration``, which PEP 479
    turns into a ``RuntimeError`` once it crosses a generator frame — and the
    dataset stack is full of them, so the type would depend on the call site.
    """
    with pytest.raises(ValueError, match="paired by position"):
        conv_preprocess(
            "voice_assistant",
            [{"from": "human", "value": "hi"}, {"from": "human", "value": "again"}],
            {"audios": [_wav_bytes(0.5, 16000)]},
        )


@pytest.fixture
def _declared_rate_source():
    """A source that asserts a rate on the item meta, contradicting the file."""
    SEED_OMNI_PREPROCESSOR_REGISTRY["_declared_rate"] = lambda conversations, example, **kwargs: (
        [["user", ("audio", None, {SAMPLING_RATE_KEY: 16000})]],
        [],
        [],
        list(example["audios"]),
    )
    yield "_declared_rate"
    del SEED_OMNI_PREPROCESSOR_REGISTRY["_declared_rate"]


@pytest.fixture
def _three_tuple_source():
    """A source registered for one test, and removed even if it fails.

    Via the local-override path rather than ``register``, which raises on a
    duplicate key and would make this file fail on a second collection while
    leaving the key visible to every later test in the process.
    """
    SEED_OMNI_PREPROCESSOR_REGISTRY["_legacy_three_tuple"] = lambda conversations, example, **kwargs: (
        [["user", ("text", "hi")]],
        [],
        [],
    )
    yield "_legacy_three_tuple"
    del SEED_OMNI_PREPROCESSOR_REGISTRY["_legacy_three_tuple"]


def test_a_declared_rate_that_contradicts_the_file_is_refused(_declared_rate_source):
    """The header wins any argument, so a disagreement is a false claim.

    Preferring the declared value would be a silent misplacement on TMRoPE's
    clock rather than a clip that sounds wrong, which is why it stops the run.
    """
    with pytest.raises(ValueError, match="but the clip decodes at"):
        process_seedomni_example({"source_name": _declared_rate_source, "audios": [_wav_bytes(0.5, 22050)]})


def test_a_preprocessor_written_before_audio_existed_still_works(_three_tuple_source):
    """The registry is an extension point, so widening it must not break it.

    A preprocessor for a private corpus lives outside this repo and returns the
    older 3-tuple; that is read as "this source has no audio".
    """
    assert conv_preprocess(_three_tuple_source, None, {}) == ([["user", ("text", "hi")]], [], [], [])


def test_the_voice_assistant_layout_is_speech_in_text_out():
    """The clip is the user's, so it is an encoder input — not a talker target.

    ``role`` is the only thing separating the two uses of ``type="audio"``, and
    the transcript is dropped: fed as text the model would answer from it and
    the tower would carry no signal at all.
    """
    items = process_seedomni_example(
        _sample([_wav_bytes(1.0, 22050)], [("human", "what is your name"), ("gpt", "I am Omni")])
    )[0]["conversation_list"]

    assert [(item.role, item.type) for item in items] == [("user", "audio"), ("assistant", "text")]
    assert "what is your name" not in str([item.value for item in items if item.type == "text"])
