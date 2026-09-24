# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal audio IO for the SeedOmni data layer.

**Loads at the source's own rate and reports it.** That is the whole design, and
it is the opposite of ``veomni/data/multimodal/audio_utils.py``, which resamples
to 16 kHz and returns the samples alone. Two reasons the rate cannot be dropped
here, and neither is about audio quality:

* **There is no single right rate.** One conversation can carry audio bound for
  two modules that disagree: a Qwen3-Omni thinker's tower wants 16 kHz, and the
  codec that tokenises the assistant's speech into talker targets wants 24 kHz.
  Picking one in the data layer makes the other module resample back, and
  upsampling cannot restore the band the first conversion removed.
* **The rate is what puts a clip on the shared clock.** Under TMRoPE an audio
  token costs one position id and a video's temporal row advances by wall-clock
  seconds, so a clip whose declared duration is wrong drifts against the video
  track — 22.05 kHz read as 16 kHz makes a 4 s clip claim 5.54 s, a 38% error.

So the samples travel with ``meta["sampling_rate"]`` and each module converts to
what it needs (``Qwen3OmniAudioPreprocessor._to_encoder_rate``,
``Qwen3OmniCodecPreprocessor``). Resampling is a module decision because the
target rate is a module property.
"""

from __future__ import annotations

import os
import tempfile
from io import BytesIO
from typing import ByteString, Union

import numpy as np
import torch


AudioInput = Union[np.ndarray, ByteString, str]

# What an item declares its rate as, on ``ConversationItem.meta``. The same
# spelling the audio tower reads, the codec requires, and ``OmniInferencer``
# needs before writing a wav — one key for the whole graph.
SAMPLING_RATE_KEY = "sampling_rate"


def load_audio(audio: AudioInput, audio_sampling_rate: int | None = None) -> tuple[np.ndarray, int]:
    """Load one clip as ``(mono float32 samples, rate)`` at its native rate.

    ``audio_sampling_rate`` is *not* a resample request — it is the rate of a
    bare sample array, which carries none. A path or bytes stream states its own
    and this argument is ignored for them.
    """
    if isinstance(audio, np.ndarray):
        if audio_sampling_rate is None:
            raise ValueError(
                "load_audio: a raw sample array declares no rate, so one must be passed as "
                "audio_sampling_rate. Downstream modules resample to what they need and cannot guess."
            )
        # Channel axis unknown for a bare array, so it is guessed; see
        # ``_to_mono_float32``. A decoded file does not need the guess.
        return _to_mono_float32(audio), int(audio_sampling_rate)

    if isinstance(audio, (bytes, bytearray)):
        source: object = BytesIO(bytes(audio))
    elif isinstance(audio, str):
        if audio.startswith(("http://", "https://")):
            # Fetched here rather than handed to a decoder, so that a URL means
            # the same thing it does in ``fetch_images``.
            import requests

            # Timeout, because this runs on a dataloader worker: a server that
            # accepts and then stalls would stop the training loop producing
            # batches with no exception to name the cause.
            response = requests.get(audio, timeout=(5, 30))
            response.raise_for_status()
            source = BytesIO(response.content)
        elif not os.path.exists(audio):
            raise FileNotFoundError(f"Audio path does not exist: {audio}")
        else:
            source = audio
    else:
        raise NotImplementedError(f"Unsupported audio input type: {type(audio).__name__}")

    import soundfile

    try:
        # ``always_2d=False`` keeps a mono file 1-D; ``dtype="float32"`` is the
        # only conversion done here, and ``sf.read`` never resamples.
        samples, rate = soundfile.read(source, dtype="float32", always_2d=False)
    except soundfile.LibsndfileError:
        # libsndfile covers wav / flac / ogg / mp3 but not the ffmpeg-only
        # containers (m4a, aac, wma), which ``veomni.data.multimodal`` reaches through
        # librosa's audioread backend. ``sr=None`` is what keeps the native
        # rate — librosa's own default would resample to 22.05 kHz and defeat
        # the point of this file.
        return _decode_via_audioread(source)

    # ``sf.read`` is documented to give ``(frames, channels)``, so the axis is
    # known rather than guessed. It matters for a degenerate clip: a stereo file
    # of one frame has fewer frames than channels, and a guess would read its
    # two channels as two samples.
    return _to_mono_float32(samples, channel_axis=1), int(rate)


def _decode_via_audioread(source: str | BytesIO) -> tuple[np.ndarray, int]:
    """Second-chance decode for a container libsndfile does not know.

    ``audioread`` picks among ffmpeg / gstreamer / coreaudio, so which one runs
    depends on the box; with none installed it raises ``NoBackendError``, which
    names no container. It takes a filename rather than a stream, so a bytes
    source is spilled — worth the write because a parquet corpus stores its
    clips inline, making bytes the *likely* form for an m4a source rather than
    the exotic one. The suffix below is cosmetic: the container is sniffed from
    content, not from the name.

    This route is deprecated upstream (librosa 1.0 drops it) and ``audioread``
    imports modules removed in Python 3.13, so it is a second reason this
    project's ``<3.13`` pin cannot move. Revisit when the librosa pin does.
    """
    import librosa

    if isinstance(source, str):
        samples, rate = librosa.load(source, sr=None, mono=True)
        return _to_mono_float32(samples), int(rate)

    source.seek(0)
    with tempfile.NamedTemporaryFile(suffix=".audio") as spill:
        spill.write(source.read())
        spill.flush()
        samples, rate = librosa.load(spill.name, sr=None, mono=True)
    return _to_mono_float32(samples), int(rate)


def _to_mono_float32(samples: np.ndarray, channel_axis: int | None = None) -> np.ndarray:
    """Mono float32, averaging channels.

    Downmixing here rather than downstream because this is the layer that knows
    the file's channel layout. The modules take mono and reject anything else
    (they have no basis for choosing a mix), so a stereo file that reached them
    unmixed would simply fail.

    ``channel_axis`` is passed when the layout is known and guessed otherwise —
    the guess assumes the channel axis is the shorter one, which is true of any
    clip longer than a handful of samples.
    """
    array = np.asarray(samples, dtype=np.float32)
    if array.size == 0:
        # Averaging an empty axis yields NaN with only a warning, and NaN
        # samples surface much later as a NaN loss with nothing pointing back
        # to the file that produced them.
        raise ValueError("load_audio: clip is empty — it carries no samples to encode or score against.")
    if array.ndim == 1:
        return array
    if array.ndim != 2:
        raise ValueError(f"load_audio: expected 1-D or 2-D samples, got shape {tuple(array.shape)}")
    # The guess: average over whichever axis is *not* the shorter one, since a
    # real clip has far more frames than channels. Getting this backwards
    # collapses the clip to one value per channel in both orientations, which
    # the empty-clip guard above would not catch.
    axis = channel_axis if channel_axis is not None else int(array.shape[0] >= array.shape[1])
    return array.mean(axis=axis).astype(np.float32, copy=False)


def fetch_audios(audios: list[AudioInput], **kwargs) -> list[tuple[np.ndarray, int]]:
    """Load a list of clips, each as ``(samples, rate)`` at its own native rate.

    A repeated ref is decoded once and copied on reuse, as ``fetch_images``
    does. The copy is not redundant: the two rows holding one clip can be
    ``role="user"`` and ``role="assistant"``, which are read by different
    modules, and the alternative — one shared array — makes the first in-place
    write anyone adds corrupt the other module's input silently.
    """
    audio_sampling_rate = kwargs.get("audio_sampling_rate")
    cache: dict = {}
    out: list[tuple[np.ndarray, int]] = []
    for audio in audios:
        key = audio if isinstance(audio, (str, bytes)) else id(audio)
        if key in cache:
            samples, rate = cache[key]
            out.append((samples.copy(), rate))
        else:
            cache[key] = load_audio(audio, audio_sampling_rate=audio_sampling_rate)
            out.append(cache[key])
    return out


def save_audio(path: str, waveform: torch.Tensor, sampling_rate: int | None) -> None:
    """Write one generated waveform, however the emitting module shaped it.

    The inverse of :func:`load_audio`, and it keeps that function's contract on
    both counts. The rate must come from the module that produced the samples —
    a default here would write a file that plays at the wrong speed, which is
    indistinguishable from a model that generated it wrong. And an empty clip is
    refused rather than written, the same way ``_to_mono_float32`` refuses one on
    the way in: a 44-byte header-only wav looks like a file that was written.

    ``soundfile`` picks the container from ``path``'s extension, and for ``.wav``
    the default subtype is ``PCM_16``. Samples outside ``[-1, 1]`` are therefore
    hard-clipped on write, so a vocoder that returns unnormalised audio produces
    a file that is quietly distorted rather than one that fails.

    The two conversions before the checks:

    * ``squeeze`` — a vocoder hands back ``(batch, channel, samples)`` and
      soundfile takes only 1-D or 2-D, so this is what lets a caller pass the
      tensor through unshaped. It is a convenience, not a guard: a real clip
      left unsqueezed raises rather than writing anything.
    * ``float()`` — under mixed precision the waveform arrives ``bfloat16``,
      which numpy has no dtype for: ``Tensor.numpy()`` raises
      ``TypeError: Got unsupported ScalarType BFloat16``.
    """
    import soundfile

    if sampling_rate is None:
        raise ValueError(
            f"save_audio: the audio item for {path} carries no meta[{SAMPLING_RATE_KEY!r}]; the emitting "
            f"module must declare the rate it generated at."
        )
    if not torch.is_tensor(waveform):
        raise TypeError(f"save_audio: the audio item for {path} is a {type(waveform).__name__}, expected a tensor.")
    samples = waveform.detach().float().cpu().squeeze()
    # ``squeeze`` takes a one-sample clip down to a scalar; it is still mono, and
    # reporting it as a batching problem below would send the reader hunting for
    # a bug that is not there.
    if samples.ndim == 0:
        samples = samples.reshape(1)
    if samples.ndim != 1:
        raise ValueError(
            f"save_audio: the audio item for {path} has shape {tuple(waveform.shape)}, which is not a single "
            f"mono waveform. Emit one item per utterance."
        )
    if samples.numel() == 0:
        raise ValueError(
            f"save_audio: the audio item for {path} carries no samples. The emitting module produced an empty "
            f"waveform — writing it would leave a playable-looking file with nothing in it."
        )
    soundfile.write(path, samples.numpy(), int(sampling_rate))


__all__ = ["AudioInput", "SAMPLING_RATE_KEY", "fetch_audios", "load_audio", "save_audio"]
