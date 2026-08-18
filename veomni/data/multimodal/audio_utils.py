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

import io
from fractions import Fraction
from io import BytesIO
from typing import ByteString, List, Optional, Tuple, Union

import av
import numpy as np
import torch

from ...utils import logging


logger = logging.get_logger(__name__)


AudioInput = Union[
    np.ndarray,
    ByteString,
    str,
]


def load_audio_bytes_from_path(audio_path: str):
    import soundfile as sf

    audio, sample_rate = sf.read(audio_path)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()


def save_audio_bytes_to_file(audio_bytes, output_path):
    import soundfile as sf

    audio_bytes = io.BytesIO(audio_bytes)
    audio_reloaded, sample_rate = sf.read(audio_bytes)
    sf.write(output_path, audio_reloaded, samplerate=sample_rate)


def load_audio_bytes_from_array(audio_array: np.ndarray, sample_rate: int):
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()


def load_audio_bytes(audio: Union[str, np.ndarray, bytes], sample_rate: Optional[int] = None):
    if isinstance(audio, str):
        return load_audio_bytes_from_path(audio)
    elif isinstance(audio, np.ndarray):
        if sample_rate is None:
            raise ValueError("sample_rate must be provided when audio is a numpy array")
        return load_audio_bytes_from_array(audio, sample_rate)
    elif isinstance(audio, bytes):
        return audio
    else:
        raise ValueError("audio must be a string, numpy array, or bytes")


def load_audio_from_bytes(audio_bytes: bytes, sample_rate: int = 16000, **kwargs):
    import librosa

    with BytesIO(audio_bytes) as wav_io:
        audio, _ = librosa.load(wav_io, sr=sample_rate)
    return audio


def load_audio_from_path(audio_path: str, sample_rate: int = 16000, **kwargs):
    import audioread
    import librosa

    if audio_path.startswith("http://") or audio_path.startswith("https://"):
        return librosa.load(audioread.ffdec.FFmpegAudioFile(audio_path), sr=sample_rate)[0]
    else:
        return librosa.load(audio_path, sr=sample_rate)[0]


def load_audio(audios: AudioInput, **kwargs):
    if isinstance(audios, str):
        return load_audio_from_path(audios)
    elif isinstance(audios, bytes):
        return load_audio_from_bytes(audios, **kwargs)
    else:
        raise NotImplementedError


def fetch_audios(audios: List[AudioInput], **kwargs):
    audios = [load_audio(audio, **kwargs) for audio in audios]
    return audios


def extract_audio_from_video(
    video_input: Union[str, bytes], max_duration_seconds: Optional[float] = None
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Extract audio from video file using PyAV.

    Args:
        video_input: Video file path (str) or video bytes
        max_duration_seconds: Maximum audio duration to extract (prevents OOM).
                            If None, uses video duration + 1 second buffer.

    Returns:
        Tuple containing:
            - audio: Mono audio array (np.ndarray) or None if no audio stream
            - audio_fps: Audio sample rate (int) or None if no audio stream

    Raises:
        Exception: If PyAV fails to open the video container
    """
    audio, audio_fps = None, None

    try:
        # Open video container with PyAV
        import av

        container_input = io.BytesIO(video_input) if isinstance(video_input, bytes) else video_input
        container = av.open(container_input)

        # Check if video has audio streams
        if len(container.streams.audio) > 0:
            audio_stream = container.streams.audio[0]
            audio_fps = audio_stream.rate

            # Prevent OOM: limit audio buffer size
            if max_duration_seconds is None:
                # Use video duration if available, otherwise default to 60 seconds
                video_duration = container.duration / av.time_base if container.duration else 60.0
                max_duration_seconds = video_duration + 1.0

            max_samples = int(max_duration_seconds * audio_fps)

            audio_frames_list = []
            current_samples = 0

            # Decode audio frames
            for frame in container.decode(audio_stream):
                frame_np = frame.to_ndarray()
                audio_frames_list.append(frame_np)
                current_samples += frame_np.shape[1]
                if current_samples >= max_samples:
                    break

            # Concatenate and convert to mono
            if len(audio_frames_list) > 0:
                aframes = np.concatenate(audio_frames_list, axis=1)
                # Convert multi-channel to mono
                if aframes.shape[0] > 1:
                    aframes = np.mean(aframes, axis=0)
                else:
                    aframes = aframes[0]
                audio = aframes

        container.close()

    except Exception as e:
        logger.warning(f"Failed to extract audio from video: {e}")
        audio = None
        audio_fps = None

    return audio, audio_fps


def save_audio_tensor_to_file(
    audio: Union[torch.Tensor, np.ndarray],
    output_path: str,
    sample_rate: int = 32000,
):
    """Save audio tensor or numpy array to WAV file.

    Args:
        audio: Audio data. Supports shapes:
            - (T,) mono
            - (C, T) multi-channel
            - (T, C) multi-channel (auto-detected when C <= 8)
        sample_rate: Audio sample rate in Hz.
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    if audio.ndim == 2:
        # (C, T) -> (T, C) for soundfile (expects samples-first)
        if audio.shape[0] <= 8 and audio.shape[1] > 8:
            audio = audio.T

    import soundfile as sf

    sf.write(output_path, audio, samplerate=sample_rate)


def convert_to_stereo(audio_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert audio to stereo.
    Supports [C, T] or [B, C, T]. Duplicate mono, keep stereo.
    """
    if audio_tensor.size(-2) == 1:
        return audio_tensor.repeat(1, 2, 1) if audio_tensor.dim() == 3 else audio_tensor.repeat(2, 1)
    return audio_tensor


def _resample_audio(
    container: av.container.Container, audio_stream: av.audio.AudioStream, frame_in: av.AudioFrame
) -> None:
    cc = audio_stream.codec_context

    # Use the encoder's format/layout/rate as the *target*
    target_format = cc.format or "fltp"  # AAC → usually fltp
    target_layout = cc.layout or "stereo"
    target_rate = cc.sample_rate or frame_in.sample_rate

    audio_resampler = av.audio.resampler.AudioResampler(
        format=target_format,
        layout=target_layout,
        rate=target_rate,
    )

    audio_next_pts = 0
    for rframe in audio_resampler.resample(frame_in):
        if rframe.pts is None:
            rframe.pts = audio_next_pts
        audio_next_pts += rframe.samples
        rframe.sample_rate = frame_in.sample_rate
        container.mux(audio_stream.encode(rframe))

    # flush audio encoder
    for packet in audio_stream.encode():
        container.mux(packet)


def _write_audio(
    container: av.container.Container,
    audio_stream: av.audio.AudioStream,
    samples: torch.Tensor,
    audio_sample_rate: int,
) -> None:
    if samples.ndim == 1:
        samples = samples.unsqueeze(0)
    samples = convert_to_stereo(samples)
    assert samples.ndim == 2 and samples.shape[0] == 2, "audio samples must be [C, S] or [S], C must be 1 or 2"
    samples = samples.T
    # Convert to int16 packed for ingestion; resampler converts to encoder fmt.
    if samples.dtype != torch.int16:
        samples = torch.clip(samples, -1.0, 1.0)
        samples = (samples * 32767.0).to(torch.int16)

    frame_in = av.AudioFrame.from_ndarray(
        samples.contiguous().reshape(1, -1).cpu().numpy(),
        format="s16",
        layout="stereo",
    )
    frame_in.sample_rate = audio_sample_rate

    _resample_audio(container, audio_stream, frame_in)


def _prepare_audio_stream(container: av.container.Container, audio_sample_rate: int) -> av.audio.AudioStream:
    """
    Prepare the audio stream for writing.
    """
    audio_stream = container.add_stream("aac")
    supported_sample_rates = audio_stream.codec_context.codec.audio_rates
    if supported_sample_rates:
        best_rate = min(supported_sample_rates, key=lambda x: abs(x - audio_sample_rate))
        if best_rate != audio_sample_rate:
            print(f"Using closest supported audio sample rate: {best_rate}")
    else:
        best_rate = audio_sample_rate
    audio_stream.codec_context.sample_rate = best_rate
    audio_stream.codec_context.layout = "stereo"
    audio_stream.codec_context.time_base = Fraction(1, best_rate)
    return audio_stream
