import io
from io import BytesIO
from typing import ByteString, List, Optional, Tuple, Union

import audioread
import av
import librosa
import numpy as np
import soundfile as sf

from ...utils import logging


logger = logging.get_logger(__name__)


if not hasattr(av, "AVError"):
    try:
        from av.error import AVError  # noqa: F401
    except (ImportError, AttributeError):
        av.AVError = OSError


AudioInput = Union[
    np.ndarray,
    ByteString,
    str,
]


def load_audio_bytes_from_path(audio_path: str):
    audio, sample_rate = sf.read(audio_path)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()


def save_audio_bytes_to_file(audio_bytes, output_path):
    audio_bytes = io.BytesIO(audio_bytes)
    audio_reloaded, sample_rate = sf.read(audio_bytes)
    sf.write(output_path, audio_reloaded, samplerate=sample_rate)


def load_audio_bytes_from_array(audio_array: np.ndarray, sample_rate: int):
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
    with BytesIO(audio_bytes) as wav_io:
        audio, _ = librosa.load(wav_io, sr=sample_rate)
    return audio


def load_audio_from_path(audio_path: str, sample_rate: int = 16000, **kwargs):
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
