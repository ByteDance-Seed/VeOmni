import io
from io import BytesIO
from typing import ByteString, List, Optional, Union

import audioread
import librosa
import numpy as np
import soundfile as sf

from .file_utils import resolve_relative_path


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
        audio_path = resolve_relative_path(audio_path, kwargs.get("train_path"))
        return librosa.load(audio_path, sr=sample_rate)[0]


def load_audio(audios: AudioInput, **kwargs):
    if isinstance(audios, str):
        return load_audio_from_path(audios, **kwargs)
    elif isinstance(audios, bytes):
        return load_audio_from_bytes(audios, **kwargs)
    else:
        raise NotImplementedError


def fetch_audios(audios: List[AudioInput], **kwargs):
    audios = [load_audio(audio, **kwargs) for audio in audios]
    return audios
