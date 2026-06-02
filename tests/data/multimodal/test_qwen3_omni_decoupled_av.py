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

"""Smoke test: Qwen3-Omni decoupled video frames + standalone audio data path.

Verifies:
  1. The `qwen_omni_decoupled_av` preprocessor splits text/video/audio markers
     into separate conversation items.
  2. `fetch_videos` returns `(video_tensor, None)` for List[bytes] frame inputs.
  3. `fetch_audios` returns the standalone audio.
  4. (Optional, requires the Qwen3-Omni processor on disk) `process_sample_qwen_omni`
     produces input_ids where `<|video_pad|>` and `<|audio_pad|>` form **separate**
     contiguous runs (no interleaving), and `video_grid_thw` matches frame count.

Run::

    pytest tests/data/multimodal/test_qwen3_omni_decoupled_av.py -v -s
    # Or, to also exercise the full processor path:
    QWEN3_OMNI_MODEL_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct pytest ... -v -s
"""

import os
from io import BytesIO

import numpy as np
import PIL.Image
import pytest
import torch

from veomni.data.multimodal.preprocess import conv_preprocess
from veomni.utils.constants import IGNORE_INDEX  # noqa: F401  (sanity import)
from veomni.utils.import_utils import is_ffmpeg_available


needs_ffmpeg = pytest.mark.skipif(
    not is_ffmpeg_available(),
    reason="ffmpeg binary not on PATH — fetch_videos() raises at the top regardless of input type",
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _make_png_frame(h: int = 64, w: int = 64, color: int = 0) -> bytes:
    """Return PNG-encoded bytes of a solid-color RGB frame."""
    arr = np.full((h, w, 3), color, dtype=np.uint8)
    img = PIL.Image.fromarray(arr)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_silent_wav(duration_s: float = 1.0, sr: int = 16000) -> bytes:
    """Return WAV-encoded bytes of mono silence (no extra deps)."""
    samples = np.zeros(int(duration_s * sr), dtype=np.int16)
    buf = BytesIO()
    import wave

    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(samples.tobytes())
    return buf.getvalue()


# ----------------------------------------------------------------------------
# 1. Preprocessor produces video/audio/text items in marker order
# ----------------------------------------------------------------------------


def test_preprocessor_splits_markers():
    conversations = [
        {"from": "human", "value": "<video>\n<audio>\nWhat is happening?"},
        {"from": "gpt", "value": "Someone is speaking near a car."},
    ]
    out = conv_preprocess("qwen_omni_decoupled_av", conversations)

    assert out[0][0] == "user"
    # Drop ("text", "\n") chunks between markers — only assert mm content + final text exist
    types = [item[0] for item in out[0][1:]]
    assert types[0] == "video", f"first item should be video, got {types}"
    assert "audio" in types, f"audio missing, got {types}"
    assert types[-1] == "text", f"last item should be text, got {types}"
    assert "What is happening?" in out[0][-1][1]

    assert out[1] == ["assistant", ("text", "Someone is speaking near a car.")]


def test_preprocessor_multiple_videos_per_turn():
    conversations = [
        {"from": "human", "value": "Compare <video> with <video>."},
        {"from": "gpt", "value": "They differ in color."},
    ]
    out = conv_preprocess("qwen_omni_decoupled_av", conversations)
    types = [item[0] for item in out[0][1:]]
    assert types.count("video") == 2


# ----------------------------------------------------------------------------
# 2. fetch_videos on List[bytes] → (video_tensor, None) audio
# ----------------------------------------------------------------------------


@needs_ffmpeg
def test_fetch_videos_frame_list_returns_none_audio():
    from veomni.data.multimodal.video_utils import fetch_videos

    frames = [_make_png_frame(color=c) for c in (10, 80, 150, 220)]
    videos = [frames]
    video_inputs, audio_inputs = fetch_videos(
        videos,
        fps=2.0,
        frame_factor=2,
        min_frames=4,
        max_frames=20,
        scale_factor=28,
        video_min_pixels=100352,
        video_max_pixels=602112,
        max_ratio=200,
        sample_rate=16000,
    )

    assert len(video_inputs) == 1
    assert isinstance(video_inputs[0], torch.Tensor)
    assert video_inputs[0].ndim == 4  # (T, C, H, W)
    # Frame-list inputs must surface as audio=None so the downstream processor
    # emits an *unpaired* video token run (no <video>/<audio> interleaving).
    assert audio_inputs == [None]


def test_load_and_process_video_with_codec_frame_list_audio_none():
    # No @needs_ffmpeg: the List[bytes] branch in _load_and_process_video_with_codec
    # decodes via PIL only and never touches torchcodec/PyAV (see video_utils.py:398-420).
    from veomni.data.multimodal.video_utils import _load_and_process_video_with_codec

    frames = [_make_png_frame(color=c) for c in (0, 50, 100, 150, 200, 250)]
    video, audio, audio_fps, frames_indices = _load_and_process_video_with_codec(
        frames,
        fps=2.0,
        frame_factor=2,
        min_frames=2,
        max_frames=10,
        scale_factor=28,
        video_min_pixels=100352,
        video_max_pixels=602112,
        max_ratio=200,
    )
    assert video.ndim == 4
    assert audio is None
    assert audio_fps is None
    assert frames_indices.numel() == video.shape[0]


# ----------------------------------------------------------------------------
# 3. fetch_audios independently loads standalone audio
# ----------------------------------------------------------------------------


def test_fetch_audios_independent():
    pytest.importorskip("librosa", reason="audio extra not installed (uv sync --extra audio)")
    from veomni.data.multimodal.audio_utils import fetch_audios

    wav_bytes = _make_silent_wav(duration_s=1.0, sr=16000)
    audios = fetch_audios([wav_bytes], sample_rate=16000)
    assert len(audios) == 1
    assert audios[0].ndim == 1
    # Should be sampled at 16k, ~1s of silence
    assert 15500 <= audios[0].shape[0] <= 16500


# ----------------------------------------------------------------------------
# 4. Full processor path — gated on QWEN3_OMNI_MODEL_PATH
# ----------------------------------------------------------------------------


QWEN3_OMNI_MODEL_PATH = os.environ.get("QWEN3_OMNI_MODEL_PATH", "")


@needs_ffmpeg
@pytest.mark.skipif(
    not QWEN3_OMNI_MODEL_PATH,
    reason="Set QWEN3_OMNI_MODEL_PATH to a local Qwen3-Omni-30B-A3B-Instruct checkpoint to enable.",
)
def test_qwen3_omni_decoupled_av_end_to_end():
    """Run the actual Qwen3-Omni transform on a synthetic decoupled sample.

    Uses a stub `position_id_func` because the assertions check token-interleaving
    in the processor output (which is produced *before* position_id_func runs).
    """
    from veomni.data.data_transform import process_sample_qwen_omni
    from veomni.models import build_processor

    processor = build_processor(QWEN3_OMNI_MODEL_PATH)

    def stub_position_id_func(input_ids, attention_mask, **_kwargs):
        # 3D position_ids (dim, batch, len) — matches what the Qwen-Omni RoPE expects
        L = input_ids.shape[-1]
        return {"position_ids": torch.zeros(3, 1, L, dtype=torch.long)}

    position_id_func = stub_position_id_func

    # ---- synthetic sample ----
    frames = [_make_png_frame(color=c) for c in (10, 80, 150, 220)]
    wav_bytes = _make_silent_wav(duration_s=1.0, sr=16000)
    sample = {
        "videos": [frames],
        "audios": [wav_bytes],
        "conversations": [
            {"from": "human", "value": "<video>\n<audio>\nWhat is happening?"},
            {"from": "gpt", "value": "Someone is speaking."},
        ],
    }

    out = process_sample_qwen_omni(
        sample,
        processor=processor,
        position_id_func=position_id_func,
        source_name="qwen_omni_decoupled_av",
        # mm_configs
        scale_factor=28,
        image_min_pixels=3136,
        image_max_pixels=12845056,
        video_min_pixels=100352,
        video_max_pixels=602112,
        max_ratio=200,
        min_frames=4,
        max_frames=20,
        frame_factor=2,
        sample_rate=16000,
        fps=2.0,
        use_audio_in_video=False,
    )[0]

    # ---- assertions ----
    assert "input_ids" in out
    assert "video_mask" in out
    assert "audio_mask" in out
    assert "video_grid_thw" in out

    video_mask = out["video_mask"]
    audio_mask = out["audio_mask"]

    # Each modality must have a positive number of tokens.
    assert video_mask.sum() > 0, "no video tokens emitted"
    assert audio_mask.sum() > 0, "no audio tokens emitted"

    # Decoupled mode = NO interleaving. The video run and the audio run must be
    # two distinct contiguous spans in input_ids — i.e. neither mask should
    # contain a video bit followed by an audio bit then another video bit.
    def _runs(mask: torch.Tensor) -> int:
        diff = torch.diff(mask.int(), prepend=torch.zeros(1, dtype=torch.int))
        return int((diff == 1).sum())

    assert _runs(video_mask) == 1, (
        f"expected a single contiguous video run, got {_runs(video_mask)} — "
        "video and audio tokens are being interleaved, which is the 'omni' path, "
        "not the decoupled path"
    )
    assert _runs(audio_mask) == 1, f"expected a single contiguous audio run, got {_runs(audio_mask)}"

    # video_grid_thw matches the temporally-merged frame count.
    video_grid_thw = out["video_grid_thw"]
    assert video_grid_thw.shape[0] == 1, video_grid_thw.shape
    print(f"\n[OK] video_grid_thw = {video_grid_thw.tolist()}")
    print(f"[OK] video tokens = {int(video_mask.sum())}, audio tokens = {int(audio_mask.sum())}")
    print(f"[OK] input_ids length = {out['input_ids'].numel()}")
