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

"""Smoke test: Qwen3-Omni offline-A/V data path (paired frames + audio).

Verifies:
  1. The `qwen_omni_offline_av` preprocessor splits inline `<image>` /
     `<video>` / `<audio>` markers into separate conversation items.
  2. `_dict_to_video_audio` on a ``{"frames": [...], "audio": <wav_bytes>}``
     dict surfaces a 4-D video tensor *and* a non-empty mono audio array, so
     the downstream processor sees the result as an audio-enabled video.
  3. (Optional, gated on QWEN3_OMNI_MODEL_PATH) `process_sample_qwen_omni`
     produces an `input_ids` where `<|video_pad|>` and `<|audio_pad|>` form
     **interleaved** runs (the omni layout — not contiguous spans), and
     `video_grid_thw` matches the sampled frame count.

Run::

    pytest tests/data/multimodal/test_qwen3_omni_offline_av.py -v -s
    # Or, to also exercise the full processor path:
    QWEN3_OMNI_MODEL_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct pytest ... -v -s
"""

import os
import wave
from io import BytesIO

import numpy as np
import PIL.Image
import pytest
import torch

from veomni.data.multimodal.preprocess import conv_preprocess
from veomni.utils.constants import IGNORE_INDEX  # noqa: F401  (sanity import)


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
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(samples.tobytes())
    return buf.getvalue()


def _av_dict(num_frames: int = 4, duration_s: float = 1.0, sr: int = 16000) -> dict:
    """Build a synthetic paired-A/V dict in the offline-A/V sample shape."""
    return {
        "frames": [_make_png_frame(color=int(255 * i / max(1, num_frames - 1))) for i in range(num_frames)],
        "audio": _make_silent_wav(duration_s=duration_s, sr=sr),
        "video_fps": 2.0,
        "audio_fps": sr,
    }


# ----------------------------------------------------------------------------
# 1. Preprocessor split markers in declaration order
# ----------------------------------------------------------------------------


def test_preprocessor_splits_video_marker():
    out = conv_preprocess(
        "qwen_omni_offline_av",
        [
            {"from": "human", "value": "<video>\nWhat is happening?"},
            {"from": "gpt", "value": "Someone is speaking near a car."},
        ],
    )
    assert out[0][0] == "user"
    types = [item[0] for item in out[0][1:]]
    assert types[0] == "video", f"first item should be video, got {types}"
    assert "audio" not in types, "no <audio> marker expected for the paired-AV turn"
    assert types[-1] == "text"
    assert "What is happening?" in out[0][-1][1]

    assert out[1] == ["assistant", ("text", "Someone is speaking near a car.")]


def test_preprocessor_video_plus_standalone_audio_and_image():
    out = conv_preprocess(
        "qwen_omni_offline_av",
        [
            {"from": "human", "value": "<video>\nDescribe this. Look at <image> and listen to <audio>."},
            {"from": "gpt", "value": "ok"},
        ],
    )
    types = [item[0] for item in out[0][1:]]
    assert types.count("video") == 1
    assert types.count("audio") == 1
    assert types.count("image") == 1


def test_preprocessor_multiple_videos_per_turn():
    out = conv_preprocess(
        "qwen_omni_offline_av",
        [
            {"from": "human", "value": "Compare <video> with <video>."},
            {"from": "gpt", "value": "They differ in color."},
        ],
    )
    types = [item[0] for item in out[0][1:]]
    assert types.count("video") == 2


# ----------------------------------------------------------------------------
# 2. _dict_to_video_audio surfaces both frames and audio
# ----------------------------------------------------------------------------


def test_dict_to_video_audio_paired_av_returns_non_none_audio():
    # No ffmpeg / torchcodec needed — this code path goes through PIL + soundfile only.
    from veomni.data.multimodal.video_utils import _dict_to_video_audio

    av = _av_dict(num_frames=4, duration_s=1.0, sr=16000)
    video, video_fps, audio, audio_fps = _dict_to_video_audio(av)

    assert isinstance(video, torch.Tensor)
    assert video.ndim == 4  # (T, C, H, W)
    assert video.shape[0] == 4
    assert video.shape[1] == 3

    # The point of the paired-A/V shape: audio is *not* None, so the downstream
    # processor will interleave video/audio tokens via the omni path.
    assert audio is not None, "paired AV must surface a non-None audio array"
    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert audio.shape[0] > 0

    assert video_fps == 2.0
    assert audio_fps == 16000


def test_dict_to_video_audio_missing_frames_and_video_raises():
    from veomni.data.multimodal.video_utils import _dict_to_video_audio

    with pytest.raises(ValueError, match="must contain either 'video'"):
        _dict_to_video_audio({"audio": _make_silent_wav()})


def test_dict_to_video_audio_ndarray_audio_without_fps_raises():
    """ndarray audio without explicit audio_fps would otherwise be silently
    dropped by fetch_videos (audio is not None and audio_fps is not None gate),
    which would inversely flip the recipe back to the non-interleaved path.
    """
    from veomni.data.multimodal.video_utils import _dict_to_video_audio

    frames = [_make_png_frame(color=c) for c in (10, 80, 150, 220)]
    audio_array = np.zeros(16000, dtype=np.float32)  # ndarray, no audio_fps

    with pytest.raises(ValueError, match="requires an explicit `audio_fps`"):
        _dict_to_video_audio({"frames": frames, "audio": audio_array})


def test_dict_to_video_audio_ndarray_audio_with_fps_ok():
    from veomni.data.multimodal.video_utils import _dict_to_video_audio

    frames = [_make_png_frame(color=c) for c in (10, 80, 150, 220)]
    audio_array = np.zeros(16000, dtype=np.float32)

    video, video_fps, audio, audio_fps = _dict_to_video_audio(
        {"frames": frames, "audio": audio_array, "audio_fps": 16000}
    )
    assert audio is audio_array  # passed through unchanged
    assert audio_fps == 16000


def test_dict_to_video_audio_frames_layout_honors_default_video_fps():
    """Frame dicts without `video_fps` must fall back to the caller's configured
    fps (plumbed from `mm_configs.fps`), not a hard-coded 2.0 — otherwise an
    `mm_configs.fps: 1.0` recipe would re-sample the offline frames as if they
    were captured at 2 fps and silently halve the frame count downstream.
    """
    from veomni.data.multimodal.video_utils import _dict_to_video_audio

    frames = [_make_png_frame(color=c) for c in (10, 80, 150, 220)]

    _, video_fps, _, _ = _dict_to_video_audio({"frames": frames, "audio": _make_silent_wav()}, default_video_fps=1.0)
    assert video_fps == 1.0


def test_dict_to_video_audio_video_layout_ndarray_audio_no_fps_silent_drop():
    """The decoded-video (`{"video": ndarray}`) layout predates the offline-A/V
    recipe and historically dropped ndarray audio without `audio_fps` silently
    (via the downstream `audio_fps is not None` gate). The stricter guard added
    for the new `frames` layout must not regress that path.
    """
    from veomni.data.multimodal.video_utils import _dict_to_video_audio

    video_np = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    audio_array = np.zeros(16000, dtype=np.float32)  # ndarray, no audio_fps

    _, _, audio, audio_fps = _dict_to_video_audio({"video": video_np, "audio": audio_array})
    assert audio is audio_array
    assert audio_fps is None


# ----------------------------------------------------------------------------
# 3. Full processor path — gated on QWEN3_OMNI_MODEL_PATH
# ----------------------------------------------------------------------------


QWEN3_OMNI_MODEL_PATH = os.environ.get("QWEN3_OMNI_MODEL_PATH", "")


@pytest.mark.skipif(
    not QWEN3_OMNI_MODEL_PATH,
    reason="Set QWEN3_OMNI_MODEL_PATH to a local Qwen3-Omni-30B-A3B-Instruct processor dir to enable.",
)
def test_qwen3_omni_offline_av_end_to_end():
    """Run the actual Qwen3-Omni transform on a synthetic paired-A/V sample.

    Uses a stub `position_id_func` because the assertions check the
    video/audio token interleaving in the processor output, which is produced
    *before* position_id_func runs.
    """
    from veomni.data.data_transform import process_sample_qwen_omni
    from veomni.models import build_processor

    processor = build_processor(QWEN3_OMNI_MODEL_PATH)

    def stub_position_id_func(input_ids, attention_mask, **_kwargs):
        L = input_ids.shape[-1]
        return {"position_ids": torch.zeros(3, 1, L, dtype=torch.long)}

    sample = {
        # 3 s of audio + 4 sampled frames → enough audio tokens to robustly
        # exceed one temporal video chunk, so the interleaved pattern shows up
        # as at least one audio chunk *between* two video chunks (i.e.
        # video_runs > 1).
        "videos": [_av_dict(num_frames=4, duration_s=3.0, sr=16000)],
        "conversations": [
            {"from": "human", "value": "<video>\nWhat is happening?"},
            {"from": "gpt", "value": "Someone is speaking."},
        ],
    }

    out = process_sample_qwen_omni(
        sample,
        processor=processor,
        position_id_func=stub_position_id_func,
        source_name="qwen_omni_offline_av",
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
        use_audio_in_video=True,
    )[0]

    # ---- assertions ----
    assert "input_ids" in out
    assert "video_mask" in out
    assert "audio_mask" in out
    assert "video_grid_thw" in out

    video_mask = out["video_mask"]
    audio_mask = out["audio_mask"]

    assert video_mask.sum() > 0, "no video tokens emitted"
    assert audio_mask.sum() > 0, "no audio tokens emitted"

    # Paired-AV (omni) mode: video and audio tokens interleave inside one
    # <vision_bos> ... <vision_eos> block. Specifically, the video mask is
    # *not* a single contiguous span — it gets broken by audio chunks.
    def _runs(mask: torch.Tensor) -> int:
        diff = torch.diff(mask.int(), prepend=torch.zeros(1, dtype=torch.int))
        return int((diff == 1).sum())

    video_runs = _runs(video_mask)
    audio_runs = _runs(audio_mask)
    # Omni-path proof: the video span is broken into multiple runs by interleaved
    # audio chunks. (audio_runs can validly be 1 — when the audio tail is shorter
    # than one temporal video chunk it forms a single trailing run; what's
    # diagnostic is that audio appears *between* video runs at all.)
    assert video_runs > 1, (
        f"expected the video mask to be broken into multiple runs by interleaved audio chunks, "
        f"got {video_runs} run — looks like the per-position audio is empty and the processor "
        "fell through to the non-interleaved path"
    )

    video_grid_thw = out["video_grid_thw"]
    assert video_grid_thw.shape[0] == 1, video_grid_thw.shape
    print(f"\n[OK] video_grid_thw = {video_grid_thw.tolist()}")
    print(
        f"[OK] video tokens = {int(video_mask.sum())} ({video_runs} runs); "
        f"audio tokens = {int(audio_mask.sum())} ({audio_runs} runs)"
    )
    print(f"[OK] input_ids length = {out['input_ids'].numel()}")
