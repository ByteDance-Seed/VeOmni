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


@pytest.fixture(params=["qwen2_5_omni", "qwen3_omni_moe"])
def omni_processor(request):
    """Real processors with a local tokenizer; no checkpoint download required."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2VLImageProcessor,
        Qwen2VLVideoProcessor,
        WhisperFeatureExtractor,
    )

    from veomni.models.transformers.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor
    from veomni.models.transformers.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor

    tokens = {
        "image_token": "<|image_pad|>",
        "video_token": "<|video_pad|>",
        "audio_token": "<|audio_pad|>",
        "vision_bos_token": "<|vision_start|>",
        "vision_eos_token": "<|vision_end|>",
        "audio_bos_token": "<|audio_start|>",
        "audio_eos_token": "<|audio_end|>",
    }
    special = [*tokens.values(), "<|im_start|>", "<|im_end|>", "user", "assistant", "system", "separator"]
    vocab = {token: i for i, token in enumerate(["<unk>", "<pad>", *special])}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=special,
    )
    for key, value in tokens.items():
        setattr(tokenizer, key, value)
    processor_cls = Qwen2_5OmniProcessor if request.param == "qwen2_5_omni" else Qwen3OmniMoeProcessor
    return processor_cls(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(size={"shortest_edge": 56**2, "longest_edge": 56**2}),
        video_processor=Qwen2VLVideoProcessor(size={"shortest_edge": 56**2, "longest_edge": 56**2}),
        feature_extractor=WhisperFeatureExtractor(feature_size=128),
        chat_template=(
            "{% for m in messages %}{{ '<|im_start|>' + m['role'] }}"
            "{% for c in m['content'] %}"
            "{% if c['type'] == 'video' %}<|vision_start|><|video_pad|><|vision_end|>"
            "{% else %}{{ c['text'] }}{% endif %}"
            "{% endfor %}<|im_end|>{% endfor %}"
        ),
    )


@pytest.mark.parametrize("duration", [12.0, 15.0, 30.0])
def test_omni_transform_uses_effective_fps(omni_processor, monkeypatch, duration):
    from functools import partial
    from types import SimpleNamespace

    from veomni.data.data_transform import process_sample_qwen_omni
    from veomni.models.loader import MODELING_REGISTRY
    from veomni.utils.constants import AUDIO_INPUT_INDEX, IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX

    is_qwen2_5 = type(omni_processor).__name__ == "Qwen2_5OmniProcessor"
    family = "qwen2_5_omni" if is_qwen2_5 else "qwen3_omni_moe"
    thinker_cls = MODELING_REGISTRY[family]("ThinkerForConditionalGeneration")
    model = SimpleNamespace(
        config=SimpleNamespace(
            image_token_id=IMAGE_INPUT_INDEX,
            video_token_id=VIDEO_INPUT_INDEX,
            audio_token_id=AUDIO_INPUT_INDEX,
            vision_start_token_id=omni_processor.tokenizer.convert_tokens_to_ids(omni_processor.vision_bos_token),
            audio_start_token_id=omni_processor.tokenizer.convert_tokens_to_ids(omni_processor.audio_bos_token),
            position_id_per_seconds=25 if is_qwen2_5 else 13,
            seconds_per_chunk=2.0,
        ),
        spatial_merge_size=omni_processor.video_processor.merge_size,
        get_llm_pos_ids_for_vision=partial(thinker_cls.get_llm_pos_ids_for_vision, None),
        get_chunked_index=partial(thinker_cls.get_chunked_index, None),
    )

    def no_resampling(*args, **kwargs):
        pytest.fail("Already sampled video must not be sampled again")

    monkeypatch.setattr(omni_processor.video_processor, "do_sample_frames", True)
    monkeypatch.setattr(omni_processor.video_processor, "sample_frames", no_resampling)

    def position_ids(input_ids, second_per_grids, **kwargs):
        assert second_per_grids.tolist() == pytest.approx([duration / 10])
        positions, _ = thinker_cls.get_rope_index(
            model, input_ids=input_ids, second_per_grids=second_per_grids, **kwargs
        )
        return {"position_ids": positions.squeeze(1)}

    sample = {
        "videos": [
            {
                "video": np.zeros((int(duration * 30), 8, 8, 3), dtype=np.uint8),
                "video_fps": 30.0,
                "audio": np.zeros(int(duration * 16000), dtype=np.float32),
                "audio_fps": 16000,
            }
        ],
        "conversations": [{"from": "human", "value": "<video>question"}, {"from": "gpt", "value": "answer"}],
    }
    result = process_sample_qwen_omni(
        sample,
        processor=omni_processor,
        position_id_func=position_ids,
        source_name="qwen_omni_offline_av",
        fps=2.0,
        min_frames=4,
        max_frames=20,
        frame_factor=2,
        scale_factor=28,
        video_min_pixels=56**2,
        video_max_pixels=56**2,
    )[0]
    assert result["video_mask"].sum() == result["video_grid_thw"].prod() // 4
    assert result["audio_mask"].sum() > 0
    assert "video_second_per_grid" not in result
    assert "video_metadata" not in result

    # Fractional video timing must give each modality its own RoPE positions.
    positions = result["position_ids"]
    video_positions = positions[:, result["video_mask"]]
    audio_positions = positions[:, result["audio_mask"]]
    start = video_positions[0, 0]
    t, h, w = result["video_grid_thw"][0].tolist()
    h //= model.spatial_merge_size
    w //= model.spatial_merge_size
    expected_h = torch.arange(h, dtype=positions.dtype).repeat_interleave(w).repeat(t) + start
    expected_w = torch.arange(w, dtype=positions.dtype).repeat(h * t) + start
    torch.testing.assert_close(video_positions[1], expected_h)
    torch.testing.assert_close(video_positions[2], expected_w)
    expected_audio = torch.arange(audio_positions.shape[1], dtype=positions.dtype) + start
    torch.testing.assert_close(audio_positions, expected_audio.unsqueeze(0).expand(3, -1))


def test_omni_presampled_video_without_metadata_uses_requested_fps(omni_processor):
    result = omni_processor(
        text="<|vision_start|><|video_pad|><|vision_end|>",
        videos=[np.zeros((4, 3, 56, 56), dtype=np.uint8)],
        audios=[None],
        fps=2.0,
        do_sample_frames=False,
        return_tensors="pt",
    )
    assert result["video_second_per_grid"].tolist() == [1.0]


def test_omni_per_video_fps_preserves_interleaving(omni_processor):
    video_text = "<|vision_start|><|video_pad|><|vision_end|>"
    video = np.zeros((4, 3, 56, 56), dtype=np.uint8)
    audio = np.zeros(4 * 16000, dtype=np.float32)
    video_metadata = [
        {
            "fps": 30.0,
            "total_num_frames": 60,
            "frames_indices": [0, 20, 39, 59],
        },
        {
            "fps": 30.0,
            "total_num_frames": 240,
            "frames_indices": [0, 80, 159, 239],
        },
    ]
    result = omni_processor(
        text=video_text + "separator" + video_text,
        videos=[video, video],
        audios=[None, audio],
        video_metadata=video_metadata,
        fps=2.0,
        do_sample_frames=False,
        return_tensors="pt",
    )
    reference = omni_processor(
        text=video_text,
        videos=[video],
        audios=[audio],
        video_metadata=[video_metadata[1]],
        do_sample_frames=False,
        return_tensors="pt",
    )
    assert result["video_second_per_grid"].tolist() == [1.0, 4.0]
    ids = result["input_ids"][0]
    separator = (ids == omni_processor.tokenizer.convert_tokens_to_ids("separator")).nonzero().item()
    # A preceding video without audio must not shift the second video's timing.
    assert torch.equal(ids[separator + 1 :], reference["input_ids"][0])
    assert "video_metadata" not in result
