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

"""One decode path for audio, video and video-with-sound, on both sides.

These pin the contract that ``veomni/data/seed_omni/utils/media.py`` exists to
hold: whatever a modality's refs are, ``fetch_media`` returns
``[(payload, meta), ...]`` and ``build_conversation`` carries the meta onto the
item — so a request built for inference states exactly what a training sample
states about the same clip.

The regression behind them: request building used to have its own image-only
loader that attached no metadata, so an audio item built for inference carried
no sampling rate, and ``videos`` raised ``NotImplementedError`` while the data
layer had been decoding clips for training all along.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import soundfile as sf
import torch
from PIL import Image

from veomni.data.seed_omni.utils.media import fetch_media
from veomni.data.seed_omni.utils.media_metadata import (
    AUDIO_METADATA_KEY,
    VIDEO_METADATA_KEY,
    AudioMetadata,
    VideoMetadata,
)
from veomni.data.seed_omni.utils.video import VideoInputs, save_video
from veomni.models.seed_omni.processing_omni import OmniProcessor, _as_list
from veomni.models.seed_omni.utils.conversation import build_conversation


_RATE = 24_000
_FPS = 24.0
_FRAMES = 12


def _tone(seconds: float, rate: int = _RATE) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(seconds * rate), endpoint=False, dtype=np.float32)
    return 0.2 * np.sin(2 * np.pi * 440.0 * t)


def _frames(num: int = _FRAMES) -> torch.Tensor:
    """``(T, C, H, W)`` uint8 with a per-frame ramp, so a decode that drops or
    reorders frames is visible rather than passing on shape alone."""
    clip = torch.zeros(num, 3, 32, 32, dtype=torch.uint8)
    for idx in range(num):
        clip[idx] = idx * 8
    return clip


def _clip_meta(num: int = _FRAMES) -> dict:
    """The meta a generated clip carries: its own source, every frame kept."""
    return {
        VIDEO_METADATA_KEY: VideoMetadata(
            fps=_FPS, total_num_frames=num, frames_indices=list(range(num)), requested_fps=_FPS
        )
    }


@pytest.fixture
def wav_path(tmp_path):
    path = tmp_path / "speech.wav"
    sf.write(str(path), _tone(0.5), _RATE)
    return str(path)


@pytest.fixture
def silent_clip_path(tmp_path):
    path = tmp_path / "silent.mp4"
    save_video(str(path), _frames(), _clip_meta())
    return str(path)


@pytest.fixture
def sounding_clip_path(tmp_path):
    path = tmp_path / "sounding.mp4"
    seconds = _FRAMES / _FPS
    save_video(
        str(path),
        VideoInputs(video=_frames(), audio=_tone(seconds)),
        {**_clip_meta(), AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=_RATE)},
    )
    return str(path)


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (48, 32), color=(10, 20, 30)).save(str(path))
    return str(path)


def test_an_audio_request_states_the_rate_it_decoded(wav_path):
    """The rate is the one fact a waveform cannot state about itself.

    Its consumers (a 16 kHz tower, a 24 kHz codec) each resample for themselves
    and need to know what they were handed; there is no default that is safe to
    assume, because a wrong rate is indistinguishable from a bad clip.
    """
    ((samples, meta),) = fetch_media({"audio": [wav_path]}, "t")["audio"]

    assert samples.ndim == 1 and samples.size > 0
    assert meta[AUDIO_METADATA_KEY].sampling_rate == _RATE
    assert meta[AUDIO_METADATA_KEY].num_samples == samples.size


def test_a_silent_video_request_states_its_frame_timeline(silent_clip_path):
    ((payload, meta),) = fetch_media({"video": [silent_clip_path]}, "t", {"fps": _FPS})["video"]

    assert isinstance(payload, VideoInputs)
    assert not payload.has_audio
    assert payload.video.ndim == 4
    assert meta[VIDEO_METADATA_KEY].fps == pytest.approx(_FPS, rel=0.05)
    assert AUDIO_METADATA_KEY not in meta


def test_a_clip_with_sound_stays_one_item_carrying_both_streams(sounding_clip_path):
    """One clip is one item, sound included — not a video item plus an audio item.

    The two streams share a timeline the backbone has to interleave; splitting
    them across two items throws away the alignment that made them one clip.
    """
    fetched = fetch_media({"video": [sounding_clip_path]}, "t", {"fps": _FPS, "use_audio_in_video": True})

    assert list(fetched) == ["video"]
    ((payload, meta),) = fetched["video"]
    assert payload.has_audio
    assert meta[VIDEO_METADATA_KEY].fps == pytest.approx(_FPS, rel=0.05)
    assert meta[AUDIO_METADATA_KEY].sampling_rate > 0


def test_the_same_clip_carries_no_sound_unless_it_was_asked_for(sounding_clip_path):
    """Sound is opt-in: a vision-only model must not pay to decode a track it
    cannot read, so the flag — not the file — decides."""
    ((payload, meta),) = fetch_media({"video": [sounding_clip_path]}, "t", {"fps": _FPS})["video"]

    assert not payload.has_audio
    assert AUDIO_METADATA_KEY not in meta


def test_an_image_request_decodes_to_the_tensor_the_training_path_uses(image_path):
    """Pinned because the two sides used to disagree: request building produced
    PIL images while the training transform produced uint8 pixel tensors."""
    ((payload, meta),) = fetch_media({"image": [image_path]}, "t")["image"]

    assert isinstance(payload, torch.Tensor)
    assert payload.dtype == torch.uint8
    assert payload.shape == (3, 32, 48)
    assert meta == {}


def test_a_request_naming_all_three_formats_gets_all_three(wav_path, sounding_clip_path, image_path):
    media = fetch_media(
        {"image": [image_path], "audio": [wav_path], "video": [sounding_clip_path]},
        "t",
        {"fps": _FPS, "use_audio_in_video": True},
    )
    parts = build_conversation(prompt="describe", media=media)

    assert [p.type for p in parts] == ["image", "audio", "video", "text"]
    assert parts[1].meta[AUDIO_METADATA_KEY].sampling_rate == _RATE
    assert parts[2].meta[VIDEO_METADATA_KEY].total_num_frames > 0
    assert parts[2].value.has_audio


def test_a_processor_request_carries_a_sounding_clip_end_to_end(sounding_clip_path):
    """The headline capability, through the public entry point.

    ``videos`` used to raise ``NotImplementedError`` here, so this is the path a
    caller actually takes rather than the fetchers underneath it.
    """

    class _NoopPreprocessor:
        def __call__(self, batch, inference=False, **kwargs) -> None:
            del batch, inference, kwargs

    processor = OmniProcessor({"a": _NoopPreprocessor()})

    conversation = processor(
        text="what is said in this clip",
        videos=sounding_clip_path,
        mm_configs={"fps": _FPS, "use_audio_in_video": True},
    )["conversation_list"]

    assert [item.type for item in conversation] == ["video", "text"]
    assert conversation[0].value.has_audio
    assert conversation[0].meta[VIDEO_METADATA_KEY].fps == pytest.approx(_FPS, rel=0.05)
    assert conversation[0].meta[AUDIO_METADATA_KEY].sampling_rate > 0


def test_an_unused_modality_costs_no_decode(wav_path):
    """The empty lists a request builder always passes must not reach a fetcher —
    otherwise an image-only deployment pays for the video stack to answer a
    question with no video in it."""
    media = fetch_media({"image": [], "audio": [wav_path], "video": []}, "t")

    assert list(media) == ["audio"]


def test_a_knob_a_modality_ignores_is_ignored_not_rejected(wav_path):
    """One flat bag of knobs serves every modality, which only works if each
    fetcher drops what it does not read."""
    media = fetch_media(
        {"audio": [wav_path]}, "t", {"fps": _FPS, "video_max_pixels": 1024, "use_audio_in_video": True}
    )

    assert media["audio"][0][1][AUDIO_METADATA_KEY].sampling_rate == _RATE


def test_a_modality_nothing_can_decode_is_named_not_dropped():
    """Dropping it would leave the media turns unpaired and surface much later
    as a count mismatch naming the wrong cause."""
    with pytest.raises(ValueError, match=r"my caller: refs declared for \['point_cloud'\]"):
        fetch_media({"point_cloud": ["a.ply"]}, "my caller")


def test_a_missing_decode_stack_is_named_rather_than_dropping_the_clip():
    """Refuse the request instead of returning ``[]``.

    The empty list this used to return turns a missing system dependency into a
    text-only answer to a question about a video (inference) or a pairing count
    mismatch reported against the preprocessor (training).
    """
    import veomni.utils.import_utils as import_utils

    module = importlib.import_module("veomni.data.seed_omni.utils.media")
    original = import_utils.is_video_audio_available
    import_utils.is_video_audio_available = lambda: False
    try:
        importlib.reload(module)
        with pytest.raises(RuntimeError, match="torchcodec"):
            module.fetch_media({"video": ["clip.mp4"]}, "t")
    finally:
        import_utils.is_video_audio_available = original
        importlib.reload(module)


@pytest.mark.parametrize(
    "given, expected_len",
    [
        ("one.png", 1),
        (b"raw-bytes", 1),
        (bytearray(b"raw-bytes"), 1),
        (memoryview(b"raw-bytes"), 1),
        (["a.png", "b.png"], 2),
        (None, 0),
        ([], 0),
    ],
    ids=["str", "bytes", "bytearray", "memoryview", "list", "none", "empty"],
)
def test_a_single_ref_needs_no_list(given, expected_len):
    """A bytes-backed ref is a sequence and would otherwise iterate into its
    elements — one inline clip becoming a few thousand integer "refs".

    ``bytearray`` and ``memoryview`` are the shapes a parquet corpus storing
    media inline actually produces, and the data layer accepts both
    (``load_image`` takes ``bytearray``; ``AudioInput`` is a ``ByteString``).
    """
    assert len(_as_list(given)) == expected_len


@pytest.mark.parametrize(
    "payload",
    [np.zeros(16, dtype=np.float32), torch.zeros(3, 4, 4), Image.new("RGB", (4, 4))],
    ids=["waveform", "pixels", "pil"],
)
def test_already_decoded_media_is_one_ref_not_a_list_of_rows(payload):
    """``_as_list`` only decides arity — whether a fetcher then accepts the
    payload is its own business (see the round trip below)."""
    listed = _as_list(payload)

    assert len(listed) == 1
    assert listed[0] is payload


def test_a_waveform_array_is_accepted_only_with_the_rate_it_is_in():
    """The documented boundary for already-decoded audio.

    An array states no rate, and there is no safe default to assume, so the
    caller has to say — the same refusal ``save_audio`` makes on the way out.
    """
    samples = np.zeros(1600, dtype=np.float32)

    with pytest.raises(ValueError, match="declares no rate"):
        fetch_media({"audio": [samples]}, "t")

    ((decoded, meta),) = fetch_media({"audio": [samples]}, "t", {"audio_sampling_rate": 16_000})["audio"]
    assert decoded.size == samples.size
    assert meta[AUDIO_METADATA_KEY].sampling_rate == 16_000


def test_a_pixel_tensor_is_refused_rather_than_silently_reinterpreted():
    """Pixel tensors are not a ref type: the image fetcher decodes refs and PIL
    images only. Pinned so the docstring's claim stays true."""
    with pytest.raises(NotImplementedError, match="Unsupported image input type"):
        fetch_media({"image": [torch.zeros(3, 4, 4)]}, "t")


# --- The write side: one saver per fetcher, each handed the item's whole meta ---


def test_every_modality_that_can_be_read_can_be_written():
    """``MEDIA_SAVERS`` is keyed like ``MEDIA_FETCHERS``.

    A modality added to one table and not the other either decodes into items
    ``finalize`` silently skips, or writes items nothing could have read back.
    """
    from veomni.data.seed_omni.utils.media import MEDIA_FETCHERS, MEDIA_SAVERS

    assert set(MEDIA_SAVERS) == set(MEDIA_FETCHERS)


def test_a_generated_image_round_trips_as_pil_or_as_the_loaders_tensor(tmp_path):
    """The vision decoder emits PIL; the load side produces ``(C, H, W) uint8``.

    ``save_image`` is the inverse of the latter, so writing what was loaded and
    reading it back has to give the same pixels.
    """
    from veomni.data.seed_omni.utils.image import fetch_images, save_image

    source = Image.new("RGB", (24, 16), color=(10, 20, 30))
    pil_path, tensor_path = tmp_path / "pil.png", tmp_path / "tensor.png"

    save_image(str(pil_path), source, {})
    (loaded,) = fetch_images([str(pil_path)])
    save_image(str(tensor_path), loaded, {})
    (reloaded,) = fetch_images([str(tensor_path)])

    assert torch.equal(loaded, reloaded)


def test_a_float_image_is_refused_rather_than_truncated_to_black(tmp_path):
    from veomni.data.seed_omni.utils.image import save_image

    path = tmp_path / "generated_image_0.png"
    with pytest.raises(ValueError, match="expected uint8"):
        save_image(str(path), torch.rand(3, 8, 8), {})
    assert not path.exists()


def test_the_inferencer_hands_each_saver_the_items_whole_meta(tmp_path):
    """The loop picks nothing out of meta; the rate reaches the file only
    because ``save_audio`` reads it there itself."""
    from veomni.trainer.omni.omni_inferencer import _save_generated_media

    rate = 16_000
    items = [
        {"type": "text", "value": "hello"},
        {
            "type": "audio",
            "value": torch.from_numpy(_tone(0.25, rate)),
            "meta": {AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=rate)},
        },
        {"type": "image", "value": Image.new("RGB", (8, 8))},
        {"type": "audio", "value": None},
    ]

    written = _save_generated_media(items, str(tmp_path))

    assert written == {"image": 1, "audio": 1}
    assert sf.info(str(tmp_path / "generated_audio_0.wav")).samplerate == rate
    assert (tmp_path / "generated_image_0.png").exists()
