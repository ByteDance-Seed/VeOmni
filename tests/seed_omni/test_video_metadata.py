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

"""The timeline SeedOmni's media items hand downstream.

``VideoInputs`` used to carry a single ``video_fps`` scalar set to the rate
:func:`load_video` was *asked* for, which is both the wrong number (stride
rounding and ``max_frames`` make the kept frames sparser than asked for) and
the wrong axis. Dividing source-frame indices by a target rate is the bug
`PR #1165 <https://github.com/ByteDance-Seed/VeOmni/pull/1165>`_ fixed on the
BaseTrainer path: frame 300 of a 30 fps clip is 10 s, but 300/2 reads as 150 s.

So these tests pin two things, which together are what makes that bug
unwritable here:

* the source axis is what survives (``fps`` / ``total_num_frames`` /
  ``frames_indices``), and the metadata does the division itself, so no call
  site ever pairs indices with a rate;
* it lives on the item's ``meta``, not inside the payload, because an encoder
  overwrites the payload while the backbone reads the timeline after that.

The audio half is here too: sound lifted out of a clip has to land on the same
wall clock as its frames, and the two spans have to stay comparable.
"""

import contextlib

import numpy as np
import pytest
import torch
from PIL import Image

from veomni.data.seed_omni.utils import video as _video_module
from veomni.data.seed_omni.utils.audio import fetch_audios
from veomni.data.seed_omni.utils.video import (
    AUDIO_METADATA_KEY,
    VIDEO_METADATA_KEY,
    AudioMetadata,
    VideoInputs,
    VideoMetadata,
    _sample_frame_indices,
    load_video,
)
from veomni.models.seed_omni.utils.conversation import ConversationItem


def _metadata_for(total_frames: int, source_fps: float, requested_fps: float, max_frames: int | None = None):
    """Build the metadata exactly as ``load_video``'s decoder branch does.

    Goes through the real sampler so the test covers the interaction between
    stride rounding, ``max_frames`` thinning and the derived rate — rather than
    asserting against hand-written indices that could drift from the sampler.
    """
    indices = _sample_frame_indices(total_frames, source_fps, requested_fps, max_frames)
    return VideoMetadata(
        fps=source_fps,
        total_num_frames=total_frames,
        frames_indices=indices,
        requested_fps=requested_fps,
    )


@contextlib.contextmanager
def _stub_decoder(average_fps: float | None, num_frames: int, duration_seconds: float | None):
    """Run ``load_video``'s decoder branch without a real container.

    Yields the monkeypatch context so a test can stub the audio extractor too.
    """

    class _Decoder:
        def __init__(self, *args, **kwargs):
            self.metadata = type(
                "M",
                (),
                {"average_fps": average_fps, "num_frames": num_frames, "duration_seconds": duration_seconds},
            )()

        def get_frames_at(self, indices):
            return type("F", (), {"data": torch.zeros(len(indices), 3, 2, 2, dtype=torch.uint8)})()

    import torchcodec.decoders

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torchcodec.decoders, "VideoDecoder", _Decoder)
        patch.setattr(_video_module, "is_ffmpeg_available", lambda: True)
        yield patch


def _video_item(
    video_metadata: VideoMetadata,
    frames: torch.Tensor,
    waveform: np.ndarray | None = None,
    audio_metadata: AudioMetadata | None = None,
) -> ConversationItem:
    """A ``"video"`` item as the transform hands it to a vision preprocessor."""
    meta = {VIDEO_METADATA_KEY: video_metadata}
    if audio_metadata is not None:
        meta[AUDIO_METADATA_KEY] = audio_metadata
    return ConversationItem(type="video", role="user", value=VideoInputs(video=frames, audio=waveform), meta=meta)


def test_frame_times_come_from_the_source_rate_not_the_requested_one():
    """The #1165 regression, stated as the arithmetic it got wrong.

    A 30 fps clip trimmed to 2 fps keeps source frames 0, 15, 30, ... Those are
    0 s, 0.5 s, 1 s apart on the source clock. Divided by the requested 2 fps
    they would read as 0 s, 7.5 s, 15 s — the clip stretched 15x.
    """
    meta = _metadata_for(total_frames=61, source_fps=30.0, requested_fps=2.0)

    assert meta.frames_indices == list(range(0, 61, 15))
    assert meta.frame_timestamps()[:3] == [0.0, 0.5, 1.0]
    # The number that would have produced the wrong answer is still recorded,
    # just never the divisor.
    assert meta.requested_fps == 2.0
    assert [i / meta.requested_fps for i in meta.frames_indices[:3]] == [0.0, 7.5, 15.0]


def test_frame_times_stay_on_the_source_clock_when_the_stride_rounds():
    # 25 fps source, want 2 fps -> stride round(12.5) = 12, so the kept frames
    # sit 0.48 s apart, not 0.5 s. Reading the request back would claim 0.5 s
    # and drift by a frame every couple of seconds.
    meta = _metadata_for(total_frames=101, source_fps=25.0, requested_fps=2.0)

    assert meta.frame_timestamps()[:3] == [0.0, 12 / 25.0, 24 / 25.0]


def test_frame_times_survive_max_frames_thinning_the_clip():
    """``max_frames`` spaces frames unevenly, so there is no single rate to
    divide by — only the per-frame source indices give the right times."""
    # 21 candidate frames thinned to 4 — the picks land on source frames
    # 0, 105, 195, 300, which are *not* evenly spaced.
    meta = _metadata_for(total_frames=301, source_fps=30.0, requested_fps=2.0, max_frames=4)

    times = meta.frame_timestamps()
    assert times == [idx / 30.0 for idx in meta.frames_indices]
    # Still spans the whole 10 s clip, just with 4 frames instead of 21.
    assert times == [0.0, 3.5, 6.5, 10.0]
    gaps = {round(b - a, 6) for a, b in zip(times, times[1:])}
    assert len(gaps) > 1, "uneven spacing is the point: a single rate cannot express it"


@pytest.mark.parametrize("bad", [None, 0, 0.0, -24, float("nan"), float("inf"), "30", True])
def test_metadata_without_a_real_rate_cannot_be_built(bad):
    """``fps`` is the divisor behind every time this type hands out, and the HF
    base substitutes 24 for a missing one. Refusing at construction means no
    accessor ever sees a missing rate, and neither does an HF processor handed
    this metadata. Substituting ``requested_fps`` is the #1165 bug, so the
    message rules it out by name."""
    with pytest.raises(ValueError, match="not a frame rate.*requested_fps"):
        VideoMetadata(fps=bad, total_num_frames=4, frames_indices=[0, 1, 2, 3], requested_fps=2.0)


def test_metadata_is_the_hf_type_and_fills_what_the_hf_type_leaves_open():
    """Inherits ``transformers.video_utils.VideoMetadata``, so an HF video
    processor takes it as is. The fields HF leaves optional are derived from the
    rate rather than left for a consumer to default: the full frame range, and
    the clip's span."""
    from transformers.video_utils import VideoMetadata as HFVideoMetadata

    meta = VideoMetadata(fps=24, total_num_frames=48)

    assert isinstance(meta, HFVideoMetadata)
    assert meta.frames_indices == list(range(48))
    assert meta.duration == 2.0
    # HF's own accessors now agree with ours instead of falling back to 24.
    assert meta.sampled_fps == 24 and meta.timestamps == meta.frame_timestamps()


def test_frame_times_fold_into_patches_the_way_the_vision_tower_does():
    """The tower encodes ``temporal_patch_size`` frames per patch and pads the
    tail by repeating the last frame; each patch gets its mid-time. Same
    arithmetic as ``Qwen3VLChatTemplate._calculate_timestamps``."""
    meta = VideoMetadata(fps=30.0, total_num_frames=91, frames_indices=[0, 15, 30, 45, 60], requested_fps=2.0)

    # 5 frames, patches of 2 -> pad to 6 by repeating frame 60 (2.0 s).
    assert meta.frame_timestamps(temporal_patch_size=2) == [0.25, 1.25, 2.0]
    # Padding must not have leaked back into the caller's metadata.
    assert meta.frames_indices == [0, 15, 30, 45, 60]


def test_load_video_records_the_timeline_for_a_pre_decoded_frame_list():
    """A frame list at its declared rate is a source clip like any other: 20
    frames taken at 10 fps, trimmed to 2 fps, keep every fifth frame and time
    them on the 10 fps clock."""
    frames = [Image.new("RGB", (2, 2)) for _ in range(20)]

    payload, meta = load_video(frames, fps=2.0, frames_fps=10.0)

    video_metadata = meta[VIDEO_METADATA_KEY]
    assert video_metadata.fps == 10.0
    assert video_metadata.total_num_frames == 20
    assert video_metadata.frames_indices == [0, 5, 10, 15]
    assert video_metadata.frame_timestamps() == [0.0, 0.5, 1.0, 1.5]
    assert video_metadata.requested_fps == 2.0
    assert payload.video.shape[0] == 4
    # A silent clip states no audio metadata at all, rather than an empty one a
    # consumer would have to tell apart from a real 0 Hz track.
    assert payload.audio is None and AUDIO_METADATA_KEY not in meta


def test_the_timeline_outlives_the_payload_the_encoder_overwrites():
    """Why the metadata sits on ``meta``: the vision preprocessor replaces
    ``value`` with patches, and the backbone still needs the clock afterwards."""
    item = type(
        "Item",
        (),
        {
            "value": VideoInputs(video=torch.zeros(2, 3, 2, 2, dtype=torch.uint8)),
            "meta": {VIDEO_METADATA_KEY: VideoMetadata(fps=30.0, total_num_frames=31, frames_indices=[0, 15])},
        },
    )()

    item.value = torch.zeros(8, 16)  # what an encoder leaves behind

    assert item.meta[VIDEO_METADATA_KEY].frame_timestamps() == [0.0, 0.5]


def test_in_video_audio_states_its_rate_the_same_way_a_standalone_clip_does():
    """One spelling for one quantity: a reader after a sample rate does not have
    to know whether the sound arrived as its own item or inside a clip.

    Both sides go through the real loaders — comparing a hand-written
    ``AudioMetadata`` against ``fetch_audios`` would pass even if ``load_video``
    wrote a different key, or none.
    """
    waveform = np.ones((8000,), dtype=np.float32)
    with _stub_decoder(average_fps=30.0, num_frames=61, duration_seconds=2.0) as patch:
        patch.setattr(
            _video_module, "extract_audio_from_video", lambda video, max_duration_seconds=None: (waveform, 16000)
        )
        in_video, in_video_meta = load_video("clip.mp4", fps=2.0, use_audio_in_video=True)

    standalone_payload, standalone_meta = fetch_audios(
        [np.ones((8000,), dtype=np.float32)], audio_sampling_rate=16000
    )[0]

    assert in_video.has_audio
    assert in_video_meta[AUDIO_METADATA_KEY] == standalone_meta[AUDIO_METADATA_KEY]
    # Length rides along because ``value`` becomes mel features at encode time
    # and the sample count is no longer recoverable from it.
    assert standalone_meta[AUDIO_METADATA_KEY].num_samples == standalone_payload.shape[0]
    assert standalone_meta[AUDIO_METADATA_KEY].duration_seconds == 0.5


def test_the_two_streams_of_one_clip_state_spans_that_can_be_compared():
    """Frames and sound are both measured from the container's zero, so the
    clip's two halves are comparable — which is how a caller notices a sound
    track that stops early (ordinary) or one our OOM cap cut short (our bug).
    """
    # 30 fps, 301 frames = 10 s of video; 16 kHz x 160000 samples = 10 s of sound.
    video_metadata = _metadata_for(total_frames=301, source_fps=30.0, requested_fps=2.0)
    item = _video_item(
        video_metadata,
        torch.zeros(len(video_metadata.frames_indices), 3, 2, 2, dtype=torch.uint8),
        waveform=np.zeros((160000,), dtype=np.float32),
        audio_metadata=AudioMetadata(sampling_rate=16000, num_samples=160000),
    )

    video_span = item.meta[VIDEO_METADATA_KEY].duration
    audio_span = item.meta[AUDIO_METADATA_KEY].duration_seconds
    assert video_span == pytest.approx(10.0, abs=1 / 30.0)
    assert audio_span == 10.0
    # The frames sampled out of it never claim to sit past the clip's end.
    assert item.meta[VIDEO_METADATA_KEY].frame_timestamps()[-1] <= video_span


def test_a_clip_whose_container_hides_its_duration_still_caps_audio_by_the_video():
    """The OOM cap used to fall back to a flat 60 s, which silently shortened
    the sound of any longer clip while keeping every video frame. The video's
    own span (frames / rate) is the right bound and is known even here."""
    captured = {}

    def fake_extract(video, max_duration_seconds=None):
        captured["cap"] = max_duration_seconds
        return np.zeros((16000,), dtype=np.float32), 16000

    # 30 fps x 3600 frames = 120 s, container states no duration.
    with _stub_decoder(average_fps=30.0, num_frames=3600, duration_seconds=None) as patch:
        patch.setattr(_video_module, "extract_audio_from_video", fake_extract)
        _, meta = load_video("clip.mp4", fps=2.0, use_audio_in_video=True)

    assert captured["cap"] == pytest.approx(121.0)  # 120 s of video + 1 s buffer
    assert meta[VIDEO_METADATA_KEY].duration == 120.0


@pytest.mark.parametrize("use_audio_in_video", [False, True])
def test_a_container_with_no_frame_rate_is_refused_naming_the_container(use_audio_in_video):
    """Without a rate the kept frames have no times, and the sound has no clock
    to sit on — with or without the sound track. Refused before decoding
    anything, with a message about the clip rather than about the dataclass."""
    with _stub_decoder(average_fps=None, num_frames=3600, duration_seconds=120.0):
        with pytest.raises(ValueError, match="container states no frame rate"):
            load_video("clip.mp4", fps=2.0, use_audio_in_video=use_audio_in_video)


def test_the_tmrope_scale_comes_from_the_frames_kept_not_the_rate_requested():
    """``seconds_per_patch`` is what scales TMRoPE's temporal row, so an error
    here misplaces every position id in and after the clip.

    Upstream computes ``temporal_patch_size / requested_fps``. A 25 fps source
    trimmed to "2 fps" lands on a stride of 12, i.e. frames 0.48 s apart, so
    that formula overstates each patch by 4%.
    """
    meta = _metadata_for(total_frames=101, source_fps=25.0, requested_fps=2.0)

    assert meta.seconds_per_patch(temporal_patch_size=2) == pytest.approx(2 * 12 / 25.0)
    assert meta.seconds_per_patch(temporal_patch_size=2) != 2 / meta.requested_fps
    # Consistent with the per-frame times: patch mid-times advance by exactly
    # one patch span.
    patch_times = meta.frame_timestamps(temporal_patch_size=2)
    assert patch_times[1] - patch_times[0] == pytest.approx(meta.seconds_per_patch(2))


def test_the_tmrope_scale_keeps_the_clip_span_right_when_max_frames_thins_it():
    """``max_frames`` spaces frames unevenly, and a single scalar cannot express
    that. The mean is used, which keeps the clip's total span right — an
    unthinned rate would compress the clip to a fraction of its real length."""
    meta = _metadata_for(total_frames=301, source_fps=30.0, requested_fps=2.0, max_frames=4)

    # 4 frames over the 10 s clip -> patches of 1 span 10/3 s on average.
    assert meta.seconds_per_patch() == pytest.approx(10.0 / 3.0)
    assert meta.seconds_per_patch() * (4 - 1) == pytest.approx(meta.frame_timestamps()[-1])
    # The requested rate would have claimed 0.5 s per frame, i.e. a 1.5 s clip.
    assert 1 / meta.requested_fps == 0.5


@pytest.mark.parametrize("bad", [0, -1, 2.5, "2"])
def test_a_nonsense_patch_size_is_refused_rather_than_raising_from_the_arithmetic(bad):
    meta = VideoMetadata(fps=30.0, total_num_frames=31, frames_indices=[0, 15], requested_fps=2.0)

    with pytest.raises(ValueError, match="temporal_patch_size"):
        meta.frame_timestamps(temporal_patch_size=bad)


# --- The generated side: one clip, one file, sound included --------------------


def _generated_clip(num_frames: int = 6, fps: float = 24.0) -> tuple[torch.Tensor, VideoMetadata]:
    """A clip as a model would emit it, with the metadata that describes it.

    A generated clip is its own source, so ``fps`` is the playback rate and every
    frame is kept — which is what makes it expressible in the same
    ``VideoMetadata`` a decoded clip uses, with no second type for the output side.
    """
    frames = torch.randint(0, 255, (num_frames, 3, 16, 16), dtype=torch.uint8)
    meta = VideoMetadata(
        fps=fps,
        total_num_frames=num_frames,
        frames_indices=list(range(num_frames)),
        requested_fps=fps,
    )
    return frames, meta


def _fps_meta(fps) -> dict:
    """An emitted video item's meta declaring ``fps`` and nothing else.

    Set after construction, the way a producer could on the mutable dataclass,
    so ``save_video``'s own check is what gets exercised for a bad rate."""
    meta = VideoMetadata(fps=24, total_num_frames=0, frames_indices=[], requested_fps=24)
    meta.fps = fps
    return {VIDEO_METADATA_KEY: meta}


def test_a_generated_clips_metadata_describes_its_own_timeline():
    """No output-side type: the source axis degenerates onto the clip itself."""
    frames, meta = _generated_clip(num_frames=6, fps=24.0)

    assert meta.duration == pytest.approx(6 / 24)
    # Every frame kept means the timestamps are the frames' own times, so a
    # producer gets a meaningful timeline out of the same accessors.
    assert meta.frame_timestamps() == pytest.approx([i / 24 for i in range(len(frames))])


def test_a_generated_clip_with_no_fps_is_refused_rather_than_written_at_a_guess(tmp_path):
    """Same rule as the audio side: a wrong rate plays at the wrong speed, and
    nothing downstream can tell that apart from a bad generation."""
    frames, _ = _generated_clip()

    with pytest.raises(ValueError, match=VIDEO_METADATA_KEY):
        _video_module.save_video(str(tmp_path / "generated_video_0.mp4"), frames, {})


def test_a_sound_track_with_no_rate_is_refused_rather_than_inferred_from_the_frames(tmp_path):
    """``write_video_audio`` would derive the rate from ``samples / duration``.

    That always produces *a* rate, so the speech comes out stretched to whatever
    makes the two tracks the same length instead of failing.
    """
    frames, meta = _generated_clip()
    clip = VideoInputs(video=frames, audio=np.zeros(8000, dtype=np.float32))

    with pytest.raises(ValueError, match=AUDIO_METADATA_KEY):
        _video_module.save_video(str(tmp_path / "generated_video_0.mp4"), clip, {VIDEO_METADATA_KEY: meta})


@pytest.mark.parametrize("bad", [0, 0.0, -24, float("nan"), float("inf"), "abc", [24], True])
def test_a_rate_that_is_not_a_frame_rate_is_named_rather_than_failing_inside_the_muxer(bad, tmp_path):
    """``write_video_audio`` divides by the rate and then casts it.

    So every bad shape used to fail somewhere downstream naming something else:
    ``0`` a ZeroDivisionError, a negative an avcodec error, a string an ``int()``
    error. ``True`` is the quiet one — it is an ``int``, so it wrote a 1 fps file.
    """
    frames, _ = _generated_clip()

    with pytest.raises(ValueError, match="not a positive frame rate"):
        _video_module.save_video(str(tmp_path / "generated_video_0.mp4"), frames, _fps_meta(bad))

    assert not list(tmp_path.iterdir())


def test_float_frames_are_refused_rather_than_truncated_to_black(tmp_path):
    """The write casts with ``.to(torch.uint8)``.

    A VAE decoder — the usual producer here — emits floats in [0, 1], every one of
    which truncates to 0: a playable, entirely black clip. Which scale to use is
    the producer's call, and getting it wrong looks exactly like a bad generation.
    """
    frames, meta = _generated_clip()

    with pytest.raises(ValueError, match="expected uint8"):
        _video_module.save_video(
            str(tmp_path / "generated_video_0.mp4"), frames.float() / 255.0, {VIDEO_METADATA_KEY: meta}
        )

    assert not list(tmp_path.iterdir())


def test_a_fractional_generated_rate_is_refused_rather_than_truncated(tmp_path):
    """``write_video_audio`` stores the container rate as ``int(fps)``.

    So 23.976 would be written at 23 and a sub-1 fps clip at 0 — a file that
    plays at the wrong speed, which is the same failure a missing rate causes and
    is just as indistinguishable from a bad generation.
    """
    frames, _ = _generated_clip()

    with pytest.raises(ValueError, match="truncated to 23"):
        _video_module.save_video(str(tmp_path / "generated_video_0.mp4"), frames, _fps_meta(23.976))

    assert not list(tmp_path.iterdir())


def test_a_pre_decoded_frame_list_without_a_declared_rate_is_refused_rather_than_defaulted():
    """A list of frames carries no timing, so its caller states the rate.

    No default: the HF base's is 24, and ``requested_fps`` is the #1165 bug, and
    either puts every frame on an invented clock that reads as fact.
    """
    frames = [Image.new("RGB", (32, 32)) for _ in range(8)]

    with pytest.raises(ValueError, match="frames_fps"):
        _video_module.load_video(frames, fps=2.0)


def test_the_declared_frame_rate_reaches_load_video_through_fetch_videos():
    """``frames_fps`` is an ``mm_configs`` knob, and ``fetch_media`` hands the
    knobs to ``fetch_videos`` — which has to pass this one on."""
    frames = [Image.new("RGB", (2, 2)) for _ in range(4)]

    [(_, meta)] = _video_module.fetch_videos([frames], fps=2.0, frames_fps=4.0)

    assert meta[VIDEO_METADATA_KEY].fps == 4.0
    assert meta[VIDEO_METADATA_KEY].frames_indices == [0, 2]


def test_an_empty_generated_clip_is_refused_rather_than_written_as_a_bare_container(tmp_path):
    """Symmetry with ``save_audio``, which refuses an empty waveform by name."""
    path = tmp_path / "generated_video_0.mp4"
    with pytest.raises(ValueError, match="no frames"):
        _video_module.save_video(str(path), torch.zeros(0, 3, 16, 16, dtype=torch.uint8), _fps_meta(24.0))
    assert not path.exists()


def test_a_generated_clip_with_sound_is_one_muxed_file_not_a_video_and_a_stray_wav(tmp_path):
    """The whole point of carrying both streams on one item.

    Read back through the decode path rather than trusting the writer: this
    asserts the container really holds an audio stream at the declared rate, so a
    silently dropped or resampled track fails here.
    """
    av = pytest.importorskip("av")
    frames, meta = _generated_clip(num_frames=12, fps=24.0)
    rate = 16000
    samples = int(len(frames) / meta.fps * rate)
    # bfloat16 because that is what a vocoder emits under mixed precision, and
    # numpy has no dtype for it — the save path has to cast before it writes.
    waveform = torch.zeros(samples, dtype=torch.bfloat16)
    path = tmp_path / "generated_video_0.mp4"

    _video_module.save_video(
        str(path),
        VideoInputs(video=frames, audio=waveform),
        {VIDEO_METADATA_KEY: meta, AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=rate)},
    )

    with av.open(str(path)) as container:
        video_streams = container.streams.video
        audio_streams = container.streams.audio
        assert len(video_streams) == 1 and len(audio_streams) == 1
        assert audio_streams[0].sample_rate == rate
        assert video_streams[0].codec_context.framerate == meta.fps
    # No companion .wav: the sound went into the clip, not beside it.
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_a_silent_generated_clip_needs_no_carrier(tmp_path):
    """A bare frame tensor is accepted, so a video-only model states nothing
    about sound rather than wrapping ``audio=None`` to say so."""
    av = pytest.importorskip("av")
    frames, meta = _generated_clip(num_frames=4, fps=8.0)
    path = tmp_path / "generated_video_0.mp4"

    _video_module.save_video(str(path), frames, {VIDEO_METADATA_KEY: meta})

    with av.open(str(path)) as container:
        assert len(container.streams.video) == 1
        assert not container.streams.audio


def test_the_inferencer_writes_a_generated_clip_from_the_items_own_meta(tmp_path):
    """End to end through the arm that reads the item, not just the writer.

    The rates reach the file only if the save arm looks them up under the two
    metadata keys; a typo there would still write a playable file at the global
    default, which is the failure this pins.
    """
    av = pytest.importorskip("av")
    from veomni.trainer.omni.omni_inferencer import _save_generated_media

    frames, meta = _generated_clip(num_frames=12, fps=24.0)
    rate = 16000
    item = {
        "type": "video",
        "value": VideoInputs(video=frames, audio=torch.zeros(int(len(frames) / meta.fps * rate))),
        "meta": {VIDEO_METADATA_KEY: meta, AUDIO_METADATA_KEY: AudioMetadata(sampling_rate=rate, num_samples=8000)},
    }

    written = _save_generated_media([item], str(tmp_path))

    assert written == {"video": 1}
    with av.open(str(tmp_path / "generated_video_0.mp4")) as container:
        assert container.streams.audio[0].sample_rate == rate
        assert container.streams.video[0].codec_context.framerate == meta.fps
