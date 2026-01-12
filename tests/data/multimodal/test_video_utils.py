import os
import time
from io import BytesIO

import av
import numpy as np
import PIL.Image
import pytest
import torch

from veomni.data.multimodal.video_utils import (
    fetch_videos,
    fetch_videos_metadata,
    load_video,
    load_video_from_bytes_list,
    smart_audio_nframes,
    smart_video_nframes,
)


# Make sure to place a sample.mp4 file in tests/data/assets
VIDEO_PATH = os.path.join(os.environ["CI_SAMPLES_DIR"], "sample.mp4")

# Skip tests if the sample video file doesn't exist
pytestmark = pytest.mark.skipif(not os.path.exists(VIDEO_PATH), reason=f"Test video not found at {VIDEO_PATH}")


def assert_video_output_valid(video: torch.Tensor, audio: np.ndarray = None, **kwargs):
    """
    Assert that video and audio outputs are valid and reasonable.

    Args:
        video: Video tensor (T, C, H, W)
        audio: Optional audio array
        **kwargs: Processing parameters used
    """
    # Check video tensor shape and type
    assert video.ndim == 4, f"Video must be 4D (T, C, H, W), got {video.ndim}D"
    T, C, H, W = video.shape
    assert C == 3, f"Video must have 3 channels (RGB), got {C}"

    # Check video dimensions are reasonable
    assert T > 0, f"Video must have at least 1 frame, got {T}"
    assert H > 0 and W > 0, f"Video dimensions must be positive, got H={H}, W={W}"

    # Check frame count constraints
    if "min_frames" in kwargs and kwargs["min_frames"] is not None:
        assert T >= kwargs["min_frames"], f"Video has {T} frames, expected >= {kwargs['min_frames']}"
    if "max_frames" in kwargs and kwargs["max_frames"] is not None:
        assert T <= kwargs["max_frames"], f"Video has {T} frames, expected <= {kwargs['max_frames']}"

    # Check scale_factor constraint
    if "scale_factor" in kwargs and kwargs["scale_factor"] is not None:
        scale_factor = kwargs["scale_factor"]
        assert H % scale_factor == 0, f"Height {H} must be divisible by scale_factor {scale_factor}"
        assert W % scale_factor == 0, f"Width {W} must be divisible by scale_factor {scale_factor}"

    # Check pixel value range (should be in [0, 255] for uint8 or reasonable float range)
    assert video.min() >= 0, f"Video has negative pixel values: min={video.min()}"
    if video.dtype == torch.uint8:
        assert video.max() <= 255, f"Video uint8 values exceed 255: max={video.max()}"

    # Check pixel constraints
    if "video_min_pixels" in kwargs and kwargs["video_min_pixels"] is not None:
        pixels = H * W
        assert pixels >= kwargs["video_min_pixels"], (
            f"Video has {pixels} pixels, expected >= {kwargs['video_min_pixels']}"
        )
    if "video_max_pixels" in kwargs and kwargs["video_max_pixels"] is not None:
        pixels = H * W
        assert pixels <= kwargs["video_max_pixels"], (
            f"Video has {pixels} pixels, expected <= {kwargs['video_max_pixels']}"
        )

    # Check audio if present
    if audio is not None:
        assert audio.ndim == 1, f"Audio must be 1D, got {audio.ndim}D"
        assert len(audio) > 0, "Audio array must not be empty"

        # Check audio-video duration consistency
        fps = kwargs.get("fps")
        sample_rate = kwargs.get("sample_rate")
        if fps is not None and sample_rate is not None:
            video_duration = T / fps
            audio_duration = len(audio) / sample_rate
            # Allow for some tolerance (one frame's duration)
            tolerance = 1.0 / fps
            assert abs(video_duration - audio_duration) < tolerance, (
                f"Mismatch in duration. Video: {video_duration:.2f}s, Audio: {audio_duration:.2f}s, "
                f"Difference: {abs(video_duration - audio_duration):.2f}s, Tolerance: {tolerance:.2f}s"
            )


def test_fetch_videos_from_path():
    """
    Test fetch_videos with file path input (str).
    """
    video_paths = [VIDEO_PATH]

    kwargs = {
        "fps": 1,
        "video_min_pixels": 224 * 224,
        "scale_factor": 14,
        "use_audio_in_video": True,
        "sample_rate": 16000,
    }

    videos, audios = fetch_videos(video_paths, **kwargs)

    assert len(videos) == 1, f"Expected 1 video, got {len(videos)}"
    assert len(audios) == 1, f"Expected 1 audio, got {len(audios)}"

    assert_video_output_valid(videos[0], audios[0], **kwargs)


def test_fetch_videos_from_bytes():
    """
    Test fetch_videos with bytes input (ByteString).
    """
    with open(VIDEO_PATH, "rb") as f:
        video_bytes = f.read()

    video_inputs = [video_bytes]

    kwargs = {
        "fps": 1,
        "video_min_pixels": 224 * 224,
        "scale_factor": 14,
        "use_audio_in_video": True,
        "sample_rate": 16000,
    }

    videos, audios = fetch_videos(video_inputs, **kwargs)

    assert len(videos) == 1, f"Expected 1 video, got {len(videos)}"
    assert len(audios) == 1, f"Expected 1 audio, got {len(audios)}"

    assert_video_output_valid(videos[0], audios[0], **kwargs)


def test_fetch_videos_without_audio():
    """
    Test fetch_videos with use_audio_in_video=False.
    """
    video_paths = [VIDEO_PATH]

    kwargs = {
        "fps": 1,
        "video_min_pixels": 224 * 224,
        "scale_factor": 14,
        "use_audio_in_video": False,
    }

    videos, audios = fetch_videos(video_paths, **kwargs)

    assert len(videos) == 1, f"Expected 1 video, got {len(videos)}"
    assert len(audios) == 1, f"Expected 1 audio, got {len(audios)}"
    assert audios[0] is None, "Audio should be None when use_audio_in_video=False"

    assert_video_output_valid(videos[0], **kwargs)


def test_fetch_videos_with_frame_constraints():
    """
    Test fetch_videos with min_frames and max_frames constraints.
    """
    video_paths = [VIDEO_PATH]

    kwargs = {
        "fps": 2,
        "min_frames": 8,
        "max_frames": 16,
        "frame_factor": 4,
        "video_min_pixels": 224 * 224,
        "scale_factor": 14,
    }

    videos, audios = fetch_videos(video_paths, **kwargs)

    assert len(videos) == 1, f"Expected 1 video, got {len(videos)}"

    # Check frame count is within constraints and divisible by frame_factor
    T = videos[0].shape[0]
    assert T >= kwargs["min_frames"], f"Video has {T} frames, expected >= {kwargs['min_frames']}"
    assert T <= kwargs["max_frames"], f"Video has {T} frames, expected <= {kwargs['max_frames']}"
    assert T % kwargs["frame_factor"] == 0, (
        f"Frame count {T} must be divisible by frame_factor {kwargs['frame_factor']}"
    )

    assert_video_output_valid(videos[0], audios[0], **kwargs)


@pytest.mark.benchmark
def test_benchmark_fetch_videos_from_path():
    """
    Benchmark fetch_videos with file path input.

    Measures:
    - Processing time
    - Frames per second throughput
    - Memory efficiency
    """
    video_paths = [VIDEO_PATH]

    kwargs = {
        "fps": 1,
        "video_min_pixels": 224 * 224,
        "scale_factor": 14,
        "use_audio_in_video": True,
        "sample_rate": 16000,
    }

    # Run multiple iterations for stable benchmark
    num_runs = 5
    durations = []
    videos, audios = None, None

    for _ in range(num_runs):
        start_time = time.perf_counter()
        videos, audios = fetch_videos(video_paths, **kwargs)
        durations.append((time.perf_counter() - start_time) * 1000)  # Convert to ms

    # Calculate statistics
    avg_time = sum(durations) / len(durations)
    min_time = min(durations)
    max_time = max(durations)
    std_dev = (sum((t - avg_time) ** 2 for t in durations) / len(durations)) ** 0.5

    # Validate results
    assert len(videos) == 1
    assert_video_output_valid(videos[0], audios[0], **kwargs)

    # Print benchmark info
    print(f"\n{'=' * 60}")
    print("Video Processing Benchmark Results")
    print(f"{'=' * 60}")
    print(f"Number of runs: {num_runs}")
    print(f"Average time: {avg_time:.2f} ms")
    print(f"Min time: {min_time:.2f} ms")
    print(f"Max time: {max_time:.2f} ms")
    print(f"Std dev: {std_dev:.2f} ms")
    print(f"Video shape: {videos[0].shape}")
    print(f"Audio samples: {len(audios[0]) if audios[0] is not None else 0}")
    print(f"Total pixels processed: {videos[0].shape[0] * videos[0].shape[2] * videos[0].shape[3]}")
    print(f"{'=' * 60}")


@pytest.mark.benchmark
def test_benchmark_fetch_videos_different_resolutions():
    """
    Benchmark video processing at different resolutions.

    Tests processing performance with various pixel constraints.
    """
    video_paths = [VIDEO_PATH]

    test_configs = [
        {"name": "Low (224x224)", "video_min_pixels": 224 * 224},
        {"name": "Medium (448x448)", "video_min_pixels": 448 * 448},
        {"name": "High (896x896)", "video_min_pixels": 896 * 896},
    ]

    num_runs = 5

    print(f"\n{'=' * 95}")
    print(f"{'Resolution':<20} {'Frames':<10} {'Shape':<20} {'Avg Time (ms)':<18} {'Std Dev (ms)':<15} {'FPS':<10}")
    print(f"{'=' * 95}")

    for config in test_configs:
        kwargs = {
            "fps": 2,
            "video_min_pixels": config["video_min_pixels"],
            "scale_factor": 14,
            "use_audio_in_video": False,
        }

        durations = []
        video = None
        for _ in range(num_runs):
            start_time = time.perf_counter()
            videos, _ = fetch_videos(video_paths, **kwargs)
            durations.append((time.perf_counter() - start_time) * 1000)
            video = videos[0]

        avg_time = sum(durations) / len(durations)
        std_dev = (sum((t - avg_time) ** 2 for t in durations) / len(durations)) ** 0.5

        T, C, H, W = video.shape
        fps_throughput = T / (avg_time / 1000) if avg_time > 0 else 0

        print(
            f"{config['name']:<20} {T:<10} {f'{H}x{W}':<20} {avg_time:<18.2f} {std_dev:<15.2f} {fps_throughput:<10.2f}"
        )

        assert_video_output_valid(video, **kwargs)

    print(f"{'=' * 95}")


@pytest.mark.benchmark
def test_benchmark_audio_processing():
    """
    Benchmark audio extraction and resampling performance.
    """
    video_paths = [VIDEO_PATH]

    configs = [
        {"name": "No Audio", "use_audio_in_video": False},
        {"name": "With Audio (16kHz)", "use_audio_in_video": True, "sample_rate": 16000},
        {"name": "With Audio (24kHz)", "use_audio_in_video": True, "sample_rate": 24000},
    ]

    num_runs = 5

    print(f"\n{'=' * 85}")
    print(f"{'Config':<25} {'Avg Time (ms)':<18} {'Std Dev (ms)':<15} {'Audio Samples':<15}")
    print(f"{'=' * 85}")

    for config in configs:
        kwargs = {
            "fps": 1,
            "video_min_pixels": 224 * 224,
            "scale_factor": 14,
        }
        kwargs.update({k: v for k, v in config.items() if k != "name"})

        durations = []
        videos, audios = None, None
        for _ in range(num_runs):
            start_time = time.perf_counter()
            videos, audios = fetch_videos(video_paths, **kwargs)
            durations.append((time.perf_counter() - start_time) * 1000)

        avg_time = sum(durations) / len(durations)
        std_dev = (sum((t - avg_time) ** 2 for t in durations) / len(durations)) ** 0.5

        audio_samples = len(audios[0]) if audios[0] is not None else 0

        print(f"{config['name']:<25} {avg_time:<18.2f} {std_dev:<15.2f} {audio_samples:<15}")

        assert_video_output_valid(videos[0], audios[0], **kwargs)

    print(f"{'=' * 85}")


def test_fetch_videos_from_bytes_list():
    """
    Test fetch_videos with list of bytes input (image frames).
    """
    # Load video and extract frames as JPEG bytes
    with open(VIDEO_PATH, "rb") as f:
        video_bytes = f.read()

    container = av.open(BytesIO(video_bytes))
    video_frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i >= 5:  # Take first 5 frames
            break
        video_frames.append(frame)
    container.close()

    # Convert frames to JPEG bytes
    frame_bytes_list = []
    for frame in video_frames:
        img = PIL.Image.fromarray(frame.to_rgb().to_ndarray())
        buf = BytesIO()
        img.save(buf, format="JPEG")
        frame_bytes_list.append(buf.getvalue())

    kwargs = {
        "fps": 2.0,  # Need to specify fps for bytes list
        "video_min_pixels": 224 * 224,
        "scale_factor": 14,
    }

    videos, audios = fetch_videos([frame_bytes_list], **kwargs)

    assert len(videos) == 1, f"Expected 1 video, got {len(videos)}"
    assert len(audios) == 1, f"Expected 1 audio, got {len(audios)}"
    assert audios[0] is None, "Audio should be None for bytes list input"
    assert_video_output_valid(videos[0], **kwargs)


def test_fetch_videos_from_numpy_array():
    """
    Test fetch_videos with numpy array of bytes.
    """
    # Load video and extract frames
    with open(VIDEO_PATH, "rb") as f:
        video_bytes = f.read()

    container = av.open(BytesIO(video_bytes))
    video_frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i >= 3:  # Take first 3 frames
            break
        video_frames.append(frame)
    container.close()

    # Convert frames to JPEG bytes
    frame_bytes_list = []
    for frame in video_frames:
        img = PIL.Image.fromarray(frame.to_rgb().to_ndarray())
        buf = BytesIO()
        img.save(buf, format="JPEG")
        frame_bytes_list.append(buf.getvalue())

    # Convert to numpy array
    frame_array = np.array(frame_bytes_list, dtype=object)

    kwargs = {
        "fps": 2.0,
        "video_min_pixels": 224 * 224,
        "scale_factor": 14,
    }

    videos, audios = fetch_videos([frame_array], **kwargs)

    assert len(videos) == 1, f"Expected 1 video, got {len(videos)}"
    assert_video_output_valid(videos[0], **kwargs)


def test_fetch_videos_metadata():
    """
    Test fetch_videos_metadata returns full metadata with smart processing.
    Note: In tingyang's version, passing 'fps' in kwargs triggers direct return mode.
    To test smart processing, we should NOT pass 'fps' in kwargs.
    """
    video_paths = [VIDEO_PATH]

    # NOT passing 'fps' here to ensure smart processing is applied
    kwargs = {
        "video_min_pixels": 224 * 224,
        "scale_factor": 14,
        "use_audio_in_video": True,
        "sample_rate": 16000,
        "min_frames": 4,
        "max_frames": 16,
        "frame_factor": 4,
    }

    videos, video_meta, audios, audio_meta = fetch_videos_metadata(video_paths, **kwargs)

    assert len(videos) == 1, f"Expected 1 video, got {len(videos)}"
    assert len(video_meta) == 1, f"Expected 1 video metadata, got {len(video_meta)}"

    # Check video metadata
    assert "fps" in video_meta[0], "Video metadata should contain 'fps'"
    assert "total_num_frames" in video_meta[0], "Video metadata should contain 'total_num_frames'"
    assert video_meta[0]["total_num_frames"] == videos[0].shape[0], "Metadata frame count should match tensor"

    # Check audio metadata if present
    if audios[0] is not None:
        assert audio_meta[0] is not None, "Audio metadata should not be None when audio is present"
        assert "fps" in audio_meta[0], "Audio metadata should contain 'fps'"
        assert "total_num_frames" in audio_meta[0], "Audio metadata should contain 'total_num_frames'"

    # Verify video was properly processed with smart_resize and smart_video_nframes
    T, C, H, W = videos[0].shape
    assert C == 3, "Should have 3 RGB channels"
    assert H % kwargs["scale_factor"] == 0, f"Height {H} should be divisible by scale_factor {kwargs['scale_factor']}"
    assert W % kwargs["scale_factor"] == 0, f"Width {W} should be divisible by scale_factor {kwargs['scale_factor']}"
    assert T % kwargs["frame_factor"] == 0, (
        f"Frame count {T} should be divisible by frame_factor {kwargs['frame_factor']}"
    )


def test_fetch_videos_metadata_direct_return():
    """
    Test fetch_videos_metadata with fps parameter (direct return mode).
    This mode bypasses smart processing and returns raw loaded data.
    """
    video_paths = [VIDEO_PATH]

    kwargs = {
        "fps": 2.0,  # Passing fps triggers direct return mode
        "use_audio_in_video": False,
    }

    videos, video_meta, audios, audio_meta = fetch_videos_metadata(video_paths, **kwargs)

    assert len(videos) == 1, f"Expected 1 video, got {len(videos)}"
    assert len(video_meta) == 1, f"Expected 1 video metadata, got {len(video_meta)}"

    # In direct return mode, metadata fps should match what we passed
    assert "fps" in video_meta[0], "Video metadata should contain 'fps'"
    assert "total_num_frames" in video_meta[0], "Video metadata should contain 'total_num_frames'"

    # Video shape should match metadata
    assert video_meta[0]["total_num_frames"] == videos[0].shape[0]


def test_smart_video_nframes_explicit_frames():
    """
    Test smart_video_nframes with explicit frames parameter.
    """
    video, video_meta, _, _ = load_video(VIDEO_PATH)

    target_frames = 12
    processed_video, processed_meta = smart_video_nframes(
        video, video_meta["fps"], frames=target_frames, frame_factor=4
    )

    # Should get exactly 12 frames (multiple of 4)
    assert processed_video.shape[0] == target_frames, (
        f"Expected {target_frames} frames, got {processed_video.shape[0]}"
    )
    assert processed_meta["total_num_frames"] == target_frames
    assert "fps" in processed_meta, "Processed metadata should contain 'fps'"


def test_smart_video_nframes_metadata():
    """
    Test that smart_video_nframes returns correct metadata.
    """
    video, video_meta, _, _ = load_video(VIDEO_PATH)

    processed_video, processed_meta = smart_video_nframes(
        video, video_meta["fps"], fps=2, min_frames=8, frame_factor=4
    )

    # Check metadata structure
    assert "fps" in processed_meta, "Metadata should contain 'fps'"
    assert "total_num_frames" in processed_meta, "Metadata should contain 'total_num_frames'"

    # Check metadata accuracy
    assert processed_meta["total_num_frames"] == processed_video.shape[0]
    assert processed_meta["fps"] > 0, "FPS should be positive"


def test_smart_audio_nframes_conditional_resample():
    """
    Test that smart_audio_nframes only resamples when necessary.
    """
    _, _, audio, audio_meta = load_video(VIDEO_PATH, use_audio_in_video=True)

    if audio is not None and audio_meta is not None:
        original_fps = audio_meta["fps"]

        # Test 1: Same sample rate - should not resample
        processed_audio1, meta1 = smart_audio_nframes(audio, original_fps, sample_rate=int(original_fps))
        assert meta1["fps"] == int(original_fps), "FPS should match requested sample rate"
        assert "total_num_frames" in meta1, "Metadata should contain 'total_num_frames'"

        # Test 2: Different sample rate - should resample
        processed_audio2, meta2 = smart_audio_nframes(audio, original_fps, sample_rate=16000)
        assert meta2["fps"] == 16000, "FPS should be 16000 after resampling"
        assert "total_num_frames" in meta2, "Metadata should contain 'total_num_frames'"

        if original_fps != 16000:
            # Length should change if resampling occurred
            assert len(processed_audio2) != len(audio), "Audio length should change after resampling"


def test_smart_audio_nframes_metadata():
    """
    Test that smart_audio_nframes returns correct metadata.
    """
    _, _, audio, audio_meta = load_video(VIDEO_PATH, use_audio_in_video=True)

    if audio is not None and audio_meta is not None:
        processed_audio, processed_meta = smart_audio_nframes(audio, audio_meta["fps"], sample_rate=16000)

        # Check metadata structure
        assert "fps" in processed_meta, "Metadata should contain 'fps'"
        assert "total_num_frames" in processed_meta, "Metadata should contain 'total_num_frames'"

        # Check metadata accuracy
        assert processed_meta["total_num_frames"] == len(processed_audio)
        assert processed_meta["fps"] == 16000


def test_load_video_from_bytes_list_empty():
    """
    Test that empty frame list raises ValueError.
    """
    with pytest.raises(ValueError, match="empty"):
        load_video_from_bytes_list([], fps=2.0)


def test_load_video_metadata_structure():
    """
    Test that load_video returns proper metadata structure.
    """
    video, video_meta, audio, audio_meta = load_video(VIDEO_PATH, use_audio_in_video=True)

    # Check video metadata
    assert video_meta is not None, "Video metadata should not be None"
    assert isinstance(video_meta, dict), "Video metadata should be a dictionary"
    assert "fps" in video_meta, "Video metadata should contain 'fps'"
    assert "total_num_frames" in video_meta, "Video metadata should contain 'total_num_frames'"
    assert video_meta["total_num_frames"] == video.shape[0], "Metadata frame count should match tensor"

    # Check audio metadata if audio is present
    if audio is not None:
        assert audio_meta is not None, "Audio metadata should not be None when audio is present"
        assert isinstance(audio_meta, dict), "Audio metadata should be a dictionary"
        assert "fps" in audio_meta, "Audio metadata should contain 'fps'"
        assert "total_num_frames" in audio_meta, "Audio metadata should contain 'total_num_frames'"


def test_metadata_consistency():
    """
    Test that metadata is consistent throughout the processing pipeline.
    This test verifies that fetch_videos applies proper constraints and metadata is accurate.
    """
    video_paths = [VIDEO_PATH]

    # Use fetch_videos (not fetch_videos_metadata with fps) to ensure smart processing
    kwargs = {
        "fps": 2,  # Target FPS for smart_video_nframes
        "min_frames": 8,
        "max_frames": 16,
        "frame_factor": 4,
        "video_min_pixels": 224 * 224,
        "scale_factor": 14,
    }

    # Use regular fetch_videos which always applies smart processing
    videos, _ = fetch_videos(video_paths, **kwargs)

    # Verify frame count respects constraints
    T = videos[0].shape[0]
    assert T >= kwargs["min_frames"], f"Frame count {T} should be >= {kwargs['min_frames']}"
    assert T <= kwargs["max_frames"], f"Frame count {T} should be <= {kwargs['max_frames']}"
    assert T % kwargs["frame_factor"] == 0, f"Frame count {T} should be divisible by {kwargs['frame_factor']}"

    # Also test with fetch_videos_metadata (without fps to avoid direct return mode)
    kwargs_no_fps = {k: v for k, v in kwargs.items() if k != "fps"}
    videos2, video_meta, _, _ = fetch_videos_metadata(video_paths, **kwargs_no_fps)

    # Verify metadata frame count matches actual tensor
    assert videos2[0].shape[0] == video_meta[0]["total_num_frames"], (
        f"Video tensor has {videos2[0].shape[0]} frames but metadata says {video_meta[0]['total_num_frames']}"
    )

    # Verify fps is positive
    assert video_meta[0]["fps"] > 0, f"FPS should be positive, got {video_meta[0]['fps']}"
