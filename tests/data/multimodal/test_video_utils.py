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
from veomni.utils.import_utils import is_ffmpeg_available


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
        if "sample_rate" in kwargs:
            # Audio length should be reasonable (at least 1 sample per video frame)
            expected_min_samples = T
            assert len(audio) >= expected_min_samples, (
                f"Audio has {len(audio)} samples, expected >= {expected_min_samples}"
            )


@pytest.mark.skipif(not (is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available")
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


@pytest.mark.skipif(not (is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available")
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


@pytest.mark.skipif(not (is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available")
def test_fetch_videos_from_pil_images():
    """
    Test fetch_videos with PIL Image list input (List[PIL.Image.Image]).
    """
    # Create dummy PIL images
    from PIL import Image

    images = [
        Image.new("RGB", (640, 480), color=(255, 0, 0)),
        Image.new("RGB", (640, 480), color=(0, 255, 0)),
        Image.new("RGB", (640, 480), color=(0, 0, 255)),
    ]

    video_inputs = [images]

    kwargs = {
        "fps": 2,
        "video_min_pixels": 224 * 224,
        "scale_factor": 14,
    }

    videos, audios = fetch_videos(video_inputs, **kwargs)

    assert len(videos) == 1, f"Expected 1 video, got {len(videos)}"
    assert len(audios) == 1, f"Expected 1 audio, got {len(audios)}"
    assert audios[0] is None, "PIL image input should not have audio"

    assert_video_output_valid(videos[0], **kwargs)


@pytest.mark.skipif(not (is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available")
def test_fetch_videos_from_dict():
    """
    Test fetch_videos with dict input (Dict[str, np.ndarray]).
    """
    # Create dummy video array (T, H, W, C) format
    video_array = np.random.randint(0, 255, size=(10, 480, 640, 3), dtype=np.uint8)
    audio_array = np.random.randn(16000 * 2).astype(np.float32)  # 2 seconds of audio

    video_dict = {"video": video_array, "audio": audio_array, "video_fps": 30.0, "audio_fps": 16000}

    video_inputs = [video_dict]

    kwargs = {"fps": 2, "video_min_pixels": 224 * 224, "scale_factor": 14, "sample_rate": 16000}

    videos, audios = fetch_videos(video_inputs, **kwargs)

    assert len(videos) == 1, f"Expected 1 video, got {len(videos)}"
    assert len(audios) == 1, f"Expected 1 audio, got {len(audios)}"

    assert_video_output_valid(videos[0], audios[0], **kwargs)


@pytest.mark.skipif(not (is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available")
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


@pytest.mark.skipif(not (is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available")
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


@pytest.mark.skipif(not (is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available")
def test_fetch_videos_multiple_inputs():
    """
    Test fetch_videos with multiple video inputs of different types.
    """
    # Prepare different input types
    from PIL import Image

    with open(VIDEO_PATH, "rb") as f:
        video_bytes = f.read()

    images = [
        Image.new("RGB", (640, 480), color=(255, 0, 0)),
        Image.new("RGB", (640, 480), color=(0, 255, 0)),
    ]

    video_inputs = [VIDEO_PATH, video_bytes, images]

    kwargs = {
        "fps": 1,
        "video_min_pixels": 224 * 224,
        "scale_factor": 14,
        "use_audio_in_video": True,
        "sample_rate": 16000,
    }

    videos, audios = fetch_videos(video_inputs, **kwargs)

    assert len(videos) == 3, f"Expected 3 videos, got {len(videos)}"
    assert len(audios) == 3, f"Expected 3 audios, got {len(audios)}"

    # Verify each output
    for i, (video, audio) in enumerate(zip(videos, audios)):
        assert_video_output_valid(video, audio, **kwargs)


@pytest.mark.skipif(not (is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available")
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


@pytest.mark.skipif(not (is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available")
def test_fetch_videos_metadata():
    """
    Test fetch_videos_metadata returns full metadata with smart processing.
    """
    video_paths = [VIDEO_PATH]

    kwargs = {
        "fps": 2,
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
    assert "frames_indices" in video_meta[0], "Video metadata should contain 'frames_indices'"
    assert video_meta[0]["total_num_frames"] == videos[0].shape[0], "Metadata frame count should match tensor"

    # Check frames_indices
    frames_indices = video_meta[0]["frames_indices"]
    assert frames_indices is not None, "frames_indices should not be None"
    assert isinstance(frames_indices, torch.Tensor), (
        f"frames_indices should be torch.Tensor, got {type(frames_indices)}"
    )
    assert len(frames_indices) == videos[0].shape[0], (
        f"frames_indices length {len(frames_indices)} should match video frame count {videos[0].shape[0]}"
    )
    # Indices should be non-decreasing (may have duplicates due to padding)
    assert (frames_indices[1:] >= frames_indices[:-1]).all(), "frames_indices should be non-decreasing"

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


@pytest.mark.skipif(not (is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available")
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


@pytest.mark.skipif(not (is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available")
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


@pytest.mark.skipif(not (is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available")
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


@pytest.mark.skipif(not (is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available")
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


@pytest.mark.skipif(not (is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available")
def test_load_video_from_bytes_list_empty():
    """
    Test that empty frame list raises ValueError.
    """
    with pytest.raises(ValueError, match="empty"):
        load_video_from_bytes_list([], fps=2.0)


@pytest.mark.skipif(not (is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available")
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
    assert "frames_indices" in video_meta, "Video metadata should contain 'frames_indices'"
    assert video_meta["total_num_frames"] == video.shape[0], "Metadata frame count should match tensor"

    # Check frames_indices
    frames_indices = video_meta["frames_indices"]
    assert frames_indices is not None, "frames_indices should not be None"
    assert isinstance(frames_indices, torch.Tensor), (
        f"frames_indices should be torch.Tensor, got {type(frames_indices)}"
    )
    assert len(frames_indices) == video.shape[0], "frames_indices length should match video frame count"

    # Check audio metadata if audio is present
    if audio is not None:
        assert audio_meta is not None, "Audio metadata should not be None when audio is present"
        assert isinstance(audio_meta, dict), "Audio metadata should be a dictionary"
        assert "fps" in audio_meta, "Audio metadata should contain 'fps'"
        assert "total_num_frames" in audio_meta, "Audio metadata should contain 'total_num_frames'"


@pytest.mark.skipif(not (is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available")
def test_frames_indices_for_qwen3vl():
    """
    Test that frames_indices are correctly returned for Qwen3-VL timestamp calculation.

    This test validates:
    1. frames_indices are present in metadata
    2. indices correspond to actual sampled frames from original video
    3. indices can be used for accurate timestamp calculation
    """
    video_paths = [VIDEO_PATH]

    kwargs = {
        "fps": 2.0,  # Sample at 2 FPS
        "min_frames": 4,
        "max_frames": 16,
        "frame_factor": 2,
        "video_min_pixels": 224 * 224,
        "scale_factor": 14,
    }

    videos, video_meta, _, _ = fetch_videos_metadata(video_paths, **kwargs)

    # Extract metadata
    meta = video_meta[0]
    frames_indices = meta["frames_indices"]
    fps = meta["fps"]
    total_frames = meta["total_num_frames"]

    # Validate frames_indices structure
    assert frames_indices is not None, "frames_indices should not be None"
    assert isinstance(frames_indices, torch.Tensor), "frames_indices should be a torch.Tensor"
    assert frames_indices.dtype == torch.long, f"frames_indices should be torch.long, got {frames_indices.dtype}"
    assert len(frames_indices) == total_frames, "frames_indices length should match total_num_frames"

    # Validate frames_indices values
    assert frames_indices.min() >= 0, "All frame indices should be >= 0"
    assert (frames_indices[1:] >= frames_indices[:-1]).all(), "Frame indices should be non-decreasing"

    # Simulate Qwen3-VL timestamp calculation (simplified version)
    # This replicates the logic in multimodal_chat_template.py
    merge_size = 2
    indices_list = frames_indices.tolist()

    # Pad to merge_size if needed (as done in Qwen3VLChatTemplate._calculate_timestamps)
    if len(indices_list) % merge_size != 0:
        indices_list.extend([indices_list[-1]] * (merge_size - len(indices_list) % merge_size))

    # Convert to timestamps
    timestamps = [idx / fps for idx in indices_list]

    # Merge and average
    merged_timestamps = [
        (timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)
    ]

    # Validate timestamps
    assert len(merged_timestamps) > 0, "Should have at least one timestamp"
    assert all(t >= 0 for t in merged_timestamps), "All timestamps should be >= 0"
    assert merged_timestamps == sorted(merged_timestamps), "Timestamps should be in ascending order"

    print("\n--- Qwen3-VL Timestamp Calculation Test ---")
    print(f"Original frames sampled: {len(frames_indices)}")
    print(f"Frame indices: {frames_indices.tolist()[:10]}{'...' if len(frames_indices) > 10 else ''}")
    print(f"FPS: {fps}")
    print(f"Calculated timestamps: {merged_timestamps}")
    print("-------------------------------------------")


@pytest.mark.skipif(not (is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available")
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


@pytest.mark.skipif(not (is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available")
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


@pytest.mark.skipif(not (is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available")
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
