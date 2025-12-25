import os

import numpy as np
import pytest
import torch
from PIL import Image

from veomni.data.multimodal.video_utils import TORCHCODEC_AVAILABLE, fetch_videos, is_ffmpeg_available


# Make sure to place a sample.mp4 file in tests/data/assets
VIDEO_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "sample.mp4")

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


@pytest.mark.skipif(
    not (TORCHCODEC_AVAILABLE and is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available"
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


@pytest.mark.skipif(
    not (TORCHCODEC_AVAILABLE and is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available"
)
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


@pytest.mark.skipif(
    not (TORCHCODEC_AVAILABLE and is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available"
)
def test_fetch_videos_from_pil_images():
    """
    Test fetch_videos with PIL Image list input (List[PIL.Image.Image]).
    """
    # Create dummy PIL images
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


@pytest.mark.skipif(
    not (TORCHCODEC_AVAILABLE and is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available"
)
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


@pytest.mark.skipif(
    not (TORCHCODEC_AVAILABLE and is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available"
)
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


@pytest.mark.skipif(
    not (TORCHCODEC_AVAILABLE and is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available"
)
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


@pytest.mark.skipif(
    not (TORCHCODEC_AVAILABLE and is_ffmpeg_available()), reason="torchcodec or ffmpeg is not available"
)
def test_fetch_videos_multiple_inputs():
    """
    Test fetch_videos with multiple video inputs of different types.
    """
    # Prepare different input types
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
