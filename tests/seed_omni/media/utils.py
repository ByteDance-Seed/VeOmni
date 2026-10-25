"""Skip guard for tests that write and decode a real clip."""

import os
from types import ModuleType

import pytest

from veomni.utils.import_utils import is_ffmpeg_available


def require_video_stack() -> ModuleType:
    """Return ``veomni.data.seed_omni.utils.video``, or skip unless a real clip
    can be written (``av``) and decoded (ffmpeg + torchcodec) here.

    The ``npu_aarch64`` extra ships without torchcodec. An installed torchcodec
    raises ``RuntimeError`` on import when ffmpeg's or CUDA's shared libraries
    are missing, but also when it does not match the installed torch, so CI
    fails on it instead of skipping.
    """
    pytest.importorskip("av")
    if not is_ffmpeg_available():
        pytest.skip("ffmpeg is not available")
    try:
        import torchcodec.decoders  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"torchcodec is not installed: {exc}")
    except RuntimeError as exc:
        if os.environ.get("CI"):
            raise
        pytest.skip(f"torchcodec cannot load: {str(exc).splitlines()[0]}")
    from veomni.data.seed_omni.utils import video

    return video
