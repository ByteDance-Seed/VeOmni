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


"""Import utils"""

import importlib.metadata
import importlib.util
import subprocess
from functools import lru_cache
from typing import TYPE_CHECKING, Dict

from packaging import version


if TYPE_CHECKING:
    from packaging.version import Version


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _get_package_version(name: str) -> "Version":
    try:
        return version.parse(importlib.metadata.version(name))
    except Exception:
        return version.parse("0.0.0")


_PACKAGE_FLAGS: Dict[str, bool] = {
    "flash_attn": _is_package_available("flash_attn"),
    "liger_kernel": _is_package_available("liger_kernel"),
    "torch_npu": _is_package_available("torch_npu"),
    "diffusers": _is_package_available("diffusers"),
    "av": _is_package_available("av"),
    "librosa": _is_package_available("librosa"),
    "soundfile": _is_package_available("soundfile"),
    "triton": _is_package_available("triton"),
    "veomni_patch": _is_package_available("veomni_patch"),
}


def is_flash_attn_2_available() -> bool:
    return _PACKAGE_FLAGS["flash_attn"]


def is_liger_kernel_available() -> bool:
    return _PACKAGE_FLAGS["liger_kernel"]


def is_torch_npu_available() -> bool:
    return _PACKAGE_FLAGS["torch_npu"]


def is_diffusers_available() -> bool:
    return _PACKAGE_FLAGS["diffusers"]


def is_fused_moe_available() -> bool:
    import torch

    return torch.cuda.is_available() and not _PACKAGE_FLAGS["torch_npu"] and _PACKAGE_FLAGS["triton"]


def is_video_audio_available() -> bool:
    return _PACKAGE_FLAGS["av"] and _PACKAGE_FLAGS["librosa"] and _PACKAGE_FLAGS["soundfile"]


@lru_cache
def is_torch_version_greater_than(value: str) -> bool:
    return _get_package_version("torch") >= version.parse(value)


@lru_cache
def is_transformers_version_greater_or_equal_to(value: str) -> bool:
    return _get_package_version("transformers") >= version.parse(value)


def is_veomni_patch_available() -> bool:
    return _PACKAGE_FLAGS["veomni_patch"]


def is_linux_aarch64_platform() -> bool:
    import platform

    return platform.system() == "Linux" and platform.machine().lower() == "aarch64"


_FFMPEG_AVAILABLE = None


def is_ffmpeg_available() -> bool:
    """Check if ffmpeg is available (required for torchcodec and URL downloads).

    Supported FFmpeg versions: 4-8.
    """
    global _FFMPEG_AVAILABLE
    if _FFMPEG_AVAILABLE is None:
        try:
            subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, text=True)
            _FFMPEG_AVAILABLE = True
        except (FileNotFoundError, subprocess.CalledProcessError):
            _FFMPEG_AVAILABLE = False
    return _FFMPEG_AVAILABLE
