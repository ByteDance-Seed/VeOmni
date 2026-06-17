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

"""Minimal image IO for the SeedOmni V2 data layer.

This is intentionally tiny: **load** the image and, if it's huge, do a single
**aspect-preserving downscale** so a giant source image doesn't blow up memory /
IPC. That's it.

It deliberately does **not** do ``smart_resize`` (patch-aligned / min-max-pixel
rounding) — that is a model-specific decision owned by the vision module's
processor (e.g. ``Qwen2VLImageProcessor``), which receives the pixels and does
its own resize + patchify + normalize. Keeping the data layer model-agnostic is
the whole point of SeedOmni V2 (see ``docs/seed_omni/design.md`` "Layer 2").
"""

from __future__ import annotations

import os
from io import BytesIO
from typing import ByteString, Union

import numpy as np
import requests
import torch
from PIL import Image


ImageInput = Union[Image.Image, ByteString, str]


def load_image(image: ImageInput) -> Image.Image:
    """Load an image (path / URL / bytes / PIL) into an RGB :class:`PIL.Image`."""
    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, str):
        if image.startswith(("http://", "https://")):
            response = requests.get(image, stream=True)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        else:
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image path does not exist: {image}")
            img = Image.open(image)
    elif isinstance(image, (bytes, bytearray)):
        img = Image.open(BytesIO(image))
    else:
        raise NotImplementedError(f"Unsupported image input type: {type(image).__name__}")
    return img.convert("RGB")


def resize_to_max_pixels(image: Image.Image, max_pixels: int | None) -> Image.Image:
    """Aspect-preserving downscale so ``H * W <= max_pixels`` (OOM guard only).

    No-op when ``max_pixels`` is ``None`` or the image already fits. Never
    upscales — small images are left for the processor to handle.
    """
    if not max_pixels:
        return image
    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = (max_pixels / (w * h)) ** 0.5
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))))


def fetch_images(images: list[ImageInput], image_max_pixels: int | None = None, **kwargs) -> list[torch.Tensor]:
    """Load + (optionally) OOM-cap a list of images, returning ``(C, H, W) uint8``
    tensors ready to carry on a conversation item.

    ``image_max_pixels`` is the only knob; everything else (``**kwargs``) is
    ignored here and handled by the vision module's processor downstream."""
    del kwargs
    return [pil_to_uint8_tensor(resize_to_max_pixels(load_image(image), image_max_pixels)) for image in images]


def pil_to_uint8_tensor(image: Image.Image) -> torch.Tensor:
    """Convert an RGB PIL image to a ``(C, H, W) uint8`` torch tensor.

    No normalization, no float conversion, no channel-mean subtraction —
    those are vision-encoder-specific decisions and live in the encoder
    module's ``pre_forward``/forward.  Keeping pixels as uint8 makes the
    dataloader-worker → main-process IPC roughly 4x cheaper than float32
    and preserves all original pixel information.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    # ``np.array`` (vs ``np.asarray``) forces a writable copy so torch
    # doesn't warn about non-writable tensors when we later .permute().
    arr = np.array(image, dtype=np.uint8)  # (H, W, C)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (C, H, W)
    return tensor


__all__ = ["ImageInput", "load_image", "resize_to_max_pixels", "fetch_images", "pil_to_uint8_tensor"]
