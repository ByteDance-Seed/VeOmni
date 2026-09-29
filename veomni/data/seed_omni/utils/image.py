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

"""Minimal image IO for the SeedOmni data layer.

This is intentionally tiny: **load** the image and, if it's huge, do a single
**aspect-preserving downscale** so a giant source image doesn't blow up memory /
IPC. That's it.

It deliberately does **not** do ``smart_resize`` (patch-aligned / min-max-pixel
rounding) — that is a model-specific decision owned by the vision module's
processor (e.g. ``Qwen2VLImageProcessor``), which receives the pixels and does
its own resize + patchify + normalize. Keeping the data layer model-agnostic is
the whole point.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from io import BytesIO
from typing import Any, ByteString, Union

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
            response = requests.get(image, timeout=(5, 30))
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

    A repeated ref (the same path / bytes object appearing more than once in the
    list) is decoded once and cloned on reuse, so duplicates don't pay a second
    decode and each item carries an independent tensor. ``image_max_pixels`` is
    the only other knob; everything else (``**kwargs``) is ignored here and
    handled by the vision module's processor downstream."""
    del kwargs
    cache: dict = {}
    out: list[torch.Tensor] = []
    for image in images:
        # Hashable refs (str / bytes) key by value; other refs key by identity.
        key = image if isinstance(image, (str, bytes)) else id(image)
        if key in cache:
            out.append(cache[key].clone())
        else:
            tensor = pil_to_uint8_tensor(resize_to_max_pixels(load_image(image), image_max_pixels))
            cache[key] = tensor
            out.append(tensor)
    return out


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


def save_image(path: str, image: Image.Image | torch.Tensor, meta: Mapping[str, Any] | None = None) -> None:
    """Write one generated image.

    Takes a PIL image (what a vision decoder emits) or the ``(C, H, W) uint8``
    tensor :func:`pil_to_uint8_tensor` produces on the way in. ``meta`` is the
    item's whole ``ConversationItem.meta``, accepted so every saver has one call
    shape; an image states nothing there that changes how it is written.

    A non-uint8 tensor is refused rather than cast, for the reason
    :func:`~.video.save_video` refuses one: a decoder's ``[0, 1]`` floats all
    truncate to 0, and a black image looks like a bad generation.
    """
    del meta
    if isinstance(image, Image.Image):
        image.save(path)
        return
    if not torch.is_tensor(image):
        raise TypeError(f"save_image: the image item for {path} is a {type(image).__name__}, expected a PIL image.")
    if image.ndim != 3 or image.shape[0] not in (1, 3, 4):
        raise ValueError(
            f"save_image: the image item for {path} has shape {tuple(image.shape)}, which is not one "
            f"(C, H, W) image. Emit one item per image."
        )
    if image.dtype != torch.uint8:
        raise ValueError(
            f"save_image: the image item for {path} holds {image.dtype} pixels, expected uint8. Convert to "
            f"8-bit pixels first (a [0, 1] float image needs scaling by 255, not a cast)."
        )
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    Image.fromarray(array[..., 0] if array.shape[-1] == 1 else array).save(path)


__all__ = ["ImageInput", "load_image", "resize_to_max_pixels", "fetch_images", "pil_to_uint8_tensor", "save_image"]
