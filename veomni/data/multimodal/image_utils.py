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


import io
import math
from io import BytesIO
from typing import ByteString, List, Union

import numpy as np
import requests
from PIL import Image


ImageInput = Union[
    Image.Image,
    np.ndarray,
    ByteString,
    str,
]


def load_image_bytes_from_path(image_path: str):
    image = Image.open(image_path).convert("RGB")
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    return image_bytes.getvalue()


def save_image_bytes_to_file(image_bytes, output_path):
    image_bytes = io.BytesIO(image_bytes)
    image = Image.open(image_bytes).convert("RGB")
    image.save(output_path)


def smart_resize(
    image: Image.Image,
    scale_factor: int = None,
    image_min_pixels: int = None,
    image_max_pixels: int = None,
    max_ratio: int = None,
    **kwargs,
):
    width, height = image.size
    if max_ratio is not None:
        ratio = max(width, height) / min(width, height)
        if ratio > max_ratio:
            raise ValueError(f"absolute aspect ratio must be smaller than {max_ratio}, got {ratio}")

    if scale_factor is not None:
        h_bar = max(scale_factor, round(height / scale_factor) * scale_factor)
        w_bar = max(scale_factor, round(width / scale_factor) * scale_factor)
    else:
        h_bar = height
        w_bar = width

    if image_max_pixels is not None and h_bar * w_bar > image_max_pixels:
        beta = math.sqrt((height * width) / image_max_pixels)
        if scale_factor is not None:
            h_bar = math.floor(height / beta / scale_factor) * scale_factor
            w_bar = math.floor(width / beta / scale_factor) * scale_factor
        else:
            h_bar = math.floor(height / beta)
            w_bar = math.floor(width / beta)
    if image_min_pixels is not None and h_bar * w_bar < image_min_pixels:
        beta = math.sqrt(image_min_pixels / (height * width))
        if scale_factor is not None:
            h_bar = math.ceil(height * beta / scale_factor) * scale_factor
            w_bar = math.ceil(width * beta / scale_factor) * scale_factor
        else:
            h_bar = math.ceil(height * beta)
            w_bar = math.ceil(width * beta)
    image = image.resize((w_bar, h_bar))
    return image


def load_image_from_path(image: str, **kwargs):
    if image.startswith("http://") or image.startswith("https://"):
        response = requests.get(image, stream=True)
        image_obj = Image.open(BytesIO(response.content))
    else:
        image_obj = Image.open(image)
    return image_obj.convert("RGB")


def load_image_from_bytes(image: bytes, **kwargs):
    return Image.open(BytesIO(image)).convert("RGB")


def load_image(image: ImageInput, **kwargs):
    if isinstance(image, str):
        return load_image_from_path(image, **kwargs)
    elif isinstance(image, bytes):
        return load_image_from_bytes(image, **kwargs)
    else:
        raise NotImplementedError


def fetch_images(images: List[ImageInput], **kwargs):
    images = [load_image(image) for image in images]
    max_image_nums = kwargs.get("max_image_nums", len(images))
    images = images[:max_image_nums]
    images = [smart_resize(image, **kwargs) for image in images]
    return images
