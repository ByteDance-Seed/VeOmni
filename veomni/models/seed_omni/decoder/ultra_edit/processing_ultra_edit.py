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

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from transformers import BaseImageProcessor, BatchFeature
from transformers.image_transforms import resize
from transformers.image_utils import (
    ImageInput,
    get_image_size,
    make_list_of_images,
    to_numpy_array,
)


def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """
        Copied from qwen2vl.imageprocessor.smart_resize.
        Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class UltraEditProcessor(BaseImageProcessor):
    model_input_names = ["images"]

    def __init__(self, vae_scale_factor: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.vae_scale_factor = vae_scale_factor
        self.max_pixels = 512
        self.patch_size = 14
        self.merge_size = 2
        self.image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    def to_dict(self):
        encoder_dict = super().to_dict()
        encoder_dict.pop("image_processor")
        return encoder_dict

    def transform(self, image: ImageInput) -> np.array:
        height, width = get_image_size(image)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=self.patch_size * self.merge_size,
            max_pixels=self.max_pixels * self.max_pixels,
        )
        image = resize(image, size=(resized_height, resized_width))
        image = self.image_processor.preprocess(image)
        return image

    def preprocess(
        self,
        images: Optional[ImageInput] = None,
        return_tensors: str = "pt",
    ) -> torch.Tensor:
        images = make_list_of_images(images)
        images = [to_numpy_array(image) for image in images]

        pixel_values, image_mask, image_grid_thw = [], [], []

        for image in images:
            pixel_value = self.transform(image)
            image_grid_thw.append(
                torch.tensor([1, pixel_value.shape[2] // self.patch_size, pixel_value.shape[3] // self.patch_size])
            )
            pad_height = (self.max_pixels - pixel_value.shape[2]) // 2
            pad_width = (self.max_pixels - pixel_value.shape[3]) // 2
            mask_img = torch.ones_like(pixel_value)
            padding = (pad_width, pad_width, pad_height, pad_height)
            pixel_value = F.pad(pixel_value, padding, mode="constant", value=0)
            mask_img = F.pad(mask_img, padding, mode="constant", value=0)
            pixel_values.append(pixel_value)
            image_mask.append(mask_img)

        pixel_values = torch.cat(pixel_values, dim=0)
        image_mask = torch.cat(image_mask, dim=0)
        image_grid_thw = torch.stack(image_grid_thw, dim=0)
        return BatchFeature(
            data={"features": pixel_values, "mask": image_mask, "image_grid_thw": image_grid_thw},
            tensor_type=return_tensors,
        )
