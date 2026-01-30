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

from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import resize
from transformers.image_utils import make_list_of_images, to_numpy_array

from ....utils import logging


logger = logging.get_logger(__name__)

ImageType = Union[np.ndarray, torch.Tensor, Image.Image]


def expand2square(np_img: np.ndarray, background_color):
    pil_img = Image.fromarray(np_img)
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class JanusImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        image_size: int,
        min_size: int = 14,
        image_mean: Union[Tuple[float, float, float], List[float]] = (
            0.48145466,
            0.4578275,
            0.40821073,
        ),
        image_std: Union[Tuple[float, float, float], List[float]] = (
            0.26862954,
            0.26130258,
            0.27577711,
        ),
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean
        self.image_std = image_std
        self.min_size = min_size
        self.do_normalize = do_normalize

        if image_mean is None:
            self.background_color = (127, 127, 127)
        else:
            self.background_color = tuple([int(x * 255) for x in image_mean])

    def resize(self, pil_img: np.ndarray) -> np.ndarray:
        height, width, _ = pil_img.shape
        max_size = max(width, height)

        size = [
            max(int(height / max_size * self.image_size), self.min_size),
            max(int(width / max_size * self.image_size), self.min_size),
        ]

        if width <= 0 or height <= 0 or size[0] <= 0 or size[1] <= 0:
            print(f"orig size = {pil_img.shape}, new size = {size}")
            raise ValueError("Invalid size!")
        pil_img = resize(pil_img, size)

        pil_img = expand2square(pil_img, self.background_color)
        return to_numpy_array(pil_img)

    def preprocess(self, images, image_type: str = "input", return_tensors: str = "pt", **kwargs) -> BatchFeature:
        images = make_list_of_images(images)
        images = [to_numpy_array(image) for image in images]
        if image_type == "output":
            images = [resize(image, (self.image_size, self.image_size)) for image in images]
            images = [self.rescale(image=image, scale=self.rescale_factor) for image in images]
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]
        elif image_type == "input":
            images: List[np.ndarray] = [self.resize(image) for image in images]
            images = [self.rescale(image=image, scale=self.rescale_factor) for image in images]
            # normalize
            if self.do_normalize:
                images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]
        else:
            raise ValueError(f"image_type = {image_type} is not supported!")
        images = [np.transpose(image, (2, 0, 1)) for image in images]
        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)

    @property
    def default_shape(self):
        return [3, self.image_size, self.image_size]
