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

from typing import Optional

import numpy as np
import torch
from transformers import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput, make_list_of_images, to_numpy_array


class CosmosProcessor(BaseImageProcessor):
    model_input_names = ["images"]

    def __init__(self, max_pixels=None, **kwargs):
        super().__init__(**kwargs)
        self.max_pixels = max_pixels

    def pad_image_batch(self, batch: np.ndarray, spatial_align: int = 16) -> tuple[np.ndarray, list[int]]:
        """Pads a batch of images to be divisible by `spatial_align`.

        Args:
            batch: The batch of images to pad, layout BxHxWx3, in any range.
            align: The alignment to pad to.
        Returns:
            The padded batch and the crop region.
        """
        height, width = batch.shape[1:3]
        align = spatial_align
        height_to_pad = (align - height % align) if height % align != 0 else 0
        width_to_pad = (align - width % align) if width % align != 0 else 0

        crop_region = [
            height_to_pad >> 1,
            width_to_pad >> 1,
            height + (height_to_pad >> 1),
            width + (width_to_pad >> 1),
        ]
        batch = np.pad(
            batch,
            (
                (0, 0),
                (height_to_pad >> 1, height_to_pad - (height_to_pad >> 1)),
                (width_to_pad >> 1, width_to_pad - (width_to_pad >> 1)),
                (0, 0),
            ),
            mode="constant",
        )
        return batch, crop_region

    def unpad_image_batch(self, batch: np.ndarray, crop_region: list[int]) -> np.ndarray:
        """Unpads image with `crop_region`.

        Args:
            batch: A batch of numpy images, layout BxHxWxC.
            crop_region: [y1,x1,y2,x2] top, left, bot, right crop indices.

        Returns:
            np.ndarray: Cropped numpy image, layout BxHxWxC.
        """
        assert len(crop_region) == 4, "crop_region should be len of 4."
        y1, x1, y2, x2 = crop_region
        return batch[..., y1:y2, x1:x2, :]

    def transform(
        self,
        image: np.array,
    ):
        images = np.expand_dims(image, axis=0)
        padded_input_image, crop_region = self.pad_image_batch(images)
        padded_input_image = padded_input_image.astype(np.float32) / 127.5 - 1.0
        padded_input_image = np.transpose(padded_input_image, (0, 3, 1, 2))
        return padded_input_image, crop_region

    def postprocess(
        self,
        images: Optional[ImageInput] = None,
        crop_region=None,
    ):
        images = images.permute(1, 2, 0)
        images = (images + 1) * 127.5
        images = images.detach().cpu().numpy().astype(np.uint8)
        images = self.unpad_image_batch(images, crop_region=crop_region)
        return images

    def preprocess(
        self,
        images: Optional[ImageInput] = None,
        return_tensors: str = "pt",
    ) -> torch.Tensor:
        raise NotImplementedError  # TODO: any_res?
        images = make_list_of_images(images)
        images = [to_numpy_array(image) for image in images]
        pixel_values, crop_region, num_image_tokens = [], [], []
        for image in images:
            pixel_value, crop_region = self.transform(image)
            pixel_values.append(pixel_value)
            crop_region.append(crop_region)

        num_image_tokens = [1024]
        return BatchFeature(
            data={"features": pixel_values, "num_tokens": num_image_tokens, "crop_region": crop_region},
            tensor_type=return_tensors,
        )
