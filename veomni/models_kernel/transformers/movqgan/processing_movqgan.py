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
# limitations under the License.from typing import Optional

from typing import Optional

import numpy as np
import torch
from transformers import BatchFeature, ProcessorMixin
from transformers.image_transforms import center_crop, resize
from transformers.image_utils import ImageInput, PILImageResampling, make_list_of_images, to_numpy_array


class MoVQGANProcessor(ProcessorMixin):
    attributes = []

    def __init__(self, image_size: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size

    def transform(self, image):
        crop_size = min(image.shape[0], image.shape[1])
        image = center_crop(image, (crop_size, crop_size), input_data_format="channels_last")
        image = resize(
            image,
            (self.image_size, self.image_size),
            resample=PILImageResampling.BICUBIC,
            reducing_gap=1,
            input_data_format="channels_last",
        )
        image = image.astype(np.float32) / 127.5 - 1.0
        return np.transpose(image, [2, 0, 1])

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        return_tensors: str = "pt",
    ) -> torch.Tensor:
        images = make_list_of_images(images)
        images = [to_numpy_array(image) for image in images]
        pixel_values = [self.transform(image) for image in images]
        return BatchFeature(
            data={"features": pixel_values},
            tensor_type=return_tensors,
        )
