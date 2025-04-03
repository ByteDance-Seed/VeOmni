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

import torch
from PIL import Image
from transformers import BatchFeature
from transformers.image_utils import ImageInput

from ....transformers.movqgan.processing_movqgan import MoVQGANProcessor
from ..base import BaseDecoderProcessorMixin


class MoVQGANDecoderProcessor(BaseDecoderProcessorMixin, MoVQGANProcessor):
    valid_kwargs = BaseDecoderProcessorMixin.valid_kwargs + ["image_size"]

    def __init__(self, token_size=[1, 32, 32], token_num=1024, image_size=256, **kwargs):
        super().__init__(token_size=token_size, token_num=token_num, image_size=image_size, **kwargs)

    def process(
        self,
        images: Optional[ImageInput] = None,
        return_tensors: str = "pt",
    ) -> BatchFeature:
        features = super().__call__(images=images, return_tensors=return_tensors)["features"]
        image_grid_thw = torch.tensor([self.token_size] * len(images)).type(torch.int32)
        num_image_tokens = [self.token_num] * len(images)
        return BatchFeature(
            data={"features": features, "num_tokens": num_image_tokens, "grid_thw": image_grid_thw},
            tensor_type=return_tensors,
        )

    def postprocess(
        self,
        images: Optional[ImageInput] = None,
        return_tensors: str = "Image",
    ):
        return_image_list = []
        for image in images:
            post_image = image.permute(1, 2, 0)
            post_image = (post_image + 1) * 127.5
            post_image = post_image.to(dtype=torch.uint8)
            if return_tensors == "Image":
                post_image = post_image.detach().cpu().numpy()
                post_image = Image.fromarray(post_image)
            return_image_list.append(post_image)
        if return_tensors != "Image":
            return_image_list = torch.stack(return_image_list, dim=0)
        return return_image_list
