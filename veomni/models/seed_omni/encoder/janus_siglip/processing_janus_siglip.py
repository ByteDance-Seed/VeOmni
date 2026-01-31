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

import inspect
from typing import List, Optional

from transformers import BatchFeature
from transformers.image_utils import ImageInput

from ....transformers.janus.image_processing_janus import JanusImageProcessor
from ..base import BaseEncoderProcessorMixin


class JanusSiglipEncoderProcessor(BaseEncoderProcessorMixin, JanusImageProcessor):
    valid_kwargs = BaseEncoderProcessorMixin.valid_kwargs + list(
        inspect.signature(JanusImageProcessor.__init__).parameters.keys()
    )

    def __init__(
        self,
        token_size=[1, 24, 24],
        token_num=576,
        image_size: int = 384,
        min_size: int = 14,
        image_mean: List[float] = (
            0.48145466,
            0.4578275,
            0.40821073,
        ),
        image_std: List[float] = (
            0.26862954,
            0.26130258,
            0.27577711,
        ),
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        **kwargs,
    ):
        BaseEncoderProcessorMixin.__init__(self, token_num=token_num, token_size=token_size, **kwargs)
        JanusImageProcessor.__init__(
            self,
            image_size=image_size,
            min_size=min_size,
            image_mean=image_mean,
            image_std=image_std,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            **kwargs,
        )

    def process(self, images: Optional[ImageInput] = None, return_tensors: str = "pt", **kwargs) -> BatchFeature:
        features = super().preprocess(images=images, image_type="input", return_tensors=return_tensors)["pixel_values"]
        num_image_tokens = [self.token_num] * len(images)
        return BatchFeature(
            data={"features": features, "num_tokens": num_image_tokens},
            tensor_type=return_tensors,
        )
